# Cosmos-Reason-2-Speed-Blocker
-> This project provides a patch to fix the embedding layer slowness on Qwen3VL when processing images or videos
-> Result: time to first token from ~150 s to 2.19 s (before the first word)

---

## 1 · Context

### 1.1 The big picture and the task

I wanted to show, hands-on, how NVIDIA's **Cosmos Reason 2** (a VLM: Vision-Language Model) understands driving scenes. I was using the model `Qwen3VL` I passed **10 clips** from NVIDIA's PhysicalAI-AV dataset (night, rain, road works, pedestrians) through the model on my RTX 4090 laptop (16 GB) and asked it to answer the same 3 questions for each clip:

1. *Describe what happens in this clip, in time order.*
2. *Is it safe for the ego vehicle to change lanes to the left at the end of this clip? Explain.*
3. *Which road user matters most for the ego vehicle's next action, and why?*

For every answer, I measured the speed and memory, then graded it by hand.

### 1.2 How I implemented it (to reproduce)

1. **Environment:** `conda create -n env_cosmos python=3.11`, then `pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128` and `pip install transformers==4.57.3 accelerate==1.12.0 av==16.1.0 pillow==12.0.0` (NVIDIA's pinned versions), then `hf auth login`.
2. Implement Cosmos 2 in your local machine: https://github.com/marcelpatrick/Run-Cosmos-Reason-2-on-Driving-Clips 
    2.1. **`cr2.py`:** `load()` loads the model in fp16; `ask()` sends the video then the question and times the first word (`ttft_s`), the writing speed and peak memory.
    2.2. **`bench.py`:** loops 10 clips × 3 questions and writes `results\benchmark.csv`.
3. **Reproduce the problem:** Run the smoke test with only one image: `python smoke_image.py` (one photo) or `python bench.py`, then look at `ttft_s`.

More on project: https://github.com/marcelpatrick/Run-Cosmos-Reason-2-on-Driving-Clips 
---

## 2 · The problem

### 2.1 Symptom

When I ran the 10 clips through the model, I noticed it was taking a very long time (~1hr) to analyze only 10 clips with ~8sec each.

Then I ran the smoke test `python smoke_image.py` which takes only 1 image and describes it. The same problem was observed. I noticed on the performance metrics that every answer waited **~100–150s** before the first word (**time to first token - ttft**), even for one photo (107 s). Once it started, writing was normal (~30 tokens/s), and memory was fine (~7 GB). So the model wasn't slow at *writing*; something before the first word was.

smoke_image.py:
```py
# smoke_image.py: one image, one question, all numbers printed
from pathlib import Path
from cr2 import load, ask

IMG = Path(r"C:\projects\av_world_models\test\street.jpg")
model, proc = load()

r = ask(model, proc, IMG, "What should a driver pay attention to in this scene, and why?",
        kind="image")
print("-" * 60)
print(r["answer"])
print("-" * 60)
for k in ("input_tokens", "output_tokens", "prep_s", "ttft_s", "tok_per_s",
          "total_s", "peak_vram_gb", "reserved_vram_gb", "hit_limit"):
    print(f"{k:18}{r[k]}")
```

![Before the fix: bench.py](img/1_before_bench.png)

### 2.2 Root cause
- The problem was a major lag when converting the image to tokens (time before generating the first word)

#### 2.2.1 How I found it

** I measured each stage.** using a diagnostic script `diag.py` puts a stopwatch (PyTorch hooks) around the vision encoder only: **122.08 of 149.08 s** were inside it.

diag.py
```py
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  diag.py · Where do the ~130 s before the first word go?                       ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  WHAT  Puts a stopwatch around ONE layer, the vision encoder (pixels → tokens),║
# ║        then asks short questions on a photo and a clip and prints that layer's ║
# ║        time next to the total wait for the first word.                         ║
# ║  WHY   Fixing the memory spill didn't change the speed, so I stopped guessing  ║
# ║        and measured each stage. Result: 153.4 of 154.3 s in the vision encoder ║
# ║        → matches a known PyTorch 2.9.0 slow path for its first (Conv3d) layer. ║
# ║  RUN   conda activate env_cosmos → cd C:\projects\av_world_models             ║
# ║        python diag.py                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Summary
#   To find why answers took two minutes, diag attaches two hooks, small timers PyTorch
#    runs just before and just after a chosen layer, to the vision encoder. Then it asks
#    one question on a photo and on a clip, and prints the vision time next to the total
#    wait: 153 of 154 seconds were in that one step.
#
# FAQ
#   · How did you find the bottleneck?        → STEP 2 (hooks)
#   · Why call torch.cuda.synchronize()?      → STEP 2 (GPU work is asynchronous)


# diag.py: where do the ~130 s before the first word go?
import time

import torch

from cr2 import load, ask

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 · Inputs: one photo and one clip, then load the model
# ──────────────────────────────────────────────────────────────────────────────
ROOT = r"C:\projects\av_world_models"
IMG = ROOT + r"\test\street.jpg"
VID = ROOT + r"\clips\clip70.mp4"

model, proc = load()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 · Attach two hooks (timers) to the vision encoder
#   A hook is a function PyTorch runs automatically when a layer runs:
#   pre-hook = just before, forward hook = just after. They must RETURN NOTHING,
#   otherwise PyTorch replaces the layer's input/output (my first version's bug).
#   synchronize() waits for the GPU to finish, because GPU work runs in the background;
#   without it the stopwatch would stop before the GPU was actually done.
# ──────────────────────────────────────────────────────────────────────────────
vis = model.model.visual                     # the vision encoder (frames → tokens)
t = {}


def start_timer(module, inputs):             # runs just before the vision encoder
    torch.cuda.synchronize()
    t["v0"] = time.perf_counter()            # returns nothing → inputs untouched


def stop_timer(module, inputs, output):      # runs just after the vision encoder
    torch.cuda.synchronize()
    t["vision_s"] = time.perf_counter() - t["v0"]   # returns nothing → output untouched


vis.register_forward_pre_hook(start_timer)
vis.register_forward_hook(stop_timer)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 · Warm-up (not printed): pay one-off GPU setup costs first
# ──────────────────────────────────────────────────────────────────────────────
ask(model, proc, IMG, "Hi.", kind="image", reasoning=False, max_new_tokens=4)   # warm-up

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 · Three tests: a photo, the clip at 1 fps, the clip at 2 fps
#   Short answers (32 tokens, no reasoning) so almost all time is the "reading" phase.
#   Output line: input tokens · prep · vision encoder · first token · total
# ──────────────────────────────────────────────────────────────────────────────
tests = [
    ("image", dict(media_path=IMG, kind="image")),
    ("video fps=1", dict(media_path=VID, fps=1)),
    ("video fps=2", dict(media_path=VID, fps=2)),
]

for label, kw in tests:
    t.clear()
    r = ask(model, proc, question="Describe the scene in one sentence.",
            reasoning=False, max_new_tokens=32, **kw)
    print(f"{label:12} input_tokens={r['input_tokens']:5}  prep={r['prep_s']:6.2f}s  "
          f"vision={t.get('vision_s', float('nan')):7.2f}s  "
          f"first_token={r['ttft_s']}s  total={r['total_s']}s")
```

<img width="1502" height="618" alt="image" src="https://github.com/user-attachments/assets/95fc5e06-7efc-4f49-9530-7281817b3d35" />


#### 2.2.2 What caused it, and why: A PyTorch version issue

Before the language model can read an image, the **vision encoder** turns pixels into tokens. Its first layer is a **3D convolution (Conv3d)** that cuts frames into 16×16-pixel patches (2 frames deep). In **PyTorch 2.9.0**, the version NVIDIA's setup pins, this Conv3d takes a **very slow GPU code path in half precision (fp16)**, a known regression reported by other Qwen3-VL users: 
. https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940
. https://github.com/pytorch/pytorch/issues/166643
. https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210

The giveaway: the GPU showed ~97% "busy" while drawing only ~34–54 W of 175 W. It was mostly waiting, not computing. The model, the weights and the laptop were fine; one library code path was the bottleneck.

---

## 3 · The solution 

There are 2 types of fixes: 
  1. Swap the PyTorch version from 2.9.0 to 2.9.1.
  2. Keep PyTorch version 2.9.0 and implement a patch

I chose the 2nd because it allowed me to keep PyTorch's version pinned by NVIDIA and I was afraid that changing a version mid-project would break something else. 

### The Patch:

This Conv3d's **kernel equals its stride**: each patch is seen exactly once, with no overlap. That makes it mathematically identical to a plain **matrix multiply**, so I replaced only that layer's forward step with one.

Befor loading the model: **Paste just above `def load(`**:

```python
# ══════════════════════════════════════════════════════════════════════════════
# Patch to make shortten time to first token (ttft)
# ══════════════════════════════════════════════════════════════════════════════

### PASTE:
import types
def _fast_patch_embed(model):
    """PyTorch 2.9.0 runs this Conv3d slowly in fp16. kernel == stride, so it is a matrix multiply."""
    pe = model.model.visual.patch_embed
    def forward(self, hidden_states):
        w = self.proj.weight
        x = hidden_states.reshape(-1, w[0].numel()).to(w.dtype)
        return torch.nn.functional.linear(x, w.reshape(w.shape[0], -1), self.proj.bias)
    pe.forward = types.MethodType(forward, pe)
###

# ══════════════════════════════════════════════════════════════════════════════
# load(): put the model and its processor on the GPU, once
# ══════════════════════════════════════════════════════════════════════════════
def load(max_vision_tokens: int = 8192):

```
**Paste inside `load()`, right after `model = ...from_pretrained(...)`:**
```python
def load(max_vision_tokens: int = 8192):
 
    # ── STEP 1: Load model + processor. 
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )

    ### PASTE: apply patch to the model to reduce ttft
    _fast_patch_embed(model)
    ###

    # ── STEP 2 · Load the processor  
    proc = transformers.Qwen3VLProcessor.from_pretrained(MODEL)
```

If you are following this tutorial (https://github.com/marcelpatrick/Run-Cosmos-Reason-2-on-Driving-Clips), these changes are to be applied inside file `cr2.py`

### Result
**Result:** the first word arrives after **2.19 s** instead of 107–153 s (~50–70× faster). It worked because matrix multiplies run on PyTorch's fast, well-optimized GPU path. The fix is the same maths, the same weights, and changes only one layer.

<img width="1498" height="547" alt="image" src="https://github.com/user-attachments/assets/e7ca0670-c860-4385-9680-67787db2d83c" />


