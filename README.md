# Cosmos-Reason-2-Speed-Blocker
-> from ~150 s to 2.19 s before the first word

---

## 1 · Context

### 1.1 The big picture and the task

I wanted to show, hands-on, how NVIDIA's **Cosmos Reason 2** (a 2B vision-language model) understands driving scenes. I passed **10 clips** from NVIDIA's PhysicalAI-AV dataset (night, rain, road works, pedestrians) through the model on my RTX 4090 laptop (16 GB) and asked each clip the same 3 questions:

1. *Describe what happens in this clip, in time order.*
2. *Is it safe for the ego vehicle to change lanes to the left at the end of this clip? Explain.*
3. *Which road user matters most for the ego vehicle's next action, and why?*

For every answer I measured the speed and memory, then graded it by hand.

### 1.2 How I implemented it (to reproduce)

1. **Environment:** `conda create -n env_cosmos python=3.11`, then `pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128` and `pip install transformers==4.57.3 accelerate==1.12.0 av==16.1.0 pillow==12.0.0` (NVIDIA's pinned versions), then `hf auth login`.
2. **`cr2.py`:** `load()` loads the model in fp16; `ask()` sends the video then the question and times the first word (`ttft_s`), the writing speed and peak memory.
3. **`bench.py`:** loops 10 clips × 3 questions and writes `results\benchmark.csv`.
4. **Reproduce the problem:** `python smoke_image.py` (one photo) or `python bench.py`, then look at `ttft_s`.

---

## 2 · The problem

### 2.1 Symptom

Every answer waited **~130–150 s before the first word**, even for one photo (107 s). Once it started, writing was normal (~30 tokens/s), and memory was fine (~7 GB). So the model wasn't slow at *writing*; something before the first word was.

![Before the fix: bench.py](img/1_before_bench.png)

### 2.2 Root cause

#### 2.2.1 How I found it

1. **The first suspect was memory.** Task Manager showed GPU memory overflowing into system RAM (15.6 GB dedicated + 4.6 GB shared). I fixed that with `torch.cuda.empty_cache()`, but **the wait didn't change**, so memory wasn't the cause.
2. **Then I measured each stage.** `diag.py` puts a stopwatch (PyTorch hooks) around the vision encoder only: **153.4 of 154.3 s** were inside it.

![Memory spill (fixed, but not the cause)](img/task_manager_memory_spill.png)
![diag.py: the vision encoder takes almost the whole wait](img/2_diag.png)

#### 2.2.2 What caused it, and why

Before the language model can read an image, the **vision encoder** turns pixels into tokens. Its first layer is a **3D convolution (Conv3d)** that cuts frames into 16×16-pixel patches (2 frames deep). In **PyTorch 2.9.0**, the version NVIDIA's setup pins, this Conv3d takes a **very slow GPU code path in half precision (fp16)**, a known regression reported by other Qwen3-VL users. The giveaway: the GPU showed ~97% "busy" while drawing only ~34–54 W of 175 W. It was mostly waiting, not computing. The model, the weights and the laptop were fine; one library code path was the bottleneck.

---

## 3 · The solution, and why it worked

This Conv3d's **kernel equals its stride**: each patch is seen exactly once, with no overlap. That makes it mathematically identical to a plain **matrix multiply**, so I replaced only that layer's forward step with one.

**Paste in `cr2.py`, just above `def load(`** (`import torch` is already there):
```python
import types

def _fast_patch_embed(model):
    pe = model.model.visual.patch_embed
    def forward(self, hidden_states):
        w = self.proj.weight
        x = hidden_states.reshape(-1, w[0].numel()).to(w.dtype)
        return torch.nn.functional.linear(x, w.reshape(w.shape[0], -1), self.proj.bias)
    pe.forward = types.MethodType(forward, pe)
```
**Paste inside `load()`, right after `model = ...from_pretrained(...)`:**
```python
    _fast_patch_embed(model)
```

**Result:** the first word arrives after **2.19 s** instead of 107–153 s (~50–70× faster). It worked because matrix multiplies run on PyTorch's fast, well-optimized GPU path. The fix is the same maths, the same weights, and changes only one layer.

![Before and after](img/3_before_after.png)
