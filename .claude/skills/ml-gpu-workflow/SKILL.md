---
name: ml-gpu-workflow
description: |
  ML architecture → training → GPU deployment → inference workflow for the ACP ML Architecture Builder.
  Use when the user asks to:
  - Train a model on a GPU / RunPod pod
  - Upload training scripts to the pod
  - Set up dependencies on the pod
  - Run inference on the GPU
  - Check what packages are installed on the pod
  - Deploy the full ML pipeline end-to-end
---

# ML GPU Workflow — ACP Architecture Builder

This skill guides you through the complete workflow: design → train setup → GPU prep → training → inference.

## Tool Reference

| Tool | Purpose |
|------|---------|
| `generate-training-code` | Generate training scripts with W&B integration |
| `show-training-deploy` | **Show GPU deployment widget after generate-training-code** |
| `check-gpu-packages` | Read what pip packages are installed on the pod |
| `setup-gpu` | Install missing packages (smart diff, skips already-installed) |
| `upload-scripts` | SCP model.py / data.py / train.py / inference.py to pod |
| `generate-inference-code` | Generate task-aware inference.py |
| `run-inference` | Upload + execute inference on GPU pod, return JSON |
| `get-training-code` | Retrieve previously generated scripts |

---

## Standard GPU Deployment Workflow

### Phase 1: Generate Scripts + Show Deploy Panel
```
1. prepare-train           → Opens Train tab in architecture builder
2. (user picks dataset, configures training in Train tab)
3. generate-training-code  → Creates model.py / data.py / train.py / requirements.txt
4. show-training-deploy    → ← MANDATORY: shows GPU deployment widget immediately
```

The `training-deploy` widget handles GPU setup interactively:
- Shows generated file list + W&B project
- Pod ID input + "Deploy All" one-click button
- Step-by-step: check packages → install deps → upload scripts
- Inference generation + quick run at the bottom

### Phase 2: Manual GPU Setup (if not using widget)
```
1. check-gpu-packages     → Read what's already installed
2. setup-gpu              → Install ONLY what is missing (reads requirements.txt + pip freeze)
3. upload-scripts         → SCP all scripts to /workspace
```

### Phase 3: Training
```
SSH into pod and run:
  cd /workspace && python3 train.py

W&B run URL is printed to stdout:
  WANDB_RUN_URL: https://wandb.ai/...
```

### Phase 4: Inference
```
Option A — Via MCP tool:
  run-inference → uploads + executes → returns JSON

Option B — Widget:
  Switch to "Infer" tab → enter input → click "Run on GPU"

Option C — Manual SSH:
  echo '{"input": "text"}' | python3 /workspace/inference.py
```

---

## Package Policy (CRITICAL)

**NEVER install packages that are likely already installed.**

RunPod PyTorch images (`runpod/pytorch:*`) ship with:
- torch, torchvision, torchaudio
- numpy, scipy, matplotlib
- Pillow (PIL), requests, aiohttp
- cuda drivers, cudnn

**Always call `check-gpu-packages` first.** The `setup-gpu` tool does this automatically when `force` is not set.

```
# DO:
check-gpu-packages → see what's missing → setup-gpu (installs only missing)

# DON'T:
setup-gpu with force=true  (reinstalls everything, wastes time)
```

---

## Common Input Formats for Inference

| Task | Input Type | Example payload |
|------|-----------|----------------|
| LLM / NLP | `text` | `{"input": "Once upon a time"}` |
| Vision | `image_url` | `{"image_url": "https://..."}` |
| Vision | `image_base64` | `{"image_base64": "<base64>"}` |
| Tabular | `tabular` | `{"input": [5.1, 3.5, 1.4, 0.2]}` |

---

## W&B Integration

`generate-training-code` with `wandb: { project: "my-model" }` automatically adds:
- `wandb.init()` with full hyperparameter config
- Per-epoch `wandb.log()` with train/val loss
- `wandb.save()` on checkpoint
- Prints `WANDB_RUN_URL: <url>` to stdout (capture this from SSH output)

---

## Error Patterns

| Error | Likely cause | Fix |
|-------|-------------|-----|
| SCP fails | SSH key not added to pod | Run `runpodctl ssh add-key` or check `~/.ssh/id_ed25519.pub` |
| `ModuleNotFoundError: model` | model.py not uploaded | Call `upload-scripts` |
| `FileNotFoundError: checkpoint.pt` | No checkpoint saved yet | inference runs with random weights — train first |
| pip install timeout | Large packages on slow connection | Increase `timeoutMs` in setup-gpu |

---

## Quick Reference: Full Pipeline (one-shot)

When the user says "set up training and deploy to GPU", do ALL of these:

1. `generate-training-code` (if not already done)
2. `generate-inference-code` (auto-detects task type)
3. `check-gpu-packages` → note what's missing
4. `setup-gpu` → installs missing packages
5. `upload-scripts` → uploads all files to /workspace
6. Tell user: "Scripts are ready. SSH in and run `python3 train.py`"
