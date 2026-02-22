# GPU and ML MCP
Visual, end‑to‑end ML model design inside an MCP server: drag blocks to build a network, validate shapes, generate runnable PyTorch, and push training/inference to a RunPod GPU pod — all from a single interactive widget.

## What It Does

- Drag‑and‑drop architecture builder with connectors, properties panel, and composite blocks.
- Graph validation for shape compatibility before code generation.
- One‑click PyTorch generation (model + training/inference scaffolds).
- Training setup with dataset presets, optimizer/loss selection, and script generation.
- RunPod integration for GPU setup, script upload, and remote inference.

## How It Works (MCP Flow)

1. Call `design-architecture` (optional) to auto‑layout a graph.
2. Call `render-model-builder` to open the visual builder.
3. Build/edit the architecture in the UI and save automatically.
4. Call `generate-pytorch-code` for runnable PyTorch.
5. Use training/inference tools to generate scripts and run on RunPod.

## Tech Stack

- MCP server built with `mcp-use`
- React + TypeScript UI widgets (Vite)
- Zod for schemas and validation
- Tailwind for styling
- RunPod via `runpodctl` + SSH

## Quickstart

```bash
npm install
npm run dev
```

Open `http://localhost:3000/inspector` and invoke `render-model-builder`.

## Environment

- `RUNPOD_API_KEY` (optional): enables RunPod tools and auto‑configures `runpodctl`.
- `WORKOS_SUBDOMAIN`, `WORKOS_CLIENT_ID`, `WORKOS_API_KEY` (optional): enable OAuth.

## MCP Tools (Core)

- `render-model-builder`: opens the visual architecture builder widget.
- `generate-pytorch-code`: validates and generates PyTorch from the saved graph.
- `generate-training-code`: creates model/train/data scripts + requirements.
- `generate-inference-code`: creates inference script for the current model.
- `show-next-steps`: action panel after design (generate code, train, edit, explain).
- `validate-graph`: returns graph validation errors.

## MCP Tools (RunPod)

- `runpod-login-deploy`: configure `runpodctl`, create a pod, and add SSH keys.
- `runpod-pod-run`: run a single command on a pod over SSH.
- `setup-gpu`: install Python deps on a pod using generated requirements.
- `upload-scripts`: write generated scripts to a pod via SSH.
- `run-inference`: upload and execute inference on a pod.

## Project Structure

- `index.ts`: MCP server, tool definitions, graph validation, code generation.
- `resources/ml-architecture-builder/`: main visual builder widget.
- `resources/training-deploy.tsx`: training setup + RunPod deploy UI.
- `resources/next-steps.tsx`: post‑design action panel widget.

## Deployment

```bash
npm run deploy
```

The `Dockerfile` installs `runpodctl` for RunPod‑enabled deployments.