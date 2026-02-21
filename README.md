# MCP Server built with mcp-use

This is an MCP server project bootstrapped with [`create-mcp-use-app`](https://mcp-use.com/docs/typescript/getting-started/quickstart).

## Getting Started

First, run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000/inspector](http://localhost:3000/inspector) with your browser to test your server.

You can start building by editing the entry file. Add tools, resources, and prompts — the server auto-reloads as you edit.

## Learn More

To learn more about mcp-use and MCP:

- [mcp-use Documentation](https://mcp-use.com/docs/typescript/getting-started/quickstart) — guides, API reference, and tutorials

## Runpod MCP Tools

The server exposes a `runpod-login-deploy` tool that:

1. Configures `runpodctl` with your API key.
2. Creates a pod using `runpodctl create pod`.

Requirements:

- `runpodctl` installed and available on your `PATH`.

Inputs (high level):

- `apiKey` (required)
- `gpuType` (required)
- `imageName` or `templateId` (required)
- Optional flags like `gpuCount`, `secureCloud`, `communityCloud`, `volumeSize`, `env`, `ports`

The server also exposes:

- `runpod-pod-ssh-info`: generates a fresh SSH keypair, adds the public key to Runpod, and returns the public key.
- `runpod-pod-run`: runs a single command on a pod over SSH and returns stdout/stderr.

Common inputs:

- `podId` (required)
- `ensureKey` (optional): add an SSH key via `runpodctl ssh add-key`
- `generateKey` (optional, `runpod-pod-ssh-info`): generate a new keypair first (default true)
- `keyPath` (optional, `runpod-pod-ssh-info`): private key path (default `~/.ssh/id_ed25519`)
- `key` / `keyFile` (optional): SSH public key contents or file path (if you don't want to generate)
- `host` / `port` / `user` (optional overrides if SSH details cannot be parsed)
- `identityFile` (optional, `runpod-pod-run`): SSH private key path
- `allocatePty` (optional, `runpod-pod-run`): request a PTY (default true)
- `timeoutMs` (optional, `runpod-pod-run`): SSH process timeout in ms (default 15000)
- `mode` (optional, `runpod-pod-run`): `proxy` (ssh.runpod.io, default) or `direct` (public-IP SSH)
- `skipHostKeyCheck` (optional, `runpod-pod-run`): disable strict host key checking

## Deploy on Manufact Cloud

```bash
npm run deploy
```
