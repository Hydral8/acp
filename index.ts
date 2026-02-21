import { MCPServer, object, text, widget } from "mcp-use/server";
import { execFile, spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { z } from "zod";

const server = new MCPServer({
  name: "acp",
  title: "ML Architecture Builder",
  version: "1.0.0",
  description: "Visual ML architecture builder — drag blocks, connect layers, generate PyTorch code",
  baseUrl: process.env.MCP_URL || "http://localhost:3000",
  favicon: "favicon.ico",
  icons: [{ src: "icon.svg", mimeType: "image/svg+xml", sizes: ["512x512"] }],
});

// ── Schemas ────────────────────────────────────────────────────────────────

const innerNodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  parameters: z.record(z.string(), z.unknown()),
});

const innerEdgeSchema = z.object({
  id: z.string(),
  sourceId: z.string(),
  targetId: z.string(),
});

const compositeSchema = z.object({
  label: z.string(),
  nodes: z.array(innerNodeSchema),
  edges: z.array(innerEdgeSchema),
  inputNodeId: z.string(),
  outputNodeId: z.string(),
}).optional();

const nodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  parameters: z.record(z.string(), z.unknown()),
  composite: compositeSchema,
});

const edgeSchema = z.object({
  id: z.string(),
  sourceId: z.string(),
  targetId: z.string(),
});

const graphSchema = z.object({
  nodes: z.array(nodeSchema),
  edges: z.array(edgeSchema),
});

// ── Tool: render-model-builder ─────────────────────────────────────────────

const execFileAsync = promisify(execFile);

type RunpodctlResult = {
  stdout: string;
  stderr: string;
};

async function runpodctl(args: string[]): Promise<RunpodctlResult> {
  try {
    const { stdout, stderr } = await execFileAsync("runpodctl", args, {
      env: process.env,
    });
    return { stdout: stdout.trim(), stderr: stderr.trim() };
  } catch (err) {
    const error = err as NodeJS.ErrnoException & {
      stderr?: string;
      stdout?: string;
    };

    if (error.code === "ENOENT") {
      throw new Error(
        "runpodctl not found. Install it and ensure it is on PATH."
      );
    }

    const stderr = (error.stderr ?? "").toString().trim();
    const stdout = (error.stdout ?? "").toString().trim();
    const message = stderr || stdout || error.message || "Unknown error";
    throw new Error(`runpodctl failed: ${message}`);
  }
}

type SshInfo = {
  host?: string;
  port?: number;
  user?: string;
  command?: string;
  raw?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function scanForSshFields(value: unknown, acc: SshInfo): void {
  if (!isRecord(value) && !Array.isArray(value)) return;
  if (Array.isArray(value)) {
    for (const item of value) scanForSshFields(item, acc);
    return;
  }

  for (const [key, val] of Object.entries(value)) {
    const lower = key.toLowerCase();
    if (typeof val === "string") {
      if (!acc.command && lower.includes("ssh") && lower.includes("command")) {
        acc.command = val;
      }
      if (!acc.host && (lower.includes("sshhost") || lower.includes("host"))) {
        acc.host = val;
      }
      if (!acc.user && (lower.includes("sshuser") || lower === "user")) {
        acc.user = val;
      }
      if (!acc.port && (lower.includes("sshport") || lower === "port")) {
        const parsed = Number(val);
        if (Number.isFinite(parsed)) acc.port = parsed;
      }
    } else if (typeof val === "number") {
      if (!acc.port && (lower.includes("sshport") || lower === "port")) {
        acc.port = val;
      }
    }
    if (isRecord(val) || Array.isArray(val)) scanForSshFields(val, acc);
  }
}

function parseSshInfo(raw: string): SshInfo {
  const info: SshInfo = { raw };

  const trimmed = raw.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      scanForSshFields(parsed, info);
    } catch {
      // ignore JSON parsing errors
    }
  }

  const sshCommandMatch =
    raw.match(/ssh\s+-p\s+(\d+)\s+([^\s@]+)@([^\s]+)/i) ||
    raw.match(/ssh\s+([^\s@]+)@([^\s]+)\s+-p\s+(\d+)/i) ||
    raw.match(/ssh\s+([^\s@]+)@([^\s]+)/i);

  if (sshCommandMatch) {
    info.command = info.command || sshCommandMatch[0];
    if (!info.user) info.user = sshCommandMatch[2] ?? sshCommandMatch[1];
    if (!info.host)
      info.host =
        sshCommandMatch[3] ?? sshCommandMatch[2] ?? sshCommandMatch[1];
    if (!info.port) {
      const port =
        sshCommandMatch[1] && sshCommandMatch[0].includes("-p")
          ? Number(sshCommandMatch[1])
          : Number(sshCommandMatch[3]);
      if (Number.isFinite(port)) info.port = port;
    }
  }

  if (!info.host) {
    const hostMatch = raw.match(/ssh\s*host\s*[:=]\s*(\S+)/i);
    if (hostMatch) info.host = hostMatch[1];
  }
  if (!info.user) {
    const userMatch = raw.match(/ssh\s*user\s*[:=]\s*(\S+)/i);
    if (userMatch) info.user = userMatch[1];
  }
  if (!info.port) {
    const portMatch = raw.match(/ssh\s*port\s*[:=]\s*(\d+)/i);
    if (portMatch) info.port = Number(portMatch[1]);
  }

  if (!info.host || !info.port) {
    const portMapMatch = raw.match(
      /(\d{1,3}(?:\.\d{1,3}){3}):(\d+)->\d+\s*\((?:[^\)]*)\)/i
    );
    if (portMapMatch) {
      if (!info.host) info.host = portMapMatch[1];
      if (!info.port) info.port = Number(portMapMatch[2]);
    }
  }

  if (info.host && info.host.includes(":") && !info.port) {
    const [host, port] = info.host.split(":");
    const parsedPort = Number(port);
    if (host) info.host = host;
    if (Number.isFinite(parsedPort)) info.port = parsedPort;
  }

  return info;
}

async function ensureRunpodSshKey(key?: string, keyFile?: string) {
  const args = ["ssh", "add-key"];
  if (key) args.push("--key", key);
  if (keyFile) args.push("--key-file", keyFile);
  return runpodctl(args);
}

async function getPodSshInfo(
  podId: string,
  overrides?: { host?: string; port?: number; user?: string }
): Promise<SshInfo> {
  if (overrides?.host || overrides?.port || overrides?.user) {
    const host = overrides.host;
    const port = overrides.port;
    const user = overrides.user;
    const command =
      host && user
        ? `ssh -p ${port ?? 22} ${user}@${host}`
        : undefined;
    return { host, port, user, command };
  }

  const { stdout } = await runpodctl(["get", "pod", podId, "--allfields"]);
  const info = parseSshInfo(stdout);
  return info;
}

function resolveHomePath(filePath?: string): string | undefined {
  if (!filePath) return undefined;
  if (filePath === "~") return homedir();
  if (filePath.startsWith("~/")) return path.join(homedir(), filePath.slice(2));
  return filePath;
}

async function generateSshKeyPair(privateKeyPath: string): Promise<void> {
  await fs.mkdir(path.dirname(privateKeyPath), { recursive: true });
  await fs.rm(privateKeyPath, { force: true });
  await fs.rm(`${privateKeyPath}.pub`, { force: true });

  await execFileAsync("ssh-keygen", [
    "-t",
    "ed25519",
    "-f",
    privateKeyPath,
    "-N",
    "",
    "-C",
    "runpod",
  ]);
}

async function readPublicKey(
  key?: string,
  keyFile?: string
): Promise<{ publicKey?: string; keyFile?: string }> {
  if (key) return { publicKey: key.trim() || key, keyFile };

  const resolvedKeyFile = resolveHomePath(keyFile);
  if (resolvedKeyFile) {
    const data = await fs.readFile(resolvedKeyFile, "utf8");
    return { publicKey: data.trim(), keyFile: resolvedKeyFile };
  }

  const defaultKeyFile = path.join(homedir(), ".ssh", "id_ed25519.pub");
  try {
    const data = await fs.readFile(defaultKeyFile, "utf8");
    return { publicKey: data.trim(), keyFile: defaultKeyFile };
  } catch {
    return {};
  }
}

server.tool(
  {
    name: "runpod-login-deploy",
    description:
      "Configure runpodctl with an API key, then create a Runpod Pod.",
    schema: z
      .object({
        apiKey: z.string().describe("Runpod API key"),
        apiUrl: z
          .string()
          .optional()
          .describe("Optional custom Runpod API URL"),
        name: z.string().optional().describe("Pod name"),
        gpuType: z
          .string()
          .optional()
          .default("NVIDIA GeForce RTX 4090")
          .describe("GPU type (long form, e.g. NVIDIA GeForce RTX 4090)"),
        gpuCount: z.number().int().positive().optional().default(1),
        secureCloud: z.boolean().optional(),
        communityCloud: z.boolean().optional(),
        imageName: z
          .string()
          .optional()
          .default(
            "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
          )
          .describe("Docker image to use for the Pod"),
        templateId: z
          .string()
          .optional()
          .describe("Template ID to use for Pod configuration"),
        containerDiskSize: z.number().int().positive().optional(),
        volumeSize: z.number().int().positive().optional(),
        volumePath: z.string().optional(),
        networkVolumeId: z.string().optional(),
        cost: z.number().positive().optional(),
        mem: z.number().int().positive().optional(),
        vcpu: z.number().int().positive().optional(),
        env: z
          .array(z.string())
          .optional()
          .describe("Environment variables, e.g. KEY=VALUE"),
        args: z.string().optional().describe("Container startup args"),
        ports: z
          .array(z.string())
          .optional()
          .describe("Ports, e.g. 8888/http or 22/tcp"),
      })
      .refine((value) => value.imageName || value.templateId, {
        message: "Provide either imageName or templateId",
        path: ["imageName"],
      })
      .refine((value) => !(value.secureCloud && value.communityCloud), {
        message: "Only one of secureCloud or communityCloud can be true",
        path: ["secureCloud"],
      }),
    outputSchema: z.object({
      configStatus: z.string(),
      createStdout: z.string().optional(),
      createStderr: z.string().optional(),
    }),
  },
  async (input) => {
    const configArgs = ["config", "--apiKey", input.apiKey];
    if (input.apiUrl) {
      configArgs.push("--apiUrl", input.apiUrl);
    }
    await runpodctl(configArgs);

    const createArgs = ["create", "pod"];
    if (input.name) createArgs.push("--name", input.name);
    createArgs.push("--gpuType", input.gpuType);
    if (input.gpuCount) createArgs.push("--gpuCount", String(input.gpuCount));
    if (input.secureCloud) createArgs.push("--secureCloud");
    if (input.communityCloud) createArgs.push("--communityCloud");
    if (input.imageName) createArgs.push("--imageName", input.imageName);
    if (input.templateId) createArgs.push("--templateId", input.templateId);
    if (input.containerDiskSize)
      createArgs.push("--containerDiskSize", String(input.containerDiskSize));
    if (input.volumeSize)
      createArgs.push("--volumeSize", String(input.volumeSize));
    if (input.volumePath) createArgs.push("--volumePath", input.volumePath);
    if (input.networkVolumeId)
      createArgs.push("--networkVolumeId", input.networkVolumeId);
    if (input.cost) createArgs.push("--cost", String(input.cost));
    if (input.mem) createArgs.push("--mem", String(input.mem));
    if (input.vcpu) createArgs.push("--vcpu", String(input.vcpu));
    if (input.env) {
      for (const envVar of input.env) {
        createArgs.push("--env", envVar);
      }
    }
    if (input.args) createArgs.push("--args", input.args);
    if (input.ports) {
      for (const port of input.ports) {
        createArgs.push("--ports", port);
      }
    }

    const { stdout, stderr } = await runpodctl(createArgs);

    return object({
      configStatus: "runpodctl configured",
      createStdout: stdout || undefined,
      createStderr: stderr || undefined,
    });
  }
);

server.tool(
  {
    name: "runpod-pod-ssh-info",
    description:
      "Generate a fresh SSH keypair, add the public key to Runpod, and return the public key.",
    schema: z.object({
      podId: z
        .string()
        .optional()
        .describe("Runpod pod ID (optional; not used by this tool)"),
      ensureKey: z
        .boolean()
        .optional()
        .describe("If true, add an SSH key to Runpod"),
      generateKey: z
        .boolean()
        .optional()
        .describe("If true, generate a new SSH keypair first (default: true)"),
      keyPath: z
        .string()
        .optional()
        .describe("Path for the private key (default: ~/.ssh/id_ed25519)"),
      key: z
        .string()
        .optional()
        .describe("Public SSH key contents to add (optional)"),
      keyFile: z
        .string()
        .optional()
        .describe("Path to public SSH key file to add (optional)"),
    }),
    outputSchema: z.object({
      publicKey: z.string().optional(),
      keyFile: z.string().optional(),
      identityFile: z.string().optional(),
      addKeyStdout: z.string().optional(),
      addKeyStderr: z.string().optional(),
    }),
  },
  async (input) => {
    const identityFile =
      resolveHomePath(input.keyPath) ??
      path.join(homedir(), ".ssh", "id_ed25519");

    if (input.generateKey !== false) {
      await generateSshKeyPair(identityFile);
    }

    const publicKeyFile = `${identityFile}.pub`;
    const { publicKey, keyFile } = await readPublicKey(
      input.key,
      input.keyFile ?? publicKeyFile
    );

    if (!publicKey) {
      throw new Error(
        "No public SSH key found. Provide `key` or `keyFile`, or create ~/.ssh/id_ed25519.pub."
      );
    }

    let addKeyStdout: string | undefined;
    let addKeyStderr: string | undefined;
    if (input.ensureKey !== false) {
      const result = await ensureRunpodSshKey(publicKey, keyFile);
      addKeyStdout = result.stdout || undefined;
      addKeyStderr = result.stderr || undefined;
    }

    return object({
      publicKey,
      keyFile,
      identityFile,
      addKeyStdout,
      addKeyStderr,
    });
  }
);

server.tool(
  {
    name: "runpod-pod-run",
    description:
      "Run a single shell command on a pod over SSH and return stdout/stderr.",
    schema: z.object({
      podId: z.string().describe("Runpod pod ID"),
      command: z.string().describe("Shell command to run on the pod"),
      ensureKey: z
        .boolean()
        .optional()
        .describe("If true, ensure an SSH key is added to Runpod"),
      key: z
        .string()
        .optional()
        .describe("Public SSH key contents to add (optional)"),
      keyFile: z
        .string()
        .optional()
        .describe("Path to public SSH key file to add (optional)"),
      host: z
        .string()
        .optional()
        .describe("Override host if runpodctl output lacks SSH info"),
      port: z.number().int().positive().optional().describe("Override port"),
      user: z.string().optional().describe("Override user"),
      identityFile: z
        .string()
        .optional()
        .describe("Path to SSH private key file for the connection"),
      allocatePty: z
        .boolean()
        .optional()
        .default(true)
        .describe("If true, request a PTY for the SSH session"),
      mode: z
        .enum(["proxy", "direct"])
        .optional()
        .describe("proxy = ssh.runpod.io (default), direct = public-IP SSH"),
      timeoutMs: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Kill the SSH command after this many milliseconds (default: 15000)"),
      skipHostKeyCheck: z
        .boolean()
        .optional()
        .default(true)
        .describe("If true, disables strict host key checking"),
    }),
    outputSchema: z.object({
      sshCommand: z.string(),
      stdout: z.string().optional(),
      stderr: z.string().optional(),
    }),
  },
  async (input) => {
    if (input.ensureKey) {
      await ensureRunpodSshKey(input.key, input.keyFile);
    }

    const info = await getPodSshInfo(input.podId, {
      host: input.host,
      port: input.port,
      user: input.user,
    });

    const host = info.host;
    const port = info.port ?? 22;
    const user = info.user ?? "root";
    if (!host) {
      throw new Error(
        "Unable to determine SSH host. Provide host/port/user overrides."
      );
    }

    const mode = input.mode ?? "proxy";
    const allocatePty = input.allocatePty !== false;
    const baseArgs = ["-p", String(port)];
    if (input.identityFile) baseArgs.push("-i", input.identityFile);
    if (allocatePty) baseArgs.push("-tt");
    if (input.skipHostKeyCheck) {
      baseArgs.push(
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null"
      );
    }

    if (mode === "direct") {
      const remoteCommand = allocatePty
        ? `bash -lc ${JSON.stringify(input.command)}`
        : input.command;
      const sshArgs = [...baseArgs, `${user}@${host}`, remoteCommand];
      const { stdout, stderr } = await execFileAsync("ssh", sshArgs, {
        env: process.env,
        timeout: input.timeoutMs ?? 15000,
      });
      return object({
        sshCommand: `ssh ${sshArgs.map((arg) => JSON.stringify(arg)).join(" ")}`,
        stdout: stdout.trim() || undefined,
        stderr: stderr.trim() || undefined,
      });
    }

    // proxy mode: open interactive session, send command via stdin, enforce timeout
    const sshArgs = [...baseArgs, `${user}@${host}`];
    const child = spawn("ssh", sshArgs, {
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    const timeoutMs = input.timeoutMs ?? 15000;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);

    child.stdin.write(`${input.command}\nexit\n`);
    child.stdin.end();

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    child.stdout.on("data", (chunk) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk) => stderrChunks.push(chunk));

    await new Promise<void>((resolve, reject) => {
      child.on("error", reject);
      child.on("close", (code, signal) => {
        clearTimeout(timer);
        if (timedOut) {
          return reject(
            new Error(`SSH command timed out after ${timeoutMs} ms`)
          );
        }
        if (code !== 0) {
          return reject(
            new Error(
              `SSH command failed with code ${code ?? "null"} signal ${
                signal ?? "null"
              }`
            )
          );
        }
        resolve();
      });
    });

    const stdout = Buffer.concat(stdoutChunks).toString("utf8").trim();
    const stderr = Buffer.concat(stderrChunks).toString("utf8").trim();

    return object({
      sshCommand: `ssh ${sshArgs.map((arg) => JSON.stringify(arg)).join(" ")}`,
      stdout: stdout || undefined,
      stderr: stderr || undefined,
    });
  }
);

server.tool(
  {
    name: "render-model-builder",
    description: "Open the visual ML architecture builder to design a neural network",
    schema: z.object({}),
    widget: {
      name: "ml-architecture-builder",
      invoking: "Opening architecture builder…",
      invoked: "Architecture builder ready",
    },
  },
  async () => {
    return widget({
      props: {},
      output: text(
        "ML Architecture Builder is ready. " +
        "Drag blocks from the left panel onto the canvas, connect them by dragging from output ports (bottom) to input ports (top), " +
        "then click 'Generate Model' to produce PyTorch code."
      ),
    });
  }
);

// ── Tool: generate-pytorch-code ────────────────────────────────────────────

server.tool(
  {
    name: "generate-pytorch-code",
    description: "Generate runnable PyTorch model code from a graph JSON produced by the architecture builder",
    schema: z.object({
      graph: graphSchema.describe("The architecture graph with nodes and edges"),
    }),
    outputSchema: z.object({
      code: z.string(),
      errors: z.array(z.string()),
    }),
  },
  async ({ graph }) => {
    const errors = validateGraph(graph);
    if (errors.length > 0) {
      return object({ code: "", errors });
    }
    const code = generatePyTorchCode(graph);
    return object({ code, errors: [] });
  }
);

// ── Tool: validate-graph ───────────────────────────────────────────────────

server.tool(
  {
    name: "validate-graph",
    description: "Validate an architecture graph and return any errors",
    schema: z.object({
      graph: graphSchema.describe("The architecture graph"),
    }),
    outputSchema: z.object({
      valid: z.boolean(),
      errors: z.array(z.string()),
    }),
  },
  async ({ graph }) => {
    const errors = validateGraph(graph);
    return object({ valid: errors.length === 0, errors });
  }
);

// ── Validation ─────────────────────────────────────────────────────────────

const OPTIMIZER_TYPES = new Set(["SGD", "Adam"]);
const LOSS_TYPES      = new Set(["MSELoss", "CrossEntropyLoss"]);
const TRAINING_TYPES  = new Set([...OPTIMIZER_TYPES, ...LOSS_TYPES]);

type GraphInput = z.infer<typeof graphSchema>;
type NodeInput  = z.infer<typeof nodeSchema>;
type EdgeInput  = z.infer<typeof edgeSchema>;

function validateGraph(graph: GraphInput): string[] {
  const errors: string[] = [];
  const inputNodes = graph.nodes.filter(n => n.type === "Input");
  if (inputNodes.length === 0) errors.push("Graph must have exactly one Input block.");
  if (inputNodes.length > 1)  errors.push("Graph must have exactly one Input block (found multiple).");
  if (!graph.nodes.some(n => OPTIMIZER_TYPES.has(n.type))) errors.push("Graph must include an Optimizer block (SGD or Adam).");
  if (!graph.nodes.some(n => LOSS_TYPES.has(n.type)))      errors.push("Graph must include a Loss block (MSELoss or CrossEntropyLoss).");
  return errors;
}

// ── PyTorch Code Generator ─────────────────────────────────────────────────

function topologicalSort(nodes: NodeInput[], edges: EdgeInput[]): NodeInput[] {
  const ids = new Set(nodes.map(n => n.id));
  const inDeg = new Map<string, number>();
  const adj   = new Map<string, string[]>();

  for (const n of nodes) { inDeg.set(n.id, 0); adj.set(n.id, []); }

  for (const e of edges) {
    if (!ids.has(e.sourceId) || !ids.has(e.targetId)) continue;
    adj.get(e.sourceId)!.push(e.targetId);
    inDeg.set(e.targetId, (inDeg.get(e.targetId) ?? 0) + 1);
  }

  const queue = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
  const sorted: NodeInput[] = [];

  while (queue.length > 0) {
    const id = queue.shift()!;
    const node = nodes.find(n => n.id === id)!;
    if (node) sorted.push(node);
    for (const next of (adj.get(id) ?? [])) {
      inDeg.set(next, inDeg.get(next)! - 1);
      if (inDeg.get(next) === 0) queue.push(next);
    }
  }

  return sorted;
}

function p(params: Record<string, unknown>, key: string, def: unknown = 0): unknown {
  return params[key] ?? def;
}

// Generate a nn.Module subclass for a composite (pre-built or custom) block
function generateCompositeClass(node: NodeInput, className: string): string {
  const q = node.parameters;
  const lines: string[] = [];

  if (node.type === "TransformerBlock") {
    const d = p(q, "d_model", 512);
    const nh = p(q, "nhead", 8);
    const ff = p(q, "dim_feedforward", 2048);
    const drop = p(q, "dropout", 0.1);
    lines.push(`class ${className}(nn.Module):`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    lines.push(`        self.self_attn = nn.MultiheadAttention(${d}, ${nh}, dropout=${drop}, batch_first=True)`);
    lines.push(`        self.ff = nn.Sequential(`);
    lines.push(`            nn.Linear(${d}, ${ff}), nn.GELU(), nn.Dropout(${drop}),`);
    lines.push(`            nn.Linear(${ff}, ${d}), nn.Dropout(${drop})`);
    lines.push(`        )`);
    lines.push(`        self.norm1 = nn.LayerNorm(${d})`);
    lines.push(`        self.norm2 = nn.LayerNorm(${d})`);
    lines.push(`    def forward(self, x):`);
    lines.push(`        a, _ = self.self_attn(x, x, x)`);
    lines.push(`        x = self.norm1(x + a)`);
    lines.push(`        x = self.norm2(x + self.ff(x))`);
    lines.push(`        return x`);
  } else if (node.type === "ConvBNReLU") {
    const ic = p(q, "in_channels", 3);
    const oc = p(q, "out_channels", 64);
    const ks = p(q, "kernel_size", 3);
    const st = p(q, "stride", 1);
    const pd = p(q, "padding", 1);
    lines.push(`class ${className}(nn.Module):`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    lines.push(`        self.conv = nn.Conv2d(${ic}, ${oc}, kernel_size=${ks}, stride=${st}, padding=${pd}, bias=False)`);
    lines.push(`        self.bn   = nn.BatchNorm2d(${oc})`);
    lines.push(`        self.act  = nn.ReLU(inplace=True)`);
    lines.push(`    def forward(self, x):`);
    lines.push(`        return self.act(self.bn(self.conv(x)))`);
  } else if (node.type === "ResNetBlock") {
    const ch = p(q, "channels", 64);
    const st = p(q, "stride", 1);
    lines.push(`class ${className}(nn.Module):`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    lines.push(`        self.conv1 = nn.Conv2d(${ch}, ${ch}, 3, stride=${st}, padding=1, bias=False)`);
    lines.push(`        self.bn1   = nn.BatchNorm2d(${ch})`);
    lines.push(`        self.conv2 = nn.Conv2d(${ch}, ${ch}, 3, padding=1, bias=False)`);
    lines.push(`        self.bn2   = nn.BatchNorm2d(${ch})`);
    lines.push(`        self.act   = nn.ReLU(inplace=True)`);
    lines.push(`    def forward(self, x):`);
    lines.push(`        residual = x`);
    lines.push(`        out = self.act(self.bn1(self.conv1(x)))`);
    lines.push(`        out = self.bn2(self.conv2(out))`);
    lines.push(`        return self.act(out + residual)`);
  } else if (node.type === "MLPBlock") {
    const inf = p(q, "in_features", 512);
    const hf  = p(q, "hidden_features", 2048);
    const outf = p(q, "out_features", 512);
    const drop = p(q, "dropout", 0.0);
    lines.push(`class ${className}(nn.Module):`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    lines.push(`        self.fc1  = nn.Linear(${inf}, ${hf})`);
    lines.push(`        self.act  = nn.GELU()`);
    lines.push(`        self.drop = nn.Dropout(${drop})`);
    lines.push(`        self.fc2  = nn.Linear(${hf}, ${outf})`);
    lines.push(`    def forward(self, x):`);
    lines.push(`        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))`);
  } else if (node.type === "Custom" && node.composite) {
    // Generate a class from the stored subgraph
    const comp = node.composite;
    const subNodes = comp.nodes as NodeInput[];
    const subEdges = comp.edges as EdgeInput[];
    const subSorted = topologicalSort(subNodes, subEdges);

    const subIncoming: Record<string, string[]> = {};
    for (const e of subEdges) {
      if (!subIncoming[e.targetId]) subIncoming[e.targetId] = [];
      subIncoming[e.targetId].push(e.sourceId);
    }

    const subVarMap: Record<string, string> = {};
    const subInit: string[] = [];
    const subFwd: string[] = [];
    let si = 0;

    for (const sn of subSorted) {
      const ins = (subIncoming[sn.id] ?? []).map(id => subVarMap[id]).filter(Boolean);
      const inV = sn.id === comp.inputNodeId ? "x" : (ins[0] ?? "x");
      const outV = sn.id === comp.inputNodeId ? "x" : `h${si}`;
      if (sn.id !== comp.inputNodeId) si++;
      subVarMap[sn.id] = outV;
      emitNode(sn, inV, outV, ins, subInit, subFwd, si);
    }

    const lastV = subVarMap[comp.outputNodeId] ?? "x";
    lines.push(`class ${className}(nn.Module):  # ${comp.label}`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    for (const l of subInit) lines.push(`        ${l}`);
    if (subInit.length === 0) lines.push(`        pass`);
    lines.push(`    def forward(self, x):`);
    for (const l of subFwd) lines.push(`        ${l}`);
    lines.push(`        return ${lastV}`);
  }

  return lines.join("\n");
}

// Emit init + forward lines for a single node (used for both main graph and custom subgraphs)
function emitNode(
  node: NodeInput,
  inVar: string,
  outVar: string,
  inputs: string[],
  initLines: string[],
  forwardLines: string[],
  li: number,
): void {
  const q = node.parameters;
  switch (node.type) {
    case "Input":
      forwardLines.push(`# input — expected shape: ${JSON.stringify(q.shape ?? [])}`);
      break;
    case "Linear": {
      const name = `fc${li - 1}`;
      initLines.push(`self.${name} = nn.Linear(${p(q,"in_features",128)}, ${p(q,"out_features",64)}, bias=${q.bias !== false ? "True" : "False"})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "Conv2D": {
      const name = `conv${li - 1}`;
      initLines.push(`self.${name} = nn.Conv2d(${p(q,"in_channels",1)}, ${p(q,"out_channels",32)}, kernel_size=${p(q,"kernel_size",3)}, stride=${p(q,"stride",1)}, padding=${p(q,"padding",0)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "Flatten": {
      const name = `flatten${li - 1}`;
      initLines.push(`self.${name} = nn.Flatten(start_dim=${p(q,"start_dim",1)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "BatchNorm": {
      const name = `bn${li - 1}`;
      initLines.push(`self.${name} = nn.BatchNorm1d(${p(q,"num_features",64)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "Dropout": {
      const name = `drop${li - 1}`;
      initLines.push(`self.${name} = nn.Dropout(p=${p(q,"p",0.5)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "Tokenizer": {
      forwardLines.push(`# ${outVar}: tokenize input → token IDs  (vocab=${p(q,"vocab_size",30000)}, max_len=${p(q,"max_length",512)})`);
      forwardLines.push(`# ${outVar} = tokenizer(text, return_tensors='pt', max_length=${p(q,"max_length",512)}, truncation=True, padding='max_length')['input_ids']`);
      break;
    }
    case "Embedding": {
      const name = `emb${li - 1}`;
      initLines.push(`self.${name} = nn.Embedding(${p(q,"num_embeddings",30000)}, ${p(q,"embedding_dim",512)}, padding_idx=${p(q,"padding_idx",0)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})  # [batch, seq] → [batch, seq, ${p(q,"embedding_dim",512)}]`);
      break;
    }
    case "SinePE": {
      const name = `sinepe${li - 1}`;
      initLines.push(`self.${name} = SinusoidalPE(d_model=${p(q,"d_model",512)}, max_len=${p(q,"max_len",512)}, dropout=${p(q,"dropout",0.1)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "RoPE": {
      const name = `rope${li - 1}`;
      initLines.push(`self.${name} = RotaryEmbedding(dim=${p(q,"dim",64)}, max_seq_len=${p(q,"max_seq_len",2048)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})  # apply RoPE to query/key tensor`);
      break;
    }
    case "LearnedPE": {
      const name = `lepe${li - 1}`;
      initLines.push(`self.${name} = nn.Embedding(${p(q,"max_len",512)}, ${p(q,"d_model",512)})`);
      forwardLines.push(`_pos_${li-1} = torch.arange(${inVar}.size(1), device=${inVar}.device).unsqueeze(0)`);
      forwardLines.push(`${outVar} = ${inVar} + self.${name}(_pos_${li-1})`);
      break;
    }
    case "LayerNorm": {
      const name = `ln${li - 1}`;
      initLines.push(`self.${name} = nn.LayerNorm(${p(q,"normalized_shape",512)})`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
    case "MultiHeadAttn": {
      const name = `attn${li - 1}`;
      initLines.push(`self.${name} = nn.MultiheadAttention(${p(q,"embed_dim",512)}, ${p(q,"num_heads",8)}, batch_first=True)`);
      forwardLines.push(`${outVar}, _ = self.${name}(${inVar}, ${inVar}, ${inVar})`);
      break;
    }
    case "ReLU":    { const n = `relu${li-1}`;    initLines.push(`self.${n} = nn.ReLU()`);                        forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
    case "GELU":    { const n = `gelu${li-1}`;    initLines.push(`self.${n} = nn.GELU()`);                        forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
    case "Sigmoid": { const n = `sigmoid${li-1}`; initLines.push(`self.${n} = nn.Sigmoid()`);                     forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
    case "Tanh":    { const n = `tanh${li-1}`;    initLines.push(`self.${n} = nn.Tanh()`);                        forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
    case "Softmax": { const n = `softmax${li-1}`; initLines.push(`self.${n} = nn.Softmax(dim=${p(q,"dim",-1)})`); forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
    case "ResidualAdd": {
      const in2 = inputs[1] ?? inVar;
      forwardLines.push(`${outVar} = ${inVar} + ${in2}  # residual`);
      break;
    }
    case "Concatenate": {
      const allIn = inputs.length >= 2 ? inputs.join(", ") : `${inVar}, ${inVar}`;
      forwardLines.push(`${outVar} = torch.cat([${allIn}], dim=${p(q,"dim",1)})`);
      break;
    }
    // Pre-built composite types — instantiated as their generated subclass
    case "TransformerBlock":
    case "ConvBNReLU":
    case "ResNetBlock":
    case "MLPBlock":
    case "Custom": {
      const name = `block${li - 1}`;
      // class name will have been generated already; just reference it by convention
      initLines.push(`self.${name} = ${outVar}_cls()  # see class definition above`);
      forwardLines.push(`${outVar} = self.${name}(${inVar})`);
      break;
    }
  }
}

const COMPOSITE_TYPES = new Set(["TransformerBlock", "ConvBNReLU", "ResNetBlock", "MLPBlock", "Custom"]);

// ── PE helper class templates ───────────────────────────────────────────────

const SINUSOIDAL_PE_CLASS = `\
class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])`;

const ROTARY_EMBEDDING_CLASS = `\
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding — RoPE (Su et al. 2021)."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    def forward(self, x):
        # x: [batch, seq, dim]  — apply to query or key before attention
        seq = x.shape[1]
        t = torch.arange(seq, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)  # [1, seq, dim]
        return x * emb.cos() + self._rotate_half(x) * emb.sin()`;

function generatePyTorchCode(graph: GraphInput): string {
  const allNodes  = graph.nodes;
  const allEdges  = graph.edges;
  const layerNodes = allNodes.filter(n => !TRAINING_TYPES.has(n.type));
  const layerEdges = allEdges.filter(e =>
    layerNodes.some(n => n.id === e.sourceId) &&
    layerNodes.some(n => n.id === e.targetId)
  );

  const optimizerNode = allNodes.find(n => OPTIMIZER_TYPES.has(n.type));
  const lossNode      = allNodes.find(n => LOSS_TYPES.has(n.type));

  const sorted = topologicalSort(layerNodes, layerEdges);

  // Build incoming edges map
  const incoming: Record<string, string[]> = {};
  for (const e of layerEdges) {
    if (!incoming[e.targetId]) incoming[e.targetId] = [];
    incoming[e.targetId].push(e.sourceId);
  }

  const varMap: Record<string, string> = {};
  const classBlocks:  string[] = [];   // helper class definitions

  // Inject PE helper classes up front if needed
  if (layerNodes.some(n => n.type === "SinePE"))  classBlocks.push(SINUSOIDAL_PE_CLASS);
  if (layerNodes.some(n => n.type === "RoPE"))    classBlocks.push(ROTARY_EMBEDDING_CLASS);
  const initLines:    string[] = [];
  const forwardLines: string[] = [];
  let li = 0;
  let ci = 0; // composite class counter

  for (const node of sorted) {
    const inputs = (incoming[node.id] ?? []).map(id => varMap[id]).filter(Boolean);
    const inVar  = inputs[0] ?? "x";
    const outVar = node.type === "Input" ? "x" : `h${li}`;
    if (node.type !== "Input") li++;
    varMap[node.id] = outVar;

    if (COMPOSITE_TYPES.has(node.type)) {
      // Generate a standalone class for this composite block
      const className = `CompositeBlock${ci++}`;
      const classDef = generateCompositeClass(node, className);
      if (classDef) classBlocks.push(classDef);
      const attrName = `block_${li - 1}`;
      initLines.push(`self.${attrName} = ${className}()`);
      forwardLines.push(`${outVar} = self.${attrName}(${inVar})`);
    } else {
      emitNode(node, inVar, outVar, inputs, initLines, forwardLines, li);
    }
  }

  const lastVar = sorted.length > 0 ? (varMap[sorted[sorted.length - 1].id] ?? "x") : "x";

  const indent2 = (lines: string[]) => lines.map(l => `        ${l}`).join("\n") || "        pass";
  const indent1 = (lines: string[]) => lines.map(l => `        ${l}`).join("\n") || "        return x";

  const optLine = optimizerNode
    ? (optimizerNode.type === "SGD"
      ? `optimizer = torch.optim.SGD(model.parameters(), lr=${p(optimizerNode.parameters,"lr",0.01)}, momentum=${p(optimizerNode.parameters,"momentum",0.9)})`
      : `optimizer = torch.optim.Adam(model.parameters(), lr=${p(optimizerNode.parameters,"lr",0.001)})`)
    : "# optimizer not configured";

  const lossLine = lossNode
    ? (lossNode.type === "MSELoss" ? "criterion = nn.MSELoss()" : "criterion = nn.CrossEntropyLoss()")
    : "# loss not configured";

  const helperSection = classBlocks.length > 0
    ? classBlocks.join("\n\n") + "\n\n\n"
    : "";

  const needsMath = layerNodes.some(n => n.type === "SinePE");

  return `import torch
import torch.nn as nn${needsMath ? '\nimport math' : ''}


${helperSection}class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
${indent2(initLines)}

    def forward(self, x):
${indent1([...forwardLines, `return ${lastVar}`])}


# ── Setup ──────────────────────────────────────────────────────────────────
model = GeneratedModel()
${optLine}
${lossLine}


# ── Training Step ──────────────────────────────────────────────────────────
def train_step(batch_x, batch_y):
    model.train()
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(batch_x, batch_y):
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
        loss = criterion(output, batch_y)
    return loss.item()
`;
}

// ── Start ──────────────────────────────────────────────────────────────────

server.listen().then(() => console.log("ML Architecture Builder MCP server running"));
