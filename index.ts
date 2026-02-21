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
  codeOverride: z.string().optional(),
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
      "Configure runpodctl with an API key, create a Runpod Pod, and (by default) ensure an SSH key is added.",
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
        ensureKey: z
          .boolean()
          .optional()
          .describe(
            "If true, ensure an SSH key is added to Runpod (default: true)"
          ),
        generateKey: z
          .boolean()
          .optional()
          .describe(
            "If true, generate a new SSH keypair first (default: true)"
          ),
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
      publicKey: z.string().optional(),
      keyFile: z.string().optional(),
      identityFile: z.string().optional(),
      addKeyStdout: z.string().optional(),
      addKeyStderr: z.string().optional(),
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

    const identityFile =
      resolveHomePath(input.keyPath) ??
      path.join(homedir(), ".ssh", "id_ed25519");

    let publicKey: string | undefined;
    let keyFile: string | undefined;
    let addKeyStdout: string | undefined;
    let addKeyStderr: string | undefined;

    if (input.ensureKey !== false) {
      if (input.generateKey !== false) {
        await generateSshKeyPair(identityFile);
      }

      const publicKeyFile = `${identityFile}.pub`;
      const keyData = await readPublicKey(
        input.key,
        input.keyFile ?? publicKeyFile
      );
      publicKey = keyData.publicKey;
      keyFile = keyData.keyFile;

      if (!publicKey) {
        throw new Error(
          "No public SSH key found. Provide `key` or `keyFile`, or create ~/.ssh/id_ed25519.pub."
        );
      }

      const result = await ensureRunpodSshKey(publicKey, keyFile);
      addKeyStdout = result.stdout || undefined;
      addKeyStderr = result.stderr || undefined;
    }

    return object({
      configStatus: "runpodctl configured",
      createStdout: stdout || undefined,
      createStderr: stderr || undefined,
      publicKey,
      keyFile,
      identityFile: input.ensureKey !== false ? identityFile : undefined,
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

    const mode = input.mode ?? "proxy";

    const podIdWithSuffix = input.podId.endsWith("-64411bc6")
      ? input.podId
      : `${input.podId}-64411bc6`;
    const host =
      input.host ??
      (mode === "proxy" ? "ssh.runpod.io" : info.host);
    const user =
      input.user ??
      (mode === "proxy" ? podIdWithSuffix : info.user ?? "root");
    const port =
      input.port ??
      (mode === "proxy" ? 22 : info.port ?? 22);
    if (!host) {
      throw new Error(
        "Unable to determine SSH host. Provide host/port/user overrides."
      );
    }

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

let designedLayout: ReturnType<typeof autoLayout> | null = null;

server.tool(
  {
    name: "render-model-builder",
    description: "Open the visual ML architecture builder. If design-architecture was called beforehand, the designed layout is preloaded. Otherwise opens a blank canvas for manual editing.",
    schema: z.object({}),
    widget: {
      name: "ml-architecture-builder",
      invoking: "Opening architecture builder…",
      invoked: "Architecture builder ready",
    },
  },
  async () => {
    const layout = designedLayout;
    designedLayout = null; // consume — next blank call gets empty canvas
    const hasDesign = layout !== null && layout.nodes.length > 0;
    return widget({
      props: hasDesign ? { initialNodes: layout!.nodes, initialEdges: layout!.edges } : {},
      output: text(
        hasDesign
          ? `Architecture loaded with ${layout!.nodes.length} layer${layout!.nodes.length !== 1 ? 's' : ''} and ${layout!.edges.length} connection${layout!.edges.length !== 1 ? 's' : ''}. IMPORTANT: Now call show-next-steps immediately to display the action panel to the user.`
          : "ML Architecture Builder is ready. Drag blocks from the left panel onto the canvas, connect them by dragging between ports, then click 'Generate Model' to produce PyTorch code."
      ),
    });
  }
);

// ── Tool: generate-pytorch-code ────────────────────────────────────────────

server.tool(
  {
    name: "generate-pytorch-code",
    description: "Generate runnable PyTorch model code from a graph. If graph is omitted, uses the last saved design from the visual builder.",
    schema: z.object({
      graph: graphSchema.optional().describe("Architecture graph (nodes + edges). Omit to use the current saved design."),
    }),
    outputSchema: z.object({
      code: z.string(),
      errors: z.array(z.string()),
    }),
  },
  async ({ graph: graphArg }) => {
    const graph = (graphArg && graphArg.nodes.length > 0) ? graphArg : savedDesign;
    if (!graph || graph.nodes.length === 0) {
      return object({ code: "", errors: ["No architecture found — design one first."] });
    }
    const errors = validateGraph(graph);
    if (errors.length > 0) {
      return object({ code: "", errors });
    }
    const code = generatePyTorchCode(graph);
    return object({ code, errors: [] });
  }
);

// ── Node metadata for simulation ─────────────────────────────────────────────

const LAYER_LABELS: Record<string, string> = {
  Input: 'Input', Linear: 'Linear', Conv2D: 'Conv2D', Flatten: 'Flatten',
  BatchNorm: 'BatchNorm', Dropout: 'Dropout', LayerNorm: 'LayerNorm',
  MultiHeadAttn: 'MultiHead Attn', Tokenizer: 'Tokenizer', Embedding: 'Embedding',
  SinePE: 'Sine PE', RoPE: 'RoPE', LearnedPE: 'Learned PE',
  ReLU: 'ReLU', GELU: 'GELU', Sigmoid: 'Sigmoid', Tanh: 'Tanh', Softmax: 'Softmax',
  ResidualAdd: 'Residual Add', Concatenate: 'Concatenate',
  TransformerBlock: 'Transformer', ConvBNReLU: 'Conv-BN-ReLU',
  ResNetBlock: 'ResNet Block', MLPBlock: 'MLP Block', Custom: 'Custom',
};

const LAYER_CATS: Record<string, string> = {
  Input: 'core', Linear: 'core', Conv2D: 'core', Flatten: 'core',
  BatchNorm: 'core', Dropout: 'core', LayerNorm: 'core',
  MultiHeadAttn: 'core', Tokenizer: 'core', Embedding: 'core',
  SinePE: 'core', RoPE: 'core', LearnedPE: 'core',
  ReLU: 'activation', GELU: 'activation', Sigmoid: 'activation',
  Tanh: 'activation', Softmax: 'activation',
  ResidualAdd: 'structural', Concatenate: 'structural',
  TransformerBlock: 'composite', ConvBNReLU: 'composite',
  ResNetBlock: 'composite', MLPBlock: 'composite', Custom: 'composite',
};

// ── Training helpers ───────────────────────────────────────────────────────

type TaskType = 'nlp_lm' | 'nlp_classification' | 'vision' | 'tabular' | 'rlhf' | 'unknown';
type DatasetCategory = 'llm' | 'vlm' | 'rlhf' | 'cv' | 'tabular';

interface TaskAnalysis {
  taskType: TaskType;
  category: DatasetCategory;
  suggestedLoss: string;
  suggestedOptimizer: string;
  description: string;
}

function detectTask(graph: GraphInput): TaskAnalysis {
  const types = new Set(graph.nodes.map(n => n.type));
  const hasNLP    = types.has('Tokenizer') || types.has('Embedding') || types.has('TransformerBlock') || types.has('MultiHeadAttn');
  const hasVision = types.has('Conv2D')    || types.has('ConvBNReLU') || types.has('ResNetBlock');
  const hasSoftmax = types.has('Softmax');
  const hasRLHFPair = types.has('Sigmoid') && (types.has('Embedding') || types.has('Linear'));

  if (hasNLP && hasRLHFPair && !hasSoftmax) {
    return { taskType: 'rlhf', category: 'rlhf', suggestedLoss: 'BCEWithLogitsLoss', suggestedOptimizer: 'AdamW', description: 'RLHF / reward model (NLP with preference pairs)' };
  }
  if (hasNLP) {
    if (hasSoftmax) return { taskType: 'nlp_classification', category: 'llm', suggestedLoss: 'CrossEntropyLoss', suggestedOptimizer: 'AdamW', description: 'NLP classification / token prediction' };
    return { taskType: 'nlp_lm', category: 'llm', suggestedLoss: 'CrossEntropyLoss', suggestedOptimizer: 'AdamW', description: 'Language model / causal LM' };
  }
  if (hasVision && (types.has('Embedding') || types.has('TransformerBlock'))) {
    return { taskType: 'vision', category: 'vlm', suggestedLoss: 'CrossEntropyLoss', suggestedOptimizer: 'AdamW', description: 'Vision-language model' };
  }
  if (hasVision) {
    return { taskType: 'vision', category: 'cv', suggestedLoss: 'CrossEntropyLoss', suggestedOptimizer: 'Adam', description: 'Computer vision / image classification' };
  }
  const hasLinear = types.has('Linear') || types.has('MLPBlock');
  if (hasLinear) {
    const hasMSE = types.has('MSELoss');
    return { taskType: 'tabular', category: 'tabular', suggestedLoss: hasMSE ? 'MSELoss' : 'CrossEntropyLoss', suggestedOptimizer: 'Adam', description: 'Tabular / structured data model' };
  }
  return { taskType: 'unknown', category: 'llm', suggestedLoss: 'CrossEntropyLoss', suggestedOptimizer: 'Adam', description: 'Unknown task type' };
}

function computeShapeAnnotations(graph: GraphInput): Record<string, string> {
  const annotations: Record<string, string> = {};
  const sorted  = topologicalSort(graph.nodes, graph.edges);
  const incoming = new Map<string, string[]>();
  for (const n of graph.nodes) incoming.set(n.id, []);
  for (const e of graph.edges) incoming.get(e.targetId)?.push(e.sourceId);

  const shapes = new Map<string, number[] | null>();
  for (const node of sorted) {
    const inIds = incoming.get(node.id) ?? [];
    const inShape = inIds.length > 0 ? (shapes.get(inIds[0]) ?? null) : null;
    const q = node.parameters;
    const v = (k: string, fb = 0) => (typeof q[k] === 'number' ? (q[k] as number) : fb);
    let out: number[] | null = null;
    switch (node.type) {
      case 'Input':    out = (q.shape as number[] | undefined) ?? null; break;
      case 'Linear': case 'MLPBlock': {
        const o = v('out_features', 64);
        out = inShape ? [...inShape.slice(0, -1), o] : null; break;
      }
      case 'Conv2D': case 'ConvBNReLU': {
        if (inShape && inShape.length >= 3) {
          const oc = v('out_channels', 32), k = v('kernel_size', 3), s = v('stride', 1), pad = v('padding', 0);
          const ci = inShape.length >= 4 ? 1 : 0;
          const o2 = [...inShape]; o2[ci] = oc;
          for (let d = ci + 1; d < o2.length; d++) o2[d] = Math.floor((o2[d] + 2 * pad - k) / s + 1);
          out = o2;
        } break;
      }
      case 'Flatten': {
        if (inShape) {
          const sd = v('start_dim', 1);
          const flat = inShape.slice(sd).reduce((a, b) => a * b, 1);
          out = [...inShape.slice(0, sd), flat];
        } break;
      }
      case 'Tokenizer': out = [v('max_length', 512)]; break;
      case 'Embedding': out = inShape ? [...inShape, v('embedding_dim', 512)] : null; break;
      default: out = inShape; break;
    }
    shapes.set(node.id, out);
    if (out) annotations[node.id] = `[${out.join('×')}]`;
  }
  return annotations;
}

// ── Tool: prepare-train ────────────────────────────────────────────────────

server.tool(
  {
    name: "prepare-train",
    description:
      "Select a training dataset and preview data flowing through the model. " +
      "Pass the graph from render-model-builder to animate the real architecture — " +
      "omit it to use a demo graph.",
    schema: z.object({
      graph: graphSchema.optional().describe(
        "Architecture graph from the model builder (nodes + edges). " +
        "When provided the simulation shows the actual model layers."
      ),
    }),
    widget: {
      name: "ml-architecture-builder",
      invoking: "Loading training setup…",
      invoked: "Training setup ready",
    },
  },
  async ({ graph }) => {
    // Use provided graph, or fall back to savedDesign, or empty
    const g = (graph && graph.nodes.length > 0) ? graph : savedDesign ?? null;
    const hasDesign = g !== null && g.nodes.length > 0;

    return widget({
      props: hasDesign
        ? { initialNodes: g!.nodes, initialEdges: g!.edges, initialMode: 'train' }
        : { initialMode: 'train' },
      output: text(
        hasDesign
          ? `Training setup ready. ${g!.nodes.length} layer architecture loaded — switch to Train tab to pick a dataset and generate scripts.`
          : "Training setup opened. No model design found — build an architecture first in Design mode, then switch to Train."
      ),
    });
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

// ── Tool: generate-training-code ───────────────────────────────────────────

server.tool(
  {
    name: 'generate-training-code',
    description: 'Generate modular Python training files (model.py, data.py, train.py) from the current model design and chosen training config. Reads the current graph from savedDesign automatically.',
    schema: z.object({
      dataset: z.object({
        name: z.string().describe("Human-readable dataset name"),
        source: z.enum(['huggingface', 'torchvision', 'custom']).describe("Where the dataset comes from"),
        hfId: z.string().optional().describe("HuggingFace dataset id, e.g. 'roneneldan/TinyStories'"),
        torchvisionName: z.string().optional().describe("torchvision class name, e.g. 'MNIST', 'CIFAR10'"),
      }),
      taskType: z.string().optional().describe("Detected task type from prepare-train"),
      optimizer: z.enum(['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'LBFGS']).describe("Optimizer to use"),
      loss: z.enum(['CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss', 'NLLLoss', 'L1Loss', 'HuberLoss', 'KLDivLoss', 'CTCLoss']).describe("Loss function"),
      hyperparams: z.object({
        lr:           z.number().describe("Learning rate"),
        batch_size:   z.number().describe("Batch size"),
        epochs:       z.number().describe("Training epochs"),
        weight_decay: z.number().optional().describe("Weight decay (default 0)"),
      }),
    }),
    outputSchema: z.object({ modelPy: z.string(), dataPy: z.string(), trainPy: z.string(), summary: z.string() }),
  },
  async ({ dataset, taskType, optimizer, loss, hyperparams }) => {
    const graph = savedDesign;
    const modelPy = graph ? generatePyTorchCode(graph as GraphInput) : '# No model design found. Build one with design-architecture first.\n';
    const dataPy  = generateDataPy(dataset, taskType ?? 'unknown', hyperparams.batch_size);
    const trainPy = generateTrainPy(optimizer, loss, hyperparams);
    // Persist on server so get-training-code can retrieve it later
    savedTrainingCode = {
      modelPy, dataPy, trainPy,
      meta: {
        dataset: dataset.name,
        optimizer,
        loss,
        taskType: taskType ?? 'unknown',
        generatedAt: new Date().toISOString(),
      },
    };
    return object({
      modelPy,
      dataPy,
      trainPy,
      summary: `Generated and saved model.py (${modelPy.split('\n').length} lines), data.py (${dataPy.split('\n').length} lines), train.py (${trainPy.split('\n').length} lines). Use get-training-code to retrieve them again.`,
    });
  }
);

// ── Tool: get-training-code ─────────────────────────────────────────────────

server.tool(
  {
    name: 'get-training-code',
    description: 'Retrieve the most recently generated training scripts (model.py, data.py, train.py) that were saved by generate-training-code.',
    schema: z.object({
      file: z.enum(['model', 'data', 'train', 'all']).optional().describe("Which file to return. Omit or use 'all' for all three."),
    }),
    outputSchema: z.object({
      modelPy: z.string().optional(),
      dataPy:  z.string().optional(),
      trainPy: z.string().optional(),
      meta:    z.object({ dataset: z.string(), optimizer: z.string(), loss: z.string(), taskType: z.string(), generatedAt: z.string() }).optional(),
      found:   z.boolean(),
    }),
  },
  async ({ file = 'all' }) => {
    if (!savedTrainingCode) {
      return object({ found: false });
    }
    const { modelPy, dataPy, trainPy, meta } = savedTrainingCode;
    return object({
      found: true,
      meta,
      modelPy: file === 'all' || file === 'model' ? modelPy : undefined,
      dataPy:  file === 'all' || file === 'data'  ? dataPy  : undefined,
      trainPy: file === 'all' || file === 'train' ? trainPy : undefined,
    });
  }
);

// ── Training code-gen helpers ───────────────────────────────────────────────

function generateDataPy(
  dataset: { name: string; source: string; hfId?: string; torchvisionName?: string },
  taskType: string,
  batchSize: number,
): string {
  const bs = batchSize;
  if (dataset.source === 'huggingface') {
    const hfId = dataset.hfId ?? 'dataset/name';
    const isNLP = taskType.startsWith('nlp') || taskType === 'rlhf';
    return `"""data.py — HuggingFace dataset loader for ${dataset.name}"""
from datasets import load_dataset
from torch.utils.data import DataLoader
${isNLP ? "from transformers import AutoTokenizer" : "import torchvision.transforms as T"}

DATASET_ID  = "${hfId}"
BATCH_SIZE  = ${bs}
MAX_LENGTH  = 512   # tokens${isNLP ? `
TOKENIZER   = "gpt2"  # swap for your tokenizer

def get_dataloaders(batch_size=BATCH_SIZE, hf_token=None):
    ds = load_dataset(DATASET_ID, token=hf_token)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    train_ds = ds["train"].map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    val_ds   = ds.get("validation") or ds.get("test")
    if val_ds:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    train_ds.set_format("torch")
    if val_ds: val_ds.set_format("torch")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size) if val_ds else None,
    )` : `

def get_dataloaders(batch_size=BATCH_SIZE, hf_token=None):
    ds = load_dataset(DATASET_ID, token=hf_token)
    # TODO: define transforms and adjust column names for your dataset
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = ds["train"]
    val_ds   = ds.get("validation") or ds.get("test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size) if val_ds else None,
    )`}
`;
  }

  if (dataset.source === 'torchvision') {
    const cls = dataset.torchvisionName ?? 'CIFAR10';
    const isGrayscale = cls === 'MNIST' || cls === 'FashionMNIST';
    const norm = isGrayscale ? '(0.5,), (0.5,)' : '(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)';
    return `"""data.py — torchvision ${cls} loader"""
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

BATCH_SIZE = ${bs}

def get_dataloaders(batch_size=BATCH_SIZE):
    transform = T.Compose([T.ToTensor(), T.Normalize(${norm})])
    train_set = torchvision.datasets.${cls}(root="./data", train=True,  download=True, transform=transform)
    val_set   = torchvision.datasets.${cls}(root="./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2),
    )
`;
  }

  // custom / fallback
  return `"""data.py — custom dataset loader"""
from torch.utils.data import Dataset, DataLoader
import torch

BATCH_SIZE = ${bs}

class CustomDataset(Dataset):
    def __init__(self, split="train"):
        # TODO: load your data here
        self.data   = torch.randn(1000, 3, 32, 32)  # placeholder
        self.labels = torch.randint(0, 10, (1000,))  # placeholder

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.data[i], self.labels[i]

def get_dataloaders(batch_size=BATCH_SIZE):
    return (
        DataLoader(CustomDataset("train"), batch_size=batch_size, shuffle=True),
        DataLoader(CustomDataset("val"),   batch_size=batch_size),
    )
`;
}

function generateTrainPy(
  optimizer: string,
  loss: string,
  hp: { lr: number; batch_size: number; epochs: number; weight_decay?: number },
): string {
  const wd = hp.weight_decay ?? 0;
  const optArgs = optimizer === 'SGD'
    ? `model.parameters(), lr=${hp.lr}, momentum=0.9, weight_decay=${wd}`
    : `model.parameters(), lr=${hp.lr}, weight_decay=${wd}`;
  const lossInit = loss === 'MSELoss'
    ? 'nn.MSELoss()'
    : loss === 'BCEWithLogitsLoss'
    ? 'nn.BCEWithLogitsLoss()'
    : 'nn.CrossEntropyLoss()';

  return `"""train.py — training loop"""
import torch
import torch.nn as nn
from model import Model
from data import get_dataloaders

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LR        = ${hp.lr}
BATCH     = ${hp.batch_size}
EPOCHS    = ${hp.epochs}
SAVE_PATH = "checkpoint_best.pt"


def train():
    model     = Model().to(DEVICE)
    optimizer = torch.optim.${optimizer}(${optArgs})
    criterion = ${lossInit}
    train_loader, val_loader = get_dataloaders(batch_size=BATCH)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validation ────────────────────────────────────────────────────
        val_info = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    val_loss += criterion(model(x), y).item()
            val_loss /= len(val_loader)
            val_info = f"  val={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                val_info += " ✓saved"

        print(f"Epoch {epoch:03d}/{EPOCHS}  train={train_loss:.4f}{val_info}")

    torch.save(model.state_dict(), "model_final.pt")
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
`;
}

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
  // ── Code override: user-supplied forward line takes precedence ──────────────
  if (node.codeOverride) {
    const line = node.codeOverride.trim()
      .replace(/\{in\}/g,  inVar)
      .replace(/\{out\}/g, outVar);
    forwardLines.push(`# ${node.type} (custom override)`);
    forwardLines.push(line);
    return;
  }

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

// ── design-architecture helpers ────────────────────────────────────────────

const NODE_W = 140;
const NODE_H = 64;

type BlockCategory = 'core' | 'activation' | 'structural' | 'training' | 'composite';
interface BlockTypeDef {
  label: string; category: BlockCategory;
  defaultParams: Record<string, unknown>;
  hasInput: boolean; hasOutput: boolean;
  description: string;
}

const BLOCK_CATALOG: Record<string, BlockTypeDef> = {
  // ── Core Layers ──────────────────────────────────────────────────────────
  Input:            { label: 'Input',          category: 'core', hasInput: false, hasOutput: true,  defaultParams: { shape: [1, 28, 28] },                                                              description: 'Source node. shape: output tensor dims (no batch dim)' },
  Linear:           { label: 'Linear',         category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { in_features: 128, out_features: 64, bias: true },                                  description: 'Fully-connected y = xW^T + b. Transforms last dim in_features → out_features' },
  Conv2D:           { label: 'Conv2D',         category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 0 },         description: '2D convolution. Out spatial: floor((H+2P-K)/S+1)' },
  Flatten:          { label: 'Flatten',        category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { start_dim: 1 },                                                                    description: 'Flatten from start_dim onward into one dimension' },
  BatchNorm:        { label: 'BatchNorm',      category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { num_features: 64 },                                                                description: 'Batch normalization. num_features must match channel count' },
  Dropout:          { label: 'Dropout',        category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { p: 0.5 },                                                                          description: 'Randomly zeros elements with probability p during training' },
  LayerNorm:        { label: 'LayerNorm',      category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { normalized_shape: 512 },                                                           description: 'Layer normalization over last dim. normalized_shape must match last dim' },
  MultiHeadAttn:    { label: 'MultiHead Attn', category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { embed_dim: 512, num_heads: 8 },                                                    description: 'Multi-head self-attention. embed_dim must match last dim and be divisible by num_heads' },
  Tokenizer:        { label: 'Tokenizer',      category: 'core', hasInput: false, hasOutput: true,  defaultParams: { vocab_size: 30000, max_length: 512 },                                              description: 'Text tokenizer. Outputs integer token IDs with shape [max_length]' },
  Embedding:        { label: 'Embedding',      category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { num_embeddings: 30000, embedding_dim: 512, padding_idx: 0 },                       description: 'Lookup table. Input: token IDs → Output: [..., embedding_dim]' },
  SinePE:           { label: 'Sine PE',        category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { d_model: 512, max_len: 512, dropout: 0.1 },                                        description: 'Fixed sinusoidal positional encoding (Vaswani 2017). Passthrough shape' },
  RoPE:             { label: 'RoPE',           category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { dim: 64, max_seq_len: 2048 },                                                      description: 'Rotary positional encoding (Su 2021, LLaMA-style). Passthrough shape' },
  LearnedPE:        { label: 'Learned PE',     category: 'core', hasInput: true,  hasOutput: true,  defaultParams: { max_len: 512, d_model: 512 },                                                      description: 'BERT-style learned positional embeddings. Passthrough shape' },
  // ── Activations ──────────────────────────────────────────────────────────
  ReLU:             { label: 'ReLU',           category: 'activation', hasInput: true, hasOutput: true, defaultParams: {},           description: 'max(0,x). Passthrough shape' },
  GELU:             { label: 'GELU',           category: 'activation', hasInput: true, hasOutput: true, defaultParams: {},           description: 'Gaussian Error Linear Unit. Common in transformers. Passthrough shape' },
  Sigmoid:          { label: 'Sigmoid',        category: 'activation', hasInput: true, hasOutput: true, defaultParams: {},           description: 'σ(x) = 1/(1+e^-x). Output in (0,1). Passthrough shape' },
  Tanh:             { label: 'Tanh',           category: 'activation', hasInput: true, hasOutput: true, defaultParams: {},           description: 'tanh(x). Output in (-1,1). Passthrough shape' },
  Softmax:          { label: 'Softmax',        category: 'activation', hasInput: true, hasOutput: true, defaultParams: { dim: -1 }, description: 'Softmax over dim. Converts logits to probabilities. Passthrough shape' },
  // ── Structural ───────────────────────────────────────────────────────────
  ResidualAdd:      { label: 'Residual Add',   category: 'structural', hasInput: true, hasOutput: true, defaultParams: {},           description: 'Identity skip connection. Adds input to itself. Passthrough shape' },
  Concatenate:      { label: 'Concatenate',    category: 'structural', hasInput: true, hasOutput: true, defaultParams: { dim: 1 },  description: 'Concatenate tensors along dim' },
  // ── Training Config ──────────────────────────────────────────────────────
  SGD:              { label: 'SGD Optimizer',     category: 'training', hasInput: false, hasOutput: false, defaultParams: { lr: 0.01, momentum: 0.9 }, description: 'SGD optimizer. Standalone config node (no data flow)' },
  Adam:             { label: 'Adam Optimizer',    category: 'training', hasInput: false, hasOutput: false, defaultParams: { lr: 0.001 },              description: 'Adam optimizer. Standalone config node (no data flow)' },
  MSELoss:          { label: 'MSE Loss',          category: 'training', hasInput: false, hasOutput: false, defaultParams: {},                          description: 'MSE loss for regression. Standalone config node' },
  CrossEntropyLoss: { label: 'CrossEntropy Loss', category: 'training', hasInput: false, hasOutput: false, defaultParams: {},                          description: 'Cross-entropy loss + softmax for classification. Standalone config node' },
  // ── Composite ────────────────────────────────────────────────────────────
  TransformerBlock: { label: 'Transformer',   category: 'composite', hasInput: true, hasOutput: true, defaultParams: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },                          description: 'Transformer encoder block: MultiHeadAttn + LayerNorm + FFN + LayerNorm' },
  ConvBNReLU:       { label: 'Conv-BN-ReLU',  category: 'composite', hasInput: true, hasOutput: true, defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 },             description: 'Conv2D → BatchNorm → ReLU fused block. Standard CNN building block' },
  ResNetBlock:      { label: 'ResNet Block',   category: 'composite', hasInput: true, hasOutput: true, defaultParams: { channels: 64, stride: 1 },                                                              description: 'Residual block: two Conv-BN-ReLU layers with identity skip connection' },
  MLPBlock:         { label: 'MLP Block',      category: 'composite', hasInput: true, hasOutput: true, defaultParams: { in_features: 512, hidden_features: 2048, out_features: 512, dropout: 0.0 },           description: 'Two-layer MLP: Linear → GELU → Dropout → Linear. Used in transformer FFN' },
};

// Derived default params lookup (used in auto-layout and shape engine fallback)
const BLOCK_DEFAULT_PARAMS: Record<string, Record<string, unknown>> = Object.fromEntries(
  Object.entries(BLOCK_CATALOG).map(([k, v]) => [k, v.defaultParams])
);

// ── Runtime custom type registries ─────────────────────────────────────────

interface CustomTypeDef {
  label: string; category: BlockCategory;
  defaultParams: Record<string, unknown>;
  hasInput: boolean; hasOutput: boolean;
  description: string;
}
interface CustomCompositeDef {
  nodes: Array<{ id: string; type: string; parameters?: Record<string, unknown> }>;
  edges: Array<{ from: string; to: string }>;
  inputNodeId: string;
  outputNodeId: string;
}

const customTypeRegistry      = new Map<string, CustomTypeDef>();
const customCompositeRegistry = new Map<string, CustomCompositeDef>();

function getDefaultParams(type: string): Record<string, unknown> {
  return BLOCK_DEFAULT_PARAMS[type] ?? customTypeRegistry.get(type)?.defaultParams ?? {};
}

function autoLayout(
  specNodes: Array<{ id: string; type: string; parameters?: Record<string, unknown> }>,
  specEdges: Array<{ from: string; to: string }>,
): {
  nodes: Array<{ id: string; type: string; x: number; y: number; parameters: Record<string, unknown> }>;
  edges: Array<{ id: string; sourceId: string; targetId: string }>;
} {
  const nodeIds = new Set(specNodes.map(n => n.id));
  const adj    = new Map<string, string[]>();
  const inDeg  = new Map<string, number>();
  for (const n of specNodes) { adj.set(n.id, []); inDeg.set(n.id, 0); }
  for (const e of specEdges) {
    if (!nodeIds.has(e.from) || !nodeIds.has(e.to)) continue;
    adj.get(e.from)!.push(e.to);
    inDeg.set(e.to, (inDeg.get(e.to) ?? 0) + 1);
  }

  // BFS rank assignment
  const rank    = new Map<string, number>();
  const visited = new Set<string>();
  let layer = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
  let r = 0;
  while (layer.length > 0) {
    const next: string[] = [];
    for (const id of layer) {
      if (visited.has(id)) continue;
      visited.add(id);
      rank.set(id, r);
      for (const nxt of adj.get(id) ?? []) {
        inDeg.set(nxt, inDeg.get(nxt)! - 1);
        if (inDeg.get(nxt) === 0) next.push(nxt);
      }
    }
    layer = next; r++;
  }
  for (const n of specNodes) if (!rank.has(n.id)) rank.set(n.id, 0);

  // Group by rank → assign positions
  const byRank = new Map<number, string[]>();
  for (const n of specNodes) {
    const rr = rank.get(n.id) ?? 0;
    if (!byRank.has(rr)) byRank.set(rr, []);
    byRank.get(rr)!.push(n.id);
  }

  const COL_W = NODE_W + 60;   // horizontal spacing (for parallel branches)
  const ROW_H = NODE_H + 50;   // vertical spacing between ranks
  const PAD   = 60;

  const posMap = new Map<string, { x: number; y: number }>();
  for (const [rr, ids] of byRank.entries()) {
    // Centre parallel nodes horizontally when a rank has multiple items
    const totalW = ids.length * COL_W - (COL_W - NODE_W);
    const startX = PAD - totalW / 2 + NODE_W / 2;
    for (let i = 0; i < ids.length; i++) {
      posMap.set(ids[i], {
        x: startX + i * COL_W,
        y: PAD + rr * ROW_H,
      });
    }
  }

  const nodes = specNodes.map(n => {
    const comp = customCompositeRegistry.get(n.type);
    if (comp) {
      // Expand named composite into a Custom node with embedded sub-graph
      return {
        id: n.id, type: 'Custom',
        x: posMap.get(n.id)?.x ?? 0, y: posMap.get(n.id)?.y ?? 0,
        parameters: {},
        composite: {
          label: n.type,
          nodes: comp.nodes.map((cn, i) => ({
            id: cn.id, type: cn.type, x: 0, y: i * 104,
            parameters: { ...getDefaultParams(cn.type), ...(cn.parameters ?? {}) },
          })),
          edges: comp.edges.map((ce, i) => ({ id: `ci${i}`, sourceId: ce.from, targetId: ce.to })),
          inputNodeId: comp.inputNodeId,
          outputNodeId: comp.outputNodeId,
        },
      };
    }
    return {
      id: n.id, type: n.type,
      x: posMap.get(n.id)?.x ?? 0, y: posMap.get(n.id)?.y ?? 0,
      parameters: { ...getDefaultParams(n.type), ...(n.parameters ?? {}) },
    };
  });

  let eCounter = 0;
  const edges = specEdges
    .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
    .map(e => ({ id: `ae${++eCounter}`, sourceId: e.from, targetId: e.to }));

  return { nodes, edges };
}

// ── Resource: block-types://catalog ───────────────────────────────────────

server.resource(
  {
    name: 'block-types-catalog',
    uri: 'block-types://catalog',
    title: 'Block Types Catalog',
    description: 'All available ML block types with default parameters, categories, input/output rules, and descriptions. Read this before calling design-architecture or create-block-type.',
  },
  async () => object({
    builtIn: BLOCK_CATALOG,
    custom: Object.fromEntries(customTypeRegistry.entries()),
    composites: Object.fromEntries(
      [...customCompositeRegistry.entries()].map(([name, def]) => [name, {
        ...def,
        label: name,
        category: 'composite',
        hasInput: true,
        hasOutput: true,
        description: `User-defined composite block`,
      }])
    ),
  })
);

// ── Resource: layer-builder://guide ───────────────────────────────────────

server.resource(
  {
    name: 'layer-builder-guide',
    uri: 'layer-builder://guide',
    title: 'Layer Builder Guide',
    description: 'How to define custom layers that integrate fully with the visual graph: editable parameters, connection ports, shape display, and code generation.',
  },
  async () => text(`# Layer Builder Integration Guide

## Parameters (Properties Panel)

Every key in \`defaultParams\` becomes an editable input row in the Properties Panel.

| JS type of default value | UI control | Notes |
|--------------------------|------------|-------|
| \`number\`               | Number input | Step 1 (or 0.001 if < 1) |
| \`boolean\`              | Text input ("true"/"false") | |
| \`string\`               | Text input | |
| \`number[]\` (array)     | JSON text input | e.g. shape: [1, 28, 28] |

**Best practices:**
- Use \`number\` defaults for all dimensional params (in_features, out_channels, etc.)
- Use snake_case keys matching PyTorch convention (in_features, not inFeatures)
- Keep param names short — they display as labels with limited width

## Connection Ports

\`hasInput\`: input port (top of block) — block can receive data from upstream
\`hasOutput\`: output port (bottom of block) — block can send data downstream

Guidelines:
- \`hasInput: false\` → source/root nodes (Input, Tokenizer — no upstream)
- \`hasInput: false, hasOutput: false\` → config nodes (optimizers, losses — not in data flow)
- All other layers: \`hasInput: true, hasOutput: true\`

## Categories & Colors

| Category | Color | Use for |
|----------|-------|---------|
| \`core\` | Blue | Standard layers (Linear, Conv2D, etc.) |
| \`activation\` | Purple | Activation functions (ReLU, GELU, etc.) |
| \`structural\` | Green | Skip connections, concatenation |
| \`training\` | Amber | Optimizer and loss config nodes |
| \`composite\` | Gold | Multi-layer blocks (Transformer, ResNet, etc.) |

## Shape Propagation (shapeEngine.ts)

The visual graph automatically computes and displays output shapes, and highlights dimension mismatches.

**Built-in types** have full shape rules (e.g. Linear changes last dim, Conv2D changes spatial dims).

**Custom types defined via \`create-block-type\`** are treated as **passthrough** (outShape = inShape). This means no shape mismatch warnings but also no shape transformation display.

To get accurate shape display for a new layer type, it must be added to \`shapeEngine.ts\` in the \`computeOutput()\` function. Provide the following to the user and ask them to add it:

\`\`\`typescript
case 'YourType': {
  if (!inShape) return { outShape: null, conflict: null };
  const o = v('out_features');       // read param
  const out = [...inShape.slice(0, -1), o];  // compute new shape
  const last = inShape[inShape.length - 1];
  if (last !== v('in_features')) return { outShape: out, conflict: \`\${last} ≠ \${v('in_features')}\` };
  return { outShape: out, conflict: null };
}
\`\`\`

## Composite Blocks (create-custom-block)

Composite blocks group existing layers into a reusable named type. They render with:
- Gold/amber color (composite category)
- Internal layer flow shown in the Properties Panel (topological order)
- Detailed sub-graph view on double-click (CompositeDetailModal)

The \`inputNodeId\` and \`outputNodeId\` define which internal nodes connect to external edges.

## PyTorch Code Generation

Custom types defined via \`create-block-type\` generate as commented placeholders:
\`\`\`python
# TODO: YourType(param1=value1, param2=value2)  — not yet implemented
\`\`\`

Custom composite types generate as a call to a helper class that wraps the sub-graph.

To get full code generation, a \`case 'YourType':\` must be added to the \`emitNode()\` function in \`index.ts\`.
`)
);

// ── Tool: create-block-type ────────────────────────────────────────────────

server.tool(
  {
    name: 'create-block-type',
    description: 'Define a new custom ML layer type that will appear in the visual graph. Parameters become editable fields in the Properties Panel. Use design-architecture afterward to place it in a schematic.',
    schema: z.object({
      type: z.string().describe("Unique type identifier, PascalCase (e.g. 'GroupNorm', 'GRUCell', 'SEBlock')"),
      label: z.string().describe("Human-readable label shown on the block in the graph"),
      category: z.enum(['core', 'activation', 'structural', 'training', 'composite']).describe("Category determines block color. core=blue, activation=purple, structural=green, training=amber, composite=gold"),
      defaultParams: z.record(z.string(), z.unknown()).describe("Default parameter values. Each key becomes an editable field. Use numbers for dimensions, booleans for flags, arrays for shapes."),
      hasInput: z.boolean().describe("Does this block receive data from upstream? False for source nodes (like Input, Tokenizer)."),
      hasOutput: z.boolean().describe("Does this block produce output for downstream? False for config-only nodes (like optimizers)."),
      description: z.string().optional().describe("Optional description for the catalog"),
    }),
    outputSchema: z.object({ registered: z.boolean(), type: z.string() }),
  },
  async ({ type, label, category, defaultParams, hasInput, hasOutput, description }) => {
    customTypeRegistry.set(type, { label, category, defaultParams, hasInput, hasOutput, description: description ?? '' });
    return object({ registered: true, type });
  }
);

// ── Tool: create-custom-block ──────────────────────────────────────────────

server.tool(
  {
    name: 'create-custom-block',
    description: 'Define a named composite block type built from existing layer types. Once registered, use it in design-architecture by setting a node\'s type to this name.',
    schema: z.object({
      name: z.string().describe("Unique composite name, PascalCase (e.g. 'MyEncoder', 'VisionStem')"),
      nodes: z.array(z.object({
        id: z.string().describe("Internal node id"),
        type: z.string().describe("Layer type (from block-types://catalog)"),
        parameters: z.record(z.string(), z.unknown()).optional().describe("Parameter overrides"),
      })).describe("Internal layers of this composite block"),
      edges: z.array(z.object({
        from: z.string().describe("Source internal node id"),
        to:   z.string().describe("Target internal node id"),
      })).describe("Internal connections"),
      inputNodeId:  z.string().describe("ID of the internal node that receives external input"),
      outputNodeId: z.string().describe("ID of the internal node that provides external output"),
    }),
    outputSchema: z.object({ registered: z.boolean(), name: z.string() }),
  },
  async ({ name, nodes, edges, inputNodeId, outputNodeId }) => {
    customCompositeRegistry.set(name, { nodes, edges, inputNodeId, outputNodeId });
    return object({ registered: true, name });
  }
);

// ── Server-side in-memory design storage (latest save from widget) ─────────
let savedDesign: z.infer<typeof graphSchema> | null = null;

// ── Server-side training script storage ─────────────────────────────────────
interface SavedTrainingCode {
  modelPy: string;
  dataPy: string;
  trainPy: string;
  meta: {
    dataset: string;
    optimizer: string;
    loss: string;
    taskType: string;
    generatedAt: string;
  };
}
let savedTrainingCode: SavedTrainingCode | null = null;

// ── Tool: design-architecture ──────────────────────────────────────────────

server.tool(
  {
    name: "design-architecture",
    description: "Compute and stage a neural network architecture layout from layer/connection specs. Stores the result for render-model-builder. IMPORTANT: You MUST call render-model-builder immediately after this tool to display the visualization to the user.",
    schema: z.object({
      nodes: z.array(z.object({
        id: z.string().describe("Unique node id, e.g. 'input', 'conv1', 'relu1'"),
        type: z.string().describe("Block type — see block-types://catalog for the full list with parameters"),
        parameters: z.record(z.string(), z.unknown()).optional().describe("Override default parameters"),
      })).describe("Layers in the architecture"),
      edges: z.array(z.object({
        from: z.string().describe("Source node id"),
        to:   z.string().describe("Target node id"),
      })).describe("Connections between layers"),
      title: z.string().optional().describe("Architecture name for display"),
    }),
    outputSchema: z.object({ staged: z.boolean(), nodeCount: z.number(), edgeCount: z.number(), nextStep: z.string() }),
  },
  async ({ nodes: specNodes, edges: specEdges, title }) => {
    const layout = autoLayout(specNodes, specEdges);
    designedLayout = layout;
    return object({
      staged: true,
      nodeCount: layout.nodes.length,
      edgeCount: layout.edges.length,
      nextStep: `Architecture "${title ?? 'Untitled'}" staged with ${layout.nodes.length} layer${layout.nodes.length !== 1 ? 's' : ''} and ${layout.edges.length} connection${layout.edges.length !== 1 ? 's' : ''}. Call render-model-builder now to display it, then call show-next-steps to show the action panel.`,
    });
  }
);

// ── Tool: save-design ──────────────────────────────────────────────────────

server.tool(
  {
    name: "save-design",
    description: "Save the current architecture graph from the visual builder (called automatically by the widget on every change)",
    schema: z.object({
      graph: graphSchema.describe("Current architecture graph"),
    }),
    outputSchema: z.object({ saved: z.boolean() }),
  },
  async ({ graph }) => {
    savedDesign = graph;
    return object({ saved: true });
  }
);

// ── Tool: get-current-design ───────────────────────────────────────────────

server.tool(
  {
    name: "get-current-design",
    description: "Get the current architecture graph from the visual builder. Use this to read and then modify an existing design via design-architecture.",
    schema: z.object({}),
    outputSchema: z.object({
      graph: graphSchema.nullable(),
      message: z.string(),
    }),
  },
  async () => {
    if (!savedDesign) {
      return object({ graph: null, message: "No design saved yet — open the builder first and add some blocks." });
    }
    return object({ graph: savedDesign, message: `Current design has ${savedDesign.nodes.length} nodes and ${savedDesign.edges.length} edges.` });
  }
);

// ── Tool: set-block-code ───────────────────────────────────────────────────

server.tool(
  {
    name: "set-block-code",
    description:
      "Set a custom Python forward-pass line for a specific block in the current design. " +
      "Use {in} for the input tensor variable and {out} for the output. " +
      "Example: '{out} = torch.sigmoid(self.fc({in})) + {in}'. " +
      "Call rerender-builder afterward to show the updated visualization.",
    schema: z.object({
      nodeId: z.string().describe("ID of the node to update (from get-current-design)"),
      code: z.string().optional().describe(
        "Python forward-pass line. Use {in} for input var, {out} for output var. " +
        "Omit to remove the override and restore auto-generated code."
      ),
    }),
    outputSchema: z.object({ updated: z.boolean(), nodeId: z.string(), message: z.string() }),
  },
  async ({ nodeId, code }) => {
    if (!savedDesign) {
      return object({ updated: false, nodeId, message: "No saved design — open the builder first." });
    }
    const node = savedDesign.nodes.find(n => n.id === nodeId);
    if (!node) {
      return object({ updated: false, nodeId, message: `Node '${nodeId}' not found in saved design.` });
    }
    savedDesign = {
      ...savedDesign,
      nodes: savedDesign.nodes.map(n =>
        n.id === nodeId ? { ...n, codeOverride: code } : n
      ),
    };
    return object({
      updated: true,
      nodeId,
      message: code
        ? `Code override set on '${node.type}' (${nodeId}). Call rerender-builder to refresh the view.`
        : `Code override cleared on '${node.type}' (${nodeId}). Auto-generated code restored.`,
    });
  }
);

// ── Tool: rerender-builder ─────────────────────────────────────────────────

server.tool(
  {
    name: "rerender-builder",
    description:
      "Re-render the visual architecture builder with the current saved design. " +
      "Call this after set-block-code or any other programmatic changes to refresh the visualization.",
    schema: z.object({}),
    widget: {
      name: "ml-architecture-builder",
      invoking: "Refreshing architecture…",
      invoked: "Architecture refreshed",
    },
  },
  async () => {
    if (!savedDesign || savedDesign.nodes.length === 0) {
      return widget({
        props: {},
        output: text("No saved design found — use render-model-builder to open the builder first."),
      });
    }

    // Re-layout from saved spec (re-compute positions)
    const specNodes = savedDesign.nodes.map(n => ({
      id: n.id, type: n.type, parameters: n.parameters,
    }));
    const specEdges = savedDesign.edges.map(e => ({
      from: e.sourceId, to: e.targetId,
    }));
    const layout = autoLayout(specNodes, specEdges);

    // Carry forward codeOverride from savedDesign
    const nodes = layout.nodes.map(ln => {
      const saved = savedDesign!.nodes.find(sn => sn.id === ln.id);
      return saved?.codeOverride ? { ...ln, codeOverride: saved.codeOverride } : ln;
    });

    const overrideCount = nodes.filter(n => (n as typeof n & { codeOverride?: string }).codeOverride).length;

    return widget({
      props: { initialNodes: nodes, initialEdges: layout.edges },
      output: text(
        `Architecture refreshed — ${nodes.length} layer${nodes.length !== 1 ? 's' : ''}` +
        (overrideCount > 0 ? `, ${overrideCount} with custom code overrides` : '') +
        '. Now call show-next-steps to display the action panel.'
      ),
    });
  }
);

// ── Tool: show-next-steps ──────────────────────────────────────────────────

server.tool(
  {
    name: "show-next-steps",
    description:
      "Show a next-steps action panel after an architecture design is complete. " +
      "The user can generate code, set up training, edit the architecture, or get an explanation — all from one widget.",
    schema: z.object({}),
    widget: {
      name: "next-steps",
      invoking: "Preparing next steps…",
      invoked: "What would you like to do next?",
    },
  },
  async () => {
    const g = savedDesign;
    if (!g || g.nodes.length === 0) {
      return widget({
        props: {
          nodeCount: 0, edgeCount: 0,
          taskType: null, taskDescription: null,
          suggestedLoss: null, suggestedOptimizer: null,
          layerSummary: [],
        },
        output: text("No architecture found — design one first, then try again."),
      });
    }

    const task = detectTask(g);
    const typeCounts = new Map<string, number>();
    for (const n of g.nodes) typeCounts.set(n.type, (typeCounts.get(n.type) ?? 0) + 1);
    const layerSummary = [...typeCounts.entries()]
      .map(([type, count]) => ({ type, count }))
      .sort((a, b) => b.count - a.count);

    return widget({
      props: {
        nodeCount: g.nodes.length,
        edgeCount: g.edges.length,
        taskType: task.taskType,
        taskDescription: task.description,
        suggestedLoss: task.suggestedLoss,
        suggestedOptimizer: task.suggestedOptimizer,
        layerSummary,
      },
      output: text(
        `Architecture has ${g.nodes.length} layers and ${g.edges.length} connections. ` +
        `Detected task: ${task.description}. ` +
        `Choose an action: generate PyTorch code, set up training, edit the architecture, or get an explanation.`
      ),
    });
  }
);

// ── Prompt: architecture-workflow ─────────────────────────────────────────

server.prompt(
  {
    name: "architecture-workflow",
    description: "Canonical tool sequence for designing and displaying neural network architectures. Read this when a user asks to design, build, or modify a model.",
  },
  async () => text(`# ML Architecture Builder — Tool Workflow

When a user asks to design, create, visualise, or modify a neural network, follow this sequence:

## Step 1 — Explore available blocks (recommended)
Read the resource \`block-types://catalog\` to see all layer types, their default parameters, categories, and descriptions.
Read \`layer-builder://guide\` to understand how custom layer types integrate with the visual graph.

## Step 2 — Register custom types (if needed)
- \`create-block-type\` — define a new atomic layer type (name, params, category, connection rules)
- \`create-custom-block\` — define a named composite from existing layers (sub-graph)

## Step 3 — Design the architecture ← shape integrity AND training config are CRITICAL here
Call \`design-architecture\` with:
- \`nodes\`: list of layers, each with an \`id\` and \`type\` (and optional \`parameters\` overrides)
- \`edges\`: list of connections as \`{ from, to }\` pairs referencing node ids

### Choose a suitable loss function and optimizer based on the task
Decide these **before** designing, and tell the user your reasoning:

| Task | Loss | Optimizer | Notes |
|------|------|-----------|-------|
| Multi-class classification | CrossEntropyLoss | Adam / AdamW | Output layer: Linear → (Softmax optional, CE includes it) |
| Binary classification | BCEWithLogitsLoss | Adam | Output layer: Linear with 1 unit, no Sigmoid (loss includes it) |
| Regression | MSELoss or HuberLoss | Adam / SGD | HuberLoss is robust to outliers |
| Language modelling (causal LM) | CrossEntropyLoss | AdamW | Shift labels by 1; use weight decay |
| Sequence-to-sequence / translation | CrossEntropyLoss or CTCLoss | AdamW | CTC for alignmentfree tasks |
| RLHF / reward model | BCEWithLogitsLoss | AdamW | Pairwise preference pairs |
| Self-supervised / contrastive | KLDivLoss or custom | AdamW | Output is log-softmax for KLDiv |
| Tabular regression | MSELoss | Adam | Scale targets to [0,1] for stability |
| Tabular classification | CrossEntropyLoss | Adam or SGD | |

**Always tell the user:** which loss and optimizer you chose and why, so they can override in the training setup widget.

**Before finalising the node list you MUST verify that every dimension matches end-to-end:**

### Shape integrity rules (enforce these — the system will flag mismatches visually)

| Transition | What must match |
|------------|----------------|
| Any layer → Linear | \`in_features\` must equal the last dim of the upstream output |
| Any layer → Conv2D / ConvBNReLU | \`in_channels\` must equal the channel dim of the upstream output |
| Any layer → BatchNorm | \`num_features\` must equal channel dim (dim 1 for 4-D, dim 0 for 3-D) |
| Any layer → LayerNorm | \`normalized_shape\` must equal the last dim of the upstream output |
| Any layer → MultiHeadAttn | \`embed_dim\` must equal the last dim of the upstream output |
| Any layer → TransformerBlock | \`d_model\` must equal the last dim of the upstream output |
| Embedding → subsequent layers | Embedding adds a new last dim (\`embedding_dim\`); next layer must expect \`[..., embedding_dim]\` |
| Conv2D → Flatten → Linear | After Flatten(start_dim=1) with input \`[C, H, W]\`, Linear \`in_features\` must be \`C * H * W\` |
| ResNetBlock | \`channels\` must match the channel dim of the input |

### Shape propagation walkthrough — always do this mentally
For each node in topological order, compute the output shape, then verify the next node's parameters accept it:
1. **Input / Tokenizer** — defines the root shape (e.g. \`[512]\` tokens, \`[3, 224, 224]\` image)
2. **Embedding** — appends \`embedding_dim\` → \`[seq, embed_dim]\`
3. **Linear(in, out)** — replaces last dim: \`[..., in] → [..., out]\`. Set \`in_features = last dim of upstream\`
4. **Conv2D(in_ch, out_ch, k, s, p)** — replaces channel dim and spatial dims: \`[C,H,W] → [out_ch, (H+2p-k)/s+1, (W+2p-k)/s+1]\`
5. **Flatten(start_dim=1)** — collapses dims ≥ start_dim: \`[C,H,W] → [C*H*W]\`
6. **Passthrough layers** (ReLU, GELU, Dropout, LayerNorm, MultiHeadAttn, SinePE, RoPE, TransformerBlock etc.) — output shape = input shape

### Common mistakes to avoid
- Setting Linear \`in_features\` to the wrong value after a Flatten — always calculate \`C * H * W\` explicitly
- Forgetting that Conv2D changes spatial dimensions — a 28×28 image through Conv2D(k=3,s=1,p=0) becomes 26×26
- Mismatched channel counts between consecutive Conv2D layers — \`out_channels\` of layer N must equal \`in_channels\` of layer N+1
- Using BatchNorm \`num_features\` that doesn't match the channel count
- Stacking two Linear layers without checking the inner dimension

This computes the layout and stages it server-side. It does NOT show any visual yet.

## Step 4 — ALWAYS render immediately after ⬅ critical
Call \`render-model-builder\` with no arguments immediately after \`design-architecture\`.
This is the step that actually displays the interactive visual builder to the user.
**Do not skip this step. Do not ask the user before calling it.**

## Step 5 — Show next steps ⬅ critical
Immediately after \`render-model-builder\`, call \`show-next-steps\` to display an action panel.
This gives the user clear options: generate code, set up training, edit the architecture, or get an explanation.
**Do not skip this step. Always show next steps after rendering the architecture.**

## Step 6 — Iterative modifications
To modify an existing design:
1. \`get-current-design\` — read the current canvas state (nodes + edges)
2. Re-verify shape integrity for any changed layers
3. \`design-architecture\` — call with the updated spec
4. \`render-model-builder\` — display the updated design
5. \`show-next-steps\` — show the action panel again

## Step 7 — Generate code
Call \`generate-pytorch-code\` (graph is optional — omit to use the saved design) to produce runnable PyTorch.

## Step 8 — Training scripts
Call \`prepare-train\` to open the training setup widget, then \`generate-training-code\` to produce model.py / data.py / train.py.
Scripts are saved server-side automatically. Use \`get-training-code\` to retrieve them later (e.g. file='train' for just train.py).

---
**Summary of the mandatory three-step render sequence:**
\`design-architecture\` → stage layout → \`render-model-builder\` → show builder → \`show-next-steps\` → action panel

**Shape integrity is your responsibility before calling \`design-architecture\`. The visual system will highlight mismatches but will not auto-fix them — get dimensions right upfront.**
`)
);

// ── Start ──────────────────────────────────────────────────────────────────

server.listen().then(() => console.log("ML Architecture Builder MCP server running"));
