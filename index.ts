import { MCPServer, object, text, widget } from "mcp-use/server";
import { execFile, spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { z } from "zod";

const server = new MCPServer({
  name: "acp",
  title: "acp", // display name
  version: "1.0.0",
  description: "MCP server with MCP Apps integration",
  baseUrl: process.env.MCP_URL || "http://localhost:3000", // Full base URL (e.g., https://myserver.com)
  favicon: "favicon.ico",
  websiteUrl: "https://mcp-use.com", // Can be customized later
  icons: [
    {
      src: "icon.svg",
      mimeType: "image/svg+xml",
      sizes: ["512x512"],
    },
  ],
});

/**
 * TOOL THAT RETURNS A WIDGET
 * The `widget` config tells mcp-use which widget component to render.
 * The `widget()` helper in the handler passes props to that component.
 * Docs: https://mcp-use.com/docs/typescript/server/mcp-apps
 */

// Fruits data — color values are Tailwind bg-[] classes used by the carousel UI
const fruits = [
  { fruit: "mango", color: "bg-[#FBF1E1] dark:bg-[#FBF1E1]/10" },
  { fruit: "pineapple", color: "bg-[#f8f0d9] dark:bg-[#f8f0d9]/10" },
  { fruit: "cherries", color: "bg-[#E2EDDC] dark:bg-[#E2EDDC]/10" },
  { fruit: "coconut", color: "bg-[#fbedd3] dark:bg-[#fbedd3]/10" },
  { fruit: "apricot", color: "bg-[#fee6ca] dark:bg-[#fee6ca]/10" },
  { fruit: "blueberry", color: "bg-[#e0e6e6] dark:bg-[#e0e6e6]/10" },
  { fruit: "grapes", color: "bg-[#f4ebe2] dark:bg-[#f4ebe2]/10" },
  { fruit: "watermelon", color: "bg-[#e6eddb] dark:bg-[#e6eddb]/10" },
  { fruit: "orange", color: "bg-[#fdebdf] dark:bg-[#fdebdf]/10" },
  { fruit: "avocado", color: "bg-[#ecefda] dark:bg-[#ecefda]/10" },
  { fruit: "apple", color: "bg-[#F9E7E4] dark:bg-[#F9E7E4]/10" },
  { fruit: "pear", color: "bg-[#f1f1cf] dark:bg-[#f1f1cf]/10" },
  { fruit: "plum", color: "bg-[#ece5ec] dark:bg-[#ece5ec]/10" },
  { fruit: "banana", color: "bg-[#fdf0dd] dark:bg-[#fdf0dd]/10" },
  { fruit: "strawberry", color: "bg-[#f7e6df] dark:bg-[#f7e6df]/10" },
  { fruit: "lemon", color: "bg-[#feeecd] dark:bg-[#feeecd]/10" },
];

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
    name: "search-tools",
    description: "Search for fruits and display the results in a visual widget",
    schema: z.object({
      query: z.string().optional().describe("Search query to filter fruits"),
    }),
    widget: {
      name: "product-search-result",
      invoking: "Searching...",
      invoked: "Results loaded",
    },
  },
  async ({ query }) => {
    const results = fruits.filter(
      (f) => !query || f.fruit.toLowerCase().includes(query.toLowerCase())
    );

    // let's emulate a delay to show the loading state
    await new Promise((resolve) => setTimeout(resolve, 2000));

    return widget({
      props: { query: query ?? "", results },
      output: text(
        `Found ${results.length} fruits matching "${query ?? "all"}"`
      ),
    });
  }
);

server.tool(
  {
    name: "get-fruit-details",
    description: "Get detailed information about a specific fruit",
    schema: z.object({
      fruit: z.string().describe("The fruit name"),
    }),
    outputSchema: z.object({
      fruit: z.string(),
      color: z.string(),
      facts: z.array(z.string()),
    }),
  },
  async ({ fruit }) => {
    const found = fruits.find(
      (f) => f.fruit?.toLowerCase() === fruit?.toLowerCase()
    );
    return object({
      fruit: found?.fruit ?? fruit,
      color: found?.color ?? "unknown",
      facts: [
        `${fruit} is a delicious fruit`,
        `Color: ${found?.color ?? "unknown"}`,
      ],
    });
  }
);

server.listen().then(() => {
  console.log(`Server running`);
});
