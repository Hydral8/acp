import { MCPServer, object, text, widget } from "mcp-use/server";
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

const nodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  parameters: z.record(z.string(), z.unknown()),
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

function validateGraph(graph: z.infer<typeof graphSchema>): string[] {
  const errors: string[] = [];
  const inputNodes = graph.nodes.filter(n => n.type === "Input");
  if (inputNodes.length === 0) errors.push("Graph must have exactly one Input block.");
  if (inputNodes.length > 1)  errors.push("Graph must have exactly one Input block (found multiple).");
  if (!graph.nodes.some(n => OPTIMIZER_TYPES.has(n.type))) errors.push("Graph must include an Optimizer block (SGD or Adam).");
  if (!graph.nodes.some(n => LOSS_TYPES.has(n.type)))      errors.push("Graph must include a Loss block (MSELoss or CrossEntropyLoss).");
  return errors;
}

// ── PyTorch Code Generator ─────────────────────────────────────────────────

interface Node { id: string; type: string; parameters: Record<string, unknown> }
interface Edge { id: string; sourceId: string; targetId: string }

function topologicalSort(nodes: Node[], edges: Edge[]): Node[] {
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
  const sorted: Node[] = [];

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

function generatePyTorchCode(graph: z.infer<typeof graphSchema>): string {
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
  const initLines:    string[] = [];
  const forwardLines: string[] = [];
  let li = 0;

  for (const node of sorted) {
    const inputs = (incoming[node.id] ?? []).map(id => varMap[id]).filter(Boolean);
    const inVar  = inputs[0] ?? "x";
    const outVar = node.type === "Input" ? "x" : `h${li}`;
    if (node.type !== "Input") li++;
    varMap[node.id] = outVar;

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
      case "ReLU":    { const n = `relu${li-1}`;    initLines.push(`self.${n} = nn.ReLU()`);                      forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
      case "GELU":    { const n = `gelu${li-1}`;    initLines.push(`self.${n} = nn.GELU()`);                      forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
      case "Sigmoid": { const n = `sigmoid${li-1}`; initLines.push(`self.${n} = nn.Sigmoid()`);                   forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
      case "Tanh":    { const n = `tanh${li-1}`;    initLines.push(`self.${n} = nn.Tanh()`);                      forwardLines.push(`${outVar} = self.${n}(${inVar})`);  break; }
      case "Softmax": { const n = `softmax${li-1}`; initLines.push(`self.${n} = nn.Softmax(dim=${p(q,"dim",-1)})`); forwardLines.push(`${outVar} = self.${n}(${inVar})`); break; }

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

  return `import torch
import torch.nn as nn


class GeneratedModel(nn.Module):
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
