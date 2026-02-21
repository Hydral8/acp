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

  const COL_W = NODE_W + 80;
  const ROW_H = NODE_H + 40;
  const PAD   = 60;

  const posMap = new Map<string, { x: number; y: number }>();
  for (const [rr, ids] of byRank.entries()) {
    for (let i = 0; i < ids.length; i++) {
      posMap.set(ids[i], { x: PAD + rr * COL_W, y: PAD + i * ROW_H });
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

// ── Tool: design-architecture ──────────────────────────────────────────────

server.tool(
  {
    name: "design-architecture",
    description: "Create and render a neural network architecture in the visual builder. Specify layers (nodes) and connections (edges); the builder opens with the design pre-loaded for the user to inspect and edit.",
    schema: z.object({
      nodes: z.array(z.object({
        id: z.string().describe("Unique node id, e.g. 'input', 'conv1', 'relu1'"),
        type: z.string().describe("Block type: Input | Linear | Conv2D | Flatten | BatchNorm | Dropout | LayerNorm | MultiHeadAttn | Tokenizer | Embedding | SinePE | RoPE | LearnedPE | ReLU | GELU | Sigmoid | Tanh | Softmax | ResidualAdd | Concatenate | SGD | Adam | MSELoss | CrossEntropyLoss | TransformerBlock | ConvBNReLU | ResNetBlock | MLPBlock"),
        parameters: z.record(z.string(), z.unknown()).optional().describe("Override default parameters"),
      })).describe("Layers in the architecture"),
      edges: z.array(z.object({
        from: z.string().describe("Source node id"),
        to:   z.string().describe("Target node id"),
      })).describe("Connections between layers"),
      title: z.string().optional().describe("Architecture name shown in the output message"),
    }),
    widget: {
      name: "ml-architecture-builder",
      invoking: "Designing architecture…",
      invoked: "Architecture ready",
    },
  },
  async ({ nodes: specNodes, edges: specEdges, title }) => {
    const { nodes, edges } = autoLayout(specNodes, specEdges);
    return widget({
      props: { initialNodes: nodes, initialEdges: edges },
      output: text(
        `Architecture "${title ?? 'Untitled'}" loaded with ${nodes.length} layer${nodes.length !== 1 ? 's' : ''} ` +
        `and ${edges.length} connection${edges.length !== 1 ? 's' : ''}. ` +
        "Open the builder to inspect or edit it visually."
      ),
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

// ── Start ──────────────────────────────────────────────────────────────────

server.listen().then(() => console.log("ML Architecture Builder MCP server running"));
