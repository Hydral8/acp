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

// ── Start ──────────────────────────────────────────────────────────────────

server.listen().then(() => console.log("ML Architecture Builder MCP server running"));
