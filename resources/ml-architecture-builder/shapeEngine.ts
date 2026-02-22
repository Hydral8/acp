import { GraphNode, GraphEdge } from './types';

export type Shape = number[];

export interface ShapeFix {
  nodeId: string;
  key: string;
  value: unknown;
}

export interface NodeShapeInfo {
  inShape: Shape | null;
  outShape: Shape | null;
  conflict: string | null;
  fixSelf: ShapeFix | null;
  fixUpstream: ShapeFix | null;
}

export function fmtShape(s: Shape): string {
  return `[${s.join(',')}]`;
}

// ── Per-layer output shape computation ────────────────────────────────────

function computeOutput(
  type: string, p: Record<string, unknown>, inShape: Shape | null,
): { outShape: Shape | null; conflict: string | null; expectedDim?: { pos: 'last' | 'ch'; val: number } } {
  const v = (k: string, fb = 0) => (typeof p[k] === 'number' ? (p[k] as number) : fb);
  if (!inShape && type !== 'Input') return { outShape: null, conflict: null };

  switch (type) {
    case 'Input':
      return { outShape: (p.shape as number[]) ?? null, conflict: null };

    case 'Linear': {
      const i = v('in_features'), o = v('out_features');
      if (!inShape) return { outShape: null, conflict: null };
      const last = inShape[inShape.length - 1];
      const out = [...inShape.slice(0, -1), o];
      if (last !== i) return { outShape: out, conflict: `${last} ≠ ${i}`, expectedDim: { pos: 'last', val: i } };
      return { outShape: out, conflict: null };
    }

    case 'Conv2D': {
      const ic = v('in_channels'), oc = v('out_channels'), k = v('kernel_size', 3), s = v('stride', 1), pad = v('padding', 0);
      if (!inShape || inShape.length < 3) return { outShape: null, conflict: inShape ? 'needs 3D+' : null };
      const ci = inShape.length >= 4 ? 1 : 0;
      const out = [...inShape]; out[ci] = oc;
      for (let d = ci + 1; d < out.length; d++) out[d] = Math.floor((out[d] + 2 * pad - k) / s + 1);
      if (inShape[ci] !== ic) return { outShape: out, conflict: `${inShape[ci]}ch ≠ ${ic}ch`, expectedDim: { pos: 'ch', val: ic } };
      return { outShape: out, conflict: null };
    }

    case 'Flatten': {
      if (!inShape) return { outShape: null, conflict: null };
      const sd = v('start_dim', 1);
      const flat = inShape.slice(sd).reduce((a, b) => a * b, 1);
      return { outShape: [...inShape.slice(0, sd), flat], conflict: null };
    }

    case 'BatchNorm': {
      if (!inShape) return { outShape: null, conflict: null };
      const nf = v('num_features');
      if (inShape.length >= 2 && inShape[1] !== nf)
        return { outShape: inShape, conflict: `${inShape[1]} ≠ ${nf}`, expectedDim: { pos: 'ch', val: nf } };
      return { outShape: inShape, conflict: null };
    }

    case 'LayerNorm': {
      if (!inShape) return { outShape: null, conflict: null };
      const ns = v('normalized_shape');
      const last = inShape[inShape.length - 1];
      if (last !== ns) return { outShape: inShape, conflict: `${last} ≠ ${ns}`, expectedDim: { pos: 'last', val: ns } };
      return { outShape: inShape, conflict: null };
    }

    case 'MultiHeadAttn': {
      if (!inShape) return { outShape: null, conflict: null };
      const ed = v('embed_dim');
      const last = inShape[inShape.length - 1];
      if (last !== ed) return { outShape: inShape, conflict: `${last} ≠ ${ed}`, expectedDim: { pos: 'last', val: ed } };
      return { outShape: inShape, conflict: null };
    }

    case 'TransformerBlock': {
      if (!inShape) return { outShape: null, conflict: null };
      const dm = v('d_model');
      const last = inShape[inShape.length - 1];
      if (last !== dm) return { outShape: inShape, conflict: `${last} ≠ ${dm}`, expectedDim: { pos: 'last', val: dm } };
      return { outShape: inShape, conflict: null };
    }

    case 'ConvBNReLU': {
      const ic = v('in_channels'), oc = v('out_channels'), k = v('kernel_size', 3), s = v('stride', 1), pad = v('padding', 1);
      if (!inShape || inShape.length < 3) return { outShape: null, conflict: inShape ? 'needs 3D+' : null };
      const ci = inShape.length >= 4 ? 1 : 0;
      const out = [...inShape]; out[ci] = oc;
      for (let d = ci + 1; d < out.length; d++) out[d] = Math.floor((out[d] + 2 * pad - k) / s + 1);
      if (inShape[ci] !== ic) return { outShape: out, conflict: `${inShape[ci]}ch ≠ ${ic}ch`, expectedDim: { pos: 'ch', val: ic } };
      return { outShape: out, conflict: null };
    }

    case 'ResNetBlock': {
      if (!inShape || inShape.length < 3) return { outShape: null, conflict: inShape ? 'needs 3D+' : null };
      const ch = v('channels');
      const ci = inShape.length >= 4 ? 1 : 0;
      if (inShape[ci] !== ch) return { outShape: inShape, conflict: `${inShape[ci]}ch ≠ ${ch}ch`, expectedDim: { pos: 'ch', val: ch } };
      return { outShape: inShape, conflict: null };
    }

    case 'MLPBlock': {
      const i = v('in_features'), o = v('out_features');
      if (!inShape) return { outShape: null, conflict: null };
      const last = inShape[inShape.length - 1];
      const out = [...inShape.slice(0, -1), o];
      if (last !== i) return { outShape: out, conflict: `${last} ≠ ${i}`, expectedDim: { pos: 'last', val: i } };
      return { outShape: out, conflict: null };
    }

    case 'Tokenizer': {
      const ml = v('max_length', 512);
      return { outShape: [ml], conflict: null };
    }

    case 'Embedding': {
      if (!inShape) return { outShape: null, conflict: null };
      const ed = v('embedding_dim', 512);
      return { outShape: [...inShape, ed], conflict: null };
    }

    case 'SinePE': case 'RoPE': case 'LearnedPE':
      return { outShape: inShape, conflict: null };

    // Passthrough
    case 'Dropout': case 'ReLU': case 'GELU': case 'Sigmoid': case 'Tanh': case 'Softmax':
    case 'ResidualAdd': case 'Concatenate':
      return { outShape: inShape, conflict: null };

    default:
      return { outShape: inShape, conflict: null };
  }
}

// ── Fix helpers ───────────────────────────────────────────────────────────

function selfFix(id: string, type: string, inShape: Shape): ShapeFix | null {
  const last = inShape[inShape.length - 1];
  const ch = inShape.length >= 4 ? inShape[1] : inShape.length >= 3 ? inShape[0] : null;
  const ch2 = inShape.length >= 2 ? inShape[1] : null;

  switch (type) {
    case 'Linear': case 'MLPBlock': return { nodeId: id, key: 'in_features', value: last };
    case 'Conv2D': case 'ConvBNReLU': return ch != null ? { nodeId: id, key: 'in_channels', value: ch } : null;
    case 'BatchNorm': return ch2 != null ? { nodeId: id, key: 'num_features', value: ch2 } : null;
    case 'LayerNorm': return { nodeId: id, key: 'normalized_shape', value: last };
    case 'MultiHeadAttn': return { nodeId: id, key: 'embed_dim', value: last };
    case 'TransformerBlock': return { nodeId: id, key: 'd_model', value: last };
    case 'ResNetBlock': return ch != null ? { nodeId: id, key: 'channels', value: ch } : null;
    default: return null;
  }
}

function upFix(
  upId: string, upType: string, upParams: Record<string, unknown>,
  dim: { pos: 'last' | 'ch'; val: number },
): ShapeFix | null {
  if (dim.pos === 'last') {
    switch (upType) {
      case 'Linear': case 'MLPBlock': return { nodeId: upId, key: 'out_features', value: dim.val };
      case 'Input': {
        const s = ((upParams.shape as number[]) ?? []).slice();
        if (s.length > 0) { s[s.length - 1] = dim.val; return { nodeId: upId, key: 'shape', value: s }; }
        return null;
      }
      default: return null;
    }
  } else {
    switch (upType) {
      case 'Conv2D': case 'ConvBNReLU': return { nodeId: upId, key: 'out_channels', value: dim.val };
      default: return null;
    }
  }
}

// ── Main propagation ──────────────────────────────────────────────────────

export function propagateShapes(nodes: GraphNode[], edges: GraphEdge[]): Map<string, NodeShapeInfo> {
  const result = new Map<string, NodeShapeInfo>();
  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const n of nodes) { incoming.set(n.id, []); outgoing.set(n.id, []); }
  for (const e of edges) { incoming.get(e.targetId)?.push(e.sourceId); outgoing.get(e.sourceId)?.push(e.targetId); }

  // Kahn's topo sort
  const deg = new Map(nodes.map(n => [n.id, incoming.get(n.id)!.length]));
  const q = nodes.filter(n => deg.get(n.id) === 0).map(n => n.id);
  const sorted: string[] = [];
  while (q.length) {
    const id = q.shift()!;
    sorted.push(id);
    for (const t of outgoing.get(id) ?? []) {
      deg.set(t, (deg.get(t) ?? 1) - 1);
      if (deg.get(t) === 0) q.push(t);
    }
  }
  for (const n of nodes) if (!sorted.includes(n.id)) sorted.push(n.id);

  for (const id of sorted) {
    const node = nodeMap.get(id)!;
    const sources = incoming.get(id) ?? [];

    let inShape: Shape | null = null;
    let upNode: GraphNode | null = null;
    if (sources.length > 0) {
      upNode = nodeMap.get(sources[0]) ?? null;
      inShape = result.get(sources[0])?.outShape ?? null;
    }

    const { outShape, conflict, expectedDim } = computeOutput(node.type, node.parameters, inShape);

    let fixSelf: ShapeFix | null = null;
    let fixUpstream: ShapeFix | null = null;
    if (conflict && inShape) {
      fixSelf = selfFix(id, node.type, inShape);
      if (expectedDim && upNode) {
        fixUpstream = upFix(upNode.id, upNode.type, upNode.parameters, expectedDim);
      }
    }

    result.set(id, {
      inShape: sources.length > 0 ? inShape : null,
      outShape,
      conflict,
      fixSelf,
      fixUpstream,
    });
  }

  return result;
}
