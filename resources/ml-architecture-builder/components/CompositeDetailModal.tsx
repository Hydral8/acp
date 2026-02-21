import { GraphNode, GraphEdge, BLOCK_DEFS, CATEGORY_COLORS, NODE_WIDTH, NODE_HEIGHT } from '../types';
import { EdgeLayer } from './EdgeLayer';

// ── Auto-layout: rank-based left-to-right placement ───────────────────────

function layoutSubgraph(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
  if (nodes.length === 0) return [];

  const adj    = new Map<string, string[]>();
  const inDeg  = new Map<string, number>();
  for (const n of nodes) { adj.set(n.id, []); inDeg.set(n.id, 0); }
  for (const e of edges) {
    if (adj.has(e.sourceId) && adj.has(e.targetId)) {
      adj.get(e.sourceId)!.push(e.targetId);
      inDeg.set(e.targetId, (inDeg.get(e.targetId) ?? 0) + 1);
    }
  }

  // BFS to assign rank (column)
  const rank = new Map<string, number>();
  let layer = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
  const visited = new Set<string>();
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
    layer = next;
    r++;
  }
  // Unranked nodes (isolated / cyclic)
  for (const n of nodes) { if (!rank.has(n.id)) rank.set(n.id, 0); }

  // Group by rank
  const byRank = new Map<number, string[]>();
  for (const n of nodes) {
    const rr = rank.get(n.id) ?? 0;
    if (!byRank.has(rr)) byRank.set(rr, []);
    byRank.get(rr)!.push(n.id);
  }

  const COL_W = NODE_WIDTH + 60;
  const ROW_H = NODE_HEIGHT + 28;
  const PAD   = 30;

  const posMap = new Map<string, { x: number; y: number }>();
  for (const [rr, ids] of byRank.entries()) {
    for (let i = 0; i < ids.length; i++) {
      posMap.set(ids[i], { x: PAD + rr * COL_W, y: PAD + i * ROW_H });
    }
  }

  return nodes.map(n => ({ ...n, x: posMap.get(n.id)?.x ?? 0, y: posMap.get(n.id)?.y ?? 0 }));
}

// ── Read-only mini BlockNode ────────────────────────────────────────────────

function MiniNode({ node }: { node: GraphNode }) {
  const PORT = 9;
  const def    = BLOCK_DEFS.find(d => d.type === node.type);
  const colors = CATEGORY_COLORS[def?.category ?? 'composite'];
  const label  = node.type === 'Custom'
    ? (node.composite?.label ?? 'Custom')
    : (def?.label ?? node.type);

  return (
    <div style={{
      position: 'absolute', left: node.x, top: node.y,
      width: NODE_WIDTH, height: NODE_HEIGHT,
    }}>
      {/* Input port */}
      {(def?.hasInput ?? true) && (
        <div style={{
          position: 'absolute',
          top: -PORT / 2, left: NODE_WIDTH / 2 - PORT / 2,
          width: PORT, height: PORT, borderRadius: '50%',
          backgroundColor: colors.port, border: '2px solid #0a0a0a',
        }} />
      )}

      {/* Body */}
      <div style={{
        width: '100%', height: '100%',
        backgroundColor: colors.bg,
        border: `1.5px solid ${colors.border}70`,
        borderRadius: 8, overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
        boxShadow: `0 2px 8px rgba(0,0,0,0.5)`,
      }}>
        <div style={{ height: 3, backgroundColor: colors.border, flexShrink: 0 }} />
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '0 8px', fontSize: 11, fontWeight: 600,
          color: colors.text, textAlign: 'center', lineHeight: 1.2,
        }}>
          {label}
        </div>
      </div>

      {/* Output port */}
      {(def?.hasOutput ?? true) && (
        <div style={{
          position: 'absolute',
          bottom: -PORT / 2, left: NODE_WIDTH / 2 - PORT / 2,
          width: PORT, height: PORT, borderRadius: '50%',
          backgroundColor: colors.port, border: '2px solid #0a0a0a',
        }} />
      )}
    </div>
  );
}

// ── Modal ──────────────────────────────────────────────────────────────────

interface Props {
  node: GraphNode;
  onClose: () => void;
}

export function CompositeDetailModal({ node, onClose }: Props) {
  const comp = node.composite;
  if (!comp) return null;

  const subNodes = comp.nodes as GraphNode[];
  const subEdges = comp.edges as GraphEdge[];
  const laid = layoutSubgraph(subNodes, subEdges);

  const PAD = 40;
  const canvasW = (laid.length > 0 ? Math.max(...laid.map(n => n.x + NODE_WIDTH)) : NODE_WIDTH) + PAD;
  const canvasH = (laid.length > 0 ? Math.max(...laid.map(n => n.y + NODE_HEIGHT)) : NODE_HEIGHT) + PAD;

  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: 'fixed', inset: 0,
        backgroundColor: 'rgba(0,0,0,0.85)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 150,
      }}
    >
      <div style={{
        width: '82%', maxHeight: '78%',
        backgroundColor: '#080808',
        border: '1px solid #252525',
        borderLeft: '3px solid #ca8a04',
        borderRadius: 10,
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
        boxShadow: '0 24px 64px rgba(0,0,0,0.9)',
      }}>
        {/* Header */}
        <div style={{
          padding: '11px 14px',
          borderBottom: '1px solid #141414',
          display: 'flex', alignItems: 'center', gap: 10,
          flexShrink: 0,
          backgroundColor: '#0a0a0a',
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            backgroundColor: '#eab308', boxShadow: '0 0 8px #eab308',
          }} />
          <span style={{ fontWeight: 700, fontSize: 13, color: '#fde68a', letterSpacing: 0.2 }}>
            {comp.label}
          </span>
          <span style={{ fontSize: 10, color: '#444' }}>
            {subNodes.length} layer{subNodes.length !== 1 ? 's' : ''} · {subEdges.length} connection{subEdges.length !== 1 ? 's' : ''}
          </span>
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 9, color: '#333', letterSpacing: 0.5 }}>
            double-click to dismiss
          </span>
          <button
            onClick={onClose}
            style={{
              background: 'none', border: 'none',
              color: '#444', cursor: 'pointer', fontSize: 18,
              lineHeight: 1, padding: '0 2px',
            }}
          >
            ×
          </button>
        </div>

        {/* Graph canvas */}
        <div style={{
          flex: 1, overflow: 'auto',
          padding: '16px 20px',
          backgroundColor: '#080808',
          backgroundImage: `radial-gradient(circle, #141414 1px, transparent 1px)`,
          backgroundSize: '20px 20px',
        }}>
          <div style={{ position: 'relative', width: canvasW, height: canvasH }}>
            {/* Edge SVG */}
            <EdgeLayer
              nodes={laid}
              edges={subEdges}
              pending={null}
              onDeleteEdge={() => {}}
            />
            {/* Nodes (read-only) */}
            {laid.map(n => <MiniNode key={n.id} node={n} />)}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '8px 14px',
          borderTop: '1px solid #111',
          display: 'flex', alignItems: 'center', gap: 6,
          flexShrink: 0,
          backgroundColor: '#060606',
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            backgroundColor: '#ca8a04', opacity: 0.7,
          }} />
          <span style={{ fontSize: 10, color: '#333' }}>
            Input node: {subNodes.find(n => n.id === comp.inputNodeId)?.type ?? '—'}
          </span>
          <span style={{ fontSize: 10, color: '#2a2a2a', margin: '0 4px' }}>·</span>
          <span style={{ fontSize: 10, color: '#333' }}>
            Output node: {subNodes.find(n => n.id === comp.outputNodeId)?.type ?? '—'}
          </span>
        </div>
      </div>
    </div>
  );
}
