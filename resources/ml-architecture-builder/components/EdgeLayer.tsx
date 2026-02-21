import { GraphEdge, GraphNode, NODE_WIDTH, NODE_HEIGHT, BLOCK_DEFS, CATEGORY_COLORS, PendingConn } from '../types';

function getPortPos(node: GraphNode, port: 'input' | 'output') {
  return {
    x: node.x + NODE_WIDTH / 2,
    y: port === 'input' ? node.y : node.y + NODE_HEIGHT,
  };
}

function bezier(x1: number, y1: number, x2: number, y2: number): string {
  const dy = Math.abs(y2 - y1);
  const c = Math.max(dy * 0.5, 50);
  return `M ${x1} ${y1} C ${x1} ${y1 + c}, ${x2} ${y2 - c}, ${x2} ${y2}`;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  pending: PendingConn | null;
  onDeleteEdge: (id: string) => void;
}

export function EdgeLayer({ nodes, edges, pending, onDeleteEdge }: Props) {
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        overflow: 'visible',
        pointerEvents: 'none',
      }}
    >
      <defs>
        <marker id="arr" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
          <polygon points="0 0, 7 2.5, 0 5" fill="#4a9eff" opacity="0.5" />
        </marker>
      </defs>

      {edges.map(edge => {
        const src = nodeMap[edge.sourceId];
        const tgt = nodeMap[edge.targetId];
        if (!src || !tgt) return null;

        const s = getPortPos(src, 'output');
        const t = getPortPos(tgt, 'input');
        const path = bezier(s.x, s.y, t.x, t.y);
        const def = BLOCK_DEFS.find(d => d.type === src.type)!;
        const color = CATEGORY_COLORS[def.category].port;

        return (
          <g key={edge.id}>
            {/* Wide invisible hit target */}
            <path
              d={path}
              fill="none"
              stroke="transparent"
              strokeWidth={14}
              style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
              onClick={() => onDeleteEdge(edge.id)}
            />
            {/* Visible edge */}
            <path
              d={path}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              opacity={0.55}
              markerEnd="url(#arr)"
            />
          </g>
        );
      })}

      {/* Pending connection */}
      {pending && (() => {
        const src = nodeMap[pending.sourceId];
        if (!src) return null;
        const s = getPortPos(src, 'output');
        const def = BLOCK_DEFS.find(d => d.type === src.type)!;
        const color = CATEGORY_COLORS[def.category].port;
        const path = bezier(s.x, s.y, pending.mouseX, pending.mouseY);
        return (
          <path
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="7 3"
            opacity={0.75}
          />
        );
      })()}
    </svg>
  );
}
