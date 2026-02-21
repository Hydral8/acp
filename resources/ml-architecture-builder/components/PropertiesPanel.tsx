import { GraphNode, GraphEdge, BLOCK_DEFS, CATEGORY_COLORS } from '../types';

function topoSort(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
  const adj = new Map<string, string[]>();
  const inDeg = new Map<string, number>();
  for (const n of nodes) { adj.set(n.id, []); inDeg.set(n.id, 0); }
  for (const e of edges) { adj.get(e.sourceId)?.push(e.targetId); inDeg.set(e.targetId, (inDeg.get(e.targetId) ?? 0) + 1); }
  const queue = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
  const sorted: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!; sorted.push(id);
    for (const nxt of adj.get(id) ?? []) { inDeg.set(nxt, inDeg.get(nxt)! - 1); if (inDeg.get(nxt) === 0) queue.push(nxt); }
  }
  return sorted.map(id => nodes.find(n => n.id === id)!).filter(Boolean);
}

interface Props {
  selectedNodes: GraphNode[];
  edges: GraphEdge[];
  onParamChange: (nodeId: string, key: string, value: unknown) => void;
  onDeleteNode: (nodeId: string) => void;
}

export function PropertiesPanel({ selectedNodes, edges, onParamChange, onDeleteNode }: Props) {
  const panelStyle: React.CSSProperties = {
    width: 216,
    minWidth: 216,
    backgroundColor: '#080808',
    borderLeft: '1px solid #191919',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
  };

  const header = (
    <div style={{
      padding: '10px 12px 6px',
      fontSize: 9,
      letterSpacing: 1.5,
      color: '#3a3a3a',
      textTransform: 'uppercase',
      fontWeight: 700,
      borderBottom: '1px solid #111',
    }}>
      Properties
    </div>
  );

  // ── No selection ─────────────────────────────────────────────────────────
  if (selectedNodes.length === 0) {
    return (
      <div style={panelStyle}>
        {header}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#262626',
          fontSize: 12,
          textAlign: 'center',
          padding: '0 20px',
          gap: 6,
        }}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#262626" strokeWidth="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h6M9 12h6M9 15h4" />
          </svg>
          <span>Select a block to edit its parameters</span>
        </div>
      </div>
    );
  }

  // ── Multi-select ─────────────────────────────────────────────────────────
  if (selectedNodes.length > 1) {
    return (
      <div style={panelStyle}>
        {header}
        <div style={{ padding: 12 }}>
          <div style={{
            padding: '9px 11px',
            backgroundColor: '#1a1200',
            border: '1px solid #ca8a0440',
            borderLeft: '3px solid #ca8a04',
            borderRadius: 6,
            marginBottom: 14,
          }}>
            <div style={{ fontSize: 12, color: '#fde68a', fontWeight: 700 }}>
              {selectedNodes.length} blocks selected
            </div>
            <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
              Shift+click to add/remove · Group in toolbar
            </div>
          </div>

          <div style={{
            fontSize: 9, color: '#3a3a3a',
            letterSpacing: 1.5, textTransform: 'uppercase',
            fontWeight: 700, marginBottom: 8,
          }}>
            Selected
          </div>

          {selectedNodes.map(node => {
            const def = BLOCK_DEFS.find(d => d.type === node.type);
            const colors = CATEGORY_COLORS[def?.category ?? 'composite'];
            const label = node.type === 'Custom'
              ? (node.composite?.label ?? 'Custom Block')
              : (def?.label ?? node.type);
            return (
              <div key={node.id} style={{
                marginBottom: 6,
                padding: '5px 8px',
                backgroundColor: colors.bg,
                border: `1px solid ${colors.border}40`,
                borderLeft: `2px solid ${colors.border}`,
                borderRadius: 4,
                fontSize: 11,
                color: colors.text,
                fontWeight: 500,
              }}>
                {label}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // ── Single selection ─────────────────────────────────────────────────────
  const selectedNode = selectedNodes[0];
  const def = BLOCK_DEFS.find(d => d.type === selectedNode.type)!;
  const colors = CATEGORY_COLORS[def?.category ?? 'composite'];
  const connectionCount = edges.filter(
    e => e.sourceId === selectedNode.id || e.targetId === selectedNode.id
  ).length;
  const hasParams = Object.keys(selectedNode.parameters).length > 0;
  const displayLabel = selectedNode.type === 'Custom'
    ? (selectedNode.composite?.label ?? 'Custom Block')
    : (def?.label ?? selectedNode.type);

  return (
    <div style={panelStyle}>
      {header}

      <div style={{ padding: 12, flex: 1 }}>
        {/* Block header */}
        <div style={{
          padding: '9px 11px',
          backgroundColor: colors.headerBg,
          border: `1px solid ${colors.border}40`,
          borderLeft: `3px solid ${colors.border}`,
          borderRadius: 6,
          marginBottom: 14,
        }}>
          <div style={{ fontSize: 12, color: colors.text, fontWeight: 700 }}>{displayLabel}</div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            {def?.category ?? 'composite'} · {connectionCount} connection{connectionCount !== 1 ? 's' : ''}
          </div>
          {selectedNode.type === 'Custom' && selectedNode.composite && (
            <div style={{ fontSize: 10, color: '#888', marginTop: 4 }}>
              {selectedNode.composite.nodes.length} internal layers · double-click to inspect
            </div>
          )}
        </div>

        {/* Internal structure for Custom blocks */}
        {selectedNode.type === 'Custom' && selectedNode.composite && (() => {
          const comp = selectedNode.composite;
          const sorted = topoSort(comp.nodes as GraphNode[], comp.edges as GraphEdge[]);
          return (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase', fontWeight: 700, marginBottom: 8 }}>
                Internal Structure
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
                {sorted.map((n, i) => {
                  const d = BLOCK_DEFS.find(bd => bd.type === n.type);
                  const c = CATEGORY_COLORS[d?.category ?? 'composite'];
                  const lbl = n.type === 'Custom' ? (n.composite?.label ?? 'Custom') : (d?.label ?? n.type);
                  return (
                    <span key={n.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 3, backgroundColor: c.bg, border: `1px solid ${c.border}50`, color: c.text, fontWeight: 600, whiteSpace: 'nowrap' }}>
                        {lbl}
                      </span>
                      {i < sorted.length - 1 && <span style={{ fontSize: 9, color: '#2a2a2a' }}>→</span>}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* Parameters */}
        {hasParams && (
          <>
            <div style={{
              fontSize: 9,
              color: '#3a3a3a',
              letterSpacing: 1.5,
              textTransform: 'uppercase',
              fontWeight: 700,
              marginBottom: 8,
            }}>
              Parameters
            </div>

            {Object.entries(selectedNode.parameters).map(([key, value]) => (
              <div key={key} style={{ marginBottom: 10 }}>
                <label style={{
                  display: 'block',
                  fontSize: 10,
                  color: '#555',
                  marginBottom: 4,
                  fontWeight: 600,
                  letterSpacing: 0.3,
                }}>
                  {key}
                </label>
                <input
                  type={typeof value === 'number' ? 'number' : 'text'}
                  value={Array.isArray(value) ? JSON.stringify(value) : String(value)}
                  step={typeof value === 'number' && value < 1 ? 0.001 : 1}
                  onChange={e => {
                    let parsed: unknown = e.target.value;
                    if (typeof value === 'number') {
                      parsed = parseFloat(e.target.value);
                      if (isNaN(parsed as number)) parsed = 0;
                    } else if (typeof value === 'boolean') {
                      parsed = e.target.value === 'true';
                    } else if (Array.isArray(value)) {
                      try { parsed = JSON.parse(e.target.value); } catch { parsed = e.target.value; }
                    }
                    onParamChange(selectedNode.id, key, parsed);
                  }}
                  style={{
                    width: '100%',
                    padding: '5px 8px',
                    backgroundColor: '#0f0f0f',
                    border: `1px solid #202020`,
                    borderRadius: 4,
                    color: '#d0d0d0',
                    fontSize: 11,
                    outline: 'none',
                    boxSizing: 'border-box',
                    fontFamily: 'inherit',
                  }}
                  onFocus={e => (e.target.style.borderColor = (colors.border ?? '#888') + '80')}
                  onBlur={e => (e.target.style.borderColor = '#202020')}
                />
              </div>
            ))}
          </>
        )}

        {!hasParams && (
          <div style={{ fontSize: 11, color: '#333', marginBottom: 14 }}>
            No parameters
          </div>
        )}

        {/* Separator */}
        <div style={{ borderTop: '1px solid #141414', margin: '12px 0' }} />

        {/* Delete */}
        <button
          onClick={() => onDeleteNode(selectedNode.id)}
          style={{
            width: '100%',
            padding: '7px 0',
            backgroundColor: 'transparent',
            border: '1px solid #3d1515',
            borderRadius: 5,
            color: '#c05050',
            cursor: 'pointer',
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0.3,
            transition: 'background-color 0.1s, color 0.1s',
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#2a0a0a';
            (e.currentTarget as HTMLButtonElement).style.color = '#e06060';
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'transparent';
            (e.currentTarget as HTMLButtonElement).style.color = '#c05050';
          }}
        >
          Delete Block
        </button>
      </div>
    </div>
  );
}
