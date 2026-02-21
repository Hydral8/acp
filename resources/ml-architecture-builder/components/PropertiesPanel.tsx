import { GraphNode, GraphEdge, BLOCK_DEFS, CATEGORY_COLORS } from '../types';

interface Props {
  selectedNode: GraphNode | null;
  edges: GraphEdge[];
  onParamChange: (nodeId: string, key: string, value: unknown) => void;
  onDeleteNode: (nodeId: string) => void;
}

export function PropertiesPanel({ selectedNode, edges, onParamChange, onDeleteNode }: Props) {
  const panelStyle: React.CSSProperties = {
    width: 216,
    minWidth: 216,
    backgroundColor: '#080808',
    borderLeft: '1px solid #191919',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
  };

  if (!selectedNode) {
    return (
      <div style={panelStyle}>
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

  const def = BLOCK_DEFS.find(d => d.type === selectedNode.type)!;
  const colors = CATEGORY_COLORS[def.category];
  const connectionCount = edges.filter(
    e => e.sourceId === selectedNode.id || e.targetId === selectedNode.id
  ).length;
  const hasParams = Object.keys(selectedNode.parameters).length > 0;

  return (
    <div style={panelStyle}>
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
          <div style={{ fontSize: 12, color: colors.text, fontWeight: 700 }}>{def.label}</div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            {def.category} · {connectionCount} connection{connectionCount !== 1 ? 's' : ''}
          </div>
        </div>

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
                  onFocus={e => (e.target.style.borderColor = colors.border + '80')}
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
