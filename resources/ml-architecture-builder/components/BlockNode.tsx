import { GraphNode, BLOCK_DEFS, CATEGORY_COLORS, NODE_WIDTH, NODE_HEIGHT } from '../types';

const PORT_SIZE = 11;
const PORT_HALF = PORT_SIZE / 2;

interface Props {
  node: GraphNode;
  isSelected: boolean;
  onMouseDown: (e: React.MouseEvent, nodeId: string) => void;
  onPortMouseDown: (e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => void;
  onPortMouseUp: (e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => void;
}

export function BlockNode({ node, isSelected, onMouseDown, onPortMouseDown, onPortMouseUp }: Props) {
  const def = BLOCK_DEFS.find(d => d.type === node.type)!;
  const colors = CATEGORY_COLORS[def.category];

  const portBase: React.CSSProperties = {
    position: 'absolute',
    width: PORT_SIZE,
    height: PORT_SIZE,
    borderRadius: '50%',
    backgroundColor: colors.port,
    border: '2px solid #080808',
    cursor: 'crosshair',
    zIndex: 2,
    transition: 'transform 0.1s, box-shadow 0.1s',
  };

  return (
    <div
      style={{
        position: 'absolute',
        left: node.x,
        top: node.y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        userSelect: 'none',
        zIndex: isSelected ? 10 : 1,
      }}
    >
      {/* Input port (top-center) */}
      {def.hasInput && (
        <div
          onMouseDown={e => { e.stopPropagation(); onPortMouseDown(e, node.id, 'input'); }}
          onMouseUp={e => { e.stopPropagation(); onPortMouseUp(e, node.id, 'input'); }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.3)';
            (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 8px ${colors.port}`;
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)';
            (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
          }}
          style={{
            ...portBase,
            top: -PORT_HALF,
            left: NODE_WIDTH / 2 - PORT_HALF,
          }}
        />
      )}

      {/* Block body */}
      <div
        onMouseDown={e => onMouseDown(e, node.id)}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: colors.bg,
          border: `1.5px solid ${isSelected ? '#ffffff88' : colors.border + '70'}`,
          borderRadius: 8,
          cursor: 'move',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: isSelected
            ? `0 0 0 1.5px ${colors.border}, 0 6px 20px rgba(0,0,0,0.6)`
            : `0 2px 10px rgba(0,0,0,0.5)`,
          transition: 'box-shadow 0.15s, border-color 0.15s',
        }}
      >
        {/* Category color bar */}
        <div style={{ height: 3, backgroundColor: colors.border, flexShrink: 0 }} />

        {/* Label */}
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 10px',
          fontSize: 11.5,
          fontWeight: 600,
          color: colors.text,
          letterSpacing: 0.2,
          textAlign: 'center',
          lineHeight: 1.2,
        }}>
          {def.label}
        </div>
      </div>

      {/* Output port (bottom-center) */}
      {def.hasOutput && (
        <div
          onMouseDown={e => { e.stopPropagation(); onPortMouseDown(e, node.id, 'output'); }}
          onMouseUp={e => { e.stopPropagation(); onPortMouseUp(e, node.id, 'output'); }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.3)';
            (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 8px ${colors.port}`;
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)';
            (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
          }}
          style={{
            ...portBase,
            bottom: -PORT_HALF,
            left: NODE_WIDTH / 2 - PORT_HALF,
          }}
        />
      )}
    </div>
  );
}
