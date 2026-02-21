import { GraphNode, BLOCK_DEFS, CATEGORY_COLORS, NODE_WIDTH, NODE_HEIGHT, COMPOSITE_PREBUILT_TYPES } from '../types';
import { type NodeShapeInfo, type ShapeFix, fmtShape } from '../shapeEngine';

const PORT_SIZE = 11;
const PORT_HALF = PORT_SIZE / 2;

function fmtParams(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

function getNodeInfo(type: string, p: Record<string, unknown>): {
  inLabel: string | null; outLabel: string | null; params: number | null;
} {
  const n = (k: string, fb = 0) => (typeof p[k] === 'number' ? (p[k] as number) : fb);
  switch (type) {
    case 'Input':        return { inLabel: null, outLabel: `${JSON.stringify(p.shape)}`, params: null };
    case 'Linear':       { const i=n('in_features'),o=n('out_features'); return { inLabel:`${i}`, outLabel:`${o}`, params: i*o+(p.bias!==false?o:0) }; }
    case 'Conv2D':       { const i=n('in_channels'),o=n('out_channels'),k=n('kernel_size',3); return { inLabel:`${i}ch`, outLabel:`${o}ch`, params: o*i*k*k+o }; }
    case 'Flatten':      return { inLabel: null, outLabel: 'flat', params: null };
    case 'BatchNorm':    { const f=n('num_features'); return { inLabel:`${f}`, outLabel:`${f}`, params: 2*f }; }
    case 'Dropout':      return { inLabel: null, outLabel: null, params: null };
    case 'LayerNorm':    { const s=n('normalized_shape'); return { inLabel:`${s}`, outLabel:`${s}`, params: 2*s }; }
    case 'MultiHeadAttn':{ const d=n('embed_dim'); return { inLabel:`${d}`, outLabel:`${d}`, params: 4*d*d+4*d }; }
    case 'TransformerBlock': { const d=n('d_model'),ff=n('dim_feedforward'); const a=4*d*d+4*d; return { inLabel:`${d}`, outLabel:`${d}`, params: a+2*d*ff+d+ff+4*d }; }
    case 'ConvBNReLU':   { const i=n('in_channels'),o=n('out_channels'),k=n('kernel_size',3); return { inLabel:`${i}ch`, outLabel:`${o}ch`, params: o*i*k*k+3*o }; }
    case 'ResNetBlock':  { const c=n('channels'); return { inLabel:`${c}ch`, outLabel:`${c}ch`, params: 2*c*c*9+4*c }; }
    case 'MLPBlock':     { const i=n('in_features'),h=n('hidden_features'),o=n('out_features'); return { inLabel:`${i}`, outLabel:`${o}`, params: i*h+h+h*o+o }; }
    case 'Tokenizer':    { const ml=n('max_length',512); return { inLabel: null, outLabel: `${ml} tok`, params: null }; }
    case 'Embedding':    { const ne=n('num_embeddings',30000),ed=n('embedding_dim',512); return { inLabel: 'tokens', outLabel: `${ed}d`, params: ne*ed }; }
    case 'SinePE':       { const d=n('d_model',512); return { inLabel: `${d}d`, outLabel: `${d}d`, params: null }; }
    case 'RoPE':         { const d=n('dim',64); return { inLabel: `${d}d`, outLabel: `${d}d`, params: null }; }
    case 'LearnedPE':    { const ml=n('max_len',512),d=n('d_model',512); return { inLabel: `${d}d`, outLabel: `${d}d`, params: ml*d }; }
    case 'SGD': case 'Adam': return { inLabel: null, outLabel: null, params: null };
    default:             return { inLabel: null, outLabel: null, params: null };
  }
}

function edgeLabelStyle(isTop: boolean): React.CSSProperties {
  return {
    position: 'absolute',
    left: 0,
    width: NODE_WIDTH,
    textAlign: 'center',
    fontSize: 9,
    fontFamily: '"SF Mono", "Fira Code", monospace',
    color: '#ffffffcc',
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
    ...(isTop ? { top: -16 } : { bottom: -16 }),
  };
}

const fixBtnStyle: React.CSSProperties = {
  padding: '0 4px', fontSize: 7, fontWeight: 700,
  backgroundColor: '#3f0000', border: '1px solid #ff6b6b50',
  borderRadius: 3, color: '#ff9999', cursor: 'pointer',
  lineHeight: '14px', whiteSpace: 'nowrap',
};

interface Props {
  node: GraphNode;
  isSelected: boolean;
  shapeInfo?: NodeShapeInfo;
  onMouseDown: (e: React.MouseEvent, nodeId: string) => void;
  onPortMouseDown: (e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => void;
  onPortMouseUp: (e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => void;
  onFixConflict?: (fix: ShapeFix) => void;
  onDoubleClick?: (nodeId: string) => void;
}

export function BlockNode({ node, isSelected, shapeInfo, onMouseDown, onPortMouseDown, onPortMouseUp, onFixConflict, onDoubleClick }: Props) {
  const def = BLOCK_DEFS.find(d => d.type === node.type);
  const colors = CATEGORY_COLORS[def?.category ?? 'composite'];

  const isCustom = node.type === 'Custom';
  const isPrebuilt = COMPOSITE_PREBUILT_TYPES.has(node.type);
  const isComposite = isCustom || isPrebuilt;

  const displayLabel = isCustom
    ? (node.composite?.label ?? 'Custom Block')
    : (def?.label ?? node.type);

  const sublabel = isCustom
    ? `${node.composite?.nodes.length ?? 0} layers`
    : isPrebuilt
    ? 'composite'
    : null;

  const hasInput  = def?.hasInput  ?? true;
  const hasOutput = def?.hasOutput ?? true;
  const info = getNodeInfo(node.type, node.parameters);
  const hasConflict = !!shapeInfo?.conflict;

  const fixBtnStyle: React.CSSProperties = {
    fontSize: 8, padding: '1px 4px', borderRadius: 2,
    backgroundColor: '#3a0000', border: '1px solid #ff6b6b50',
    color: '#ff9999', cursor: 'pointer', lineHeight: 1.2,
  };

  const topLabel  = shapeInfo?.inShape ? fmtShape(shapeInfo.inShape) : info.inLabel;
  const botLabel  = shapeInfo?.outShape ? fmtShape(shapeInfo.outShape) : info.outLabel;

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
      {/* Input port + shape label (top) */}
      {hasInput && (
        <>
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
          {topLabel && (
            <div style={{ ...edgeLabelStyle(true), color: hasConflict ? '#ff6b6b' : '#ffffffcc' }}>{topLabel}</div>
          )}
        </>
      )}

      {/* Block body */}
      <div
        onMouseDown={e => onMouseDown(e, node.id)}
        onDoubleClick={e => { e.stopPropagation(); onDoubleClick?.(node.id); }}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: colors.bg,
          border: hasConflict
            ? `1.5px solid #ff6b6b90`
            : isComposite
            ? `1.5px dashed ${isSelected ? '#ffffff88' : colors.border + '90'}`
            : `1.5px solid ${isSelected ? '#ffffff88' : colors.border + '70'}`,
          borderRadius: 8,
          cursor: 'move',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: isSelected
            ? `0 0 0 1.5px ${colors.border}, 0 6px 20px rgba(0,0,0,0.6)`
            : isComposite
            ? `0 2px 14px ${colors.border}30, 0 2px 10px rgba(0,0,0,0.5)`
            : `0 2px 10px rgba(0,0,0,0.5)`,
          transition: 'box-shadow 0.15s, border-color 0.15s',
        }}
      >
        {/* Category color bar */}
        <div style={{
          height: 3,
          backgroundColor: colors.border,
          flexShrink: 0,
          backgroundImage: isComposite
            ? `repeating-linear-gradient(90deg, ${colors.border} 0px, ${colors.border} 6px, transparent 6px, transparent 10px)`
            : 'none',
        }} />

        {/* Conflict strip */}
        {hasConflict && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
            padding: '2px 4px', backgroundColor: '#2a0000', flexShrink: 0,
          }}>
            <span style={{ fontSize: 8, color: '#ff6b6b', whiteSpace: 'nowrap' }}>⚠ {shapeInfo!.conflict}</span>
            {shapeInfo!.fixSelf && (
              <button
                onMouseDown={e => e.stopPropagation()}
                onClick={e => { e.stopPropagation(); onFixConflict?.(shapeInfo!.fixSelf!); }}
                style={fixBtnStyle}
              >Fix ↓</button>
            )}
            {shapeInfo!.fixUpstream && (
              <button
                onMouseDown={e => e.stopPropagation()}
                onClick={e => { e.stopPropagation(); onFixConflict?.(shapeInfo!.fixUpstream!); }}
                style={fixBtnStyle}
              >Fix ↑</button>
            )}
          </div>
        )}

        {/* Label + params */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2px 8px',
          gap: 2,
        }}>
          <div style={{
            fontSize: 11.5,
            fontWeight: 600,
            color: colors.text,
            letterSpacing: 0.2,
            textAlign: 'center',
            lineHeight: 1.2,
          }}>
            {displayLabel}
          </div>
          {sublabel && (
            <div style={{ fontSize: 8.5, color: colors.border, letterSpacing: 0.5, opacity: 0.8 }}>
              {sublabel}
            </div>
          )}
          {info.params != null && (
            <div style={{
              fontSize: 8.5,
              color: colors.border,
              opacity: 0.7,
              letterSpacing: 0.3,
              fontWeight: 500,
              fontFamily: '"SF Mono", "Fira Code", monospace',
            }}>
              {fmtParams(info.params)} params
            </div>
          )}
        </div>
      </div>

      {/* Output port + shape label (bottom) */}
      {hasOutput && (
        <>
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
          {botLabel && (
            <div style={edgeLabelStyle(false)}>{botLabel}</div>
          )}
        </>
      )}
    </div>
  );
}
