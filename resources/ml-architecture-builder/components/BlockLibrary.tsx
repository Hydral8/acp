import { useState } from 'react';
import { BlockCategory, BLOCK_DEFS, CATEGORY_COLORS, CATEGORY_LABELS, CompositeBlock } from '../types';

const CATEGORIES: BlockCategory[] = ['composite', 'core', 'activation', 'structural', 'training'];

interface Props {
  customBlocks?: CompositeBlock[];
  onRemoveCustomBlock?: (index: number) => void;
}

export function BlockLibrary({ customBlocks = [], onRemoveCustomBlock }: Props) {
  const [collapsed, setCollapsed] = useState<Record<BlockCategory, boolean>>({
    composite: false, core: false, activation: true, structural: true, training: true,
  });

  const toggle = (cat: BlockCategory) =>
    setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }));

  const compositeColors = CATEGORY_COLORS['composite'];

  return (
    <div style={{
      width: 168, minWidth: 168,
      backgroundColor: '#080808',
      borderRight: '1px solid #191919',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        padding: '10px 12px 6px',
        fontSize: 9, letterSpacing: 1.5,
        color: '#3a3a3a', textTransform: 'uppercase', fontWeight: 700,
        borderBottom: '1px solid #111',
      }}>
        Blocks
      </div>

      {CATEGORIES.map(cat => {
        const blocks = BLOCK_DEFS.filter(d => d.category === cat && d.type !== 'Custom');
        const colors = CATEGORY_COLORS[cat];
        const isCollapsed = collapsed[cat];

        return (
          <div key={cat}>
            <div
              onClick={() => toggle(cat)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '7px 12px 5px',
                cursor: 'pointer', userSelect: 'none',
                borderBottom: '1px solid #111',
              }}
            >
              <span style={{ fontSize: 10, fontWeight: 700, color: colors.text, letterSpacing: 0.4 }}>
                {CATEGORY_LABELS[cat]}
                {cat === 'composite' && customBlocks.length > 0 && (
                  <span style={{ marginLeft: 4, fontSize: 8, color: colors.border, opacity: 0.8 }}>
                    +{customBlocks.length}
                  </span>
                )}
              </span>
              <span style={{ fontSize: 8, color: '#333' }}>{isCollapsed ? '▶' : '▼'}</span>
            </div>

            {!isCollapsed && (
              <div style={{ padding: '4px 6px 6px' }}>
                {blocks.map(def => (
                  <div
                    key={def.type}
                    draggable
                    onDragStart={e => {
                      e.dataTransfer.setData('blockType', def.type);
                      e.dataTransfer.effectAllowed = 'copy';
                    }}
                    style={{
                      margin: '2px 0',
                      padding: '6px 10px',
                      backgroundColor: colors.bg,
                      border: `1px solid ${colors.border}40`,
                      borderLeft: `2px solid ${colors.border}`,
                      borderRadius: 5,
                      cursor: 'grab',
                      fontSize: 11,
                      color: colors.text,
                      userSelect: 'none',
                      fontWeight: 500,
                    }}
                    onMouseEnter={e => (e.currentTarget.style.backgroundColor = colors.headerBg)}
                    onMouseLeave={e => (e.currentTarget.style.backgroundColor = colors.bg)}
                  >
                    {def.label}
                  </div>
                ))}

                {/* User-created custom blocks — shown in composite section */}
                {cat === 'composite' && customBlocks.map((comp, idx) => (
                  <div
                    key={`custom-${idx}`}
                    style={{
                      margin: '2px 0',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0,
                    }}
                  >
                    <div
                      draggable
                      onDragStart={e => {
                        e.dataTransfer.setData('blockType', 'Custom');
                        e.dataTransfer.setData('customBlockData', JSON.stringify(comp));
                        e.dataTransfer.effectAllowed = 'copy';
                      }}
                      style={{
                        flex: 1,
                        padding: '5px 8px',
                        backgroundColor: compositeColors.bg,
                        border: `1px dashed ${compositeColors.border}60`,
                        borderLeft: `2px solid ${compositeColors.border}`,
                        borderRight: 'none',
                        borderRadius: '5px 0 0 5px',
                        cursor: 'grab',
                        fontSize: 10.5,
                        color: compositeColors.text,
                        userSelect: 'none',
                        fontWeight: 500,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.backgroundColor = compositeColors.headerBg)}
                      onMouseLeave={e => (e.currentTarget.style.backgroundColor = compositeColors.bg)}
                      title={`${comp.label} (${comp.nodes.length} layers)`}
                    >
                      {comp.label}
                    </div>
                    {onRemoveCustomBlock && (
                      <button
                        onClick={() => onRemoveCustomBlock(idx)}
                        title="Remove from library"
                        style={{
                          padding: '5px 6px',
                          backgroundColor: compositeColors.bg,
                          border: `1px dashed ${compositeColors.border}60`,
                          borderLeft: `1px solid ${compositeColors.border}30`,
                          borderRadius: '0 5px 5px 0',
                          cursor: 'pointer',
                          color: '#555',
                          fontSize: 10,
                          lineHeight: 1,
                        }}
                        onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#c05050'; }}
                        onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555'; }}
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Grouping hint */}
      <div style={{
        marginTop: 'auto',
        padding: '8px 10px',
        borderTop: '1px solid #111',
        fontSize: 9, color: '#2a2a2a', lineHeight: 1.5,
      }}>
        Shift+click to multi-select,<br />then Group into custom block
      </div>
    </div>
  );
}
