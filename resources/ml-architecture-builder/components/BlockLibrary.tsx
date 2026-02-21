import { useState } from 'react';
import { BlockCategory, BLOCK_DEFS, CATEGORY_COLORS, CATEGORY_LABELS } from '../types';

const CATEGORIES: BlockCategory[] = ['core', 'activation', 'structural', 'training'];

export function BlockLibrary() {
  const [collapsed, setCollapsed] = useState<Record<BlockCategory, boolean>>({
    core: false, activation: false, structural: false, training: false,
  });

  const toggle = (cat: BlockCategory) =>
    setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }));

  return (
    <div style={{
      width: 168,
      minWidth: 168,
      backgroundColor: '#080808',
      borderRight: '1px solid #191919',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        padding: '10px 12px 6px',
        fontSize: 9,
        letterSpacing: 1.5,
        color: '#3a3a3a',
        textTransform: 'uppercase',
        fontWeight: 700,
        borderBottom: '1px solid #111',
      }}>
        Layers
      </div>

      {CATEGORIES.map(cat => {
        const blocks = BLOCK_DEFS.filter(d => d.category === cat);
        const colors = CATEGORY_COLORS[cat];
        const isCollapsed = collapsed[cat];

        return (
          <div key={cat}>
            <div
              onClick={() => toggle(cat)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '7px 12px 5px',
                cursor: 'pointer',
                userSelect: 'none',
                borderBottom: '1px solid #111',
              }}
            >
              <span style={{ fontSize: 10, fontWeight: 700, color: colors.text, letterSpacing: 0.4 }}>
                {CATEGORY_LABELS[cat]}
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
                      transition: 'border-color 0.12s, background-color 0.12s',
                      fontWeight: 500,
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLDivElement).style.borderLeftColor = colors.border;
                      (e.currentTarget as HTMLDivElement).style.backgroundColor = colors.headerBg;
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLDivElement).style.borderLeftColor = `${colors.border}`;
                      (e.currentTarget as HTMLDivElement).style.backgroundColor = colors.bg;
                    }}
                  >
                    {def.label}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
