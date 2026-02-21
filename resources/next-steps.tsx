import { useState, useCallback } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';

const layerSummarySchema = z.object({
  type: z.string(),
  count: z.number(),
});

const propsSchema = z.object({
  nodeCount: z.number(),
  edgeCount: z.number(),
  taskType: z.string().nullable(),
  taskDescription: z.string().nullable(),
  suggestedLoss: z.string().nullable(),
  suggestedOptimizer: z.string().nullable(),
  layerSummary: z.array(layerSummarySchema),
});
type Props = z.infer<typeof propsSchema>;

export const widgetMetadata: WidgetMetadata = {
  description: 'Next steps after architecture design — generate code, setup training, or edit',
  props: propsSchema,
  exposeAsTool: false,
  metadata: {
    invoking: 'Preparing next steps…',
    invoked: 'What would you like to do next?',
  },
};

export default function NextSteps() {
  const { props, isPending, sendFollowUpMessage } = useWidget<Props>();
  const { callToolAsync: callGenerateCode, isPending: isGenerating } = useCallTool('generate-pytorch-code');
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeAction, setActiveAction] = useState<string | null>(null);

  const handleGenerateCode = useCallback(async () => {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await callGenerateCode({} as any);
      const sc = result?.structuredContent as { code?: string; errors?: string[] } | undefined;
      if (sc?.errors?.length) {
        setGeneratedCode(`# Validation errors:\n${sc.errors.map(e => `# - ${e}`).join('\n')}`);
      } else {
        setGeneratedCode(sc?.code ?? '# Error generating code');
      }
    } catch {
      setGeneratedCode('# Failed to generate code');
    }
  }, [callGenerateCode]);

  const handleCopy = useCallback(() => {
    if (!generatedCode) return;
    navigator.clipboard.writeText(generatedCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [generatedCode]);

  const handleAction = useCallback((id: string, message: string) => {
    setActiveAction(id);
    sendFollowUpMessage(message);
  }, [sendFollowUpMessage]);

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{
          padding: 32,
          backgroundColor: '#0d0d0d',
          color: '#444',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          fontSize: 13,
        }}>
          Loading…
        </div>
      </McpUseProvider>
    );
  }

  const { nodeCount, edgeCount, taskType, taskDescription, suggestedLoss, suggestedOptimizer, layerSummary } = props;

  const actions: Array<{
    id: string;
    title: string;
    desc: string;
    icon: string;
    color: string;
    bg: string;
    border: string;
  }> = [
    {
      id: 'generate',
      title: 'Generate PyTorch Code',
      desc: 'Convert to a runnable nn.Module with training loop scaffold',
      icon: '{ }',
      color: '#4ade80',
      bg: '#071a07',
      border: '#166534',
    },
    {
      id: 'train',
      title: 'Setup Training',
      desc: 'Pick a dataset, configure hyperparameters, generate training scripts',
      icon: '▶',
      color: '#60a5fa',
      bg: '#070f1a',
      border: '#1e40af',
    },
    {
      id: 'edit',
      title: 'Edit Architecture',
      desc: 'Re-open the visual builder to modify layers or connections',
      icon: '✎',
      color: '#c084fc',
      bg: '#0f071a',
      border: '#6b21a8',
    },
    {
      id: 'explain',
      title: 'Explain Architecture',
      desc: 'Detailed breakdown of layers, data flow, and parameter counts',
      icon: '?',
      color: '#fbbf24',
      bg: '#1a1407',
      border: '#92400e',
    },
  ];

  return (
    <McpUseProvider autoSize>
      <div style={{
        backgroundColor: '#0d0d0d',
        color: '#e0e0e0',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 13,
        overflow: 'hidden',
      }}>

        {/* Summary banner */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid #1a1a1a',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            backgroundColor: '#4ade80',
            boxShadow: '0 0 8px #4ade80',
            flexShrink: 0,
          }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#d0d0d0' }}>
              Architecture Complete
            </div>
            <div style={{ fontSize: 11, color: '#666', marginTop: 3, lineHeight: 1.5 }}>
              {nodeCount} layer{nodeCount !== 1 ? 's' : ''} · {edgeCount} connection{edgeCount !== 1 ? 's' : ''}
              {taskDescription && <span> · {taskDescription}</span>}
            </div>
          </div>
          {/* Layer pills */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: '50%' }}>
            {layerSummary.slice(0, 6).map(l => (
              <span key={l.type} style={{
                fontSize: 9,
                padding: '2px 6px',
                borderRadius: 3,
                backgroundColor: '#111',
                border: '1px solid #222',
                color: '#888',
                fontWeight: 600,
                whiteSpace: 'nowrap',
              }}>
                {l.type}{l.count > 1 ? ` ×${l.count}` : ''}
              </span>
            ))}
            {layerSummary.length > 6 && (
              <span style={{
                fontSize: 9, padding: '2px 6px', borderRadius: 3,
                backgroundColor: '#111', border: '1px solid #222',
                color: '#555', fontWeight: 600,
              }}>
                +{layerSummary.length - 6} more
              </span>
            )}
          </div>
        </div>

        {/* Detected config */}
        {(suggestedLoss || suggestedOptimizer || taskType) && (
          <div style={{
            padding: '8px 20px',
            borderBottom: '1px solid #141414',
            display: 'flex',
            gap: 12,
            fontSize: 10,
            color: '#555',
          }}>
            {taskType && taskType !== 'unknown' && (
              <span>Task: <span style={{ color: '#888' }}>{taskType}</span></span>
            )}
            {suggestedLoss && (
              <span>Loss: <span style={{ color: '#888' }}>{suggestedLoss}</span></span>
            )}
            {suggestedOptimizer && (
              <span>Optimizer: <span style={{ color: '#888' }}>{suggestedOptimizer}</span></span>
            )}
          </div>
        )}

        {/* Action cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: 10,
          padding: '16px 20px',
        }}>
          {actions.map(a => {
            const isActive = activeAction === a.id;
            const isLoading = a.id === 'generate' && isGenerating;

            return (
              <button
                key={a.id}
                disabled={isLoading || isActive}
                onClick={() => {
                  if (a.id === 'generate') {
                    handleGenerateCode();
                  } else if (a.id === 'train') {
                    handleAction(a.id, 'Set up training for my current architecture. Use prepare-train with the saved design.');
                  } else if (a.id === 'edit') {
                    handleAction(a.id, 'Re-open the architecture builder so I can edit the current design. Use rerender-builder.');
                  } else if (a.id === 'explain') {
                    handleAction(a.id, 'Explain my current architecture in detail. Use get-current-design to read the graph, then describe each layer, the data flow, estimated parameter count, and any suggestions.');
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 12,
                  padding: '14px 16px',
                  backgroundColor: isActive ? '#111' : a.bg,
                  border: `1px solid ${isActive ? '#333' : a.border}`,
                  borderRadius: 8,
                  cursor: (isLoading || isActive) ? 'not-allowed' : 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s',
                  opacity: isActive ? 0.5 : 1,
                }}
              >
                <div style={{
                  width: 32, height: 32,
                  borderRadius: 6,
                  backgroundColor: `${a.color}11`,
                  border: `1px solid ${a.color}33`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 14,
                  color: a.color,
                  fontWeight: 700,
                  fontFamily: 'monospace',
                  flexShrink: 0,
                }}>
                  {isLoading ? '…' : a.icon}
                </div>
                <div>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 700,
                    color: isActive ? '#555' : a.color,
                    marginBottom: 3,
                  }}>
                    {isLoading ? 'Generating…' : isActive ? `${a.title} — triggered` : a.title}
                  </div>
                  <div style={{ fontSize: 10, color: '#555', lineHeight: 1.4 }}>
                    {a.desc}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Code modal */}
        {generatedCode && (
          <div style={{
            margin: '0 20px 16px',
            backgroundColor: '#0a0a0a',
            border: '1px solid #1e1e1e',
            borderRadius: 8,
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '8px 12px',
              borderBottom: '1px solid #141414',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                backgroundColor: '#4ade80',
                boxShadow: '0 0 4px #4ade80',
              }} />
              <span style={{ fontSize: 11, fontWeight: 600, color: '#888' }}>
                Generated PyTorch Model
              </span>
              <div style={{ flex: 1 }} />
              <button
                onClick={handleCopy}
                style={{
                  padding: '3px 8px',
                  backgroundColor: copied ? '#0a2a0a' : '#141414',
                  border: `1px solid ${copied ? '#166534' : '#252525'}`,
                  borderRadius: 3,
                  color: copied ? '#4ade80' : '#666',
                  cursor: 'pointer',
                  fontSize: 9,
                  fontWeight: 600,
                }}
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
              <button
                onClick={() => setGeneratedCode(null)}
                style={{
                  background: 'none', border: 'none',
                  color: '#444', cursor: 'pointer', fontSize: 14,
                  lineHeight: 1, padding: '0 2px',
                }}
              >
                ×
              </button>
            </div>
            <pre style={{
              maxHeight: 320,
              overflowY: 'auto',
              padding: '12px 14px',
              margin: 0,
              fontSize: 10.5,
              lineHeight: 1.6,
              color: '#a8e6a3',
              fontFamily: '"Fira Code", "Cascadia Code", "JetBrains Mono", monospace',
              whiteSpace: 'pre',
            }}>
              {generatedCode}
            </pre>
          </div>
        )}
      </div>
    </McpUseProvider>
  );
}
