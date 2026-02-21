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

const ACTIONS = [
  {
    id: 'generate',
    title: 'Generate Code',
    desc: 'PyTorch nn.Module + training loop',
    color: '#4ade80',
    border: '#166534',
    bg: '#071a07',
  },
  {
    id: 'train',
    title: 'Setup Training',
    desc: 'Dataset, hyperparams, scripts',
    color: '#60a5fa',
    border: '#1e40af',
    bg: '#070f1a',
  },
  {
    id: 'edit',
    title: 'Edit Architecture',
    desc: 'Modify layers or connections',
    color: '#c084fc',
    border: '#6b21a8',
    bg: '#0f071a',
  },
  {
    id: 'explain',
    title: 'Explain Model',
    desc: 'Layer breakdown + param count',
    color: '#fbbf24',
    border: '#92400e',
    bg: '#1a1407',
  },
] as const;

const FOLLOW_UP: Record<string, string> = {
  train: 'Set up training for my current architecture. Use prepare-train with the saved design.',
  edit: 'Re-open the architecture builder so I can edit the current design. Use rerender-builder.',
  explain: 'Explain my current architecture in detail. Use get-current-design to read the graph, then describe each layer, the data flow, estimated parameter count, and any suggestions.',
};

export default function NextSteps() {
  const { props, isPending, sendFollowUpMessage } = useWidget<Props>();
  const { callToolAsync: callGenerateCode, isPending: isGenerating } = useCallTool('generate-pytorch-code');
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [triggered, setTriggered] = useState<string | null>(null);

  const handleGenerate = useCallback(async () => {
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

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{ padding: 20, backgroundColor: '#0a0a0a', color: '#333', fontFamily: 'system-ui, sans-serif', fontSize: 12 }}>
          Loading…
        </div>
      </McpUseProvider>
    );
  }

  const { nodeCount, edgeCount, taskDescription, suggestedLoss, suggestedOptimizer, taskType } = props;

  return (
    <McpUseProvider autoSize>
      <div style={{
        backgroundColor: '#0a0a0a',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 12,
      }}>

        {/* Header */}
        <div style={{ padding: '14px 18px 10px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            backgroundColor: '#4ade80',
            boxShadow: '0 0 6px #4ade80',
          }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: '#aaa', letterSpacing: 0.3 }}>
            {nodeCount} layers · {edgeCount} edges
            {taskDescription && taskType !== 'unknown' ? ` · ${taskDescription}` : ''}
          </span>
          <div style={{ flex: 1 }} />
          {suggestedOptimizer && (
            <span style={{ fontSize: 9, color: '#555', padding: '1px 5px', border: '1px solid #1a1a1a', borderRadius: 3 }}>
              {suggestedOptimizer}
            </span>
          )}
          {suggestedLoss && (
            <span style={{ fontSize: 9, color: '#555', padding: '1px 5px', border: '1px solid #1a1a1a', borderRadius: 3 }}>
              {suggestedLoss}
            </span>
          )}
        </div>

        {/* Actions */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, padding: '4px 18px 16px' }}>
          {ACTIONS.map(a => {
            const isTrig = triggered === a.id;
            const isLoad = a.id === 'generate' && isGenerating;

            return (
              <button
                key={a.id}
                disabled={isLoad || isTrig}
                onClick={() => {
                  if (a.id === 'generate') {
                    handleGenerate();
                  } else {
                    setTriggered(a.id);
                    sendFollowUpMessage(FOLLOW_UP[a.id]);
                  }
                }}
                style={{
                  padding: '12px 10px',
                  backgroundColor: isTrig ? '#111' : a.bg,
                  border: `1px solid ${isTrig ? '#222' : a.border}`,
                  borderRadius: 8,
                  cursor: (isLoad || isTrig) ? 'default' : 'pointer',
                  textAlign: 'center',
                  opacity: isTrig ? 0.4 : 1,
                  transition: 'opacity 0.15s',
                }}
              >
                <div style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: isTrig ? '#444' : a.color,
                  marginBottom: 3,
                  whiteSpace: 'nowrap',
                }}>
                  {isLoad ? 'Generating…' : a.title}
                </div>
                <div style={{ fontSize: 9, color: '#555', lineHeight: 1.3 }}>
                  {a.desc}
                </div>
              </button>
            );
          })}
        </div>

        {/* Inline code output */}
        {generatedCode && (
          <div style={{
            margin: '0 18px 14px',
            backgroundColor: '#060606',
            border: '1px solid #1a1a1a',
            borderRadius: 6,
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '6px 10px',
              borderBottom: '1px solid #141414',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}>
              <div style={{ width: 5, height: 5, borderRadius: '50%', backgroundColor: '#4ade80' }} />
              <span style={{ fontSize: 10, fontWeight: 600, color: '#666', flex: 1 }}>PyTorch Model</span>
              <button
                onClick={handleCopy}
                style={{
                  padding: '2px 7px',
                  backgroundColor: copied ? '#0a2a0a' : '#111',
                  border: `1px solid ${copied ? '#166534' : '#222'}`,
                  borderRadius: 3,
                  color: copied ? '#4ade80' : '#555',
                  cursor: 'pointer',
                  fontSize: 9,
                  fontWeight: 600,
                }}
              >
                {copied ? 'Copied' : 'Copy'}
              </button>
              <button
                onClick={() => setGeneratedCode(null)}
                style={{ background: 'none', border: 'none', color: '#333', cursor: 'pointer', fontSize: 13, lineHeight: 1, padding: 0 }}
              >
                ×
              </button>
            </div>
            <pre style={{
              maxHeight: 280,
              overflowY: 'auto',
              padding: '10px 12px',
              margin: 0,
              fontSize: 10,
              lineHeight: 1.55,
              color: '#a8e6a3',
              fontFamily: '"Fira Code", "Cascadia Code", monospace',
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
