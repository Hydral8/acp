import { useState, useCallback, useRef, useEffect } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';
import { GraphNode, GraphEdge, BLOCK_DEFS, PendingConn, OPTIMIZER_TYPES, LOSS_TYPES } from './types';
import { Canvas } from './components/Canvas';
import { BlockLibrary } from './components/BlockLibrary';
import { PropertiesPanel } from './components/PropertiesPanel';

const propsSchema = z.object({});
type Props = z.infer<typeof propsSchema>;

export const widgetMetadata: WidgetMetadata = {
  description: 'Visual ML architecture builder with drag-and-drop blocks, graph validation, and PyTorch code generation',
  props: propsSchema,
  exposeAsTool: false,
  metadata: {
    invoking: 'Opening architecture builder...',
    invoked: 'Architecture builder ready',
  },
};

let _nodeCounter = 0;
let _edgeCounter = 0;
const newNodeId = () => `n${++_nodeCounter}`;
const newEdgeId = () => `e${++_edgeCounter}`;

export default function MLArchitectureBuilder() {
  const { isPending } = useWidget<Props>();
  const { callToolAsync: callGenerate, isPending: isGenerating } = useCallTool('generate-pytorch-code');

  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [pendingConn, setPendingConn] = useState<PendingConn | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLPreElement>(null);

  // ── Graph mutations ──────────────────────────────────────────────────────

  const addNode = useCallback((type: string, x: number, y: number) => {
    const def = BLOCK_DEFS.find(d => d.type === type)!;
    setNodes(prev => [...prev, {
      id: newNodeId(), type, x, y,
      parameters: { ...def.defaultParams },
    }]);
  }, []);

  const moveNode = useCallback((id: string, x: number, y: number) => {
    setNodes(prev => prev.map(n => n.id === id ? { ...n, x, y } : n));
  }, []);

  const addEdge = useCallback((sourceId: string, targetId: string) => {
    setEdges(prev => {
      if (prev.some(e => e.sourceId === sourceId && e.targetId === targetId)) return prev;
      return [...prev, { id: newEdgeId(), sourceId, targetId }];
    });
  }, []);

  const deleteEdge = useCallback((id: string) => {
    setEdges(prev => prev.filter(e => e.id !== id));
  }, []);

  const deleteNode = useCallback((id: string) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setEdges(prev => prev.filter(e => e.sourceId !== id && e.targetId !== id));
    setSelectedNodeId(null);
  }, []);

  // ── Delete key ───────────────────────────────────────────────────────────

  // Use a ref so the effect closure always sees the latest selectedNodeId
  const selectedNodeIdRef = useRef(selectedNodeId);
  selectedNodeIdRef.current = selectedNodeId;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return; // don't fire while editing params
      const id = selectedNodeIdRef.current;
      if (id) deleteNode(id);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteNode]);

  const updateParam = useCallback((nodeId: string, key: string, value: unknown) => {
    setNodes(prev => prev.map(n =>
      n.id === nodeId ? { ...n, parameters: { ...n.parameters, [key]: value } } : n
    ));
  }, []);

  // ── Generate ─────────────────────────────────────────────────────────────

  const handleGenerate = useCallback(async () => {
    const errs: string[] = [];
    const inputNodes = nodes.filter(n => n.type === 'Input');
    if (inputNodes.length === 0) errs.push('Add at least one Input block.');
    if (inputNodes.length > 1) errs.push('Only one Input block is allowed.');
    if (!nodes.some(n => OPTIMIZER_TYPES.has(n.type))) errs.push('Add an Optimizer block (SGD or Adam).');
    if (!nodes.some(n => LOSS_TYPES.has(n.type))) errs.push('Add a Loss block (MSELoss or CrossEntropyLoss).');
    if (errs.length > 0) { setErrors(errs); return; }
    setErrors([]);

    const graph = {
      nodes: nodes.map(n => ({ id: n.id, type: n.type, parameters: n.parameters })),
      edges: edges.map(e => ({ id: e.id, sourceId: e.sourceId, targetId: e.targetId })),
    };
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await callGenerate({ graph } as any);
      const code = (result?.structuredContent as { code?: string } | undefined)?.code ?? '# Error generating code';
      setGeneratedCode(code);
    } catch {
      setGeneratedCode('# Failed to call generate-pytorch-code tool');
    }
  }, [nodes, edges, callGenerate]);

  const handleCopy = useCallback(() => {
    if (!generatedCode) return;
    navigator.clipboard.writeText(generatedCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [generatedCode]);

  // ── Derived values ────────────────────────────────────────────────────────

  const selectedNode = nodes.find(n => n.id === selectedNodeId) ?? null;
  const optimizerNode = nodes.find(n => OPTIMIZER_TYPES.has(n.type)) ?? null;
  const lossNode = nodes.find(n => LOSS_TYPES.has(n.type)) ?? null;

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{
          height: 640,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0d0d0d',
          color: '#333',
          fontFamily: 'monospace',
          fontSize: 12,
        }}>
          Initializing...
        </div>
      </McpUseProvider>
    );
  }

  return (
    <McpUseProvider autoSize>
      <div style={{
        height: 640,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#0d0d0d',
        color: '#e0e0e0',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 13,
        overflow: 'hidden',
        position: 'relative',
      }}>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div style={{
          height: 46,
          borderBottom: '1px solid #141414',
          display: 'flex',
          alignItems: 'center',
          padding: '0 14px',
          gap: 10,
          backgroundColor: '#070707',
          flexShrink: 0,
        }}>
          {/* Title */}
          <div style={{ fontSize: 12, fontWeight: 700, color: '#d0d0d0', letterSpacing: 0.3 }}>
            ML Architecture Builder
          </div>

          <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />

          <div style={{ fontSize: 10, color: '#333' }}>
            {nodes.length} block{nodes.length !== 1 ? 's' : ''} · {edges.length} edge{edges.length !== 1 ? 's' : ''}
          </div>

          <div style={{ flex: 1 }} />

          {/* Optimizer badge */}
          <StatusBadge label={optimizerNode ? optimizerNode.type : 'No Optimizer'} active={!!optimizerNode} />
          <StatusBadge label={lossNode ? lossNode.type : 'No Loss'} active={!!lossNode} />

          <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || nodes.length === 0}
            style={{
              padding: '5px 14px',
              backgroundColor: isGenerating ? '#0a1a0a' : '#15803d',
              border: `1px solid ${isGenerating ? '#1a3a1a' : '#166534'}`,
              borderRadius: 5,
              color: isGenerating ? '#4ade80' : '#fff',
              cursor: (isGenerating || nodes.length === 0) ? 'not-allowed' : 'pointer',
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 0.3,
              opacity: nodes.length === 0 ? 0.4 : 1,
              transition: 'background-color 0.15s',
            }}
          >
            {isGenerating ? 'Generating…' : 'Generate Model'}
          </button>
        </div>

        {/* ── Validation errors ───────────────────────────────────────────── */}
        {errors.length > 0 && (
          <div style={{
            backgroundColor: '#110808',
            borderBottom: '1px solid #2a0f0f',
            padding: '5px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            flexShrink: 0,
          }}>
            {errors.map((err, i) => (
              <span key={i} style={{ fontSize: 11, color: '#c05050' }}>• {err}</span>
            ))}
            <button
              onClick={() => setErrors([])}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: 14 }}
            >
              ×
            </button>
          </div>
        )}

        {/* ── Main body ───────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>
          <BlockLibrary />

          <Canvas
            nodes={nodes}
            edges={edges}
            selectedNodeId={selectedNodeId}
            pendingConn={pendingConn}
            onAddNode={addNode}
            onMoveNode={moveNode}
            onSelectNode={setSelectedNodeId}
            onAddEdge={addEdge}
            onDeleteEdge={deleteEdge}
            onPendingConnChange={setPendingConn}
          />

          <PropertiesPanel
            selectedNode={selectedNode}
            edges={edges}
            onParamChange={updateParam}
            onDeleteNode={deleteNode}
          />
        </div>

        {/* ── Generated code modal ────────────────────────────────────────── */}
        {generatedCode && (
          <div
            onClick={e => { if (e.target === e.currentTarget) setGeneratedCode(null); }}
            style={{
              position: 'fixed',
              inset: 0,
              backgroundColor: 'rgba(0,0,0,0.88)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 100,
            }}
          >
            <div style={{
              width: '76%',
              maxHeight: '78%',
              backgroundColor: '#0a0a0a',
              border: '1px solid #202020',
              borderRadius: 10,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              boxShadow: '0 20px 60px rgba(0,0,0,0.8)',
            }}>
              {/* Modal header */}
              <div style={{
                padding: '10px 14px',
                borderBottom: '1px solid #141414',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                flexShrink: 0,
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  backgroundColor: '#4ade80',
                  boxShadow: '0 0 6px #4ade80',
                }} />
                <span style={{ fontWeight: 600, fontSize: 12, color: '#d0d0d0' }}>
                  Generated PyTorch Model
                </span>
                <div style={{ flex: 1 }} />
                <button
                  onClick={handleCopy}
                  style={{
                    padding: '4px 10px',
                    backgroundColor: copied ? '#0a2a0a' : '#141414',
                    border: `1px solid ${copied ? '#166534' : '#252525'}`,
                    borderRadius: 4,
                    color: copied ? '#4ade80' : '#888',
                    cursor: 'pointer',
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: 0.3,
                  }}
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
                <button
                  onClick={() => setGeneratedCode(null)}
                  style={{
                    background: 'none', border: 'none',
                    color: '#444', cursor: 'pointer', fontSize: 18,
                    lineHeight: 1, padding: '0 2px',
                  }}
                >
                  ×
                </button>
              </div>

              {/* Code */}
              <pre
                ref={codeRef}
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  padding: '14px 18px',
                  margin: 0,
                  fontSize: 11.5,
                  lineHeight: 1.65,
                  color: '#a8e6a3',
                  fontFamily: '"Fira Code", "Cascadia Code", "JetBrains Mono", monospace',
                  tabSize: 4,
                  whiteSpace: 'pre',
                }}
              >
                {generatedCode}
              </pre>
            </div>
          </div>
        )}
      </div>
    </McpUseProvider>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────

function StatusBadge({ label, active }: { label: string; active: boolean }) {
  return (
    <span style={{
      fontSize: 9,
      padding: '2px 7px',
      borderRadius: 3,
      backgroundColor: active ? '#0a1a0a' : '#110808',
      color: active ? '#4ade80' : '#993333',
      border: `1px solid ${active ? '#166534' : '#3d1515'}`,
      fontWeight: 600,
      letterSpacing: 0.2,
    }}>
      {label}
    </span>
  );
}
