import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';
import { GraphNode, GraphEdge, CompositeBlock, BLOCK_DEFS, PendingConn, OPTIMIZER_TYPES, LOSS_TYPES, COMPOSITE_PREBUILT_TYPES } from './types';
import { propagateShapes, type ShapeFix } from './shapeEngine';
import { Canvas } from './components/Canvas';
import { BlockLibrary } from './components/BlockLibrary';
import { PropertiesPanel } from './components/PropertiesPanel';
import { CompositeDetailModal } from './components/CompositeDetailModal';

const designEdgeSchema = z.object({
  id: z.string(),
  sourceId: z.string(),
  targetId: z.string(),
});

const designInnerNodeSchema = z.object({
  id: z.string(), type: z.string(), x: z.number(), y: z.number(),
  parameters: z.record(z.string(), z.unknown()),
});

const designCompositeSchema = z.object({
  label: z.string(),
  nodes: z.array(designInnerNodeSchema),
  edges: z.array(designEdgeSchema),
  inputNodeId: z.string(),
  outputNodeId: z.string(),
}).optional();

const designNodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  x: z.number(),
  y: z.number(),
  parameters: z.record(z.string(), z.unknown()),
  composite: designCompositeSchema,
});

const propsSchema = z.object({
  initialNodes: z.array(designNodeSchema).optional(),
  initialEdges: z.array(designEdgeSchema).optional(),
});
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
  const { isPending, props } = useWidget<Props>();
  const { callToolAsync: callGenerate, isPending: isGenerating } = useCallTool('generate-pytorch-code');
  const { callToolAsync: callSave } = useCallTool('save-design');
  const callSaveRef = useRef(callSave);
  callSaveRef.current = callSave;

  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [pendingConn, setPendingConn] = useState<PendingConn | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const [showGroupModal, setShowGroupModal] = useState(false);
  const [groupName, setGroupName] = useState('');
  const [detailNode, setDetailNode] = useState<GraphNode | null>(null);
  const [customBlocks, setCustomBlocks] = useState<CompositeBlock[]>([]);
  const codeRef = useRef<HTMLPreElement>(null);
  const initDoneRef = useRef(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Graph mutations ──────────────────────────────────────────────────────

  const addNode = useCallback((type: string, x: number, y: number, composite?: CompositeBlock) => {
    if (type === 'Custom' && composite) {
      setNodes(prev => [...prev, { id: newNodeId(), type: 'Custom', x, y, parameters: {}, composite }]);
      return;
    }
    const def = BLOCK_DEFS.find(d => d.type === type);
    setNodes(prev => [...prev, { id: newNodeId(), type, x, y, parameters: { ...(def?.defaultParams ?? {}) } }]);
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
    setSelectedIds(prev => prev.filter(s => s !== id));
  }, []);

  const deleteSelected = useCallback(() => {
    setNodes(prev => prev.filter(n => !selectedIds.includes(n.id)));
    setEdges(prev => prev.filter(e => !selectedIds.includes(e.sourceId) && !selectedIds.includes(e.targetId)));
    setSelectedIds([]);
  }, [selectedIds]);

  // ── Selection ─────────────────────────────────────────────────────────────

  const handleSelectNode = useCallback((id: string, addToSelection: boolean) => {
    setSelectedIds(prev => {
      if (addToSelection) {
        return prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id];
      }
      return [id];
    });
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedIds([]);
  }, []);

  // ── Delete key ───────────────────────────────────────────────────────────

  const selectedIdsRef = useRef(selectedIds);
  selectedIdsRef.current = selectedIds;
  const deleteSelectedRef = useRef(deleteSelected);
  deleteSelectedRef.current = deleteSelected;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (selectedIdsRef.current.length > 0) deleteSelectedRef.current();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ── Initialize from props (design-architecture preload) ───────────────────

  useEffect(() => {
    if (isPending || initDoneRef.current) return;
    initDoneRef.current = true;
    if (props?.initialNodes?.length) setNodes(props.initialNodes as GraphNode[]);
    if (props?.initialEdges?.length) setEdges(props.initialEdges as GraphEdge[]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPending]);

  // ── Auto-save current design to server (for get-current-design) ───────────

  useEffect(() => {
    if (!initDoneRef.current) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      const graph = {
        nodes: nodes.map(n => ({
          id: n.id, type: n.type, parameters: n.parameters,
          composite: n.composite ? {
            label: n.composite.label,
            nodes: n.composite.nodes.map(cn => ({ id: cn.id, type: cn.type, parameters: cn.parameters })),
            edges: n.composite.edges.map(ce => ({ id: ce.id, sourceId: ce.sourceId, targetId: ce.targetId })),
            inputNodeId: n.composite.inputNodeId,
            outputNodeId: n.composite.outputNodeId,
          } : undefined,
        })),
        edges: edges.map(e => ({ id: e.id, sourceId: e.sourceId, targetId: e.targetId })),
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      callSaveRef.current({ graph } as any).catch(() => {});
    }, 1500);
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [nodes, edges]);

  const updateParam = useCallback((nodeId: string, key: string, value: unknown) => {
    setNodes(prev => prev.map(n =>
      n.id === nodeId ? { ...n, parameters: { ...n.parameters, [key]: value } } : n
    ));
  }, []);

  // ── Shape propagation ──────────────────────────────────────────────────────

  const shapeMap = useMemo(() => propagateShapes(nodes, edges), [nodes, edges]);

  const handleFixConflict = useCallback((fix: ShapeFix) => {
    updateParam(fix.nodeId, fix.key, fix.value);
  }, [updateParam]);

  // ── Group selection into composite block ──────────────────────────────────

  const groupSelection = useCallback((label: string) => {
    if (selectedIds.length < 2) return;

    const selectedSet = new Set(selectedIds);
    const selNodes = nodes.filter(n => selectedSet.has(n.id));

    // Internal edges: both endpoints selected
    const internalEdges = edges.filter(
      e => selectedSet.has(e.sourceId) && selectedSet.has(e.targetId)
    );
    // External edges: exactly one endpoint is selected
    const externalEdges = edges.filter(
      e => (selectedSet.has(e.sourceId) !== selectedSet.has(e.targetId))
    );

    // Input node: first selected node that receives from outside
    const inputNode = selNodes.find(n =>
      externalEdges.some(e => e.targetId === n.id && !selectedSet.has(e.sourceId))
    ) ?? selNodes[0];

    // Output node: first selected node that sends to outside
    const outputNode = selNodes.find(n =>
      externalEdges.some(e => e.sourceId === n.id && !selectedSet.has(e.targetId))
    ) ?? selNodes[selNodes.length - 1];

    // Centroid position
    const cx = selNodes.reduce((s, n) => s + n.x, 0) / selNodes.length;
    const cy = selNodes.reduce((s, n) => s + n.y, 0) / selNodes.length;

    const compositeId = newNodeId();
    const compositeNode: GraphNode = {
      id: compositeId,
      type: 'Custom',
      x: cx,
      y: cy,
      parameters: {},
      composite: {
        label,
        nodes: selNodes,
        edges: internalEdges,
        inputNodeId: inputNode.id,
        outputNodeId: outputNode.id,
      },
    };

    // Redirect external edges: change selected-endpoint to compositeId
    const redirectedEdges: GraphEdge[] = externalEdges.map(e => ({
      ...e,
      id: newEdgeId(),
      sourceId: selectedSet.has(e.sourceId) ? compositeId : e.sourceId,
      targetId: selectedSet.has(e.targetId) ? compositeId : e.targetId,
    }));

    setNodes(prev => [
      ...prev.filter(n => !selectedSet.has(n.id)),
      compositeNode,
    ]);
    setEdges(prev => [
      ...prev.filter(e => !selectedSet.has(e.sourceId) && !selectedSet.has(e.targetId)),
      ...redirectedEdges,
    ]);
    setSelectedIds([compositeId]);
    // Register in library for reuse
    if (compositeNode.composite) setCustomBlocks(prev => [...prev, compositeNode.composite!]);
  }, [selectedIds, nodes, edges]);

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
      nodes: nodes.map(n => ({
        id: n.id, type: n.type, parameters: n.parameters,
        composite: n.composite ? {
          label: n.composite.label,
          nodes: n.composite.nodes.map(cn => ({ id: cn.id, type: cn.type, parameters: cn.parameters })),
          edges: n.composite.edges.map(ce => ({ id: ce.id, sourceId: ce.sourceId, targetId: ce.targetId })),
          inputNodeId: n.composite.inputNodeId,
          outputNodeId: n.composite.outputNodeId,
        } : undefined,
      })),
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

  // ── Composite detail + custom library ────────────────────────────────────

  const handleDoubleClickNode = useCallback((nodeId: string) => {
    const n = nodes.find(nd => nd.id === nodeId);
    if (n && (n.type === 'Custom' || COMPOSITE_PREBUILT_TYPES.has(n.type)) && n.composite) {
      setDetailNode(n);
    }
  }, [nodes]);

  const removeCustomBlock = useCallback((idx: number) => {
    setCustomBlocks(prev => prev.filter((_, i) => i !== idx));
  }, []);

  // ── Derived values ────────────────────────────────────────────────────────

  const selectedNodes = nodes.filter(n => selectedIds.includes(n.id));
  const optimizerNode = nodes.find(n => OPTIMIZER_TYPES.has(n.type)) ?? null;
  const lossNode = nodes.find(n => LOSS_TYPES.has(n.type)) ?? null;
  const canGroup = selectedIds.length >= 2;

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

          {/* Group button — visible when 2+ blocks selected */}
          {canGroup && (
            <button
              onClick={() => { setGroupName(''); setShowGroupModal(true); }}
              style={{
                padding: '5px 12px',
                backgroundColor: '#1a1200',
                border: '1px solid #ca8a04',
                borderRadius: 5,
                color: '#fde68a',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: 0.3,
              }}
            >
              Group {selectedIds.length} blocks
            </button>
          )}

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
          <BlockLibrary
            customBlocks={customBlocks}
            onRemoveCustomBlock={removeCustomBlock}
          />

          <Canvas
            nodes={nodes}
            edges={edges}
            selectedIds={selectedIds}
            pendingConn={pendingConn}
            shapeMap={shapeMap}
            onAddNode={addNode}
            onMoveNode={moveNode}
            onSelectNode={handleSelectNode}
            onClearSelection={handleClearSelection}
            onAddEdge={addEdge}
            onDeleteEdge={deleteEdge}
            onPendingConnChange={setPendingConn}
            onFixConflict={handleFixConflict}
            onDoubleClickNode={handleDoubleClickNode}
          />

          <PropertiesPanel
            selectedNodes={selectedNodes}
            edges={edges}
            onParamChange={updateParam}
            onDeleteNode={deleteNode}
          />
        </div>

        {/* ── Composite detail modal (double-click) ──────────────────────── */}
        {detailNode && (
          <CompositeDetailModal node={detailNode} onClose={() => setDetailNode(null)} />
        )}

        {/* ── Group name modal ─────────────────────────────────────────────── */}
        {showGroupModal && (
          <div
            onClick={e => { if (e.target === e.currentTarget) setShowGroupModal(false); }}
            style={{
              position: 'fixed', inset: 0,
              backgroundColor: 'rgba(0,0,0,0.8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              zIndex: 200,
            }}
          >
            <div style={{
              backgroundColor: '#0d0d0d',
              border: '1px solid #252525',
              borderRadius: 10,
              padding: 24,
              width: 320,
              boxShadow: '0 20px 60px rgba(0,0,0,0.8)',
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#d0d0d0', marginBottom: 16 }}>
                Name your composite block
              </div>
              <input
                autoFocus
                type="text"
                value={groupName}
                onChange={e => setGroupName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && groupName.trim()) {
                    groupSelection(groupName.trim());
                    setShowGroupModal(false);
                  }
                  if (e.key === 'Escape') setShowGroupModal(false);
                }}
                placeholder="e.g. Encoder Block"
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  backgroundColor: '#111',
                  border: '1px solid #ca8a04',
                  borderRadius: 5,
                  color: '#fde68a',
                  fontSize: 12,
                  outline: 'none',
                  boxSizing: 'border-box',
                  fontFamily: 'inherit',
                  marginBottom: 16,
                }}
              />
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button
                  onClick={() => setShowGroupModal(false)}
                  style={{
                    padding: '6px 14px',
                    backgroundColor: 'transparent',
                    border: '1px solid #333',
                    borderRadius: 5,
                    color: '#666',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    if (groupName.trim()) {
                      groupSelection(groupName.trim());
                      setShowGroupModal(false);
                    }
                  }}
                  disabled={!groupName.trim()}
                  style={{
                    padding: '6px 14px',
                    backgroundColor: groupName.trim() ? '#1a1200' : '#111',
                    border: `1px solid ${groupName.trim() ? '#ca8a04' : '#333'}`,
                    borderRadius: 5,
                    color: groupName.trim() ? '#fde68a' : '#444',
                    cursor: groupName.trim() ? 'pointer' : 'not-allowed',
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                >
                  Create Block
                </button>
              </div>
            </div>
          </div>
        )}

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
