import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';
import { GraphNode, GraphEdge, CompositeBlock, BLOCK_DEFS, PendingConn, OPTIMIZER_TYPES, LOSS_TYPES, COMPOSITE_PREBUILT_TYPES } from './types';
import { propagateShapes, type ShapeFix, type NodeShapeInfo } from './shapeEngine';
import { Canvas } from './components/Canvas';
import { BlockLibrary } from './components/BlockLibrary';
import { PropertiesPanel } from './components/PropertiesPanel';
import { CompositeDetailModal } from './components/CompositeDetailModal';

// ── Train-view types & data ────────────────────────────────────────────────

type ViewMode = 'design' | 'train' | 'infer';
type CatKey = 'llm' | 'vlm' | 'rlhf' | 'cv' | 'tabular';
interface DatasetDef { id: string; name: string; hfId: string; description: string; size: string; task: string; }
interface GeneratedFiles { modelPy: string; dataPy: string; trainPy: string; requirementsTxt: string; wandbProject?: string; }
interface GeneratedInference { inferencePy: string; taskType: string; inputFormat: string; }
interface InferResult { result?: Record<string, unknown>; rawOutput?: string; error?: string; }

type DeployStepStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped';
interface DeployStepState { status: DeployStepStatus; detail?: string; }
interface DeployStepsState {
  generate: DeployStepState;
  check: DeployStepState;
  install: DeployStepState;
  upload: DeployStepState;
}

const TRAIN_DATASETS: Record<CatKey, DatasetDef[]> = {
  llm: [
    { id: 'tinystories', name: 'TinyStories',  hfId: 'roneneldan/TinyStories',            description: 'Simple stories for small LMs',              size: '2.1 GB',  task: 'Causal LM' },
    { id: 'openwebtext', name: 'OpenWebText',   hfId: 'Skylion007/openwebtext',             description: 'Open replication of WebText corpus',        size: '40 GB',   task: 'Causal LM' },
    { id: 'minipile',    name: 'MiniPile',      hfId: 'JeanKaddour/minipile',              description: 'Compact 1M-sample subset of The Pile',      size: '6.4 GB',  task: 'Causal LM' },
    { id: 'wikitext',    name: 'WikiText-103',  hfId: 'Salesforce/wikitext',               description: 'Wikipedia articles — standard LM benchmark', size: '510 MB',  task: 'Causal LM' },
  ],
  vlm: [
    { id: 'cc3m',     name: 'Conceptual Captions', hfId: 'google-research-datasets/conceptual_captions', description: 'Image–caption pairs from Google', size: '3M pairs',  task: 'Image–Text' },
    { id: 'flickr30k',name: 'Flickr30K',            hfId: 'nlphuji/flickr30k',                            description: '31K images, 5 captions each',    size: '31K pairs', task: 'Image–Text' },
    { id: 'laion',    name: 'LAION Aesthetics',      hfId: 'laion/laion-aesthetic-v2',                     description: 'High-aesthetic LAION subset',     size: '900K pairs',task: 'Image–Text' },
  ],
  rlhf: [
    { id: 'hh-rlhf',   name: 'Anthropic HH-RLHF', hfId: 'Anthropic/hh-rlhf',                     description: 'Helpful & harmless human preferences', size: '170K pairs', task: 'RLHF' },
    { id: 'shp',        name: 'Stanford SHP',       hfId: 'stanfordnlp/SHP',                        description: 'Reddit Q&A with upvote preferences',  size: '385K pairs', task: 'RLHF' },
    { id: 'summarize',  name: 'OpenAI Summarize',   hfId: 'openai/summarize_from_feedback',          description: 'RLHF feedback on TL;DR summaries',   size: '92K pairs',  task: 'RLHF' },
  ],
  cv: [
    { id: 'mnist',     name: 'MNIST',        hfId: 'ylecun/mnist',           description: '70K handwritten digit images (28×28)',   size: '11 MB',   task: 'Classification' },
    { id: 'cifar10',   name: 'CIFAR-10',     hfId: 'uoft-cs/cifar10',        description: '60K color images (32×32, RGB), 10 classes',size: '163 MB', task: 'Classification' },
    { id: 'imagenet1k',name: 'ImageNet-1K',  hfId: 'ILSVRC/imagenet-1k',     description: '1.28M images, 1000 classes',             size: '150 GB',  task: 'Classification' },
    { id: 'food101',   name: 'Food-101',     hfId: 'ethz/food101',           description: '101K food images, 101 categories',       size: '4.7 GB',  task: 'Classification' },
  ],
  tabular: [
    { id: 'iris',      name: 'Iris',             hfId: 'scikit-learn/iris',                 description: '150 samples, 4 features, 3 classes',     size: '<1 MB',  task: 'Classification' },
    { id: 'titanic',   name: 'Titanic',           hfId: 'mstz/titanic',                      description: 'Survival prediction from passengers',    size: '<1 MB',  task: 'Classification' },
    { id: 'adult',     name: 'Adult Income',      hfId: 'scikit-learn/adult-census-income',  description: 'Census binary income classification',    size: '3.8 MB', task: 'Classification' },
    { id: 'california',name: 'California Housing',hfId: 'lhoestq/demo-wikipedia-stats',      description: 'House value regression from census data',size: '<1 MB',  task: 'Regression' },
  ],
};

const TRAIN_ACCENT:     Record<CatKey, string> = { llm: '#3b82f6', vlm: '#8b5cf6', rlhf: '#10b981', cv: '#f59e0b', tabular: '#06b6d4' };
const TRAIN_BLOB_ICON:  Record<CatKey, string> = { llm: '📝', vlm: '🖼️', rlhf: '🎯', cv: '🖼', tabular: '📊' };
const TRAIN_BLOB_LABEL: Record<CatKey, string> = { llm: 'Text Tokens', vlm: 'Image Patches', rlhf: 'Pref. Pairs', cv: 'Image Batch', tabular: 'Row Batch' };
const TRAIN_CAT_TABS: Array<{ key: CatKey; label: string }> = [
  { key: 'llm', label: 'LLM' }, { key: 'vlm', label: 'VLM' },
  { key: 'rlhf', label: 'RL/RLHF' }, { key: 'cv', label: 'Vision' }, { key: 'tabular', label: 'Tabular' },
];

// Preprocessing definition per category
const PREPROCESS_LABEL: Record<CatKey, string> = {
  llm: 'Tokenize + Pad', vlm: 'Resize + Normalize', rlhf: 'Tokenize + Pair',
  cv: 'Resize + Normalize', tabular: 'Scale + Encode',
};
// Default output shape the preprocess step emits (→ must match Input node shape)
const PREPROCESS_OUT_SHAPE: Record<CatKey, number[]> = {
  llm: [512], vlm: [3, 224, 224], rlhf: [512], cv: [3, 32, 32], tabular: [4],
};

const TRAIN_NODE_COLORS: Record<string, string> = {
  core: '#2563eb', activation: '#7c3aed', structural: '#059669', training: '#d97706', composite: '#ca8a04',
};

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
  codeOverride: z.string().optional(),
});

const propsSchema = z.object({
  initialNodes: z.array(designNodeSchema).optional(),
  initialEdges: z.array(designEdgeSchema).optional(),
  initialMode: z.enum(['design', 'train', 'infer']).optional(),
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
  const { callToolAsync: callGenTraining, isPending: isGeneratingTraining } = useCallTool('generate-training-code');
  const { callToolAsync: callGetTraining } = useCallTool('get-training-code');
  const { callToolAsync: callGenInference, isPending: isGeneratingInference } = useCallTool('generate-inference-code');
  const { callToolAsync: callRunInference, isPending: isRunningInference } = useCallTool('run-inference');
  const { callToolAsync: callCheckGpuPkgs } = useCallTool('check-gpu-packages');
  const { callToolAsync: callSetupGpu } = useCallTool('setup-gpu');
  const { callToolAsync: callUploadScripts } = useCallTool('upload-scripts');
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
  const simTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── View / UI state ──────────────────────────────────────────────────────
  const [viewMode, setViewMode] = useState<ViewMode>('design');
  const [showProps, setShowProps] = useState(true);

  // ── Train state ───────────────────────────────────────────────────────────
  const [trainCategory, setTrainCategory] = useState<CatKey>('llm');
  const [trainSelected, setTrainSelected] = useState<DatasetDef | null>(null);
  const [trainLoss, setTrainLoss] = useState('CrossEntropyLoss');
  const [trainOptimizer, setTrainOptimizer] = useState('Adam');
  const [trainLR, setTrainLR] = useState(0.001);
  const [trainBatchSize, setTrainBatchSize] = useState(32);
  const [trainEpochs, setTrainEpochs] = useState(10);
  const [simRunning, setSimRunning] = useState(false);
  const [animStep, setAnimStep] = useState(0);
  const [generatedFiles, setGeneratedFiles] = useState<GeneratedFiles | null>(null);
  const [trainCopied, setTrainCopied] = useState<'model' | 'data' | 'train' | 'req' | null>(null);
  const [trainWandbProject, setTrainWandbProject] = useState('my-model');

  // ── Infer state ───────────────────────────────────────────────────────────
  const [inferInput, setInferInput] = useState('');
  const [inferInputType, setInferInputType] = useState<'text' | 'image_url' | 'image_base64' | 'tabular'>('text');
  const [inferPodId, setInferPodId] = useState('');
  const [inferResult, setInferResult] = useState<InferResult | null>(null);
  const [inferGenerated, setInferGenerated] = useState<GeneratedInference | null>(null);
  const [inferScriptCopied, setInferScriptCopied] = useState(false);

  // ── Deploy (Run) state ────────────────────────────────────────────────────
  const [trainPodId, setTrainPodId] = useState('');
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [deployPhase, setDeployPhase] = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploySteps, setDeploySteps] = useState<DeployStepsState>({
    generate: { status: 'pending' },
    check: { status: 'pending' },
    install: { status: 'pending' },
    upload: { status: 'pending' },
  });

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
    if (props?.initialMode) setViewMode(props.initialMode as ViewMode);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPending]);

  // ── Auto-save current design to server (for get-current-design) ───────────

  useEffect(() => {
    if (!initDoneRef.current) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      const graph = {
        nodes: nodes.map(n => ({
          id: n.id, type: n.type, x: n.x, y: n.y, parameters: n.parameters,
          codeOverride: n.codeOverride,
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

  // ── Train: reset dataset when category changes ────────────────────────────
  useEffect(() => { setTrainSelected(null); }, [trainCategory]);

  // ── Train: stop sim when switching to design mode ─────────────────────────
  useEffect(() => {
    if (viewMode === 'design') { setSimRunning(false); setAnimStep(0); }
  }, [viewMode]);

  // ── Train: restore previously generated scripts from server ───────────────
  const trainScriptsFetchedRef = useRef(false);
  useEffect(() => {
    if (viewMode !== 'train' || trainScriptsFetchedRef.current || generatedFiles) return;
    trainScriptsFetchedRef.current = true;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    callGetTraining({ file: 'all' } as any).then(result => {
      const sc = result?.structuredContent as { found?: boolean; modelPy?: string; dataPy?: string; trainPy?: string; wandbProject?: string } | undefined;
      if (sc?.found && (sc.modelPy || sc.trainPy)) {
        setGeneratedFiles({ modelPy: sc.modelPy ?? '', dataPy: sc.dataPy ?? '', trainPy: sc.trainPy ?? '', requirementsTxt: '', wandbProject: sc.wandbProject });
      }
    }).catch(() => {});
  }, [viewMode]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateParam = useCallback((nodeId: string, key: string, value: unknown) => {
    setNodes(prev => prev.map(n =>
      n.id === nodeId ? { ...n, parameters: { ...n.parameters, [key]: value } } : n
    ));
  }, []);

  const updateCodeOverride = useCallback((nodeId: string, code: string | undefined) => {
    setNodes(prev => prev.map(n =>
      n.id === nodeId ? { ...n, codeOverride: code } : n
    ));
  }, []);

  // ── Shape propagation ──────────────────────────────────────────────────────

  const shapeMap = useMemo(() => propagateShapes(nodes, edges), [nodes, edges]);

  // ── Train-mode derived values ─────────────────────────────────────────────
  const trainSortedNodes = useMemo(() => {
    const dataNodes = nodes.filter(n => !OPTIMIZER_TYPES.has(n.type) && !LOSS_TYPES.has(n.type));
    const dataEdges = edges.filter(e =>
      dataNodes.some(n => n.id === e.sourceId) && dataNodes.some(n => n.id === e.targetId)
    );
    const adj = new Map<string, string[]>();
    const inDeg = new Map<string, number>();
    for (const n of dataNodes) { adj.set(n.id, []); inDeg.set(n.id, 0); }
    for (const e of dataEdges) {
      adj.get(e.sourceId)?.push(e.targetId);
      inDeg.set(e.targetId, (inDeg.get(e.targetId) ?? 0) + 1);
    }
    const queue = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
    const sorted: string[] = [];
    while (queue.length > 0) {
      const id = queue.shift()!; sorted.push(id);
      for (const nxt of adj.get(id) ?? []) {
        inDeg.set(nxt, inDeg.get(nxt)! - 1);
        if (inDeg.get(nxt) === 0) queue.push(nxt);
      }
    }
    // append any remaining (cycles / disconnected)
    for (const n of dataNodes) if (!sorted.includes(n.id)) sorted.push(n.id);
    return sorted.map(id => dataNodes.find(n => n.id === id)!).filter(Boolean);
  }, [nodes, edges]);

  const inputNode = useMemo(() => nodes.find(n => n.type === 'Input') ?? null, [nodes]);

  const trainValidations = useMemo(() => {
    const inputNodes = nodes.filter(n => n.type === 'Input');
    const hasConflicts = [...shapeMap.values()].some(v => v.conflict !== null);
    const preShape = PREPROCESS_OUT_SHAPE[trainCategory];
    const inShape = (inputNode?.parameters?.shape as number[] | undefined) ?? null;
    const shapeMatches = inShape !== null && JSON.stringify(preShape) === JSON.stringify(inShape);
    return [
      { label: 'Has one Input node', pass: inputNodes.length === 1 },
      { label: 'No shape conflicts', pass: !hasConflicts },
      { label: `Preprocess → Input [${preShape.join('×')}]`, pass: shapeMatches },
      { label: 'Dataset selected', pass: !!trainSelected },
      { label: `Optimizer: ${trainOptimizer}`, pass: !!trainOptimizer },
      { label: `Loss: ${trainLoss}`, pass: !!trainLoss },
    ];
  }, [nodes, shapeMap, trainCategory, inputNode, trainSelected, trainOptimizer, trainLoss]);

  const canGenerateTraining = trainValidations.every(v => v.pass);

  // ── Train: animation tick (after trainSortedNodes is declared) ───────────
  useEffect(() => {
    if (!simRunning) {
      if (simTimerRef.current) clearInterval(simTimerRef.current);
      return;
    }
    const totalSteps = trainSortedNodes.length + 1; // +1 for virtual preprocess node
    simTimerRef.current = setInterval(() => {
      setAnimStep(prev => (prev + 1) % totalSteps);
    }, 600);
    return () => { if (simTimerRef.current) clearInterval(simTimerRef.current); };
  }, [simRunning, trainSortedNodes.length]);

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

  const handleGenerateTraining = useCallback(async () => {
    if (!trainSelected) return;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await callGenTraining({
        // Server reads graph from savedDesign automatically — no graph field needed
        dataset: {
          name: trainSelected.name,
          source: 'huggingface',
          hfId: trainSelected.hfId,
        },
        optimizer: trainOptimizer,
        loss: trainLoss,
        hyperparams: { lr: trainLR, batch_size: trainBatchSize, epochs: trainEpochs },
        wandb: { project: trainWandbProject },
      } as any); // eslint-disable-line @typescript-eslint/no-explicit-any
      const sc = result?.structuredContent as {
        modelPy?: string; dataPy?: string; trainPy?: string;
        requirementsTxt?: string; wandbProject?: string;
      } | undefined;
      if (sc?.modelPy || sc?.trainPy) {
        setGeneratedFiles({
          modelPy: sc.modelPy ?? '', dataPy: sc.dataPy ?? '', trainPy: sc.trainPy ?? '',
          requirementsTxt: sc.requirementsTxt ?? '', wandbProject: sc.wandbProject,
        });
      }
    } catch (err) {
      console.error('[generate-training-code]', err);
    }
  }, [trainSelected, trainOptimizer, trainLoss, trainLR, trainBatchSize, trainEpochs, callGenTraining]);

  const handleTrainCopy = useCallback((key: 'model' | 'data' | 'train' | 'req') => {
    if (!generatedFiles) return;
    const text = key === 'model' ? generatedFiles.modelPy : key === 'data' ? generatedFiles.dataPy : key === 'req' ? generatedFiles.requirementsTxt : generatedFiles.trainPy;
    navigator.clipboard.writeText(text).then(() => {
      setTrainCopied(key);
      setTimeout(() => setTrainCopied(null), 1800);
    });
  }, [generatedFiles]);

  const handleGenerateInference = useCallback(async () => {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await callGenInference({} as any);
      const sc = result?.structuredContent as { inferencePy?: string; taskType?: string; inputFormat?: string } | undefined;
      if (sc?.inferencePy) {
        const gen: GeneratedInference = {
          inferencePy: sc.inferencePy,
          taskType: sc.taskType ?? 'unknown',
          inputFormat: sc.inputFormat ?? '{"input": "..."}',
        };
        setInferGenerated(gen);
        // Auto-set input type based on task
        if (sc.taskType === 'vision') setInferInputType('image_url');
        else if (sc.taskType === 'tabular') setInferInputType('tabular');
        else setInferInputType('text');
      }
    } catch (err) {
      console.error('[generate-inference-code]', err);
    }
  }, [callGenInference]);

  const handleRun = useCallback(async () => {
    if (!trainPodId) return;

    const initSteps: DeployStepsState = {
      generate: { status: generatedFiles ? 'skipped' : 'pending' },
      check:    { status: 'pending' },
      install:  { status: 'pending' },
      upload:   { status: 'pending' },
    };
    setDeploySteps(initSteps);
    setDeployPhase('running');
    setIsDeploying(true);
    setShowDeployModal(true);

    try {
      // Step 1: Generate scripts (if not already done)
      if (!generatedFiles) {
        setDeploySteps(prev => ({ ...prev, generate: { status: 'running' } }));
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const result = await callGenTraining({
            dataset: { name: trainSelected!.name, source: 'huggingface', hfId: trainSelected!.hfId },
            optimizer: trainOptimizer, loss: trainLoss,
            hyperparams: { lr: trainLR, batch_size: trainBatchSize, epochs: trainEpochs },
            wandb: { project: trainWandbProject },
          } as any);
          const sc = result?.structuredContent as { modelPy?: string; dataPy?: string; trainPy?: string; requirementsTxt?: string; wandbProject?: string } | undefined;
          if (sc?.modelPy || sc?.trainPy) {
            setGeneratedFiles({ modelPy: sc!.modelPy ?? '', dataPy: sc!.dataPy ?? '', trainPy: sc!.trainPy ?? '', requirementsTxt: sc!.requirementsTxt ?? '', wandbProject: sc!.wandbProject });
            setDeploySteps(prev => ({ ...prev, generate: { status: 'done', detail: '4 scripts generated' } }));
          } else {
            throw new Error('No scripts returned');
          }
        } catch (e) {
          setDeploySteps(prev => ({ ...prev, generate: { status: 'error', detail: String(e) } }));
          setDeployPhase('error');
          return;
        }
      } else {
        setDeploySteps(prev => ({ ...prev, generate: { status: 'skipped', detail: 'Already generated' } }));
      }

      // Step 2: Check installed packages
      setDeploySteps(prev => ({ ...prev, check: { status: 'running' } }));
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const checkResult = await callCheckGpuPkgs({ podId: trainPodId } as any);
        const sc = checkResult?.structuredContent as { packageCount?: number; hasTorch?: boolean; hasWandb?: boolean } | undefined;
        const flags = [sc?.hasTorch && 'torch', sc?.hasWandb && 'wandb'].filter(Boolean).join(', ');
        setDeploySteps(prev => ({ ...prev, check: { status: 'done', detail: `${sc?.packageCount ?? '?'} pkgs — ${flags || 'checked'}` } }));
      } catch (e) {
        setDeploySteps(prev => ({ ...prev, check: { status: 'error', detail: String(e) } }));
        setDeployPhase('error');
        return;
      }

      // Step 3: Install missing packages
      setDeploySteps(prev => ({ ...prev, install: { status: 'running' } }));
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const installResult = await callSetupGpu({ podId: trainPodId } as any);
        const sc = installResult?.structuredContent as { installed?: string[]; skipped?: string[] } | undefined;
        const installed = sc?.installed ?? [];
        const skippedCount = sc?.skipped?.length ?? 0;
        const detail = installed.length > 0
          ? `Installed: ${installed.slice(0, 3).join(', ')}${installed.length > 3 ? ` +${installed.length - 3}` : ''}`
          : `All ${skippedCount} packages already present`;
        setDeploySteps(prev => ({ ...prev, install: { status: 'done', detail } }));
      } catch (e) {
        // Non-fatal — continue to upload
        setDeploySteps(prev => ({ ...prev, install: { status: 'error', detail: String(e) } }));
      }

      // Step 4: Upload scripts
      setDeploySteps(prev => ({ ...prev, upload: { status: 'running' } }));
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const uploadResult = await callUploadScripts({ podId: trainPodId } as any);
        const sc = uploadResult?.structuredContent as { uploaded?: string[] } | undefined;
        const files = sc?.uploaded ?? [];
        setDeploySteps(prev => ({ ...prev, upload: { status: 'done', detail: files.join(', ') || 'Files uploaded' } }));
        setDeployPhase('done');
      } catch (e) {
        setDeploySteps(prev => ({ ...prev, upload: { status: 'error', detail: String(e) } }));
        setDeployPhase('error');
      }
    } finally {
      setIsDeploying(false);
    }
  }, [trainPodId, generatedFiles, trainSelected, trainOptimizer, trainLoss, trainLR, trainBatchSize, trainEpochs, trainWandbProject, callGenTraining, callCheckGpuPkgs, callSetupGpu, callUploadScripts]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunInference = useCallback(async () => {
    if (!inferPodId || !inferInput) return;
    setInferResult(null);
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await callRunInference({
        podId: inferPodId,
        input: inferInput,
        inputType: inferInputType,
      } as any);
      const sc = result?.structuredContent as InferResult | undefined;
      setInferResult(sc ?? { error: 'No response from server' });
    } catch (err) {
      setInferResult({ error: String(err) });
    }
  }, [inferPodId, inferInput, inferInputType, callRunInference]);

  // When a dataset is selected: auto-fix Input shape (creating one if missing), cascade-fix conflicts
  const handleDatasetSelect = useCallback((ds: DatasetDef | null) => {
    setTrainSelected(ds);
    if (ds) setTrainWandbProject(ds.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''));
    if (!ds) return;
    const targetShape = PREPROCESS_OUT_SHAPE[trainCategory];

    // Work directly on current nodes/edges so we can update both atomically
    let newNodes: GraphNode[] = nodes.map(n =>
      n.type === 'Input'
        ? { ...n, parameters: { ...n.parameters, shape: targetShape } }
        : n
    );
    let newEdges: GraphEdge[] = [...edges];

    // Auto-create Input node if the architecture doesn't have one
    if (!newNodes.some(n => n.type === 'Input')) {
      const inputId = newNodeId();
      const inputNode: GraphNode = { id: inputId, type: 'Input', x: 80, y: 80, parameters: { shape: targetShape } };
      newNodes = [inputNode, ...newNodes];
      // Connect it to the first node that has no incoming edge (topological root)
      const hasIncoming = new Set(newEdges.map(e => e.targetId));
      const firstNode = newNodes.slice(1).find(n => !hasIncoming.has(n.id));
      if (firstNode) {
        newEdges = [...newEdges, { id: newEdgeId(), sourceId: inputId, targetId: firstNode.id }];
      }
    }

    // Cascade: apply fixSelf for any remaining shape conflicts (two passes for chained deps)
    for (let pass = 0; pass < 2; pass++) {
      const map = propagateShapes(newNodes, newEdges);
      newNodes = newNodes.map(n => {
        const info = map.get(n.id);
        if (info?.conflict && info.fixSelf) {
          return { ...n, parameters: { ...n.parameters, [info.fixSelf.key]: info.fixSelf.value } };
        }
        return n;
      });
    }

    setNodes(newNodes);
    setEdges(newEdges);
  }, [trainCategory, nodes, edges]);

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
            ML Architect
          </div>

          <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />

          {/* Design / Train mode tabs */}
          <div style={{ display: 'flex', gap: 2, backgroundColor: '#0d0d0d', borderRadius: 5, padding: 2, border: '1px solid #1a1a1a' }}>
            {(['design', 'train', 'infer'] as ViewMode[]).map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                style={{
                  padding: '3px 10px',
                  backgroundColor: viewMode === mode ? '#1a1a1a' : 'transparent',
                  border: viewMode === mode ? '1px solid #2a2a2a' : '1px solid transparent',
                  borderRadius: 4,
                  color: viewMode === mode ? '#d0d0d0' : '#444',
                  cursor: 'pointer',
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: 0.3,
                  textTransform: 'capitalize',
                  transition: 'all 0.1s',
                }}
              >
                {mode}
              </button>
            ))}
          </div>

          <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />

          <div style={{ fontSize: 10, color: '#333' }}>
            {nodes.length} block{nodes.length !== 1 ? 's' : ''}
          </div>

          <div style={{ flex: 1 }} />

          {/* Design-mode controls */}
          {viewMode === 'design' && (
            <>
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
                  Group {selectedIds.length}
                </button>
              )}
              <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />
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
              <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />
              {/* Props toggle */}
              <button
                onClick={() => setShowProps(p => !p)}
                title={showProps ? 'Hide properties' : 'Show properties'}
                style={{
                  padding: '5px 8px',
                  backgroundColor: showProps ? '#111' : 'transparent',
                  border: `1px solid ${showProps ? '#252525' : 'transparent'}`,
                  borderRadius: 5,
                  color: showProps ? '#888' : '#333',
                  cursor: 'pointer',
                  fontSize: 12,
                  lineHeight: 1,
                }}
              >
                ⊟
              </button>
            </>
          )}

          {/* Train-mode controls */}
          {viewMode === 'train' && (
            <>
              {trainSelected && (
                <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 3, backgroundColor: '#0a1020', color: '#93c5fd', border: '1px solid #1e3a5f', fontWeight: 600 }}>
                  {trainSelected.name}
                </span>
              )}
              <span style={{ fontSize: 10, color: '#333' }}>Model · {trainSortedNodes.length} layers</span>
              <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />
              <button
                onClick={() => {
                  if (simRunning) { setSimRunning(false); setAnimStep(0); }
                  else { setAnimStep(0); setSimRunning(true); }
                }}
                style={{
                  padding: '5px 12px',
                  backgroundColor: simRunning ? '#1a0000' : '#0a1a1a',
                  border: `1px solid ${simRunning ? '#7f1d1d' : '#164e63'}`,
                  borderRadius: 5,
                  color: simRunning ? '#fca5a5' : '#67e8f9',
                  cursor: 'pointer',
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: 0.3,
                }}
              >
                {simRunning ? 'Stop' : 'Run Dummy'}
              </button>
              <button
                onClick={handleRun}
                disabled={isDeploying || (!trainPodId) || (!canGenerateTraining && !generatedFiles)}
                title={!trainPodId ? 'Enter a Pod ID in the Deploy section' : !canGenerateTraining && !generatedFiles ? 'Fix validation errors first' : 'Deploy scripts and prepare GPU pod'}
                style={{
                  padding: '5px 12px',
                  backgroundColor: isDeploying ? '#0a1a0a' : trainPodId && (canGenerateTraining || generatedFiles) ? '#15803d' : '#111',
                  border: `1px solid ${isDeploying ? '#1a3a1a' : trainPodId && (canGenerateTraining || generatedFiles) ? '#166534' : '#1e1e1e'}`,
                  borderRadius: 5,
                  color: isDeploying ? '#4ade80' : trainPodId && (canGenerateTraining || generatedFiles) ? '#fff' : '#2a2a2a',
                  cursor: isDeploying || !trainPodId || (!canGenerateTraining && !generatedFiles) ? 'not-allowed' : 'pointer',
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: 0.3,
                  transition: 'all 0.15s',
                }}
              >
                {isDeploying ? 'Deploying…' : 'Run'}
              </button>
            </>
          )}

          {/* Infer-mode controls */}
          {viewMode === 'infer' && (
            <>
              {inferGenerated && (
                <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 3, backgroundColor: '#0a1a0a', color: '#4ade80', border: '1px solid #166534', fontWeight: 600 }}>
                  {inferGenerated.taskType}
                </span>
              )}
              <span style={{ fontSize: 10, color: '#333' }}>GPU Inference</span>
            </>
          )}
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
          {viewMode === 'design' ? (
            <>
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
              {showProps && (
                <PropertiesPanel
                  selectedNodes={selectedNodes}
                  edges={edges}
                  onParamChange={updateParam}
                  onDeleteNode={deleteNode}
                  onCodeOverrideChange={updateCodeOverride}
                />
              )}
            </>
          ) : viewMode === 'train' ? (
            <TrainBody
              trainSortedNodes={trainSortedNodes}
              shapeMap={shapeMap}
              trainCategory={trainCategory}
              setTrainCategory={setTrainCategory}
              trainSelected={trainSelected}
              onSelectDataset={handleDatasetSelect}
              trainLoss={trainLoss}
              setTrainLoss={setTrainLoss}
              trainOptimizer={trainOptimizer}
              setTrainOptimizer={setTrainOptimizer}
              trainLR={trainLR}
              setTrainLR={setTrainLR}
              trainBatchSize={trainBatchSize}
              setTrainBatchSize={setTrainBatchSize}
              trainEpochs={trainEpochs}
              setTrainEpochs={setTrainEpochs}
              animStep={animStep}
              generatedFiles={generatedFiles}
              isGeneratingTraining={isGeneratingTraining}
              canGenerateTraining={canGenerateTraining}
              onGenerateTraining={handleGenerateTraining}
              trainWandbProject={trainWandbProject}
              setTrainWandbProject={setTrainWandbProject}
              trainCopied={trainCopied}
              onTrainCopy={handleTrainCopy}
              trainValidations={trainValidations}
              trainPodId={trainPodId}
              setTrainPodId={setTrainPodId}
            />
          ) : (
            <InferBody
              inferGenerated={inferGenerated}
              isGeneratingInference={isGeneratingInference}
              onGenerateInference={handleGenerateInference}
              inferInput={inferInput}
              setInferInput={setInferInput}
              inferInputType={inferInputType}
              setInferInputType={setInferInputType}
              inferPodId={inferPodId}
              setInferPodId={setInferPodId}
              inferResult={inferResult}
              isRunningInference={isRunningInference}
              onRunInference={handleRunInference}
              inferScriptCopied={inferScriptCopied}
              onCopyScript={() => {
                if (!inferGenerated) return;
                navigator.clipboard.writeText(inferGenerated.inferencePy).then(() => {
                  setInferScriptCopied(true);
                  setTimeout(() => setInferScriptCopied(false), 1800);
                });
              }}
            />
          )}
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
      {/* ── Deploy modal ────────────────────────────────────────────────── */}
      {showDeployModal && (
        <div
          onClick={e => { if (e.target === e.currentTarget && !isDeploying) setShowDeployModal(false); }}
          style={{
            position: 'fixed', inset: 0,
            backgroundColor: 'rgba(0,0,0,0.88)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 200,
          }}
        >
          <div style={{
            backgroundColor: '#080808',
            border: '1px solid #202020',
            borderRadius: 10,
            width: 420,
            padding: 0,
            boxShadow: '0 20px 60px rgba(0,0,0,0.8)',
            overflow: 'hidden',
          }}>
            {/* Modal header */}
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid #141414',
              display: 'flex', alignItems: 'center', gap: 10,
            }}>
              <div style={{
                width: 8, height: 8, borderRadius: '50%',
                backgroundColor: deployPhase === 'done' ? '#4ade80' : deployPhase === 'error' ? '#f87171' : '#3b82f6',
                boxShadow: `0 0 6px ${deployPhase === 'done' ? '#4ade80' : deployPhase === 'error' ? '#f87171' : '#3b82f644'}`,
                animation: deployPhase === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
              }} />
              <span style={{ fontWeight: 700, fontSize: 12, color: '#d0d0d0', flex: 1 }}>
                {deployPhase === 'done' ? 'Deployed Successfully' : deployPhase === 'error' ? 'Deploy Failed' : 'Deploying to RunPod'}
              </span>
              <span style={{ fontSize: 10, color: '#333', fontFamily: 'monospace' }}>{trainPodId}</span>
              {!isDeploying && (
                <button
                  onClick={() => setShowDeployModal(false)}
                  style={{ background: 'none', border: 'none', color: '#444', cursor: 'pointer', fontSize: 18, lineHeight: 1, padding: '0 2px' }}
                >×</button>
              )}
            </div>

            {/* Steps */}
            <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {([
                { key: 'generate', label: 'Generate Scripts',     icon: '⚡' },
                { key: 'check',    label: 'Check GPU Packages',   icon: '🔍' },
                { key: 'install',  label: 'Install Dependencies', icon: '📦' },
                { key: 'upload',   label: 'Upload to Pod',        icon: '🚀' },
              ] as const).map(({ key, label, icon }) => {
                const step = deploySteps[key];
                const statusIcon =
                  step.status === 'running' ? '⏳'
                  : step.status === 'done'   ? '✓'
                  : step.status === 'error'  ? '✗'
                  : step.status === 'skipped'? '─'
                  : '○';
                const statusColor =
                  step.status === 'running' ? '#60a5fa'
                  : step.status === 'done'   ? '#4ade80'
                  : step.status === 'error'  ? '#f87171'
                  : step.status === 'skipped'? '#555'
                  : '#2a2a2a';
                return (
                  <div key={key} style={{
                    display: 'flex', alignItems: 'flex-start', gap: 10,
                    padding: '8px 10px', borderRadius: 6,
                    backgroundColor: step.status === 'running' ? '#0a1020' : step.status === 'done' ? '#0a140a' : '#0a0a0a',
                    border: `1px solid ${step.status === 'running' ? '#1e3a5f' : step.status === 'done' ? '#14441a' : '#141414'}`,
                    transition: 'all 0.2s',
                  }}>
                    <span style={{ fontSize: 13, minWidth: 18, textAlign: 'center', lineHeight: '16px' }}>{icon}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 600, color: step.status === 'pending' ? '#3a3a3a' : '#d0d0d0' }}>
                          {label}
                        </span>
                        <span style={{ fontSize: 10, color: statusColor, fontWeight: 700 }}>{statusIcon}</span>
                      </div>
                      {step.detail && (
                        <div style={{ fontSize: 9.5, color: step.status === 'error' ? '#c05050' : '#4a4a4a', marginTop: 2, fontFamily: 'monospace', wordBreak: 'break-all' }}>
                          {step.detail}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Done state */}
            {deployPhase === 'done' && (
              <div style={{ margin: '0 16px 16px', padding: '10px 12px', borderRadius: 6, backgroundColor: '#0a140a', border: '1px solid #166534' }}>
                <div style={{ fontSize: 9, color: '#4ade80', fontWeight: 700, letterSpacing: 0.5, marginBottom: 6 }}>READY TO TRAIN</div>
                <div style={{ fontSize: 10, color: '#555', marginBottom: 6 }}>SSH into your pod and run:</div>
                <div style={{
                  padding: '6px 10px', borderRadius: 4,
                  backgroundColor: '#060606', border: '1px solid #1a1a1a',
                  fontSize: 11, color: '#a8e6a3', fontFamily: 'monospace',
                }}>
                  cd /workspace && python3 train.py
                </div>
                {generatedFiles?.wandbProject && (
                  <div style={{ fontSize: 9, color: '#3a6a8a', marginTop: 6 }}>
                    W&B project: <span style={{ color: '#60a5fa' }}>{generatedFiles.wandbProject}</span>
                  </div>
                )}
              </div>
            )}

            {/* Error state */}
            {deployPhase === 'error' && (
              <div style={{ margin: '0 16px 16px', padding: '8px 12px', borderRadius: 6, backgroundColor: '#110808', border: '1px solid #3d1515' }}>
                <div style={{ fontSize: 10, color: '#c05050' }}>
                  Deploy failed. Check the step above for details. Make sure your SSH key is added to the pod.
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      </div>
    </McpUseProvider>
  );
}

// ── TrainBody ──────────────────────────────────────────────────────────────

interface TrainBodyProps {
  trainSortedNodes: GraphNode[];
  shapeMap: Map<string, NodeShapeInfo>;
  trainCategory: CatKey;
  setTrainCategory: (c: CatKey) => void;
  trainSelected: DatasetDef | null;
  onSelectDataset: (d: DatasetDef | null) => void;
  trainLoss: string;
  setTrainLoss: (l: string) => void;
  trainOptimizer: string;
  setTrainOptimizer: (o: string) => void;
  trainLR: number;
  setTrainLR: (v: number) => void;
  trainBatchSize: number;
  setTrainBatchSize: (v: number) => void;
  trainEpochs: number;
  setTrainEpochs: (v: number) => void;
  animStep: number;
  generatedFiles: GeneratedFiles | null;
  isGeneratingTraining: boolean;
  canGenerateTraining: boolean;
  onGenerateTraining: () => void;
  trainWandbProject: string;
  setTrainWandbProject: (p: string) => void;
  trainCopied: 'model' | 'data' | 'train' | 'req' | null;
  onTrainCopy: (key: 'model' | 'data' | 'train' | 'req') => void;
  trainValidations: Array<{ label: string; pass: boolean }>;
  trainPodId: string;
  setTrainPodId: (v: string) => void;
}

function TrainBody({
  trainSortedNodes, shapeMap,
  trainCategory, setTrainCategory,
  trainSelected, onSelectDataset,
  trainLoss, setTrainLoss,
  trainOptimizer, setTrainOptimizer,
  trainLR, setTrainLR,
  trainBatchSize, setTrainBatchSize,
  trainEpochs, setTrainEpochs,
  animStep,
  generatedFiles, isGeneratingTraining, canGenerateTraining, onGenerateTraining,
  trainWandbProject, setTrainWandbProject,
  trainCopied, onTrainCopy,
  trainValidations,
  trainPodId, setTrainPodId,
}: TrainBodyProps) {
  const accent = TRAIN_ACCENT[trainCategory];
  const datasets = TRAIN_DATASETS[trainCategory];

  const [activeFile, setActiveFile] = useState<'model' | 'data' | 'train' | 'req'>('train');

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>

      {/* ── Left: Dataset picker ─────────────────────────────────────────── */}
      <div style={{
        width: 200, minWidth: 200,
        borderRight: '1px solid #141414',
        display: 'flex', flexDirection: 'column',
        backgroundColor: '#070707', overflowY: 'auto',
      }}>
        {/* Category tabs */}
        <div style={{ padding: '8px 8px 4px', display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {TRAIN_CAT_TABS.map(tab => (
            <button
              key={tab.key}
              onClick={() => setTrainCategory(tab.key)}
              style={{
                padding: '3px 8px',
                backgroundColor: trainCategory === tab.key ? accent + '22' : 'transparent',
                border: `1px solid ${trainCategory === tab.key ? accent + '66' : '#1e1e1e'}`,
                borderRadius: 3,
                color: trainCategory === tab.key ? accent : '#444',
                cursor: 'pointer',
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: 0.3,
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div style={{ fontSize: 9, color: '#2a2a2a', letterSpacing: 1.5, textTransform: 'uppercase', fontWeight: 700, padding: '6px 10px 4px' }}>
          Datasets
        </div>

        {/* Dataset list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 6px 8px' }}>
          {datasets.map(ds => {
            const sel = trainSelected?.id === ds.id;
            return (
              <div
                key={ds.id}
                onClick={() => onSelectDataset(sel ? null : ds)}
                style={{
                  padding: '7px 8px',
                  marginBottom: 4,
                  borderRadius: 5,
                  cursor: 'pointer',
                  backgroundColor: sel ? accent + '18' : '#0d0d0d',
                  border: `1px solid ${sel ? accent + '55' : '#161616'}`,
                  borderLeft: sel ? `3px solid ${accent}` : '3px solid transparent',
                  transition: 'all 0.1s',
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 600, color: sel ? accent : '#888', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {ds.name}
                </div>
                <div style={{ fontSize: 9, color: '#2e2e2e', marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {ds.size} · {ds.task}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Middle: Node flow ─────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', backgroundColor: '#080808', position: 'relative' }}>
        {/* Validation bar */}
        <div style={{
          display: 'flex', gap: 6, flexWrap: 'wrap',
          padding: '6px 10px',
          borderBottom: '1px solid #111',
          backgroundColor: '#060606',
        }}>
          {trainValidations.map((v, i) => (
            <span key={i} style={{
              fontSize: 8.5, padding: '2px 7px', borderRadius: 3, fontWeight: 600,
              backgroundColor: v.pass ? '#0a1a0a' : '#1a0808',
              color: v.pass ? '#4ade80' : '#f87171',
              border: `1px solid ${v.pass ? '#166534' : '#7f1d1d'}`,
            }}>
              {v.pass ? '✓' : '✗'} {v.label}
            </span>
          ))}
        </div>

        {/* Node tree */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
          <div style={{ maxWidth: 380, margin: '0 auto', padding: '0 12px' }}>
            {/* Preprocess node */}
            <TrainNodeRow
              label={PREPROCESS_LABEL[trainCategory]}
              shape={PREPROCESS_OUT_SHAPE[trainCategory]}
              color={accent}
              isActive={animStep === 0}
              isVirtual
            />
            <ConnectorLine color={accent} />

            {/* Real nodes */}
            {trainSortedNodes.map((node, i) => {
              const info = shapeMap.get(node.id);
              const def = BLOCK_DEFS.find(d => d.type === node.type);
              const label = node.type === 'Custom'
                ? (node.composite?.label ?? 'Custom')
                : (def?.label ?? node.type);
              const cat = def?.category ?? 'composite';
              const color = TRAIN_NODE_COLORS[cat] ?? '#555';
              return (
                <div key={node.id}>
                  <TrainNodeRow
                    label={label}
                    shape={info?.outShape ?? null}
                    color={color}
                    isActive={animStep === i + 1}
                    conflict={info?.conflict ?? null}
                  />
                  {i < trainSortedNodes.length - 1 && <ConnectorLine color={color} />}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Right: Config + Generate ──────────────────────────────────────── */}
      <div style={{
        width: 220, minWidth: 220,
        borderLeft: '1px solid #141414',
        display: 'flex', flexDirection: 'column',
        backgroundColor: '#070707', overflowY: 'auto',
      }}>
        {/* Training config */}
        <div style={{ padding: '10px 10px 6px' }}>
          <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase', fontWeight: 700, marginBottom: 8 }}>
            Training Config
          </div>

          <TrainConfigRow label="Optimizer">
            <select value={trainOptimizer} onChange={e => setTrainOptimizer(e.target.value)} style={selectStyle}>
              {['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adadelta', 'LBFGS'].map(o => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </TrainConfigRow>

          <TrainConfigRow label="Loss">
            <select value={trainLoss} onChange={e => setTrainLoss(e.target.value)} style={selectStyle}>
              {['CrossEntropyLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss', 'L1Loss', 'HuberLoss', 'KLDivLoss'].map(l => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </TrainConfigRow>

          <TrainConfigRow label="LR">
            <input type="number" step={0.0001} value={trainLR} onChange={e => setTrainLR(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </TrainConfigRow>

          <TrainConfigRow label="Batch">
            <input type="number" step={1} value={trainBatchSize} onChange={e => setTrainBatchSize(parseInt(e.target.value) || 1)} style={inputStyle} />
          </TrainConfigRow>

          <TrainConfigRow label="Epochs">
            <input type="number" step={1} value={trainEpochs} onChange={e => setTrainEpochs(parseInt(e.target.value) || 1)} style={inputStyle} />
          </TrainConfigRow>
        </div>

        <div style={{ borderTop: '1px solid #111', margin: '0 10px' }} />

        {/* W&B config */}
        <div style={{ padding: '8px 10px 4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
            <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase' as const, fontWeight: 700, flex: 1 }}>
              W&amp;B
            </div>
            <span style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, backgroundColor: '#0a1a2a', color: '#93c5fd', border: '1px solid #1e3a5f', fontWeight: 700 }}>
              wandb
            </span>
          </div>
          <TrainConfigRow label="Project">
            <input
              type="text"
              value={trainWandbProject}
              onChange={e => setTrainWandbProject(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
              placeholder="my-model"
              style={inputStyle}
            />
          </TrainConfigRow>
        </div>

        <div style={{ borderTop: '1px solid #111', margin: '0 10px' }} />

        {/* GPU Deploy */}
        <div style={{ padding: '8px 10px 4px' }}>
          <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 6 }}>
            GPU Deploy
          </div>
          <TrainConfigRow label="Pod ID">
            <input
              type="text"
              value={trainPodId}
              onChange={e => setTrainPodId(e.target.value.trim())}
              placeholder="abc123xyz…"
              style={{ ...inputStyle, color: trainPodId ? '#d0d0d0' : '#555', borderColor: trainPodId ? '#166534' : '#1e1e1e' }}
            />
          </TrainConfigRow>
          {trainPodId && (
            <div style={{ fontSize: 8.5, color: '#2a4a2a', marginTop: -2, marginBottom: 4, marginLeft: 54 }}>
              ✓ Pod set — click <span style={{ color: '#4ade80' }}>Run</span> to deploy
            </div>
          )}
        </div>

        <div style={{ borderTop: '1px solid #111', margin: '0 10px' }} />

        {/* Generate button */}
        <div style={{ padding: '8px 10px' }}>
          <button
            onClick={onGenerateTraining}
            disabled={!canGenerateTraining || isGeneratingTraining}
            style={{
              width: '100%', padding: '7px 0',
              backgroundColor: canGenerateTraining && !isGeneratingTraining ? accent + '22' : '#111',
              border: `1px solid ${canGenerateTraining ? accent + '66' : '#1e1e1e'}`,
              borderRadius: 5,
              color: canGenerateTraining ? accent : '#333',
              cursor: canGenerateTraining && !isGeneratingTraining ? 'pointer' : 'not-allowed',
              fontSize: 11, fontWeight: 700, letterSpacing: 0.3,
              transition: 'all 0.15s',
            }}
          >
            {isGeneratingTraining ? 'Generating…' : '⚡ Generate Scripts'}
          </button>
        </div>

        {/* Generated files */}
        {generatedFiles && (
          <>
            <div style={{ borderTop: '1px solid #111', margin: '0 10px' }} />
            <div style={{ padding: '8px 10px' }}>

              {/* W&B project link */}
              {generatedFiles.wandbProject && (
                <div style={{
                  marginBottom: 8, padding: '6px 8px', borderRadius: 4,
                  backgroundColor: '#0a1020', border: '1px solid #1e3a5f',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <span style={{ fontSize: 8, color: '#93c5fd', fontWeight: 700, letterSpacing: 0.5 }}>W&B</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 9, color: '#60a5fa', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {generatedFiles.wandbProject}
                    </div>
                    <div style={{ fontSize: 8, color: '#2a4a6a', marginTop: 1 }}>
                      run URL printed to stdout on start
                    </div>
                  </div>
                </div>
              )}

              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 6 }}>
                Output
              </div>

              {/* File tabs */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
                {([
                  { key: 'train', label: 'train.py' },
                  { key: 'model', label: 'model.py' },
                  { key: 'data',  label: 'data.py'  },
                  { key: 'req',   label: 'req.txt'  },
                ] as const).map(f => (
                  <button
                    key={f.key}
                    onClick={() => setActiveFile(f.key)}
                    style={{
                      padding: '3px 6px',
                      backgroundColor: activeFile === f.key ? '#1a1a1a' : 'transparent',
                      border: `1px solid ${activeFile === f.key ? '#2a2a2a' : 'transparent'}`,
                      borderRadius: 3,
                      color: activeFile === f.key ? '#d0d0d0' : '#444',
                      cursor: 'pointer', fontSize: 9, fontWeight: 700,
                    }}
                  >
                    {f.label}
                  </button>
                ))}
              </div>

              {/* Code preview */}
              <div style={{ position: 'relative' }}>
                <pre style={{
                  margin: 0, padding: '8px', borderRadius: 4,
                  backgroundColor: '#080808', border: '1px solid #151515',
                  fontSize: 9.5, color: '#a8e6a3', lineHeight: 1.5,
                  maxHeight: 160, overflow: 'auto', whiteSpace: 'pre', fontFamily: 'monospace',
                }}>
                  {activeFile === 'model' ? generatedFiles.modelPy
                    : activeFile === 'data'  ? generatedFiles.dataPy
                    : activeFile === 'req'   ? generatedFiles.requirementsTxt
                    : generatedFiles.trainPy}
                </pre>
                <button
                  onClick={() => onTrainCopy(activeFile)}
                  style={{
                    position: 'absolute', top: 4, right: 4,
                    padding: '2px 6px',
                    backgroundColor: trainCopied === activeFile ? '#0a2a0a' : '#141414',
                    border: `1px solid ${trainCopied === activeFile ? '#166534' : '#252525'}`,
                    borderRadius: 3,
                    color: trainCopied === activeFile ? '#4ade80' : '#555',
                    cursor: 'pointer', fontSize: 8, fontWeight: 700,
                  }}
                >
                  {trainCopied === activeFile ? '✓' : 'copy'}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── InferBody ──────────────────────────────────────────────────────────────

interface InferBodyProps {
  inferGenerated: GeneratedInference | null;
  isGeneratingInference: boolean;
  onGenerateInference: () => void;
  inferInput: string;
  setInferInput: (v: string) => void;
  inferInputType: 'text' | 'image_url' | 'image_base64' | 'tabular';
  setInferInputType: (v: 'text' | 'image_url' | 'image_base64' | 'tabular') => void;
  inferPodId: string;
  setInferPodId: (v: string) => void;
  inferResult: InferResult | null;
  isRunningInference: boolean;
  onRunInference: () => void;
  inferScriptCopied: boolean;
  onCopyScript: () => void;
}

function InferBody({
  inferGenerated, isGeneratingInference, onGenerateInference,
  inferInput, setInferInput, inferInputType, setInferInputType,
  inferPodId, setInferPodId,
  inferResult, isRunningInference, onRunInference,
  inferScriptCopied, onCopyScript,
}: InferBodyProps) {
  const canRun = !!inferPodId && !!inferInput && !!inferGenerated;

  const inputPlaceholder =
    inferInputType === 'text'         ? 'Enter text prompt…'
    : inferInputType === 'image_url'  ? 'https://example.com/image.jpg'
    : inferInputType === 'tabular'    ? '5.1, 3.5, 1.4, 0.2'
    : 'Paste base64 image string…';

  const typeAccent = '#8b5cf6'; // purple for inference

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>

      {/* ── Left: Input panel ───────────────────────────────────────────── */}
      <div style={{
        width: 240, minWidth: 240,
        borderRight: '1px solid #141414',
        display: 'flex', flexDirection: 'column',
        backgroundColor: '#070707', overflowY: 'auto',
      }}>
        {/* Header */}
        <div style={{ padding: '10px 12px 6px', fontSize: 9, letterSpacing: 1.5, color: '#3a3a3a', textTransform: 'uppercase' as const, fontWeight: 700, borderBottom: '1px solid #111' }}>
          Input
        </div>

        <div style={{ padding: '10px 10px 6px', flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>

          {/* Input type selector */}
          <div>
            <div style={{ fontSize: 9, color: '#444', marginBottom: 4, fontWeight: 600, letterSpacing: 0.3 }}>Input Type</div>
            <select
              value={inferInputType}
              onChange={e => setInferInputType(e.target.value as typeof inferInputType)}
              style={{ ...selectStyle, color: '#888' }}
            >
              <option value="text">Text / Prompt</option>
              <option value="image_url">Image URL</option>
              <option value="image_base64">Image (base64)</option>
              <option value="tabular">Tabular (CSV)</option>
            </select>
          </div>

          {/* Input field */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 9, color: '#444', marginBottom: 4, fontWeight: 600, letterSpacing: 0.3 }}>
              {inferInputType === 'text' ? 'Prompt' : inferInputType === 'image_url' ? 'Image URL' : inferInputType === 'tabular' ? 'Feature Values' : 'Base64 Data'}
            </div>
            <textarea
              value={inferInput}
              onChange={e => setInferInput(e.target.value)}
              placeholder={inputPlaceholder}
              rows={inferInputType === 'image_base64' ? 4 : 6}
              style={{
                flex: 1,
                padding: '6px 8px',
                backgroundColor: '#0f0f0f',
                border: `1px solid #1e1e1e`,
                borderRadius: 4,
                color: '#d0d0d0',
                fontSize: 10,
                outline: 'none',
                resize: 'vertical',
                fontFamily: inferInputType === 'text' ? 'inherit' : 'monospace',
                boxSizing: 'border-box' as const,
                lineHeight: 1.5,
              }}
              onFocus={e => (e.target.style.borderColor = typeAccent + '55')}
              onBlur={e => (e.target.style.borderColor = '#1e1e1e')}
            />
          </div>

          {/* Pod ID */}
          <div>
            <div style={{ fontSize: 9, color: '#444', marginBottom: 4, fontWeight: 600, letterSpacing: 0.3 }}>RunPod Pod ID</div>
            <input
              type="text"
              value={inferPodId}
              onChange={e => setInferPodId(e.target.value)}
              placeholder="abc123xyz…"
              style={{ ...inputStyle, color: inferPodId ? '#d0d0d0' : '#555' }}
            />
          </div>

          {/* Generate script button */}
          <button
            onClick={onGenerateInference}
            disabled={isGeneratingInference}
            style={{
              padding: '6px 0',
              backgroundColor: isGeneratingInference ? '#1a0a2a' : '#120a1e',
              border: `1px solid ${isGeneratingInference ? typeAccent + '44' : typeAccent + '66'}`,
              borderRadius: 5,
              color: isGeneratingInference ? typeAccent : '#b78bf5',
              cursor: isGeneratingInference ? 'not-allowed' : 'pointer',
              fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
            }}
          >
            {isGeneratingInference ? 'Generating…' : inferGenerated ? '↺ Re-generate Script' : '⚡ Generate Inference Script'}
          </button>

          {/* Run inference button */}
          <button
            onClick={onRunInference}
            disabled={!canRun || isRunningInference}
            style={{
              padding: '6px 0',
              backgroundColor: canRun && !isRunningInference ? '#0a1a0a' : '#0d0d0d',
              border: `1px solid ${canRun && !isRunningInference ? '#166534' : '#1a1a1a'}`,
              borderRadius: 5,
              color: canRun && !isRunningInference ? '#4ade80' : '#2a2a2a',
              cursor: canRun && !isRunningInference ? 'pointer' : 'not-allowed',
              fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
            }}
            title={!inferPodId ? 'Enter a Pod ID' : !inferInput ? 'Enter input' : !inferGenerated ? 'Generate script first' : 'Run on GPU pod'}
          >
            {isRunningInference ? 'Running…' : '▶ Run on GPU'}
          </button>

          {!inferGenerated && !isGeneratingInference && (
            <div style={{ fontSize: 9, color: '#2a2a2a', lineHeight: 1.5 }}>
              Generate the inference script first, then enter your input and pod ID to run.
            </div>
          )}
        </div>
      </div>

      {/* ── Right: Script + Output ───────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', backgroundColor: '#080808' }}>

        {/* Script section */}
        {inferGenerated && (
          <>
            <div style={{ padding: '6px 12px', borderBottom: '1px solid #111', display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
              <div style={{ fontSize: 9, letterSpacing: 1.5, color: '#3a3a3a', textTransform: 'uppercase' as const, fontWeight: 700 }}>inference.py</div>
              <span style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, backgroundColor: '#1a0a2a', color: typeAccent, border: `1px solid ${typeAccent}44`, fontWeight: 700 }}>
                {inferGenerated.taskType}
              </span>
              <div style={{ flex: 1 }} />
              <button
                onClick={onCopyScript}
                style={{
                  padding: '2px 8px',
                  backgroundColor: inferScriptCopied ? '#0a2a0a' : '#141414',
                  border: `1px solid ${inferScriptCopied ? '#166534' : '#252525'}`,
                  borderRadius: 3,
                  color: inferScriptCopied ? '#4ade80' : '#555',
                  cursor: 'pointer', fontSize: 8, fontWeight: 700,
                }}
              >
                {inferScriptCopied ? '✓ Copied' : 'copy'}
              </button>
            </div>
            <div style={{ padding: '4px 8px 2px', fontSize: 9, color: '#2a2a2a', borderBottom: '1px solid #0d0d0d', flexShrink: 0 }}>
              Input: <span style={{ color: '#444', fontFamily: 'monospace' }}>{inferGenerated.inputFormat}</span>
            </div>
            <pre style={{
              flex: '0 0 auto',
              maxHeight: inferResult ? '35%' : '70%',
              overflowY: 'auto',
              margin: 0,
              padding: '8px 12px',
              fontSize: 9.5, color: '#a8e6a3', lineHeight: 1.5,
              fontFamily: '"Fira Code", "Cascadia Code", monospace',
              whiteSpace: 'pre',
              borderBottom: '1px solid #111',
            }}>
              {inferGenerated.inferencePy}
            </pre>
          </>
        )}

        {/* Output section */}
        {inferResult && (
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 12px' }}>
            <div style={{ fontSize: 9, letterSpacing: 1.5, color: '#3a3a3a', textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 8 }}>
              Output
            </div>
            {inferResult.error ? (
              <div style={{
                padding: '8px 10px', borderRadius: 5,
                backgroundColor: '#1a0808', border: '1px solid #3d1515',
                fontSize: 10, color: '#c05050', fontFamily: 'monospace', lineHeight: 1.5,
              }}>
                {inferResult.error}
              </div>
            ) : inferResult.result ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {Object.entries(inferResult.result).map(([key, val]) => (
                  <div key={key} style={{
                    padding: '7px 10px', borderRadius: 4,
                    backgroundColor: '#0a0a0a', border: '1px solid #161616',
                  }}>
                    <div style={{ fontSize: 9, color: '#444', fontWeight: 700, marginBottom: 3, letterSpacing: 0.3 }}>{key}</div>
                    <div style={{ fontSize: 10.5, color: '#d0d0d0', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {Array.isArray(val)
                        ? `[${(val as number[]).map(v => typeof v === 'number' ? v.toFixed(4) : String(v)).join(', ')}]`
                        : String(val)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <pre style={{ fontSize: 9.5, color: '#a8e6a3', fontFamily: 'monospace', margin: 0 }}>
                {inferResult.rawOutput}
              </pre>
            )}
          </div>
        )}

        {/* Empty state */}
        {!inferGenerated && !inferResult && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#262626', gap: 10 }}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#262626" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10" />
              <path d="M10 8l6 4-6 4V8z" />
            </svg>
            <div style={{ fontSize: 11, textAlign: 'center', lineHeight: 1.6, maxWidth: 200 }}>
              Generate an inference script, enter input,<br />and run on your GPU pod.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TrainNodeRow({
  label, shape, color, isActive, isVirtual, conflict,
}: {
  label: string;
  shape: number[] | null;
  color: string;
  isActive: boolean;
  isVirtual?: boolean;
  conflict?: string | null;
}) {
  return (
    <div style={{
      height: 36, display: 'flex', alignItems: 'center', gap: 8,
      padding: '0 10px',
      borderRadius: 6,
      backgroundColor: isActive ? color + '1a' : isVirtual ? '#0d0d0d' : '#0a0a0a',
      border: `1px solid ${isActive ? color + '55' : isVirtual ? '#1a1a1a' : '#141414'}`,
      boxShadow: isActive ? `0 0 8px ${color}22` : 'none',
      transition: 'all 0.3s',
      marginBottom: 0,
    }}>
      {isVirtual && (
        <div style={{
          fontSize: 8, padding: '1px 5px', borderRadius: 2,
          backgroundColor: color + '22', color, border: `1px solid ${color}44`,
          fontWeight: 700, letterSpacing: 0.5, whiteSpace: 'nowrap',
        }}>
          PREP
        </div>
      )}
      <div style={{ flex: 1, fontSize: 11, fontWeight: isVirtual ? 500 : 600, color: isActive ? color : '#555', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {label}
      </div>
      {conflict && (
        <span style={{ fontSize: 8, color: '#f87171', fontWeight: 700, whiteSpace: 'nowrap' }}>⚠</span>
      )}
      {shape && (
        <span style={{ fontSize: 9, color: isActive ? color + 'cc' : '#2a2a2a', fontWeight: 600, whiteSpace: 'nowrap', fontFamily: 'monospace' }}>
          [{shape.join('×')}]
        </span>
      )}
    </div>
  );
}

function ConnectorLine({ color }: { color: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', height: 12 }}>
      <div style={{ width: 1, height: '100%', backgroundColor: color + '33' }} />
    </div>
  );
}

// ── Shared compact styles ──────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  width: '100%', padding: '4px 6px',
  backgroundColor: '#0f0f0f', border: '1px solid #1e1e1e',
  borderRadius: 4, color: '#888', fontSize: 10, outline: 'none',
  fontFamily: 'inherit', cursor: 'pointer',
};

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '4px 6px',
  backgroundColor: '#0f0f0f', border: '1px solid #1e1e1e',
  borderRadius: 4, color: '#888', fontSize: 10, outline: 'none',
  fontFamily: 'inherit', boxSizing: 'border-box',
};

function TrainConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
      <div style={{ fontSize: 9, color: '#444', fontWeight: 600, width: 48, flexShrink: 0, textAlign: 'right' }}>{label}</div>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
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
