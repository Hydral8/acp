import { useState, useEffect, useRef, useCallback } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';

const modelNodeSchema = z.object({
  id:    z.string(),
  label: z.string(),
  cat:   z.string(),
  shape: z.string().optional(),
});

const propsSchema = z.object({
  modelNodes:          z.array(modelNodeSchema).optional(),
  taskType:            z.string().nullable().optional(),
  suggestedCategory:   z.enum(['llm','vlm','rlhf','cv','tabular']).nullable().optional(),
  suggestedLoss:       z.string().optional(),
  suggestedOptimizer:  z.string().optional(),
  shapeAnnotations:    z.record(z.string(), z.string()).optional(),
});
type Props = z.infer<typeof propsSchema>;

export const widgetMetadata: WidgetMetadata = {
  description: 'Select a training dataset and preview data flowing through the model',
  props: propsSchema,
  exposeAsTool: false,
  metadata: {
    invoking: 'Loading dataset setup…',
    invoked: 'Dataset setup ready',
  },
};

// ── Dataset catalog ──────────────────────────────────────────────────────────

interface DatasetDef {
  id: string;
  name: string;
  source: 'huggingface' | 'kaggle';
  hfId: string;
  description: string;
  size: string;
  task: string;
}

const DATASETS: Record<'llm' | 'vlm' | 'rlhf' | 'cv' | 'tabular', DatasetDef[]> = {
  llm: [
    { id: 'tinystories', name: 'TinyStories', source: 'huggingface', hfId: 'roneneldan/TinyStories', description: 'Simple short stories for training small language models', size: '2.1 GB', task: 'Causal LM' },
    { id: 'openwebtext', name: 'OpenWebText', source: 'huggingface', hfId: 'Skylion007/openwebtext', description: 'Open replication of the WebText corpus used for GPT-2', size: '40 GB', task: 'Causal LM' },
    { id: 'minipile', name: 'MiniPile', source: 'huggingface', hfId: 'JeanKaddour/minipile', description: 'Compact 1M-sample subset of The Pile for fast experiments', size: '6.4 GB', task: 'Causal LM' },
    { id: 'wikitext', name: 'WikiText-103', source: 'huggingface', hfId: 'Salesforce/wikitext', description: 'Wikipedia articles — standard LM benchmark corpus', size: '510 MB', task: 'Causal LM' },
  ],
  vlm: [
    { id: 'cc3m', name: 'Conceptual Captions', source: 'huggingface', hfId: 'google-research-datasets/conceptual_captions', description: 'Web-harvested image–caption pairs from Google', size: '3 M pairs', task: 'Image–Text' },
    { id: 'flickr30k', name: 'Flickr30K', source: 'huggingface', hfId: 'nlphuji/flickr30k', description: '31 K images with 5 human-written captions each', size: '31 K pairs', task: 'Image–Text' },
    { id: 'laion', name: 'LAION Aesthetics', source: 'huggingface', hfId: 'laion/laion-aesthetic-v2', description: 'High-aesthetic image subset of LAION for VLM training', size: '900 K pairs', task: 'Image–Text' },
    { id: 'nocaps', name: 'NoCaps', source: 'huggingface', hfId: 'HuggingFaceM4/NoCaps', description: 'Novel object captioning with open-domain images', size: '15 K pairs', task: 'Image–Text' },
  ],
  rlhf: [
    { id: 'hh-rlhf', name: 'Anthropic HH-RLHF', source: 'huggingface', hfId: 'Anthropic/hh-rlhf', description: 'Helpful & harmless human preference data from Anthropic', size: '170 K pairs', task: 'RLHF' },
    { id: 'shp', name: 'Stanford SHP', source: 'huggingface', hfId: 'stanfordnlp/SHP', description: 'Reddit Q&A with upvote-based human preferences', size: '385 K pairs', task: 'RLHF' },
    { id: 'summarize', name: 'OpenAI Summarize', source: 'huggingface', hfId: 'openai/summarize_from_feedback', description: 'Human RLHF feedback on TL;DR summaries', size: '92 K pairs', task: 'RLHF' },
    { id: 'webgpt', name: 'WebGPT Comparisons', source: 'huggingface', hfId: 'openai/webgpt_comparisons', description: 'WebGPT answer quality comparison pairs', size: '19 K pairs', task: 'RLHF' },
  ],
  cv: [
    { id: 'mnist',      name: 'MNIST',              source: 'huggingface', hfId: 'ylecun/mnist',            description: '70 K handwritten digit images (28×28 grayscale), 10 classes', size: '11 MB',   task: 'Classification' },
    { id: 'cifar10',    name: 'CIFAR-10',            source: 'huggingface', hfId: 'uoft-cs/cifar10',         description: '60 K color images (32×32, RGB), 10 classes',                 size: '163 MB',  task: 'Classification' },
    { id: 'imagenet1k', name: 'ImageNet-1K',         source: 'huggingface', hfId: 'ILSVRC/imagenet-1k',      description: '1.28 M images across 1 000 classes (requires access)',        size: '150 GB',  task: 'Classification' },
    { id: 'food101',    name: 'Food-101',            source: 'huggingface', hfId: 'ethz/food101',            description: '101 000 food images across 101 categories',                   size: '4.7 GB',  task: 'Classification' },
  ],
  tabular: [
    { id: 'iris',       name: 'Iris',                source: 'huggingface', hfId: 'scikit-learn/iris',       description: '150 samples · 4 features · 3 flower species. Classic starter.', size: '< 1 MB',  task: 'Classification' },
    { id: 'titanic',    name: 'Titanic',             source: 'huggingface', hfId: 'mstz/titanic',            description: 'Survival prediction from passenger demographics',               size: '< 1 MB',  task: 'Classification' },
    { id: 'adult',      name: 'Adult Income',        source: 'huggingface', hfId: 'scikit-learn/adult-census-income', description: 'Census-based binary income classification (>50 K)',   size: '3.8 MB',  task: 'Classification' },
    { id: 'california', name: 'California Housing',  source: 'huggingface', hfId: 'lhoestq/demo-wikipedia-stats', description: 'Median house value regression from census block features', size: '< 1 MB', task: 'Regression' },
  ],
};

// ── Demo model graphs per training type ──────────────────────────────────────

interface DemoNode { id: string; label: string; cat: 'core' | 'activation' | 'composite'; }

const DEMO_GRAPHS: Record<CatKey, DemoNode[]> = {
  llm: [
    { id: 'inp',  label: 'Input',          cat: 'core' },
    { id: 'emb',  label: 'Embedding',      cat: 'core' },
    { id: 'pe',   label: 'Sine PE',        cat: 'core' },
    { id: 'attn', label: 'MultiHead Attn', cat: 'composite' },
    { id: 'ln',   label: 'LayerNorm',      cat: 'core' },
    { id: 'fc',   label: 'Linear',         cat: 'core' },
    { id: 'sm',   label: 'Softmax',        cat: 'activation' },
  ],
  vlm: [
    { id: 'inp',  label: 'Input',     cat: 'core' },
    { id: 'conv', label: 'Conv2D',    cat: 'core' },
    { id: 'bn',   label: 'BatchNorm', cat: 'core' },
    { id: 'relu', label: 'ReLU',      cat: 'activation' },
    { id: 'flat', label: 'Flatten',   cat: 'core' },
    { id: 'fc1',  label: 'Linear',    cat: 'core' },
    { id: 'sm',   label: 'Softmax',   cat: 'activation' },
  ],
  rlhf: [
    { id: 'inp',  label: 'Input',     cat: 'core' },
    { id: 'emb',  label: 'Embedding', cat: 'core' },
    { id: 'fc1',  label: 'Linear',    cat: 'core' },
    { id: 'gelu', label: 'GELU',      cat: 'activation' },
    { id: 'fc2',  label: 'Linear',    cat: 'core' },
    { id: 'sig',  label: 'Sigmoid',   cat: 'activation' },
  ],
  cv: [
    { id: 'inp',   label: 'Input',       cat: 'core' },
    { id: 'conv1', label: 'Conv2D',      cat: 'core' },
    { id: 'bn1',   label: 'BatchNorm',   cat: 'core' },
    { id: 'relu1', label: 'ReLU',        cat: 'activation' },
    { id: 'conv2', label: 'Conv2D',      cat: 'core' },
    { id: 'flat',  label: 'Flatten',     cat: 'core' },
    { id: 'fc',    label: 'Linear',      cat: 'core' },
    { id: 'sm',    label: 'Softmax',     cat: 'activation' },
  ],
  tabular: [
    { id: 'inp',  label: 'Input',   cat: 'core' },
    { id: 'fc1',  label: 'Linear',  cat: 'core' },
    { id: 'relu', label: 'ReLU',    cat: 'activation' },
    { id: 'fc2',  label: 'Linear',  cat: 'core' },
    { id: 'relu2',label: 'ReLU',    cat: 'activation' },
    { id: 'out',  label: 'Linear',  cat: 'core' },
  ],
};

const NODE_COLORS: Record<string, string> = {
  core:       '#2563eb',
  activation: '#7c3aed',
  composite:  '#ca8a04',
};

type CatKey = 'llm' | 'vlm' | 'rlhf' | 'cv' | 'tabular';
const ACCENT:     Record<CatKey, string> = { llm: '#3b82f6', vlm: '#8b5cf6', rlhf: '#10b981', cv: '#f59e0b', tabular: '#06b6d4' };
const BLOB_ICON:  Record<CatKey, string> = { llm: '📝', vlm: '🖼️', rlhf: '🎯', cv: '🖼', tabular: '📊' };
const BLOB_LABEL: Record<CatKey, string> = { llm: 'Text Tokens', vlm: 'Image Patches', rlhf: 'Pref. Pairs', cv: 'Image Batch', tabular: 'Row Batch' };
const CAT_TABS: Array<{ key: CatKey; label: string }> = [
  { key: 'llm',     label: 'LLM' },
  { key: 'vlm',     label: 'VLM' },
  { key: 'rlhf',    label: 'RL/RLHF' },
  { key: 'cv',      label: 'Vision' },
  { key: 'tabular', label: 'Tabular' },
];

const NODE_H = 38;
const EDGE_H = 22;

// ── Main widget ───────────────────────────────────────────────────────────────

interface GeneratedFiles { modelPy: string; dataPy: string; trainPy: string; }

export default function DatasetPrep() {
  const { props, isPending } = useWidget<Props>();
  const { callToolAsync: callGenTraining, isPending: isGenerating } = useCallTool('generate-training-code');

  // Dataset selection
  const [inputTab,  setInputTab]  = useState<'curated' | 'custom'>('curated');
  const [category,  setCategory]  = useState<CatKey>('llm');
  const [selected,  setSelected]  = useState<DatasetDef | null>(null);
  const [customUrl, setCustomUrl] = useState('');
  const [platform,  setPlatform]  = useState<'huggingface' | 'kaggle' | null>(null);
  const [hfToken,   setHfToken]   = useState('');
  const [kaggleKey, setKaggleKey] = useState('');

  // Loss + optimizer — seeded from AI suggestion, user-editable
  const [loss,      setLoss]      = useState('CrossEntropyLoss');
  const [optimizer, setOptimizer] = useState('Adam');

  // Hyperparameters
  const [lr,       setLr]       = useState('0.001');
  const [batchSz,  setBatchSz]  = useState('32');
  const [epochs,   setEpochs]   = useState('10');

  // Generated code output
  const [files,    setFiles]    = useState<GeneratedFiles | null>(null);
  const [codeTab,  setCodeTab]  = useState<'model' | 'data' | 'train'>('train');
  const [copied,   setCopied]   = useState(false);

  // Animation
  const [simRunning, setSimRunning] = useState(false);
  const [animStep,   setAnimStep]   = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Use real model nodes from the graph if provided, otherwise fall back to demo
  const customNodes = !isPending && props.modelNodes && props.modelNodes.length > 0
    ? (props.modelNodes as DemoNode[])
    : null;
  const demoNodes      = customNodes ?? DEMO_GRAPHS[category];
  const hasCustomGraph = !!customNodes;
  const accentColor    = ACCENT[category];
  const taskType       = !isPending ? (props.taskType ?? null) : null;
  const sugLoss        = !isPending ? (props.suggestedLoss ?? 'CrossEntropyLoss') : 'CrossEntropyLoss';
  const sugOptimizer   = !isPending ? (props.suggestedOptimizer ?? 'Adam') : 'Adam';

  // Auto-select category / loss / optimizer from detected task (seed only once)
  useEffect(() => {
    if (isPending) return;
    if (props.suggestedCategory) setCategory(props.suggestedCategory);
    if (props.suggestedLoss)      setLoss(props.suggestedLoss);
    if (props.suggestedOptimizer) setOptimizer(props.suggestedOptimizer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPending]);

  // Auto-detect platform from URL
  useEffect(() => {
    const url = customUrl.toLowerCase();
    if (url.includes('huggingface.co') || url.includes('hf.co')) setPlatform('huggingface');
    else if (url.includes('kaggle.com')) setPlatform('kaggle');
    else setPlatform(null);
  }, [customUrl]);

  // Reset sim when category changes
  useEffect(() => {
    setSimRunning(false);
    setAnimStep(-1);
    if (timerRef.current) clearTimeout(timerRef.current);
  }, [category]);

  // Animation tick — loops indefinitely
  useEffect(() => {
    if (!simRunning) return;
    const next = animStep + 1;
    const delay = next >= demoNodes.length ? 700 : 440;
    timerRef.current = setTimeout(() => setAnimStep(next >= demoNodes.length ? 0 : next), delay);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [simRunning, animStep, demoNodes.length]);

  const startSim = useCallback(() => { setAnimStep(0); setSimRunning(true); }, []);
  const stopSim  = useCallback(() => {
    setSimRunning(false); setAnimStep(-1);
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const handleGenerate = useCallback(async () => {
    const ds = inputTab === 'curated' && selected
      ? { name: selected.name, source: 'huggingface' as const, hfId: selected.hfId }
      : { name: customUrl || 'custom', source: 'custom' as const };
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callGenTraining({
        dataset: ds, taskType: taskType ?? 'unknown',
        optimizer: optimizer as 'Adam', loss: loss as 'CrossEntropyLoss',
        hyperparams: { lr: parseFloat(lr) || 0.001, batch_size: parseInt(batchSz) || 32, epochs: parseInt(epochs) || 10 },
      } as any);
      const sc = res?.structuredContent as { modelPy?: string; dataPy?: string; trainPy?: string } | undefined;
      if (sc?.modelPy) { setFiles({ modelPy: sc.modelPy, dataPy: sc.dataPy ?? '', trainPy: sc.trainPy ?? '' }); setCodeTab('train'); }
    } catch { /* silent */ }
  }, [inputTab, selected, customUrl, taskType, loss, optimizer, lr, batchSz, epochs, callGenTraining]);

  const copyCode = useCallback(() => {
    if (!files) return;
    const content = codeTab === 'model' ? files.modelPy : codeTab === 'data' ? files.dataPy : files.trainPy;
    navigator.clipboard.writeText(content).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1600); });
  }, [files, codeTab]);

  const canGenerate = (inputTab === 'curated' ? !!selected : !!customUrl.trim()) && !isGenerating;

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{ height: 580, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0d0d0d', color: '#333', fontFamily: 'monospace', fontSize: 12 }}>
          Initializing…
        </div>
      </McpUseProvider>
    );
  }

  return (
    <McpUseProvider autoSize>
      <style>{`
        @keyframes blobGlow {
          0%, 100% { transform: scale(1); }
          50%       { transform: scale(1.06); }
        }
        @keyframes nodePulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.75; }
        }
        @keyframes edgeDash {
          from { stroke-dashoffset: 16; }
          to   { stroke-dashoffset: 0; }
        }
      `}</style>
      <div style={{
        height: 580,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#0d0d0d',
        color: '#e0e0e0',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 13,
        overflow: 'hidden',
      }}>
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <div style={{ height: 42, borderBottom: '1px solid #141414', display: 'flex', alignItems: 'center', padding: '0 14px', gap: 10, backgroundColor: '#070707', flexShrink: 0 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#d0d0d0', letterSpacing: 0.3 }}>Prepare Training</span>
          <div style={{ width: 1, height: 16, backgroundColor: '#1e1e1e' }} />
          <span style={{ fontSize: 10, color: '#3a3a3a' }}>Select dataset → preview forward pass</span>
        </div>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>

          {/* LEFT: Dataset picker ─────────────────────────────────────────── */}
          <div style={{ width: 290, borderRight: '1px solid #141414', display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden' }}>

            {/* Input mode tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid #141414', flexShrink: 0 }}>
              {(['curated', 'custom'] as const).map(t => (
                <button key={t} onClick={() => setInputTab(t)} style={{
                  flex: 1, padding: '9px 0', background: 'none', border: 'none',
                  borderBottom: `2px solid ${inputTab === t ? '#3b82f6' : 'transparent'}`,
                  color: inputTab === t ? '#93c5fd' : '#444',
                  cursor: 'pointer', fontSize: 11, fontWeight: 600, letterSpacing: 0.3,
                  textTransform: 'uppercase' as const,
                }}>
                  {t === 'curated' ? 'Curated' : 'Custom URL'}
                </button>
              ))}
            </div>

            {inputTab === 'curated' ? (
              <>
                {/* Category tabs */}
                <div style={{ display: 'flex', borderBottom: '1px solid #141414', flexShrink: 0 }}>
                  {CAT_TABS.map(({ key, label }) => (
                    <button key={key} onClick={() => { setCategory(key); setSelected(null); }} style={{
                      flex: 1, padding: '8px 0', background: 'none', border: 'none',
                      borderBottom: `2px solid ${category === key ? accentColor : 'transparent'}`,
                      color: category === key ? accentColor : '#3a3a3a',
                      cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                    }}>
                      {label}
                    </button>
                  ))}
                </div>

                {/* Dataset list */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '8px 8px 0' }}>
                  {DATASETS[category].map(ds => {
                    const isSel = selected?.id === ds.id;
                    return (
                      <div key={ds.id} onClick={() => setSelected(ds)} style={{
                        padding: '10px 11px', borderRadius: 7, marginBottom: 6, cursor: 'pointer',
                        border: `1px solid ${isSel ? accentColor : '#191919'}`,
                        backgroundColor: isSel ? `${accentColor}0d` : '#090909',
                        transition: 'border-color 0.15s, background-color 0.15s',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                          <span style={{ fontSize: 11, fontWeight: 700, color: isSel ? accentColor : '#b0b0b0' }}>{ds.name}</span>
                          <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, backgroundColor: '#0f1f10', color: '#4ade80', border: '1px solid #166534', fontWeight: 600 }}>{ds.task}</span>
                        </div>
                        <div style={{ fontSize: 10, color: '#484848', marginBottom: 4, lineHeight: 1.4 }}>{ds.description}</div>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <span style={{ fontSize: 9, color: '#e07020', fontFamily: 'monospace' }}>{ds.hfId}</span>
                          <span style={{ fontSize: 9, color: '#282828' }}>{ds.size}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* HF token — shown for any HuggingFace dataset */}
                {selected && (
                  <div style={{ padding: '10px 12px', borderTop: '1px solid #141414', flexShrink: 0 }}>
                    <div style={{ fontSize: 10, color: '#484848', marginBottom: 5, display: 'flex', alignItems: 'center', gap: 5 }}>
                      <span style={{ color: '#e07020' }}>⬡</span>
                      HuggingFace Token
                      <span style={{ color: '#282828' }}>(optional for public datasets)</span>
                    </div>
                    <input
                      type="password"
                      value={hfToken}
                      onChange={e => setHfToken(e.target.value)}
                      placeholder="hf_…"
                      style={{ width: '100%', padding: '6px 10px', backgroundColor: '#111', border: '1px solid #222', borderRadius: 5, color: '#e0e0e0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, fontFamily: 'monospace' }}
                    />
                  </div>
                )}
              </>
            ) : (
              /* Custom URL panel */
              <div style={{ flex: 1, padding: '14px 12px', display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto' }}>
                <div>
                  <div style={{ fontSize: 10, color: '#484848', marginBottom: 6 }}>Dataset URL</div>
                  <input
                    type="text"
                    value={customUrl}
                    onChange={e => setCustomUrl(e.target.value)}
                    placeholder="https://huggingface.co/datasets/…  or  kaggle.com/datasets/…"
                    style={{
                      width: '100%', padding: '7px 10px', backgroundColor: '#111',
                      border: `1px solid ${platform ? accentColor : '#222'}`,
                      borderRadius: 5, color: '#e0e0e0', fontSize: 11, outline: 'none',
                      boxSizing: 'border-box' as const, fontFamily: 'monospace',
                      transition: 'border-color 0.2s',
                    }}
                  />
                </div>

                {/* Platform badge */}
                {platform && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      fontSize: 10, padding: '3px 9px', borderRadius: 4, fontWeight: 700, letterSpacing: 0.3,
                      backgroundColor: platform === 'huggingface' ? '#0f1928' : '#12100a',
                      color:           platform === 'huggingface' ? '#60a5fa' : '#fbbf24',
                      border: `1px solid ${platform === 'huggingface' ? '#1d4ed8' : '#92400e'}`,
                    }}>
                      {platform === 'huggingface' ? '⬡ HuggingFace' : '◆ Kaggle'}
                    </span>
                    <span style={{ fontSize: 9, color: '#333' }}>detected</span>
                  </div>
                )}

                {platform === 'huggingface' && (
                  <div>
                    <div style={{ fontSize: 10, color: '#484848', marginBottom: 6 }}>HuggingFace Token</div>
                    <input
                      type="password" value={hfToken} onChange={e => setHfToken(e.target.value)}
                      placeholder="hf_…"
                      style={{ width: '100%', padding: '6px 10px', backgroundColor: '#111', border: '1px solid #1d4ed8', borderRadius: 5, color: '#e0e0e0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, fontFamily: 'monospace' }}
                    />
                  </div>
                )}

                {platform === 'kaggle' && (
                  <div>
                    <div style={{ fontSize: 10, color: '#484848', marginBottom: 6 }}>Kaggle API Key</div>
                    <input
                      type="password" value={kaggleKey} onChange={e => setKaggleKey(e.target.value)}
                      placeholder="your_kaggle_api_key"
                      style={{ width: '100%', padding: '6px 10px', backgroundColor: '#111', border: '1px solid #92400e', borderRadius: 5, color: '#e0e0e0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, fontFamily: 'monospace' }}
                    />
                  </div>
                )}

                <div style={{ fontSize: 10, color: '#282828', lineHeight: 1.6 }}>
                  Supported: HuggingFace and Kaggle datasets.<br />
                  Platform is auto-detected from the URL.<br />
                  API keys are used only for private datasets.
                </div>
              </div>
            )}

            {/* Training config — Loss + Optimizer */}
            <div style={{ padding: '10px 12px 8px', borderTop: '1px solid #141414', flexShrink: 0 }}>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 8 }}>
                Training Config
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {/* Loss selector */}
                <div>
                  <div style={{ fontSize: 9, color: '#484848', marginBottom: 3 }}>Loss Function</div>
                  <select
                    value={loss}
                    onChange={e => setLoss(e.target.value)}
                    style={{ width: '100%', padding: '5px 7px', backgroundColor: '#111', border: '1px solid #222', borderRadius: 4, color: '#d0d0d0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, cursor: 'pointer', appearance: 'none' as const }}
                  >
                    {['CrossEntropyLoss','BCEWithLogitsLoss','MSELoss','L1Loss','HuberLoss','NLLLoss','KLDivLoss','CTCLoss'].map(l => (
                      <option key={l} value={l}>{l}</option>
                    ))}
                  </select>
                </div>
                {/* Optimizer selector */}
                <div>
                  <div style={{ fontSize: 9, color: '#484848', marginBottom: 3 }}>Optimizer</div>
                  <select
                    value={optimizer}
                    onChange={e => setOptimizer(e.target.value)}
                    style={{ width: '100%', padding: '5px 7px', backgroundColor: '#111', border: '1px solid #222', borderRadius: 4, color: '#d0d0d0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, cursor: 'pointer', appearance: 'none' as const }}
                  >
                    {['Adam','AdamW','SGD','RMSprop','Adadelta','Adagrad','LBFGS'].map(o => (
                      <option key={o} value={o}>{o}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Hyperparameters */}
            <div style={{ padding: '6px 12px 6px', flexShrink: 0 }}>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.5, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 8 }}>
                Hyperparameters
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
                {([
                  { label: 'Learning Rate', value: lr, set: setLr },
                  { label: 'Batch Size', value: batchSz, set: setBatchSz },
                ] as const).map(({ label, value, set }) => (
                  <div key={label}>
                    <div style={{ fontSize: 9, color: '#484848', marginBottom: 3 }}>{label}</div>
                    <input
                      type="text" value={value} onChange={e => set(e.target.value)}
                      style={{ width: '100%', padding: '5px 7px', backgroundColor: '#111', border: '1px solid #222', borderRadius: 4, color: '#d0d0d0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, fontFamily: 'monospace' }}
                    />
                  </div>
                ))}
              </div>
              <div>
                <div style={{ fontSize: 9, color: '#484848', marginBottom: 3 }}>Epochs</div>
                <input
                  type="text" value={epochs} onChange={e => setEpochs(e.target.value)}
                  style={{ width: '100%', padding: '5px 7px', backgroundColor: '#111', border: '1px solid #222', borderRadius: 4, color: '#d0d0d0', fontSize: 11, outline: 'none', boxSizing: 'border-box' as const, fontFamily: 'monospace' }}
                />
              </div>
            </div>

            {/* Generate button */}
            <div style={{ padding: '8px 12px 10px', flexShrink: 0 }}>
              <button
                disabled={!canGenerate}
                onClick={handleGenerate}
                style={{
                  width: '100%', padding: '9px 0',
                  backgroundColor: isGenerating ? '#0d0d0d' : canGenerate ? '#0a1a2a' : '#0d0d0d',
                  border: `1px solid ${isGenerating ? '#1d4ed8' : canGenerate ? '#3b82f6' : '#1a1a1a'}`,
                  borderRadius: 6,
                  color: isGenerating ? '#60a5fa' : canGenerate ? '#93c5fd' : '#2a2a2a',
                  cursor: canGenerate ? 'pointer' : 'not-allowed',
                  fontSize: 11, fontWeight: 700, letterSpacing: 0.4,
                  transition: 'all 0.15s',
                }}
              >
                {isGenerating ? '⟳ Generating…' : '⚡ Generate Training Scripts'}
              </button>
            </div>
          </div>

          {/* RIGHT: Model simulation + code output ───────────────────────── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Sim controls */}
            <div style={{ height: 38, borderBottom: '1px solid #141414', display: 'flex', alignItems: 'center', padding: '0 14px', gap: 8, backgroundColor: '#080808', flexShrink: 0 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: '#3a3a3a', letterSpacing: 0.5 }}>MODEL SIMULATION</span>
              {hasCustomGraph ? (
                <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 3, backgroundColor: '#0a1a0a', color: '#4ade80', border: '1px solid #166534', fontWeight: 700, letterSpacing: 0.3 }}>
                  YOUR MODEL · {demoNodes.length} layers
                </span>
              ) : (
                <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 3, backgroundColor: '#111', color: '#444', border: '1px solid #222', fontWeight: 600 }}>
                  demo
                </span>
              )}
              {taskType && (
                <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 3, backgroundColor: '#0a1628', color: '#60a5fa', border: '1px solid #1d4ed8', fontWeight: 700, letterSpacing: 0.3 }}>
                  {taskType}
                </span>
              )}
              {sugLoss && sugLoss !== 'CrossEntropyLoss' || sugOptimizer && sugOptimizer !== 'Adam' ? (
                <span style={{ fontSize: 9, color: '#333' }}>
                  {sugLoss} · {sugOptimizer}
                </span>
              ) : null}
              <div style={{ flex: 1 }} />
              {simRunning ? (
                <button onClick={stopSim} style={{ padding: '4px 12px', backgroundColor: '#1a0a0a', border: '1px solid #7f1d1d', borderRadius: 4, color: '#f87171', cursor: 'pointer', fontSize: 10, fontWeight: 700 }}>
                  ■ Stop
                </button>
              ) : (
                <button onClick={startSim} style={{ padding: '4px 12px', backgroundColor: '#0a1a0a', border: '1px solid #166534', borderRadius: 4, color: '#4ade80', cursor: 'pointer', fontSize: 10, fontWeight: 700 }}>
                  ▶ Run Dummy Pass
                </button>
              )}
            </div>

            {/* Simulation canvas */}
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflowY: 'auto', padding: '16px 24px', minHeight: 0 }}>
              <SimulationView
                nodes={demoNodes}
                animStep={animStep}
                simRunning={simRunning}
                category={category}
                shapeAnnotations={!isPending ? (props.shapeAnnotations ?? {}) : {}}
              />
            </div>

            {/* Code output panel — shown after generation */}
            {files && (
              <div style={{ height: 210, borderTop: '1px solid #141414', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
                {/* Tabs + copy */}
                <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid #141414', backgroundColor: '#080808', flexShrink: 0 }}>
                  {(['model', 'data', 'train'] as const).map(tab => (
                    <button key={tab} onClick={() => setCodeTab(tab)} style={{
                      padding: '7px 14px', background: 'none', border: 'none',
                      borderBottom: `2px solid ${codeTab === tab ? '#3b82f6' : 'transparent'}`,
                      color: codeTab === tab ? '#93c5fd' : '#444',
                      cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                    }}>
                      {tab === 'model' ? 'model.py' : tab === 'data' ? 'data.py' : 'train.py'}
                    </button>
                  ))}
                  <div style={{ flex: 1 }} />
                  <button onClick={copyCode} style={{
                    padding: '4px 12px', margin: '0 8px',
                    backgroundColor: copied ? '#0a1a0a' : '#111',
                    border: `1px solid ${copied ? '#166534' : '#222'}`,
                    borderRadius: 4, color: copied ? '#4ade80' : '#555',
                    cursor: 'pointer', fontSize: 9, fontWeight: 700,
                    transition: 'all 0.2s',
                  }}>
                    {copied ? '✓ Copied' : 'Copy'}
                  </button>
                </div>
                {/* Code scroll */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '10px 14px', backgroundColor: '#060606' }}>
                  <pre style={{ margin: 0, fontSize: 10, color: '#9ca3af', fontFamily: 'monospace', lineHeight: 1.6, whiteSpace: 'pre-wrap' as const, wordBreak: 'break-word' as const }}>
                    {codeTab === 'model' ? files.modelPy : codeTab === 'data' ? files.dataPy : files.trainPy}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </McpUseProvider>
  );
}

// ── Simulation view ───────────────────────────────────────────────────────────

function SimulationView({
  nodes, animStep, simRunning, category, shapeAnnotations,
}: {
  nodes: DemoNode[];
  animStep: number;
  simRunning: boolean;
  category: CatKey;
  shapeAnnotations: Record<string, string>;
}) {
  const accent    = ACCENT[category];
  const totalH    = nodes.length * NODE_H + (nodes.length - 1) * EDGE_H;
  const NODE_W    = 172;
  // SVG dimensions for edges
  const svgW      = NODE_W;
  const edgeCx    = NODE_W / 2;

  return (
    <div style={{ display: 'flex', gap: 36, alignItems: 'flex-start' }}>

      {/* Dataset blob */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, paddingTop: NODE_H / 2 - 34 }}>
        <div style={{
          width: 68, height: 68, borderRadius: '50%',
          backgroundColor: `${accent}0f`,
          border: `2px solid ${simRunning ? accent : '#1e1e1e'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: simRunning ? `0 0 20px ${accent}44, 0 0 40px ${accent}1a` : 'none',
          transition: 'border-color 0.4s, box-shadow 0.4s',
          animation: simRunning ? 'blobGlow 1.8s ease-in-out infinite' : 'none',
          flexShrink: 0,
        }}>
          <div style={{
            width: 38, height: 38, borderRadius: '50%',
            backgroundColor: simRunning ? `${accent}1a` : '#0d0d0d',
            border: `1px solid ${simRunning ? accent : '#1a1a1a'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, transition: 'background-color 0.4s, border-color 0.4s',
          }}>
            {BLOB_ICON[category]}
          </div>
        </div>

        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
          color: simRunning ? accent : '#333',
          transition: 'color 0.4s', textAlign: 'center' as const, maxWidth: 72,
        }}>
          {BLOB_LABEL[category]}
        </span>

        {simRunning && animStep >= 0 && (
          <span style={{ fontSize: 9, color: '#333', fontFamily: 'monospace' }}>
            step {animStep + 1}/{nodes.length}
          </span>
        )}
      </div>

      {/* Connector: blob → first node */}
      <div style={{
        paddingTop: NODE_H / 2 - 1,
        display: 'flex', alignItems: 'center', flexShrink: 0,
      }}>
        <div style={{
          width: 28, height: 2,
          backgroundColor: simRunning && animStep >= 0 ? accent : '#1e1e1e',
          boxShadow: simRunning && animStep >= 0 ? `0 0 6px ${accent}` : 'none',
          transition: 'background-color 0.3s, box-shadow 0.3s',
        }} />
        <div style={{
          width: 0, height: 0,
          borderTop: '5px solid transparent',
          borderBottom: '5px solid transparent',
          borderLeft: `7px solid ${simRunning && animStep >= 0 ? accent : '#1e1e1e'}`,
          marginTop: -1,
          transition: 'border-left-color 0.3s',
        }} />
      </div>

      {/* Model nodes + SVG edges */}
      <div style={{ position: 'relative', width: NODE_W, height: totalH, flexShrink: 0 }}>

        {/* SVG layer: animated edges */}
        <svg
          width={svgW}
          height={totalH}
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        >
          {nodes.map((node, idx) => {
            if (idx === 0) return null;
            const y1 = (idx - 1) * (NODE_H + EDGE_H) + NODE_H;
            const y2 = idx * (NODE_H + EDGE_H);
            const nodeColor = NODE_COLORS[node.cat] ?? '#2563eb';
            const prevColor = NODE_COLORS[nodes[idx - 1].cat] ?? '#2563eb';
            const isActive  = simRunning && animStep === idx;
            const isPast    = simRunning && animStep > idx;

            return (
              <g key={node.id}>
                {/* Background line */}
                <line
                  x1={edgeCx} y1={y1}
                  x2={edgeCx} y2={y2}
                  stroke={isPast ? `${nodeColor}44` : '#1c1c1c'}
                  strokeWidth={2}
                />
                {/* Animated flow line */}
                {isActive && (
                  <line
                    x1={edgeCx} y1={y1}
                    x2={edgeCx} y2={y2}
                    stroke={`url(#grad-${idx})`}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    style={{ animation: 'edgeDash 0.35s linear infinite' }}
                  />
                )}
                {/* Gradient def for active edge */}
                {isActive && (
                  <defs>
                    <linearGradient id={`grad-${idx}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={prevColor} />
                      <stop offset="100%" stopColor={nodeColor} />
                    </linearGradient>
                  </defs>
                )}
                {/* Glow overlay for active */}
                {isActive && (
                  <line
                    x1={edgeCx} y1={y1}
                    x2={edgeCx} y2={y2}
                    stroke={nodeColor}
                    strokeWidth={4}
                    strokeOpacity={0.2}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {/* Nodes */}
        {nodes.map((node, idx) => {
          const top       = idx * (NODE_H + EDGE_H);
          const nodeColor = NODE_COLORS[node.cat] ?? '#2563eb';
          const isActive  = simRunning && animStep === idx;
          const isPast    = simRunning && animStep > idx;

          return (
            <div
              key={node.id}
              style={{
                position: 'absolute', top, left: 0,
                width: NODE_W, height: NODE_H,
                backgroundColor: isActive ? `${nodeColor}18` : isPast ? `${nodeColor}08` : '#0a0a0a',
                border: `1px solid ${isActive ? nodeColor : isPast ? `${nodeColor}44` : '#181818'}`,
                borderRadius: 8,
                display: 'flex', alignItems: 'center', gap: 10, padding: '0 12px',
                boxSizing: 'border-box' as const,
                boxShadow: isActive ? `0 0 14px ${nodeColor}44, 0 0 28px ${nodeColor}18` : 'none',
                transition: 'background-color 0.22s, border-color 0.22s, box-shadow 0.22s',
                animation: isActive ? 'nodePulse 0.9s ease-in-out infinite' : 'none',
              }}
            >
              {/* Status dot */}
              <div style={{
                width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                backgroundColor: isActive ? nodeColor : isPast ? `${nodeColor}55` : '#1c1c1c',
                boxShadow: isActive ? `0 0 6px ${nodeColor}` : 'none',
                transition: 'background-color 0.22s, box-shadow 0.22s',
              }} />

              {/* Label + shape */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <span style={{
                  fontSize: 11,
                  fontWeight: isActive ? 700 : 400,
                  color: isActive ? '#ffffff' : isPast ? `${nodeColor}99` : '#3a3a3a',
                  transition: 'color 0.22s',
                  display: 'block',
                }}>
                  {node.label}
                </span>
                {shapeAnnotations[node.id] && (
                  <span style={{
                    fontSize: 8, fontFamily: 'monospace',
                    color: isActive ? `${nodeColor}cc` : '#2a2a2a',
                    transition: 'color 0.22s',
                    letterSpacing: 0.2,
                  }}>
                    {shapeAnnotations[node.id]}
                  </span>
                )}
              </div>

              {/* Activity indicator */}
              {isActive && (
                <span style={{
                  fontSize: 8, color: nodeColor, fontWeight: 700, letterSpacing: 2,
                  animation: 'nodePulse 0.55s ease-in-out infinite',
                  flexShrink: 0,
                }}>
                  ●●●
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Output: loss indicator after last node */}
      <div style={{
        paddingTop: totalH - NODE_H / 2 - 8,
        display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0,
        opacity: simRunning && animStep === nodes.length - 1 ? 1 : 0,
        transition: 'opacity 0.3s',
      }}>
        <div style={{ width: 0, height: 0, borderTop: '5px solid transparent', borderBottom: '5px solid transparent', borderLeft: `7px solid ${accent}`, marginRight: 4 }} />
        <div style={{
          padding: '5px 11px',
          backgroundColor: `${accent}11`,
          border: `1px solid ${accent}`,
          borderRadius: 6,
          fontSize: 10, color: accent, fontWeight: 700, letterSpacing: 0.3,
          boxShadow: `0 0 10px ${accent}33`,
        }}>
          Loss ✓
        </div>
      </div>
    </div>
  );
}
