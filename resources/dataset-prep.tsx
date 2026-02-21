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

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '4px 8px', backgroundColor: '#0c0c0c',
    border: '1px solid #1e1e1e', borderRadius: 4, color: '#c8c8c8',
    fontSize: 11, outline: 'none', boxSizing: 'border-box', fontFamily: 'monospace',
  };
  const selectStyle: React.CSSProperties = {
    ...inputStyle, cursor: 'pointer', appearance: 'none',
  };
  const rowLabel: React.CSSProperties = {
    fontSize: 9, color: '#3a3a3a', letterSpacing: 0.3, marginBottom: 3,
  };

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{ height: 560, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0a0a0a', color: '#2a2a2a', fontFamily: 'monospace', fontSize: 11 }}>
          Loading…
        </div>
      </McpUseProvider>
    );
  }

  return (
    <McpUseProvider autoSize>
      <style>{`
        @keyframes nodePulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
        @keyframes edgeDash  { from{stroke-dashoffset:12} to{stroke-dashoffset:0} }
        select option { background:#111; }
      `}</style>
      <div style={{
        height: 560, display: 'flex', flexDirection: 'column',
        backgroundColor: '#0a0a0a', color: '#c8c8c8',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 12, overflow: 'hidden',
      }}>

        {/* ── Slim header ─────────────────────────────────────────────────── */}
        <div style={{ height: 36, borderBottom: '1px solid #111', display: 'flex', alignItems: 'center', padding: '0 14px', gap: 8, flexShrink: 0 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: '#c8c8c8' }}>Prepare Training</span>
          <div style={{ width: 1, height: 12, backgroundColor: '#1e1e1e' }} />
          {taskType && <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 3, backgroundColor: '#0d1b2a', color: '#60a5fa', border: '1px solid #1d4ed850', fontWeight: 600 }}>{taskType}</span>}
          {hasCustomGraph && <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 3, backgroundColor: '#0a1a0a', color: '#4ade80', border: '1px solid #16653450', fontWeight: 600 }}>Model · {demoNodes.length} layers</span>}
          <div style={{ flex: 1 }} />
          {simRunning
            ? <button onClick={stopSim} style={{ padding: '3px 10px', background: 'none', border: '1px solid #3d1515', borderRadius: 4, color: '#f87171', cursor: 'pointer', fontSize: 10 }}>■ Stop</button>
            : <button onClick={startSim} style={{ padding: '3px 10px', background: 'none', border: '1px solid #166534', borderRadius: 4, color: '#4ade80', cursor: 'pointer', fontSize: 10 }}>▶ Run</button>
          }
        </div>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>

          {/* LEFT: controls ─────────────────────────────────────────────── */}
          <div style={{ width: 240, borderRight: '1px solid #111', display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden' }}>

            {/* Mode tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid #111', flexShrink: 0 }}>
              {(['curated', 'custom'] as const).map(t => (
                <button key={t} onClick={() => setInputTab(t)} style={{
                  flex: 1, padding: '7px 0', background: 'none', border: 'none',
                  borderBottom: `2px solid ${inputTab === t ? accentColor : 'transparent'}`,
                  color: inputTab === t ? accentColor : '#2e2e2e',
                  cursor: 'pointer', fontSize: 10, fontWeight: 600,
                }}>
                  {t === 'curated' ? 'Curated' : 'Custom URL'}
                </button>
              ))}
            </div>

            {inputTab === 'curated' ? (
              <>
                {/* Category tabs — compact pill row */}
                <div style={{ display: 'flex', gap: 4, padding: '6px 8px', borderBottom: '1px solid #111', flexShrink: 0, flexWrap: 'wrap' as const }}>
                  {CAT_TABS.map(({ key, label }) => (
                    <button key={key} onClick={() => { setCategory(key); setSelected(null); }} style={{
                      padding: '3px 8px', background: 'none', borderRadius: 3,
                      border: `1px solid ${category === key ? accentColor : '#1e1e1e'}`,
                      color: category === key ? accentColor : '#2a2a2a',
                      cursor: 'pointer', fontSize: 9, fontWeight: 700,
                    }}>{label}</button>
                  ))}
                </div>

                {/* Dataset list — compact */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '6px' }}>
                  {DATASETS[category].map(ds => {
                    const isSel = selected?.id === ds.id;
                    return (
                      <div key={ds.id} onClick={() => setSelected(ds)} style={{
                        padding: '7px 9px', borderRadius: 5, marginBottom: 4, cursor: 'pointer',
                        border: `1px solid ${isSel ? accentColor : '#141414'}`,
                        backgroundColor: isSel ? `${accentColor}0c` : 'transparent',
                        transition: 'border-color 0.12s',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 2 }}>
                          <span style={{ fontSize: 11, fontWeight: 600, color: isSel ? accentColor : '#909090' }}>{ds.name}</span>
                          <span style={{ fontSize: 8, padding: '1px 4px', borderRadius: 2, backgroundColor: '#0a1a0a', color: '#4ade80', fontWeight: 700 }}>{ds.task}</span>
                          <span style={{ fontSize: 8, color: '#252525', marginLeft: 'auto' as const }}>{ds.size}</span>
                        </div>
                        <div style={{ fontSize: 9, color: '#2e2e2e', lineHeight: 1.3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const }}>{ds.description}</div>
                        <div style={{ fontSize: 8, color: '#3a2010', fontFamily: 'monospace', marginTop: 2 }}>{ds.hfId}</div>
                      </div>
                    );
                  })}
                </div>

                {selected && (
                  <div style={{ padding: '6px 8px', borderTop: '1px solid #111', flexShrink: 0 }}>
                    <div style={rowLabel}>HuggingFace Token (optional)</div>
                    <input type="password" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder="hf_…" style={inputStyle} />
                  </div>
                )}
              </>
            ) : (
              <div style={{ flex: 1, padding: '10px 8px', display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
                <div>
                  <div style={rowLabel}>Dataset URL</div>
                  <input type="text" value={customUrl} onChange={e => setCustomUrl(e.target.value)}
                    placeholder="huggingface.co/datasets/… or kaggle.com/datasets/…"
                    style={{ ...inputStyle, border: `1px solid ${platform ? accentColor : '#1e1e1e'}` }} />
                </div>
                {platform && <span style={{ fontSize: 9, color: platform === 'huggingface' ? '#60a5fa' : '#fbbf24', fontWeight: 600 }}>
                  {platform === 'huggingface' ? '⬡ HuggingFace detected' : '◆ Kaggle detected'}
                </span>}
                {platform === 'huggingface' && <div><div style={rowLabel}>HF Token</div><input type="password" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder="hf_…" style={inputStyle} /></div>}
                {platform === 'kaggle' && <div><div style={rowLabel}>Kaggle API Key</div><input type="password" value={kaggleKey} onChange={e => setKaggleKey(e.target.value)} placeholder="kaggle_key" style={inputStyle} /></div>}
              </div>
            )}

            {/* Training config + hyperparams — compact grid */}
            <div style={{ padding: '8px', borderTop: '1px solid #111', flexShrink: 0 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5, marginBottom: 5 }}>
                <div><div style={rowLabel}>Loss</div>
                  <select value={loss} onChange={e => setLoss(e.target.value)} style={selectStyle}>
                    {['CrossEntropyLoss','BCEWithLogitsLoss','MSELoss','L1Loss','HuberLoss','NLLLoss','KLDivLoss','CTCLoss'].map(l => <option key={l} value={l}>{l}</option>)}
                  </select>
                </div>
                <div><div style={rowLabel}>Optimizer</div>
                  <select value={optimizer} onChange={e => setOptimizer(e.target.value)} style={selectStyle}>
                    {['Adam','AdamW','SGD','RMSprop','Adadelta','Adagrad','LBFGS'].map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div><div style={rowLabel}>Learning Rate</div><input type="text" value={lr} onChange={e => setLr(e.target.value)} style={inputStyle} /></div>
                <div><div style={rowLabel}>Batch Size</div><input type="text" value={batchSz} onChange={e => setBatchSz(e.target.value)} style={inputStyle} /></div>
              </div>
              <div style={{ marginBottom: 6 }}><div style={rowLabel}>Epochs</div><input type="text" value={epochs} onChange={e => setEpochs(e.target.value)} style={inputStyle} /></div>
              <button disabled={!canGenerate} onClick={handleGenerate} style={{
                width: '100%', padding: '7px 0',
                backgroundColor: canGenerate ? `${accentColor}14` : 'transparent',
                border: `1px solid ${canGenerate ? accentColor : '#1a1a1a'}`,
                borderRadius: 4, color: canGenerate ? accentColor : '#252525',
                cursor: canGenerate ? 'pointer' : 'not-allowed', fontSize: 10, fontWeight: 700,
                transition: 'all 0.12s',
              }}>
                {isGenerating ? '⟳ Generating…' : '⚡ Generate Scripts'}
              </button>
            </div>
          </div>

          {/* RIGHT: node tree + code panel ──────────────────────────────── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

            {/* Node tree scroll area */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 28px', minHeight: 0 }}>
              <SimulationView
                nodes={demoNodes}
                animStep={animStep}
                simRunning={simRunning}
                category={category}
                shapeAnnotations={props.shapeAnnotations ?? {}}
              />
            </div>

            {/* Code panel */}
            {files && (
              <div style={{ height: 200, borderTop: '1px solid #111', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid #111', flexShrink: 0 }}>
                  {(['model', 'data', 'train'] as const).map(tab => (
                    <button key={tab} onClick={() => setCodeTab(tab)} style={{
                      padding: '6px 12px', background: 'none', border: 'none',
                      borderBottom: `2px solid ${codeTab === tab ? '#3b82f6' : 'transparent'}`,
                      color: codeTab === tab ? '#93c5fd' : '#333',
                      cursor: 'pointer', fontSize: 10, fontWeight: 600,
                    }}>
                      {tab === 'model' ? 'model.py' : tab === 'data' ? 'data.py' : 'train.py'}
                    </button>
                  ))}
                  <div style={{ flex: 1 }} />
                  <button onClick={copyCode} style={{
                    padding: '3px 10px', margin: '0 8px', background: 'none',
                    border: `1px solid ${copied ? '#166534' : '#1e1e1e'}`,
                    borderRadius: 3, color: copied ? '#4ade80' : '#333',
                    cursor: 'pointer', fontSize: 9, fontWeight: 600, transition: 'all 0.15s',
                  }}>{copied ? '✓ Copied' : 'Copy'}</button>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', padding: '8px 14px', backgroundColor: '#060606' }}>
                  <pre style={{ margin: 0, fontSize: 10, color: '#6b7280', fontFamily: 'monospace', lineHeight: 1.55, whiteSpace: 'pre-wrap' as const, wordBreak: 'break-word' as const }}>
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

// ── Simulation view — clean vertical node stack ───────────────────────────────

const CONN_H = 16; // px of connector between nodes
const N_H    = 34; // node row height

function SimulationView({
  nodes, animStep, simRunning, category, shapeAnnotations,
}: {
  nodes: DemoNode[];
  animStep: number;
  simRunning: boolean;
  category: CatKey;
  shapeAnnotations: Record<string, string>;
}) {
  const accent = ACCENT[category];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', width: '100%', maxWidth: 420, margin: '0 auto' }}>

      {/* Dataset source pill */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10,
        padding: '5px 10px', borderRadius: 6,
        backgroundColor: simRunning ? `${accent}10` : 'transparent',
        border: `1px solid ${simRunning ? `${accent}40` : '#141414'}`,
        transition: 'all 0.3s',
      }}>
        <span style={{ fontSize: 14 }}>{BLOB_ICON[category]}</span>
        <span style={{ fontSize: 10, fontWeight: 600, color: simRunning ? accent : '#2a2a2a', transition: 'color 0.3s' }}>
          {BLOB_LABEL[category]}
        </span>
        {simRunning && animStep >= 0 && (
          <span style={{ fontSize: 9, color: '#2a2a2a', marginLeft: 'auto' as const, fontFamily: 'monospace' }}>
            {animStep + 1} / {nodes.length}
          </span>
        )}
      </div>

      {/* Nodes */}
      {nodes.map((node, idx) => {
        const nodeColor = NODE_COLORS[node.cat] ?? '#2563eb';
        const isActive  = simRunning && animStep === idx;
        const isPast    = simRunning && animStep > idx;
        const shape     = shapeAnnotations[node.id];

        return (
          <div key={node.id}>
            {/* Connector above (skip first) */}
            {idx > 0 && (
              <div style={{ display: 'flex', justifyContent: 'center', height: CONN_H, alignItems: 'center' }}>
                <div style={{
                  width: 1, height: '100%',
                  backgroundColor: isPast ? `${nodeColor}50` : isActive ? nodeColor : '#181818',
                  boxShadow: isActive ? `0 0 4px ${nodeColor}` : 'none',
                  transition: 'background-color 0.2s',
                  animation: isActive ? 'edgeDash 0.4s linear infinite' : 'none',
                }} />
              </div>
            )}

            {/* Node row */}
            <div style={{
              height: N_H,
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '0 12px',
              borderRadius: 6,
              border: `1px solid ${isActive ? nodeColor : isPast ? `${nodeColor}30` : '#141414'}`,
              backgroundColor: isActive ? `${nodeColor}12` : isPast ? `${nodeColor}06` : '#0c0c0c',
              boxShadow: isActive ? `0 0 12px ${nodeColor}30` : 'none',
              transition: 'border-color 0.2s, background-color 0.2s, box-shadow 0.2s',
              animation: isActive ? 'nodePulse 0.9s ease-in-out infinite' : 'none',
            }}>
              {/* Color dot */}
              <div style={{
                width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                backgroundColor: isActive ? nodeColor : isPast ? `${nodeColor}60` : '#1e1e1e',
                boxShadow: isActive ? `0 0 5px ${nodeColor}` : 'none',
                transition: 'background-color 0.2s',
              }} />

              {/* Label */}
              <span style={{
                fontSize: 11, fontWeight: isActive ? 600 : 400, flex: 1,
                color: isActive ? '#e8e8e8' : isPast ? `${nodeColor}80` : '#383838',
                transition: 'color 0.2s',
              }}>{node.label}</span>

              {/* Shape annotation */}
              {shape && (
                <span style={{
                  fontSize: 9, fontFamily: 'monospace',
                  color: isActive ? `${nodeColor}bb` : '#1e1e1e',
                  transition: 'color 0.2s',
                }}>{shape}</span>
              )}

              {/* Active pulse dots */}
              {isActive && (
                <span style={{ fontSize: 7, color: nodeColor, letterSpacing: 2, animation: 'nodePulse 0.5s ease-in-out infinite' }}>●●●</span>
              )}
            </div>
          </div>
        );
      })}

      {/* Loss output pill */}
      <div style={{
        display: 'flex', justifyContent: 'center', marginTop: 10,
        opacity: simRunning && animStep === nodes.length - 1 ? 1 : 0,
        transition: 'opacity 0.3s',
      }}>
        <div style={{
          padding: '4px 14px', borderRadius: 5,
          backgroundColor: `${accent}12`, border: `1px solid ${accent}50`,
          fontSize: 9, color: accent, fontWeight: 700, letterSpacing: 0.5,
        }}>Loss ✓</div>
      </div>
    </div>
  );
}
