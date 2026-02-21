import { useState, useEffect, useRef, useCallback } from 'react';
import { McpUseProvider, useWidget, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';

const modelNodeSchema = z.object({
  id:    z.string(),
  label: z.string(),
  cat:   z.string(),
});

const propsSchema = z.object({
  modelNodes: z.array(modelNodeSchema).optional(),
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

const DATASETS: Record<'llm' | 'vlm' | 'rlhf', DatasetDef[]> = {
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
};

// ── Demo model graphs per training type ──────────────────────────────────────

interface DemoNode { id: string; label: string; cat: 'core' | 'activation' | 'composite'; }

const DEMO_GRAPHS: Record<'llm' | 'vlm' | 'rlhf', DemoNode[]> = {
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
};

const NODE_COLORS: Record<string, string> = {
  core:       '#2563eb',
  activation: '#7c3aed',
  composite:  '#ca8a04',
};

const ACCENT:      Record<'llm' | 'vlm' | 'rlhf', string> = { llm: '#3b82f6', vlm: '#8b5cf6', rlhf: '#10b981' };
const BLOB_ICON:   Record<'llm' | 'vlm' | 'rlhf', string> = { llm: '📝', vlm: '🖼️', rlhf: '🎯' };
const BLOB_LABEL:  Record<'llm' | 'vlm' | 'rlhf', string> = { llm: 'Text Tokens', vlm: 'Image Patches', rlhf: 'Pref. Pairs' };
const CAT_TABS: Array<{ key: 'llm' | 'vlm' | 'rlhf'; label: string }> = [
  { key: 'llm',  label: 'LLM' },
  { key: 'vlm',  label: 'VLM' },
  { key: 'rlhf', label: 'RL / RLHF' },
];

const NODE_H = 38;
const EDGE_H = 22;

// ── Main widget ───────────────────────────────────────────────────────────────

export default function DatasetPrep() {
  const { props, isPending } = useWidget<Props>();

  // Dataset selection
  const [inputTab,  setInputTab]  = useState<'curated' | 'custom'>('curated');
  const [category,  setCategory]  = useState<'llm' | 'vlm' | 'rlhf'>('llm');
  const [selected,  setSelected]  = useState<DatasetDef | null>(null);
  const [customUrl, setCustomUrl] = useState('');
  const [platform,  setPlatform]  = useState<'huggingface' | 'kaggle' | null>(null);
  const [hfToken,   setHfToken]   = useState('');
  const [kaggleKey, setKaggleKey] = useState('');

  // Animation
  const [simRunning, setSimRunning] = useState(false);
  const [animStep,   setAnimStep]   = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Use real model nodes from the graph if provided, otherwise fall back to demo
  const customNodes = !isPending && props.modelNodes && props.modelNodes.length > 0
    ? (props.modelNodes as DemoNode[])
    : null;
  const demoNodes   = customNodes ?? DEMO_GRAPHS[category];
  const hasCustomGraph = !!customNodes;
  const accentColor = ACCENT[category];

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
    setSimRunning(false);
    setAnimStep(-1);
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const canPrepare = inputTab === 'curated' ? !!selected : !!customUrl.trim();

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

            {/* Prepare button */}
            <div style={{ padding: '10px 12px', borderTop: '1px solid #141414', flexShrink: 0 }}>
              <button
                disabled={!canPrepare}
                style={{
                  width: '100%', padding: '9px 0',
                  backgroundColor: canPrepare ? '#1a1200' : '#0d0d0d',
                  border: `1px solid ${canPrepare ? '#ca8a04' : '#1a1a1a'}`,
                  borderRadius: 6,
                  color: canPrepare ? '#fde68a' : '#2a2a2a',
                  cursor: canPrepare ? 'pointer' : 'not-allowed',
                  fontSize: 11, fontWeight: 700, letterSpacing: 0.4,
                  transition: 'border-color 0.15s, color 0.15s',
                }}
              >
                ▶ Prepare Dataset
              </button>
            </div>
          </div>

          {/* RIGHT: Model simulation ──────────────────────────────────────── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Sim controls */}
            <div style={{ height: 38, borderBottom: '1px solid #141414', display: 'flex', alignItems: 'center', padding: '0 14px', gap: 10, backgroundColor: '#080808', flexShrink: 0 }}>
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
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflowY: 'auto', padding: '20px 24px' }}>
              <SimulationView
                nodes={demoNodes}
                animStep={animStep}
                simRunning={simRunning}
                category={category}
              />
            </div>
          </div>
        </div>
      </div>
    </McpUseProvider>
  );
}

// ── Simulation view ───────────────────────────────────────────────────────────

function SimulationView({
  nodes, animStep, simRunning, category,
}: {
  nodes: DemoNode[];
  animStep: number;
  simRunning: boolean;
  category: 'llm' | 'vlm' | 'rlhf';
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

              {/* Label */}
              <span style={{
                fontSize: 11,
                fontWeight: isActive ? 700 : 400,
                color: isActive ? '#ffffff' : isPast ? `${nodeColor}99` : '#3a3a3a',
                transition: 'color 0.22s',
                flex: 1,
              }}>
                {node.label}
              </span>

              {/* Activity indicator */}
              {isActive && (
                <span style={{
                  fontSize: 8, color: nodeColor, fontWeight: 700, letterSpacing: 2,
                  animation: 'nodePulse 0.55s ease-in-out infinite',
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
