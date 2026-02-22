import { useState, useCallback } from 'react';
import { McpUseProvider, useWidget, useCallTool, type WidgetMetadata } from 'mcp-use/react';
import { z } from 'zod';

type StepStatus = 'idle' | 'running' | 'success' | 'error';

interface StepState {
  status: StepStatus;
  summary?: string;
  detail?: string;
}

const propsSchema = z.object({
  hasModelPy:       z.boolean().optional(),
  hasDataPy:        z.boolean().optional(),
  hasTrainPy:       z.boolean().optional(),
  hasInferencePy:   z.boolean().optional(),
  hasRequirements:  z.boolean().optional(),
  wandbProject:     z.string().optional(),
  taskType:         z.string().optional(),
  dataset:          z.string().optional(),
  optimizer:        z.string().optional(),
  loss:             z.string().optional(),
});
type Props = z.infer<typeof propsSchema>;

export const widgetMetadata: WidgetMetadata = {
  description: 'Deploy training scripts to a GPU pod — check packages, install dependencies, upload files, run inference',
  props: propsSchema,
  exposeAsTool: false,
  metadata: {
    invoking: 'Preparing deployment…',
    invoked: 'Ready to deploy',
  },
};

// ── Helpers ────────────────────────────────────────────────────────────────

const stepColor = (s: StepStatus) =>
  s === 'success' ? '#4ade80' : s === 'error' ? '#f87171' : s === 'running' ? '#60a5fa' : '#333';

const stepBg = (s: StepStatus) =>
  s === 'success' ? '#071a07' : s === 'error' ? '#1a0707' : s === 'running' ? '#070f1a' : '#0d0d0d';

const stepBorder = (s: StepStatus) =>
  s === 'success' ? '#166534' : s === 'error' ? '#7f1d1d' : s === 'running' ? '#1e40af' : '#1a1a1a';

const stepIcon = (s: StepStatus) =>
  s === 'success' ? '✓' : s === 'error' ? '✗' : s === 'running' ? '◌' : '○';

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  backgroundColor: '#0f0f0f',
  border: '1px solid #1e1e1e',
  borderRadius: 4,
  color: '#d0d0d0',
  fontSize: 11,
  outline: 'none',
  fontFamily: 'inherit',
  boxSizing: 'border-box',
};

// ── Widget ─────────────────────────────────────────────────────────────────

export default function TrainingDeploy() {
  const { props, isPending } = useWidget<Props>();

  const { callToolAsync: callCheck,    isPending: isChecking    } = useCallTool('check-gpu-packages');
  const { callToolAsync: callSetup,    isPending: isSettingUp   } = useCallTool('setup-gpu');
  const { callToolAsync: callUpload,   isPending: isUploading   } = useCallTool('upload-scripts');
  const { callToolAsync: callGenInfer, isPending: isGenInfer    } = useCallTool('generate-inference-code');
  const { callToolAsync: callRunInfer, isPending: isRunInfer    } = useCallTool('run-inference');

  const [podId,    setPodId]    = useState('');
  const [keyFile,  setKeyFile]  = useState('');
  const [remotePath, setRemotePath] = useState('/workspace');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [steps, setSteps] = useState<Record<string, StepState>>({
    check:   { status: 'idle' },
    install: { status: 'idle' },
    upload:  { status: 'idle' },
  });

  const [inferScript, setInferScript] = useState<string | null>(null);
  const [inferInput,  setInferInput]  = useState('');
  const [inferType,   setInferType]   = useState<'text' | 'image_url' | 'tabular'>('text');
  const [inferResult, setInferResult] = useState<{ result?: Record<string, unknown>; rawOutput?: string; error?: string } | null>(null);
  const [inferCopied, setInferCopied] = useState(false);
  const [deployingAll, setDeployingAll] = useState(false);

  const setStep = useCallback((key: string, s: Partial<StepState>) =>
    setSteps(prev => ({ ...prev, [key]: { ...prev[key], ...s } })), []);

  // ── Step handlers ─────────────────────────────────────────────────────────

  const handleCheck = useCallback(async () => {
    if (!podId) return;
    setStep('check', { status: 'running', detail: undefined });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callCheck({ podId, keyFile: keyFile || undefined } as any);
      const sc = res?.structuredContent as {
        packageCount?: number; hasTorch?: boolean; hasWandb?: boolean;
        hasTransformers?: boolean; hasDatasets?: boolean;
      } | undefined;
      const flags = [
        sc?.hasTorch        && 'torch',
        sc?.hasWandb        && 'wandb',
        sc?.hasTransformers && 'transformers',
        sc?.hasDatasets     && 'datasets',
      ].filter(Boolean).join(', ');
      setStep('check', {
        status: 'success',
        summary: `${sc?.packageCount ?? '?'} packages installed`,
        detail: flags ? `Found: ${flags}` : 'Standard packages checked',
      });
    } catch (err) {
      setStep('check', { status: 'error', summary: 'Check failed', detail: String(err) });
    }
  }, [podId, keyFile, callCheck, setStep]);

  const handleInstall = useCallback(async () => {
    if (!podId) return;
    setStep('install', { status: 'running', detail: undefined });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callSetup({ podId, keyFile: keyFile || undefined } as any);
      const sc = res?.structuredContent as {
        installed?: string[]; skipped?: string[]; summary?: string;
      } | undefined;
      setStep('install', {
        status: 'success',
        summary: sc?.summary ?? 'Done',
        detail: sc?.installed?.length ? `Installed: ${sc.installed.slice(0, 4).join(', ')}${sc.installed.length > 4 ? `…+${sc.installed.length - 4}` : ''}` : undefined,
      });
    } catch (err) {
      setStep('install', { status: 'error', summary: 'Install failed', detail: String(err) });
    }
  }, [podId, keyFile, callSetup, setStep]);

  const handleUpload = useCallback(async () => {
    if (!podId) return;
    setStep('upload', { status: 'running', detail: undefined });
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callUpload({ podId, remotePath, keyFile: keyFile || undefined } as any);
      const sc = res?.structuredContent as {
        uploaded?: string[]; errors?: string[]; summary?: string;
      } | undefined;
      const hasErrors = (sc?.errors?.length ?? 0) > 0;
      setStep('upload', {
        status: hasErrors ? 'error' : 'success',
        summary: sc?.summary ?? 'Uploaded',
        detail: sc?.uploaded?.length ? `Uploaded: ${sc.uploaded.join(', ')}` : undefined,
      });
    } catch (err) {
      setStep('upload', { status: 'error', summary: 'Upload failed', detail: String(err) });
    }
  }, [podId, keyFile, remotePath, callUpload, setStep]);

  const handleDeployAll = useCallback(async () => {
    if (!podId) return;
    setDeployingAll(true);
    try {
      // Run check → install → upload sequentially
      setStep('check',   { status: 'running', detail: undefined });
      setStep('install', { status: 'idle',    detail: undefined });
      setStep('upload',  { status: 'idle',    detail: undefined });

      // Step 1: check
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const r = await callCheck({ podId, keyFile: keyFile || undefined } as any);
        const sc = r?.structuredContent as { packageCount?: number; hasTorch?: boolean } | undefined;
        setStep('check', { status: 'success', summary: `${sc?.packageCount ?? '?'} packages`, detail: sc?.hasTorch ? 'torch present' : undefined });
      } catch (err) {
        setStep('check', { status: 'error', summary: 'Check failed', detail: String(err) });
        return; // stop if we can't even connect
      }

      // Step 2: install
      setStep('install', { status: 'running' });
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const r = await callSetup({ podId, keyFile: keyFile || undefined } as any);
        const sc = r?.structuredContent as { summary?: string } | undefined;
        setStep('install', { status: 'success', summary: sc?.summary ?? 'Done' });
      } catch (err) {
        setStep('install', { status: 'error', summary: 'Install failed', detail: String(err) });
        // continue to upload even if install has issues
      }

      // Step 3: upload
      setStep('upload', { status: 'running' });
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const r = await callUpload({ podId, remotePath, keyFile: keyFile || undefined } as any);
        const sc = r?.structuredContent as { uploaded?: string[]; summary?: string } | undefined;
        setStep('upload', { status: 'success', summary: sc?.summary ?? 'Uploaded', detail: sc?.uploaded?.join(', ') });
      } catch (err) {
        setStep('upload', { status: 'error', summary: 'Upload failed', detail: String(err) });
      }
    } finally {
      setDeployingAll(false);
    }
  }, [podId, keyFile, remotePath, callCheck, callSetup, callUpload, setStep]);

  const handleGenInfer = useCallback(async () => {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callGenInfer({} as any);
      const sc = res?.structuredContent as { inferencePy?: string; taskType?: string; inputFormat?: string } | undefined;
      if (sc?.inferencePy) {
        setInferScript(sc.inferencePy);
        if (sc.taskType === 'vision') setInferType('image_url');
        else if (sc.taskType === 'tabular') setInferType('tabular');
        else setInferType('text');
      }
    } catch (err) {
      console.error('[generate-inference-code]', err);
    }
  }, [callGenInfer]);

  const handleRunInfer = useCallback(async () => {
    if (!podId || !inferInput) return;
    setInferResult(null);
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const res = await callRunInfer({ podId, input: inferInput, inputType: inferType, keyFile: keyFile || undefined } as any);
      const sc = res?.structuredContent as { result?: Record<string, unknown>; rawOutput?: string; error?: string } | undefined;
      setInferResult(sc ?? { error: 'No response' });
    } catch (err) {
      setInferResult({ error: String(err) });
    }
  }, [podId, inferInput, inferType, keyFile, callRunInfer]);

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div style={{ padding: 20, backgroundColor: '#0a0a0a', color: '#333', fontFamily: 'system-ui, sans-serif', fontSize: 12 }}>
          Loading…
        </div>
      </McpUseProvider>
    );
  }

  const {
    hasModelPy, hasDataPy, hasTrainPy, hasRequirements,
    wandbProject, taskType, dataset, optimizer, loss,
  } = props;

  const canAct = !!podId;
  const allSuccess = steps.check.status === 'success' && steps.install.status === 'success' && steps.upload.status === 'success';
  const isAnyRunning = isChecking || isSettingUp || isUploading || deployingAll;

  const ACCENT = '#60a5fa'; // blue accent for deploy

  return (
    <McpUseProvider autoSize>
      <div style={{
        backgroundColor: '#0a0a0a',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: 12,
        color: '#e0e0e0',
      }}>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div style={{
          padding: '12px 18px 10px',
          borderBottom: '1px solid #141414',
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: '#aaa', letterSpacing: 0.3 }}>
            Deploy Training Scripts
          </span>
          <div style={{ flex: 1 }} />
          {dataset && (
            <span style={{ fontSize: 9, color: '#555', padding: '1px 6px', border: '1px solid #1a1a1a', borderRadius: 3 }}>
              {dataset}
            </span>
          )}
          {optimizer && (
            <span style={{ fontSize: 9, color: '#555', padding: '1px 6px', border: '1px solid #1a1a1a', borderRadius: 3 }}>
              {optimizer}
            </span>
          )}
          {loss && (
            <span style={{ fontSize: 9, color: '#555', padding: '1px 6px', border: '1px solid #1a1a1a', borderRadius: 3 }}>
              {loss}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 0 }}>

          {/* ── Left: Pod config ─────────────────────────────────────────── */}
          <div style={{
            width: 200, minWidth: 200,
            borderRight: '1px solid #141414',
            padding: '12px 14px',
            display: 'flex', flexDirection: 'column', gap: 10,
          }}>

            {/* Generated scripts list */}
            <div>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.2, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 6 }}>
                Generated
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {[
                  { name: 'model.py',        present: hasModelPy },
                  { name: 'data.py',         present: hasDataPy  },
                  { name: 'train.py',        present: hasTrainPy },
                  { name: 'requirements.txt',present: hasRequirements },
                ].map(f => (
                  <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 9, color: f.present ? '#4ade80' : '#333', fontWeight: 700 }}>
                      {f.present ? '✓' : '○'}
                    </span>
                    <span style={{ fontSize: 10, color: f.present ? '#888' : '#2a2a2a', fontFamily: 'monospace' }}>
                      {f.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* W&B badge */}
            {wandbProject && (
              <div style={{
                padding: '6px 8px', borderRadius: 4,
                backgroundColor: '#0a1020', border: '1px solid #1e3a5f',
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{ fontSize: 8, color: '#93c5fd', fontWeight: 700, letterSpacing: 0.5 }}>W&B</span>
                <span style={{ fontSize: 9, color: '#60a5fa', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {wandbProject}
                </span>
              </div>
            )}

            {/* Divider */}
            <div style={{ borderTop: '1px solid #141414' }} />

            {/* Pod ID */}
            <div>
              <label style={{ fontSize: 9, color: '#444', fontWeight: 700, display: 'block', marginBottom: 4, letterSpacing: 0.3 }}>
                Pod ID
              </label>
              <input
                type="text"
                value={podId}
                onChange={e => setPodId(e.target.value)}
                placeholder="abc123…"
                style={{ ...inputStyle, borderColor: podId ? '#2a2a2a' : '#1e1e1e' }}
              />
            </div>

            {/* Advanced toggle */}
            <button
              onClick={() => setShowAdvanced(v => !v)}
              style={{ background: 'none', border: 'none', color: '#2a2a2a', cursor: 'pointer', fontSize: 9, padding: 0, textAlign: 'left', letterSpacing: 0.3 }}
            >
              {showAdvanced ? '▼' : '▶'} Advanced
            </button>

            {showAdvanced && (
              <>
                <div>
                  <label style={{ fontSize: 9, color: '#444', fontWeight: 700, display: 'block', marginBottom: 4 }}>
                    Remote Path
                  </label>
                  <input
                    type="text"
                    value={remotePath}
                    onChange={e => setRemotePath(e.target.value)}
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 9, color: '#444', fontWeight: 700, display: 'block', marginBottom: 4 }}>
                    SSH Key File
                  </label>
                  <input
                    type="text"
                    value={keyFile}
                    onChange={e => setKeyFile(e.target.value)}
                    placeholder="~/.ssh/id_ed25519"
                    style={inputStyle}
                  />
                </div>
              </>
            )}

            {/* Deploy all */}
            <button
              onClick={handleDeployAll}
              disabled={!canAct || isAnyRunning}
              style={{
                padding: '8px 0',
                backgroundColor: canAct && !isAnyRunning ? '#0a1a2a' : '#0d0d0d',
                border: `1px solid ${canAct && !isAnyRunning ? '#1e40af' : '#1a1a1a'}`,
                borderRadius: 5,
                color: canAct && !isAnyRunning ? ACCENT : '#2a2a2a',
                cursor: canAct && !isAnyRunning ? 'pointer' : 'not-allowed',
                fontSize: 11, fontWeight: 700, letterSpacing: 0.3,
                marginTop: 4,
              }}
            >
              {deployingAll ? 'Deploying…' : allSuccess ? '✓ Deployed' : '🚀 Deploy All'}
            </button>

            {!podId && (
              <div style={{ fontSize: 9, color: '#2a2a2a', lineHeight: 1.5 }}>
                Enter your RunPod pod ID to deploy.
              </div>
            )}
          </div>

          {/* ── Right: Steps + Inference ─────────────────────────────────── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

            {/* Steps */}
            <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8, borderBottom: '1px solid #111' }}>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.2, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 2 }}>
                Deployment Steps
              </div>

              {[
                { key: 'check',   label: 'Check GPU Packages',      action: handleCheck,   loading: isChecking  },
                { key: 'install', label: 'Install Dependencies',    action: handleInstall, loading: isSettingUp },
                { key: 'upload',  label: 'Upload Scripts',          action: handleUpload,  loading: isUploading },
              ].map(({ key, label, action, loading }) => {
                const s = steps[key];
                return (
                  <div
                    key={key}
                    style={{
                      display: 'flex', alignItems: 'flex-start', gap: 10,
                      padding: '8px 10px',
                      borderRadius: 6,
                      backgroundColor: stepBg(s.status),
                      border: `1px solid ${stepBorder(s.status)}`,
                      transition: 'all 0.2s',
                    }}
                  >
                    <span style={{ fontSize: 13, color: stepColor(s.status), lineHeight: 1, flexShrink: 0, marginTop: 1 }}>
                      {loading ? '…' : stepIcon(s.status)}
                    </span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: s.status === 'idle' ? '#555' : stepColor(s.status) }}>
                        {label}
                      </div>
                      {(s.summary || s.detail) && (
                        <div style={{ fontSize: 9, color: '#555', marginTop: 2, lineHeight: 1.4 }}>
                          {s.summary}{s.detail && ` — ${s.detail}`}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={action}
                      disabled={!canAct || loading || isAnyRunning}
                      style={{
                        padding: '3px 10px',
                        backgroundColor: 'transparent',
                        border: `1px solid ${canAct && !isAnyRunning ? stepBorder(s.status) || '#252525' : '#1a1a1a'}`,
                        borderRadius: 4,
                        color: canAct && !isAnyRunning ? stepColor(s.status) || '#444' : '#2a2a2a',
                        cursor: canAct && !isAnyRunning ? 'pointer' : 'not-allowed',
                        fontSize: 9, fontWeight: 700, whiteSpace: 'nowrap' as const,
                        flexShrink: 0,
                      }}
                    >
                      {loading ? 'Running…' : s.status === 'success' ? 'Re-run' : 'Run'}
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Training command hint */}
            {steps.upload.status === 'success' && (
              <div style={{
                margin: '10px 16px 0',
                padding: '8px 10px',
                borderRadius: 5,
                backgroundColor: '#071a07',
                border: '1px solid #166534',
              }}>
                <div style={{ fontSize: 9, color: '#4ade80', fontWeight: 700, marginBottom: 4 }}>
                  Scripts uploaded — start training:
                </div>
                <pre style={{ margin: 0, fontSize: 9.5, color: '#a8e6a3', fontFamily: 'monospace' }}>
                  {`cd ${remotePath} && python3 train.py`}
                </pre>
                {wandbProject && (
                  <div style={{ fontSize: 8, color: '#2a5f2a', marginTop: 4 }}>
                    W&B run URL will be printed to stdout on start
                  </div>
                )}
              </div>
            )}

            {/* ── Inference section ──────────────────────────────────────── */}
            <div style={{ padding: '12px 16px', marginTop: 4 }}>
              <div style={{ fontSize: 9, color: '#3a3a3a', letterSpacing: 1.2, textTransform: 'uppercase' as const, fontWeight: 700, marginBottom: 8 }}>
                Inference
              </div>

              {!inferScript ? (
                <button
                  onClick={handleGenInfer}
                  disabled={isGenInfer}
                  style={{
                    padding: '6px 14px',
                    backgroundColor: '#120a1e',
                    border: '1px solid #6b21a8',
                    borderRadius: 5,
                    color: isGenInfer ? '#6b21a8' : '#c084fc',
                    cursor: isGenInfer ? 'not-allowed' : 'pointer',
                    fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                  }}
                >
                  {isGenInfer ? 'Generating…' : `⚡ Generate inference.py${taskType ? ` (${taskType})` : ''}`}
                </button>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {/* Script preview */}
                  <div style={{
                    borderRadius: 4, overflow: 'hidden',
                    border: '1px solid #1a0a2a',
                    position: 'relative',
                  }}>
                    <div style={{
                      padding: '4px 8px', backgroundColor: '#120a1e',
                      display: 'flex', alignItems: 'center', gap: 6,
                    }}>
                      <span style={{ fontSize: 9, color: '#c084fc', fontWeight: 700, flex: 1 }}>inference.py</span>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(inferScript).then(() => {
                            setInferCopied(true);
                            setTimeout(() => setInferCopied(false), 1800);
                          });
                        }}
                        style={{
                          padding: '1px 6px',
                          backgroundColor: inferCopied ? '#0a2a0a' : '#1a1a1a',
                          border: `1px solid ${inferCopied ? '#166534' : '#252525'}`,
                          borderRadius: 3,
                          color: inferCopied ? '#4ade80' : '#555',
                          cursor: 'pointer', fontSize: 8, fontWeight: 700,
                        }}
                      >
                        {inferCopied ? '✓' : 'copy'}
                      </button>
                    </div>
                    <pre style={{
                      maxHeight: 80, overflowY: 'auto', margin: 0,
                      padding: '6px 8px', fontSize: 9,
                      color: '#a8e6a3', fontFamily: 'monospace',
                      backgroundColor: '#080808', whiteSpace: 'pre',
                    }}>
                      {inferScript}
                    </pre>
                  </div>

                  {/* Input type + field */}
                  <div style={{ display: 'flex', gap: 6 }}>
                    <select
                      value={inferType}
                      onChange={e => setInferType(e.target.value as typeof inferType)}
                      style={{
                        padding: '4px 6px', backgroundColor: '#0f0f0f',
                        border: '1px solid #1e1e1e', borderRadius: 4,
                        color: '#888', fontSize: 9, outline: 'none',
                        fontFamily: 'inherit', flexShrink: 0,
                      }}
                    >
                      <option value="text">Text</option>
                      <option value="image_url">Image URL</option>
                      <option value="tabular">Tabular</option>
                    </select>
                    <input
                      type="text"
                      value={inferInput}
                      onChange={e => setInferInput(e.target.value)}
                      placeholder={
                        inferType === 'text'      ? 'Enter prompt…'
                        : inferType === 'image_url' ? 'https://…'
                        : 'f1,f2,f3…'
                      }
                      style={{ ...inputStyle, fontSize: 10 }}
                    />
                    <button
                      onClick={handleRunInfer}
                      disabled={!canAct || !inferInput || isRunInfer}
                      style={{
                        padding: '4px 10px',
                        backgroundColor: canAct && inferInput ? '#071a07' : '#0d0d0d',
                        border: `1px solid ${canAct && inferInput ? '#166534' : '#1a1a1a'}`,
                        borderRadius: 4,
                        color: canAct && inferInput ? '#4ade80' : '#2a2a2a',
                        cursor: canAct && inferInput ? 'pointer' : 'not-allowed',
                        fontSize: 9, fontWeight: 700, whiteSpace: 'nowrap' as const, flexShrink: 0,
                      }}
                    >
                      {isRunInfer ? '…' : '▶ Run'}
                    </button>
                  </div>

                  {/* Result */}
                  {inferResult && (
                    <div style={{
                      padding: '7px 9px', borderRadius: 4,
                      backgroundColor: inferResult.error ? '#1a0707' : '#071a07',
                      border: `1px solid ${inferResult.error ? '#7f1d1d' : '#166534'}`,
                    }}>
                      {inferResult.error ? (
                        <div style={{ fontSize: 9, color: '#f87171', fontFamily: 'monospace', lineHeight: 1.4 }}>
                          {inferResult.error}
                        </div>
                      ) : inferResult.result ? (
                        Object.entries(inferResult.result).map(([k, v]) => (
                          <div key={k} style={{ marginBottom: 4 }}>
                            <span style={{ fontSize: 8, color: '#2a5f2a', fontWeight: 700, letterSpacing: 0.3 }}>{k}: </span>
                            <span style={{ fontSize: 9, color: '#a8e6a3', fontFamily: 'monospace' }}>
                              {Array.isArray(v)
                                ? `[${(v as number[]).map(n => typeof n === 'number' ? n.toFixed(3) : String(n)).join(', ')}]`
                                : String(v)}
                            </span>
                          </div>
                        ))
                      ) : (
                        <pre style={{ margin: 0, fontSize: 9, color: '#a8e6a3', fontFamily: 'monospace' }}>
                          {inferResult.rawOutput}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </McpUseProvider>
  );
}
