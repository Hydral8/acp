export type BlockCategory = 'core' | 'activation' | 'structural' | 'training';

export interface BlockDef {
  type: string;
  category: BlockCategory;
  label: string;
  defaultParams: Record<string, unknown>;
  hasInput: boolean;
  hasOutput: boolean;
}

export interface GraphNode {
  id: string;
  type: string;
  x: number;
  y: number;
  parameters: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  sourceId: string;
  targetId: string;
}

export interface PendingConn {
  sourceId: string;
  mouseX: number;
  mouseY: number;
}

export const NODE_WIDTH = 132;
export const NODE_HEIGHT = 58;

export const BLOCK_DEFS: BlockDef[] = [
  // Core Layers
  { type: 'Input',       category: 'core', label: 'Input',       defaultParams: { shape: [1, 28, 28] },                                                              hasInput: false, hasOutput: true  },
  { type: 'Linear',      category: 'core', label: 'Linear',      defaultParams: { in_features: 128, out_features: 64, bias: true },                                   hasInput: true,  hasOutput: true  },
  { type: 'Conv2D',      category: 'core', label: 'Conv2D',      defaultParams: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 0 },          hasInput: true,  hasOutput: true  },
  { type: 'Flatten',     category: 'core', label: 'Flatten',     defaultParams: { start_dim: 1 },                                                                     hasInput: true,  hasOutput: true  },
  { type: 'BatchNorm',   category: 'core', label: 'BatchNorm',   defaultParams: { num_features: 64 },                                                                 hasInput: true,  hasOutput: true  },
  { type: 'Dropout',     category: 'core', label: 'Dropout',     defaultParams: { p: 0.5 },                                                                           hasInput: true,  hasOutput: true  },
  // Activations
  { type: 'ReLU',        category: 'activation', label: 'ReLU',        defaultParams: {},             hasInput: true, hasOutput: true },
  { type: 'GELU',        category: 'activation', label: 'GELU',        defaultParams: {},             hasInput: true, hasOutput: true },
  { type: 'Sigmoid',     category: 'activation', label: 'Sigmoid',     defaultParams: {},             hasInput: true, hasOutput: true },
  { type: 'Tanh',        category: 'activation', label: 'Tanh',        defaultParams: {},             hasInput: true, hasOutput: true },
  { type: 'Softmax',     category: 'activation', label: 'Softmax',     defaultParams: { dim: -1 },    hasInput: true, hasOutput: true },
  // Structural
  { type: 'ResidualAdd', category: 'structural', label: 'Residual Add', defaultParams: {},            hasInput: true, hasOutput: true },
  { type: 'Concatenate', category: 'structural', label: 'Concatenate',  defaultParams: { dim: 1 },    hasInput: true, hasOutput: true },
  // Training Config
  { type: 'SGD',              category: 'training', label: 'SGD Optimizer',     defaultParams: { lr: 0.01, momentum: 0.9 }, hasInput: false, hasOutput: false },
  { type: 'Adam',             category: 'training', label: 'Adam Optimizer',    defaultParams: { lr: 0.001 },              hasInput: false, hasOutput: false },
  { type: 'MSELoss',          category: 'training', label: 'MSE Loss',          defaultParams: {},                         hasInput: false, hasOutput: false },
  { type: 'CrossEntropyLoss', category: 'training', label: 'CrossEntropy Loss', defaultParams: {},                         hasInput: false, hasOutput: false },
];

export type CategoryColors = {
  bg: string;
  border: string;
  text: string;
  port: string;
  headerBg: string;
};

export const CATEGORY_COLORS: Record<BlockCategory, CategoryColors> = {
  core:       { bg: '#0d1b2e', border: '#2563eb', text: '#93c5fd', port: '#3b82f6', headerBg: '#1e3a5f' },
  activation: { bg: '#160d2e', border: '#7c3aed', text: '#c4b5fd', port: '#8b5cf6', headerBg: '#2e1b4e' },
  structural: { bg: '#0a1f18', border: '#059669', text: '#6ee7b7', port: '#10b981', headerBg: '#0f3830' },
  training:   { bg: '#1e1000', border: '#d97706', text: '#fcd34d', port: '#f59e0b', headerBg: '#3d2000' },
};

export const CATEGORY_LABELS: Record<BlockCategory, string> = {
  core:       'Core Layers',
  activation: 'Activations',
  structural: 'Structural',
  training:   'Training Config',
};

export const TRAINING_TYPES = new Set(['SGD', 'Adam', 'MSELoss', 'CrossEntropyLoss']);
export const OPTIMIZER_TYPES = new Set(['SGD', 'Adam']);
export const LOSS_TYPES      = new Set(['MSELoss', 'CrossEntropyLoss']);
