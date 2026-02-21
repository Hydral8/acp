import { useRef, useCallback, useState, useEffect } from 'react';
import { GraphNode, GraphEdge, NODE_WIDTH, NODE_HEIGHT, PendingConn, CompositeBlock } from '../types';
import { type NodeShapeInfo, type ShapeFix } from '../shapeEngine';
import { BlockNode } from './BlockNode';
import { EdgeLayer } from './EdgeLayer';

interface View { panX: number; panY: number; zoom: number }

interface DragState {
  type: 'pan' | 'node' | null;
  nodeId?: string;
  offsetX?: number;
  offsetY?: number;
  panStartX?: number;
  panStartY?: number;
  initPanX?: number;
  initPanY?: number;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedIds: string[];
  pendingConn: PendingConn | null;
  shapeMap?: Map<string, NodeShapeInfo>;
  onAddNode: (type: string, x: number, y: number, composite?: CompositeBlock) => void;
  onMoveNode: (id: string, x: number, y: number) => void;
  onSelectNode: (id: string, addToSelection: boolean) => void;
  onClearSelection: () => void;
  onAddEdge: (sourceId: string, targetId: string) => void;
  onDeleteEdge: (id: string) => void;
  onPendingConnChange: (conn: PendingConn | null) => void;
  onFixConflict?: (fix: ShapeFix) => void;
  onDoubleClickNode?: (nodeId: string) => void;
}

export function Canvas({
  nodes, edges, selectedIds, pendingConn, shapeMap,
  onAddNode, onMoveNode, onSelectNode, onClearSelection, onAddEdge, onDeleteEdge, onPendingConnChange,
  onFixConflict, onDoubleClickNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef      = useRef<DragState>({ type: null });
  const viewRef      = useRef<View>({ panX: 60, panY: 60, zoom: 1 });

  // Keep a ref of pendingConn so mouse-event callbacks always see the latest value
  // without depending on React re-render timing.
  const pendingConnRef = useRef<PendingConn | null>(pendingConn);
  pendingConnRef.current = pendingConn;

  const [view, setView] = useState<View>({ panX: 60, panY: 60, zoom: 1 });
  const [isDragging, setIsDragging] = useState(false);

  // Keep ref in sync with state (used inside stable callbacks below)
  viewRef.current = view;

  const selectedSet = new Set(selectedIds);

  // ── Coordinate helper ───────────────────────────────────────────────────
  const toCanvas = useCallback((clientX: number, clientY: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    const { panX, panY, zoom } = viewRef.current;
    return {
      x: (clientX - rect.left - panX) / zoom,
      y: (clientY - rect.top  - panY) / zoom,
    };
  }, []);

  // ── Wheel zoom (native listener so passive:false works reliably) ─────────
  useEffect(() => {
    const el = containerRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx   = e.clientX - rect.left;
      const my   = e.clientY - rect.top;
      const factor = e.deltaY > 0 ? 0.92 : 1.087;
      // Atomic update — all three values computed together from the same prev state
      setView(v => {
        const newZoom = Math.max(0.2, Math.min(2.5, v.zoom * factor));
        const ratio   = newZoom / v.zoom;
        return {
          zoom: newZoom,
          panX: mx + (v.panX - mx) * ratio,
          panY: my + (v.panY - my) * ratio,
        };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // ── Canvas background mousedown (pan start) ──────────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const { panX, panY } = viewRef.current;
    dragRef.current = { type: 'pan', panStartX: e.clientX, panStartY: e.clientY, initPanX: panX, initPanY: panY };
    setIsDragging(true);
    if (!e.shiftKey) onClearSelection();
    e.preventDefault();
  }, [onClearSelection]);

  // ── Block mousedown (node drag start) ───────────────────────────────────
  const handleNodeMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    if (e.button !== 0) return;
    const node = nodes.find(n => n.id === nodeId)!;
    const cp   = toCanvas(e.clientX, e.clientY);
    dragRef.current = { type: 'node', nodeId, offsetX: cp.x - node.x, offsetY: cp.y - node.y };
    setIsDragging(true);
    onSelectNode(nodeId, e.shiftKey);
    e.preventDefault();
  }, [nodes, toCanvas, onSelectNode]);

  // ── Port mousedown — start a connection from an OUTPUT port ─────────────
  const handlePortMouseDown = useCallback((e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => {
    e.stopPropagation();
    if (portType !== 'output') return;
    const node = nodes.find(n => n.id === nodeId)!;
    const conn: PendingConn = { sourceId: nodeId, mouseX: node.x + NODE_WIDTH / 2, mouseY: node.y + NODE_HEIGHT };
    pendingConnRef.current = conn;  // Set ref immediately, before React re-render
    onPendingConnChange(conn);
    e.preventDefault();
  }, [nodes, onPendingConnChange]);

  // ── Port mouseup — complete a connection on an INPUT port ────────────────
  const handlePortMouseUp = useCallback((e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => {
    e.stopPropagation();
    const conn = pendingConnRef.current;  // Read from ref, always up-to-date
    if (portType !== 'input' || !conn) return;
    if (conn.sourceId !== nodeId) {
      onAddEdge(conn.sourceId, nodeId);
    }
    pendingConnRef.current = null;
    onPendingConnChange(null);
  }, [onAddEdge, onPendingConnChange]); // No pendingConn in deps — we use the ref

  // ── Mousemove — update pan / node drag / connection line ─────────────────
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const drag = dragRef.current;

    if (drag.type === 'pan') {
      const dx = e.clientX - (drag.panStartX ?? 0);
      const dy = e.clientY - (drag.panStartY ?? 0);
      setView(v => ({ ...v, panX: (drag.initPanX ?? 0) + dx, panY: (drag.initPanY ?? 0) + dy }));
    } else if (drag.type === 'node' && drag.nodeId) {
      const cp = toCanvas(e.clientX, e.clientY);
      onMoveNode(drag.nodeId, cp.x - (drag.offsetX ?? 0), cp.y - (drag.offsetY ?? 0));
    }

    const conn = pendingConnRef.current;  // Always read latest value
    if (conn) {
      const cp = toCanvas(e.clientX, e.clientY);
      const updated = { ...conn, mouseX: cp.x, mouseY: cp.y };
      pendingConnRef.current = updated;
      onPendingConnChange(updated);
    }
  }, [toCanvas, onMoveNode, onPendingConnChange]); // No pendingConn dep — use ref

  // ── Mouseup / leave — cancel all drag operations ─────────────────────────
  const handleMouseUp = useCallback(() => {
    dragRef.current = { type: null };
    setIsDragging(false);
    if (pendingConnRef.current) {
      pendingConnRef.current = null;
      onPendingConnChange(null);
    }
  }, [onPendingConnChange]);

  // ── HTML5 drag-and-drop from block library ────────────────────────────────
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const blockType = e.dataTransfer.getData('blockType');
    if (!blockType) return;
    const rect = containerRef.current!.getBoundingClientRect();
    const { panX, panY, zoom } = viewRef.current;
    const x = (e.clientX - rect.left - panX) / zoom - NODE_WIDTH / 2;
    const y = (e.clientY - rect.top  - panY) / zoom - NODE_HEIGHT / 2;
    const customData = e.dataTransfer.getData('customBlockData');
    if (customData) {
      try { onAddNode(blockType, x, y, JSON.parse(customData) as CompositeBlock); } catch { onAddNode(blockType, x, y); }
    } else {
      onAddNode(blockType, x, y);
    }
  };

  // ── Grid background ───────────────────────────────────────────────────────
  const { panX, panY, zoom } = view;
  const gs  = 24 * zoom;
  const gox = ((panX % gs) + gs) % gs;
  const goy = ((panY % gs) + gs) % gs;

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{
        flex: 1,
        position: 'relative',
        overflow: 'hidden',
        cursor: pendingConn ? 'crosshair' : isDragging ? 'grabbing' : 'grab',
        backgroundImage: `radial-gradient(circle, #1e1e1e 1px, transparent 1px)`,
        backgroundSize: `${gs}px ${gs}px`,
        backgroundPosition: `${gox}px ${goy}px`,
        backgroundColor: '#0d0d0d',
      }}
    >
      {/* Transform wrapper */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0,
        transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
        transformOrigin: '0 0',
        width: 0, height: 0,
      }}>
        <EdgeLayer
          nodes={nodes}
          edges={edges}
          pending={pendingConn}
          onDeleteEdge={onDeleteEdge}
        />
        {nodes.map(node => (
          <BlockNode
            key={node.id}
            node={node}
            isSelected={selectedSet.has(node.id)}
            shapeInfo={shapeMap?.get(node.id)}
            onMouseDown={handleNodeMouseDown}
            onPortMouseDown={handlePortMouseDown}
            onPortMouseUp={handlePortMouseUp}
            onFixConflict={onFixConflict}
            onDoubleClick={onDoubleClickNode}
          />
        ))}
      </div>

      {/* Empty state */}
      {nodes.length === 0 && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          gap: 10, color: '#222', pointerEvents: 'none',
        }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#222" strokeWidth="1">
            <rect x="3" y="8" width="7" height="4" rx="1" />
            <rect x="14" y="12" width="7" height="4" rx="1" />
            <path d="M10 10h4M17 12V8h-4" strokeDasharray="3 2" />
          </svg>
          <span style={{ fontSize: 13 }}>Drag blocks from the library to start</span>
          <span style={{ fontSize: 11, color: '#1a1a1a' }}>
            Scroll to zoom · Drag canvas to pan · Drag output→input port to connect · Click edge to delete
          </span>
        </div>
      )}

      {/* Zoom indicator */}
      <div style={{
        position: 'absolute', bottom: 10, right: 10,
        fontSize: 10, color: '#333',
        backgroundColor: '#0a0a0a',
        border: '1px solid #1a1a1a',
        borderRadius: 4, padding: '3px 7px',
        pointerEvents: 'none',
      }}>
        {Math.round(zoom * 100)}%
      </div>
    </div>
  );
}
