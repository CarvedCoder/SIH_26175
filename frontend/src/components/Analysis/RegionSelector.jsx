/**
 * DepthWizard — RegionSelector (Phase 13, Task 13.2)
 *
 * Rubber-band rectangular region selector overlay for small-structure detail refinement.
 * Enables the user to click & drag a bounding box directly over the terrain viewer.
 *
 * Emits normalized coordinates [0, 1]²:
 *   { x_min, y_min, x_max, y_max }
 *
 * Spec §18, §65.
 * DESIGN.md: Precision instrument blue dashed stroke, corner registration marks.
 */
import { useState, useRef, useEffect } from 'react';

/**
 * @param {{
 *   active: boolean,
 *   selectedBbox: { x_min: number, y_min: number, x_max: number, y_max: number } | null,
 *   onBboxChange: (bbox: { x_min: number, y_min: number, x_max: number, y_max: number } | null) => void,
 *   onCompleteSelection?: () => void,
 * }} props
 */
export default function RegionSelector({
  active = false,
  selectedBbox = null,
  onBboxChange,
  onCompleteSelection,
}) {
  const containerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart]   = useState(null); // { x: px, y: px }
  const [dragCurrent, setDragCurrent] = useState(null); // { x: px, y: px }

  // Keyboard escape to cancel
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') {
        setIsDragging(false);
        setDragStart(null);
        setDragCurrent(null);
        if (active) onCompleteSelection?.();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [active, onCompleteSelection]);

  const handleMouseDown = (e) => {
    if (!active || e.button !== 0) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setIsDragging(true);
    setDragStart({ x, y });
    setDragCurrent({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !dragStart) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    setDragCurrent({ x, y });
  };

  const handleMouseUp = () => {
    if (!isDragging || !dragStart || !dragCurrent) {
      setIsDragging(false);
      return;
    }

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) {
      setIsDragging(false);
      return;
    }

    const x1 = Math.min(dragStart.x, dragCurrent.x);
    const x2 = Math.max(dragStart.x, dragCurrent.x);
    const y1 = Math.min(dragStart.y, dragCurrent.y);
    const y2 = Math.max(dragStart.y, dragCurrent.y);

    // Minimum selection size threshold (at least 15px)
    if (x2 - x1 >= 15 && y2 - y1 >= 15) {
      const bbox = {
        x_min: +(x1 / rect.width).toFixed(4),
        x_max: +(x2 / rect.width).toFixed(4),
        y_min: +(y1 / rect.height).toFixed(4),
        y_max: +(y2 / rect.height).toFixed(4),
      };
      onBboxChange(bbox);
      onCompleteSelection?.();
    }

    setIsDragging(false);
    setDragStart(null);
    setDragCurrent(null);
  };

  // Convert bbox [0,1] to current pixel dimensions for rendering
  const rect = containerRef.current?.getBoundingClientRect();
  const width = rect?.width ?? 0;
  const height = rect?.height ?? 0;

  let renderBox = null;
  if (isDragging && dragStart && dragCurrent) {
    const x = Math.min(dragStart.x, dragCurrent.x);
    const y = Math.min(dragStart.y, dragCurrent.y);
    const w = Math.abs(dragCurrent.x - dragStart.x);
    const h = Math.abs(dragCurrent.y - dragStart.y);
    renderBox = { x, y, w, h };
  } else if (selectedBbox && width > 0 && height > 0) {
    const x = selectedBbox.x_min * width;
    const y = selectedBbox.y_min * height;
    const w = (selectedBbox.x_max - selectedBbox.x_min) * width;
    const h = (selectedBbox.y_max - selectedBbox.y_min) * height;
    renderBox = { x, y, w, h };
  }

  // If not active and no selected bbox, render nothing
  if (!active && !selectedBbox) return null;

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: active ? 14 : 5,
        pointerEvents: active ? 'auto' : 'none',
        cursor: active ? 'crosshair' : 'default',
      }}
    >
      {/* Active selection banner instruction */}
      {active && (
        <div style={{
          position: 'absolute',
          top: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(13, 17, 23, 0.94)',
          border: '1px solid var(--dw-accent)',
          borderRadius: 'var(--dw-radius-sm)',
          padding: '8px 16px',
          fontFamily: 'var(--dw-font-ui)',
          fontSize: 13,
          fontWeight: 500,
          color: 'var(--dw-fg)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          pointerEvents: 'none',
        }}>
          <span style={{
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: 'var(--dw-accent)',
          }} />
          Click and drag a box to select high-resolution refinement region (Esc to cancel)
        </div>
      )}

      {/* SVG Canvas for drawing the rubber-band selection */}
      <svg
        style={{
          width: '100%',
          height: '100%',
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
        }}
      >
        {renderBox && renderBox.w > 2 && renderBox.h > 2 && (
          <g>
            {/* Box fill & dashed stroke */}
            <rect
              x={renderBox.x}
              y={renderBox.y}
              width={renderBox.w}
              height={renderBox.h}
              fill="rgba(59, 130, 246, 0.10)"
              stroke="var(--dw-accent)"
              strokeWidth="1.5"
              strokeDasharray="5 3"
            />

            {/* Corner registration handles */}
            <rect x={renderBox.x - 3} y={renderBox.y - 3} width={6} height={6} fill="var(--dw-accent)" />
            <rect x={renderBox.x + renderBox.w - 3} y={renderBox.y - 3} width={6} height={6} fill="var(--dw-accent)" />
            <rect x={renderBox.x - 3} y={renderBox.y + renderBox.h - 3} width={6} height={6} fill="var(--dw-accent)" />
            <rect x={renderBox.x + renderBox.w - 3} y={renderBox.y + renderBox.h - 3} width={6} height={6} fill="var(--dw-accent)" />

            {/* Dimension Badge */}
            <foreignObject
              x={renderBox.x}
              y={Math.max(0, renderBox.y - 24)}
              width={200}
              height={24}
            >
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                background: 'var(--dw-panel)',
                border: '1px solid var(--dw-accent)',
                borderRadius: 2,
                padding: '2px 8px',
                fontFamily: 'var(--dw-font-data)',
                fontSize: 11,
                color: 'var(--dw-fg)',
                whiteSpace: 'nowrap',
              }}>
                REFINE: {Math.round(renderBox.w)} × {Math.round(renderBox.h)} px
              </div>
            </foreignObject>
          </g>
        )}
      </svg>
    </div>
  );
}
