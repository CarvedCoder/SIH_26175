/**
 * DepthWizard — SemanticInspector
 *
 * Real-time 3D Semantic Inspection & Hover Inspector.
 * Uses client-side semantic sampler (in-memory Float32/Uint8 arrays):
 *   - Hovering over terrain detects semantic class & confidence instantly
 *   - Commands 3D shader to light up/highlight the entire hovered class
 *   - Detects 4-connected building regions when hovering over buildings
 *   - Displays class confidence, pixel coordinates, and scene distribution
 */

import React, { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { Layers, Eye, Building2, Trees, Milestone, Droplets, Mountain, HelpCircle, X } from 'lucide-react';
import {
  SEMANTIC_CLASSES,
  SEMANTIC_CLASS_LABELS,
  SEMANTIC_COLORS_HEX,
} from '../../api/semantic.js';
import {
  sampleSemanticAtUV,
  getBuildingConnectedComponent,
} from '../../lib/semanticSampler.js';

const CLASS_ICONS = {
  building: Building2,
  vegetation: Trees,
  road: Milestone,
  water: Droplets,
  ground: Mountain,
  other: HelpCircle,
};

const SemanticInspector = forwardRef(function SemanticInspector(
  { terrainRef, semanticData, active = false, onClose },
  ref
) {
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);
  const [pinnedRegion, setPinnedRegion] = useState(null);

  // Command shader highlight
  const setShaderHighlight = useCallback(
    (classId) => {
      terrainRef.current?.setSemanticHighlightClass?.(classId ?? -1);
    },
    [terrainRef]
  );

  // Clear highlight on unmount or deactivation
  useEffect(() => {
    if (!active) {
      setShaderHighlight(-1);
      setHoveredInfo(null);
    }
  }, [active, setShaderHighlight]);

  // Handle terrain hover event from TerrainWorkspace
  const handleTerrainHover = useCallback(
    (point) => {
      if (!active || !semanticData?.available) return;

      if (!point || point.u == null || point.v == null) {
        if (!selectedClass) {
          setShaderHighlight(-1);
        }
        setHoveredInfo(null);
        return;
      }

      const sample = sampleSemanticAtUV(semanticData, point.u, point.v);
      if (!sample) return;

      // Building connected component detection
      let buildingRegion = null;
      if (sample.className === 'building') {
        buildingRegion = getBuildingConnectedComponent(
          semanticData,
          sample.px,
          sample.py,
          15000
        );
      }

      const info = {
        ...sample,
        elevation: point.elevation,
        worldX: point.x,
        worldZ: point.z,
        buildingRegion,
      };

      setHoveredInfo(info);

      // Unless user explicitly clicked/pinned a class, highlight the hovered class
      if (selectedClass == null) {
        setShaderHighlight(sample.classId);
      }
    },
    [active, semanticData, selectedClass, setShaderHighlight]
  );

  // Handle terrain click to pin/inspect a region
  const handleTerrainClick = useCallback(
    (point) => {
      if (!active || !semanticData?.available || !point) return;
      const sample = sampleSemanticAtUV(semanticData, point.u, point.v);
      if (!sample) return;

      if (sample.className === 'building') {
        const region = getBuildingConnectedComponent(
          semanticData,
          sample.px,
          sample.py,
          25000
        );
        setPinnedRegion(region);
      } else {
        setPinnedRegion(null);
      }
      setSelectedClass(sample.classId);
      setShaderHighlight(sample.classId);
    },
    [active, semanticData, setShaderHighlight]
  );

  const handleSelectClass = (clsName) => {
    const classId = SEMANTIC_CLASSES.indexOf(clsName);
    if (selectedClass === classId) {
      // Toggle off
      setSelectedClass(null);
      setShaderHighlight(hoveredInfo ? hoveredInfo.classId : -1);
    } else {
      setSelectedClass(classId);
      setShaderHighlight(classId);
    }
  };

  const handleResetHighlight = () => {
    setSelectedClass(null);
    setPinnedRegion(null);
    setShaderHighlight(-1);
  };

  useImperativeHandle(ref, () => ({
    handleTerrainHover,
    handleTerrainClick,
    reset: handleResetHighlight,
  }));

  if (!active) return null;

  const currentClass =
    hoveredInfo?.className ??
    (selectedClass != null ? SEMANTIC_CLASSES[selectedClass] : null);
  const currentConfidence = hoveredInfo?.confidence ?? 0;
  const IconComponent = currentClass ? CLASS_ICONS[currentClass] ?? Layers : Layers;
  const classColor = currentClass ? SEMANTIC_COLORS_HEX[currentClass] : '#b6b6bd';

  return (
    <div
      data-testid="semantic-inspector-panel"
      style={{
        position: 'absolute',
        top: 68,
        left: 20,
        zIndex: 30,
        width: 310,
        background: 'rgba(9, 9, 11, 0.94)',
        border: '1px solid rgba(250, 250, 250, 0.14)',
        borderRadius: 12,
        padding: '14px 16px',
        color: '#e4e4e7',
        fontFamily: 'ui-sans-serif, system-ui, -apple-system, sans-serif',
        backdropFilter: 'blur(12px)',
        boxShadow: '0 12px 36px rgba(0, 0, 0, 0.6)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 24,
              height: 24,
              borderRadius: 6,
              background: 'rgba(250, 250, 250, 0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--dw-fg-muted)',
            }}
          >
            <Layers size={14} />
          </div>
          <div>
            <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: '#fafafa' }}>
              Semantic Inspect
            </span>
            <span style={{ display: 'block', fontSize: 10, color: 'var(--dw-fg-muted)' }}>
              Hover terrain to isolate classes
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {selectedClass != null && (
            <button
              onClick={handleResetHighlight}
              title="Reset class isolation"
              style={{
                fontSize: 10,
                padding: '2px 6px',
                borderRadius: 4,
                background: 'rgba(255,255,255,0.08)',
                border: '1px solid rgba(255,255,255,0.15)',
                color: '#d4d4d8',
                cursor: 'pointer',
              }}
            >
              Reset
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--dw-fg-muted)',
                cursor: 'pointer',
                padding: 4,
                borderRadius: 4,
              }}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Hovered Target Badge */}
      <div
        style={{
          background: 'rgba(24, 24, 27, 0.8)',
          border: `1px solid ${classColor}40`,
          borderRadius: 8,
          padding: '10px 12px',
          marginBottom: 12,
          transition: 'all 120ms ease',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: classColor,
                boxShadow: `0 0 8px ${classColor}80`,
              }}
            />
            <span style={{ fontSize: 13, fontWeight: 700, color: '#fafafa', textTransform: 'capitalize' }}>
              {currentClass ? SEMANTIC_CLASS_LABELS[currentClass] : 'Move cursor over terrain'}
            </span>
          </div>
          {currentClass && (
            <span
              style={{
                fontSize: 11,
                fontFamily: 'ui-monospace, monospace',
                fontWeight: 700,
                color: currentConfidence > 0.8 ? '#4ade80' : '#facc15',
                background: 'rgba(0,0,0,0.3)',
                padding: '2px 6px',
                borderRadius: 4,
              }}
            >
              {Math.round(currentConfidence * 100)}% conf
            </span>
          )}
        </div>

        {/* Building Region Details if detected */}
        {hoveredInfo?.buildingRegion && (
          <div
            style={{
              marginTop: 8,
              paddingTop: 8,
              borderTop: '1px dashed rgba(255,255,255,0.1)',
              fontSize: 11,
              color: '#d4d4d8',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--dw-fg-muted)' }}>
              <span>Semantic building region:</span>
              <span style={{ fontFamily: 'ui-monospace, monospace', color: '#e4e4e7' }}>
                {hoveredInfo.buildingRegion.pixelCount} px
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--dw-fg-muted)', marginTop: 2 }}>
              <span>Span:</span>
              <span style={{ fontFamily: 'ui-monospace, monospace', color: '#e4e4e7' }}>
                {hoveredInfo.buildingRegion.bounds.width} × {hoveredInfo.buildingRegion.bounds.height} px
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Class Selector / Quick Filter Toggles */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--dw-fg-ghost)', letterSpacing: 0.8, textTransform: 'uppercase', marginBottom: 6 }}>
          Semantic Classes (Click to isolate in 3D)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
          {SEMANTIC_CLASSES.map((clsName, idx) => {
            const isTarget = (selectedClass === idx) || (selectedClass == null && hoveredInfo?.className === clsName);
            const color = SEMANTIC_COLORS_HEX[clsName];
            const frac = semanticData?.meta?.class_fractions?.[clsName];
            const Icon = CLASS_ICONS[clsName] || Layers;

            return (
              <button
                key={clsName}
                onClick={() => handleSelectClass(clsName)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 8px',
                  borderRadius: 6,
                  background: isTarget ? `${color}25` : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${isTarget ? color : 'rgba(255,255,255,0.08)'}`,
                  color: isTarget ? '#ffffff' : 'var(--dw-fg-muted)',
                  fontSize: 11,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 120ms ease',
                }}
              >
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: '50%',
                    background: color,
                    flexShrink: 0,
                  }}
                />
                <span style={{ flex: 1, textTransform: 'capitalize', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {clsName}
                </span>
                {frac != null && (
                  <span style={{ fontSize: 9.5, fontFamily: 'ui-monospace, monospace', color: 'var(--dw-fg-ghost)' }}>
                    {Math.round(frac * 100)}%
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Footer Info */}
      <div
        style={{
          paddingTop: 8,
          borderTop: '1px solid rgba(255,255,255,0.08)',
          fontSize: 10,
          color: 'var(--dw-fg-ghost)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>Model: {semanticData?.meta?.model || 'CalibrationNet (6-class)'}</span>
        {hoveredInfo?.elevation != null && (
          <span style={{ fontFamily: 'ui-monospace, monospace', color: 'var(--dw-fg-muted)' }}>
            Elev: {hoveredInfo.elevation.toFixed(1)}m
          </span>
        )}
      </div>
    </div>
  );
});

export default SemanticInspector;
