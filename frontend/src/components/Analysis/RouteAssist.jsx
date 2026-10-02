/**
 * RouteAssist — Route Assist readout panel (jury round 1 + 2).
 *
 * Vehicle passability verdicts with recommended paths: pick start/end on
 * the terrain, choose the fleet, read CAN GO / CAUTION / CANNOT GO per
 * vehicle with blocking reasons, travel time and the nearest reachable
 * detour point when a destination is cut off.
 *
 * The panel is a pure readout: point picking lives in the workspace's
 * terrain click handler (mirrors HeightMeasurement's ref contract).
 */
import { forwardRef, useImperativeHandle, useState } from 'react';
import { Route, RotateCcw, AlertTriangle } from 'lucide-react';
import { VEHICLES, VERDICT_META, fmtDistance } from '../../api/route.js';

const RouteAssist = forwardRef(function RouteAssist({ active = true }, ref) {
  const [phase, setPhase] = useState('pick-start'); // pick-start | pick-end | assessing | done
  const [vehicles, setVehicles] = useState(['fire_truck']);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [units, setUnits] = useState('meters');

  useImperativeHandle(ref, () => ({
    /** Terrain click from the workspace (source-raster pixel coords). */
    handleTerrainClick(pixel) {
      if (!active) return;
      setError(null);
      if (phase === 'pick-start') {
        setResults(null);
        setPhase('pick-end');
        return { picked: 'start', pixel };
      }
      if (phase === 'pick-end') {
        setPhase('done');
        return { picked: 'end', pixel };
      }
      return null;
    },
    /** Called by the workspace with the API response. */
    setAssessment(response) {
      setResults(response);
      setUnits(response?.units ?? 'meters');
      setPhase('done');
    },
    setAssessing() {
      setPhase('assessing');
    },
    setAssessError(message) {
      setError(message);
      setPhase('done');
    },
    getPicked() {
      return { phase };
    },
    reset() {
      setPhase('pick-start');
      setResults(null);
      setError(null);
    },
    resetPicks() {
      setPhase('pick-start');
      setResults(null);
      setError(null);
    },
    /** Currently selected vehicle keys (the workspace reads these). */
    getVehicles: () => vehicles,
  }));

  const toggleVehicle = (key) => {
    setVehicles((prev) =>
      prev.includes(key)
        ? prev.length > 1
          ? prev.filter((v) => v !== key)
          : prev
        : [...prev, key],
    );
  };

  const fmtTime = (s) => {
    if (s == null) return '—';
    if (s < 90) return `${Math.round(s)} s`;
    return `${Math.round(s / 60)} min`;
  };

  return (
    <div
      data-testid="route-assist-panel"
      style={{
        width: 320,
        padding: 14,
        borderRadius: 12,
        background: 'rgba(9, 9, 11, 0.92)',
        border: '1px solid rgba(250, 250, 250, 0.14)',
        color: '#e4e4e7',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 12,
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Route size={15} style={{ color: 'var(--dw-fg)' }} />
        <span style={{ fontWeight: 700, letterSpacing: 1 }}>ROUTE ASSIST</span>
        <button
          onClick={() => {
            setPhase('pick-start');
            setResults(null);
            setError(null);
          }}
          title="Pick new start/end points"
          style={{
            marginLeft: 'auto',
            background: 'none',
            border: '1px solid var(--dw-rim-strong)',
            borderRadius: 6,
            color: '#d4d4d8',
            cursor: 'pointer',
            padding: '2px 6px',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 11,
          }}
        >
          <RotateCcw size={11} /> Reset
        </button>
      </div>

      {/* vehicle fleet selection */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ color: 'var(--dw-fg-ghost)', marginBottom: 4, fontSize: 10, letterSpacing: 1 }}>
          FLEET
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {VEHICLES.map((v) => {
            const on = vehicles.includes(v.key);
            return (
              <button
                key={v.key}
                onClick={() => toggleVehicle(v.key)}
                style={{
                  padding: '3px 8px',
                  borderRadius: 999,
                  fontSize: 11,
                  cursor: 'pointer',
                  border: `1px solid ${on ? 'var(--dw-accent)' : 'var(--dw-rim)'}`,
                  background: on ? 'var(--dw-accent-soft)' : 'transparent',
                  color: on ? 'var(--dw-accent)' : 'var(--dw-fg-ghost)',
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* picking phase hint */}
      {phase === 'pick-start' && (
        <div style={{ color: '#d4d4d8' }}>Click the START point on the terrain…</div>
      )}
      {phase === 'pick-end' && (
        <div style={{ color: '#d4d4d8' }}>
          Click the DESTINATION point…
          {vehicles.includes('rescue_chopper') && (
            <span style={{ color: 'var(--dw-fg-ghost)' }}>
              {' '}the chopper will look for a landing zone near it.
            </span>
          )}
        </div>
      )}
      {phase === 'assessing' && <div style={{ color: '#d4d4d8' }}>Assessing routes…</div>}
      {error && (
        <div style={{ color: '#f87171', display: 'flex', gap: 6, alignItems: 'center' }}>
          <AlertTriangle size={13} /> {error}
        </div>
      )}

      {/* per-vehicle verdicts */}
      {results?.vehicles?.map((v) => {
        const meta = VERDICT_META[v.verdict] ?? VERDICT_META.CANNOT_GO;
        return (
          <div
            key={v.vehicle}
            data-testid={`route-verdict-${v.vehicle}`}
            style={{
              marginTop: 10,
              padding: 10,
              borderRadius: 8,
              border: `1px solid ${meta.color}55`,
              background: 'rgba(255,255,255,0.03)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 800,
                  color: meta.color,
                  letterSpacing: 1,
                }}
              >
                {meta.glyph} {meta.label}
              </span>
              <span style={{ marginLeft: 'auto', color: 'var(--dw-fg-muted)' }}>{v.vehicle_label}</span>
            </div>

            {v.reasons?.map((r, i) => (
              <div key={i} style={{ marginTop: 4, color: 'var(--dw-fg-muted)' }}>
                • {r}
              </div>
            ))}

            {(v.path || v.detour_pixel) && (
              <div
                style={{
                  marginTop: 8,
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '2px 10px',
                  color: 'var(--dw-fg-muted)',
                }}
              >
                {v.path_length_m != null && (
                  <>
                    <span>Length</span>
                    <span style={{ color: 'var(--dw-fg)' }}>{fmtDistance(v.path_length_m)}</span>
                  </>
                )}
                {v.landing_zone?.distance_to_goal_m != null && (
                  <>
                    <span>LZ→target</span>
                    <span style={{ color: 'var(--dw-fg)' }}>
                      {fmtDistance(v.landing_zone.distance_to_goal_m)}
                    </span>
                  </>
                )}
                {v.estimated_travel_seconds != null && (
                  <>
                    <span>Est. travel</span>
                    <span style={{ color: 'var(--dw-fg)' }}>{fmtTime(v.estimated_travel_seconds)}</span>
                  </>
                )}
                {v.max_slope_on_path_deg != null && (
                  <>
                    <span>Max slope</span>
                    <span style={{ color: 'var(--dw-fg)' }}>{v.max_slope_on_path_deg}°</span>
                  </>
                )}
                {v.max_step_on_path_m != null && (
                  <>
                    <span>Max step</span>
                    <span style={{ color: 'var(--dw-fg)' }}>{v.max_step_on_path_m} m</span>
                  </>
                )}
              </div>
            )}

            {v.semantic_aware && (
              <div
                data-testid={`route-semantics-${v.vehicle}`}
                style={{
                  marginTop: 8,
                  padding: '6px 8px',
                  borderRadius: 6,
                  background: 'rgba(24, 24, 27, 0.75)',
                  border: '1px solid rgba(161, 161, 170, 0.25)',
                }}
              >
                <div style={{ fontSize: 9.5, fontWeight: 700, color: 'var(--dw-fg-muted)', letterSpacing: 0.8, marginBottom: 5 }}>
                  ROUTE SEMANTICS
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {v.road_fraction != null && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: '#d4d4d8', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#9B9B9B' }} /> Road
                      </span>
                      <span style={{ color: '#ffffff', fontWeight: 600 }}>{Math.round(v.road_fraction * 100)}%</span>
                    </div>
                  )}
                  {v.ground_fraction != null && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: '#d4d4d8', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#D2B48C' }} /> Ground
                      </span>
                      <span style={{ color: '#ffffff', fontWeight: 600 }}>{Math.round(v.ground_fraction * 100)}%</span>
                    </div>
                  )}
                  {v.vegetation_fraction != null && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: '#d4d4d8', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#2ECC71' }} /> Vegetation
                      </span>
                      <span style={{ color: '#ffffff', fontWeight: 600 }}>{Math.round(v.vegetation_fraction * 100)}%</span>
                    </div>
                  )}
                  {((v.building_fraction || 0) + (v.water_fraction || 0) > 0) && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: '#f87171', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#E74C3C' }} /> Obstacles
                      </span>
                      <span style={{ color: '#f87171', fontWeight: 600 }}>
                        {Math.round(((v.building_fraction || 0) + (v.water_fraction || 0)) * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {v.damage_aware && (
              <div
                data-testid={`route-damage-${v.vehicle}`}
                style={{
                  marginTop: 8,
                  padding: '6px 8px',
                  borderRadius: 6,
                  background: 'rgba(24, 24, 27, 0.75)',
                  border: '1px solid rgba(161, 161, 170, 0.25)',
                }}
              >
                <div style={{ fontSize: 9.5, fontWeight: 700, color: 'var(--dw-fg-muted)', letterSpacing: 0.8, marginBottom: 5 }}>
                  DAMAGE EXPOSURE
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                  <span style={{ color: '#d4d4d8', display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{
                      width: 7, height: 7, borderRadius: '50%',
                      background: (v.damage_risk_fraction ?? 0) > 0.15 ? '#E74C3C' : '#FFC107',
                    }} /> Damaged-structure exposure
                  </span>
                  <span style={{ color: '#ffffff', fontWeight: 600 }}>
                    {Math.round((v.damage_risk_fraction ?? 0) * 100)}%
                  </span>
                </div>
              </div>
            )}

            {v.landing_zone && (
              <div
                data-testid={`route-lz-${v.vehicle}`}
                style={{
                  marginTop: 8,
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: '1px dashed rgba(91,140,255,0.45)',
                  background: 'rgba(91,140,255,0.08)',
                  color: '#d4d4d8',
                }}
              >
                <div style={{ fontWeight: 700, letterSpacing: 1, fontSize: 10 }}>
                  ⛉ LANDING ZONE (pix {v.landing_zone.pixel?.x}, {v.landing_zone.pixel?.y})
                </div>
                <div style={{ marginTop: 2, color: 'var(--dw-fg-muted)' }}>
                  Slope {v.landing_zone.slope_deg}° ·
                  {' '}{v.landing_zone.distance_to_goal_m != null
                    ? `${fmtDistance(v.landing_zone.distance_to_goal_m)} from target`
                    : `${v.landing_zone.distance_to_goal_px} px from target`}
                </div>
              </div>
            )}

            {v.detour_pixel && (
              <div style={{ marginTop: 6, color: '#f1c40f' }}>
                Nearest reachable point: ({v.detour_pixel.x}, {v.detour_pixel.y})
              </div>
            )}
          </div>
        );
      })}

      {/* honest-scope disclaimer */}
      {results?.disclaimer && (
        <div
          style={{
            marginTop: 10,
            paddingTop: 8,
            borderTop: '1px solid var(--dw-rim)',
            color: 'var(--dw-fg-ghost)',
            fontSize: 10,
            lineHeight: 1.5,
          }}
        >
          {results.disclaimer}
          {units === 'pixels' && ' Non-georeferenced scene: distances are in pixel units.'}
        </div>
      )}
    </div>
  );
});

export default RouteAssist;
