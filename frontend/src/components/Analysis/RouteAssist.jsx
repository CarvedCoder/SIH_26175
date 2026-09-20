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
import { VEHICLES, VERDICT_META } from '../../api/route.js';

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
        background: 'rgba(2, 8, 20, 0.92)',
        border: '1px solid rgba(90, 140, 255, 0.25)',
        color: '#dbe7ff',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 12,
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Route size={15} style={{ color: '#5b8cff' }} />
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
            border: '1px solid rgba(90,140,255,0.3)',
            borderRadius: 6,
            color: '#8fb0ff',
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
        <div style={{ color: '#7f92b8', marginBottom: 4, fontSize: 10, letterSpacing: 1 }}>
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
                  border: `1px solid ${on ? '#5b8cff' : 'rgba(90,140,255,0.2)'}`,
                  background: on ? 'rgba(91,140,255,0.18)' : 'transparent',
                  color: on ? '#cfe0ff' : '#7f92b8',
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
        <div style={{ color: '#8fb0ff' }}>Click the START point on the terrain…</div>
      )}
      {phase === 'pick-end' && (
        <div style={{ color: '#8fb0ff' }}>Click the DESTINATION point…</div>
      )}
      {phase === 'assessing' && <div style={{ color: '#8fb0ff' }}>Assessing routes…</div>}
      {error && (
        <div style={{ color: '#ff9c9c', display: 'flex', gap: 6, alignItems: 'center' }}>
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
              <span style={{ marginLeft: 'auto', color: '#9fb4dd' }}>{v.vehicle_label}</span>
            </div>

            {v.reasons?.map((r, i) => (
              <div key={i} style={{ marginTop: 4, color: '#a9bcdc' }}>
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
                  color: '#8ea6d4',
                }}
              >
                {v.path_length_m != null && (
                  <>
                    <span>Length</span>
                    <span style={{ color: '#cfe0ff' }}>{v.path_length_m.toFixed(0)} m</span>
                  </>
                )}
                {v.estimated_travel_seconds != null && (
                  <>
                    <span>Est. travel</span>
                    <span style={{ color: '#cfe0ff' }}>{fmtTime(v.estimated_travel_seconds)}</span>
                  </>
                )}
                {v.max_slope_on_path_deg != null && (
                  <>
                    <span>Max slope</span>
                    <span style={{ color: '#cfe0ff' }}>{v.max_slope_on_path_deg}°</span>
                  </>
                )}
                {v.max_step_on_path_m != null && (
                  <>
                    <span>Max step</span>
                    <span style={{ color: '#cfe0ff' }}>{v.max_step_on_path_m} m</span>
                  </>
                )}
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
            borderTop: '1px solid rgba(90,140,255,0.15)',
            color: '#6d7f9f',
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
