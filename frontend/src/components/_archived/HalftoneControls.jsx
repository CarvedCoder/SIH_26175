import { Sliders, RefreshCw, Eye, X } from 'lucide-react';

export default function HalftoneControls({
  config,
  onChange,
  onReset,
  onClose
}) {
  const modes = [
    { id: 'color', label: 'CMYK Rosette (Color)' },
    { id: 'duotone', label: 'Risograph Duotone' },
    { id: 'mono', label: 'Monochrome Print' }
  ];

  const shapes = [
    { id: 'circle', label: 'Circle' },
    { id: 'diamond', label: 'Diamond' },
    { id: 'square', label: 'Square' },
    { id: 'line', label: 'Scanline' }
  ];

  return (
    <div className="fixed bottom-6 right-6 z-40 w-80 bg-neutral-950/90 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl text-neutral-200 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-4">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-emerald-400" />
          <span className="font-mono text-xs font-semibold uppercase tracking-wider text-white">
            Halftone Shader Tuner
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onReset}
            className="p-1 rounded hover:bg-white/10 text-neutral-400 hover:text-white transition-colors"
            title="Reset to defaults"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-white/10 text-neutral-400 hover:text-white transition-colors"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="space-y-4 text-xs">
        {/* Mode Selector */}
        <div>
          <label className="block text-neutral-400 font-mono text-[10px] uppercase tracking-wider mb-1.5">
            Screening Mode
          </label>
          <div className="grid grid-cols-3 gap-1">
            {modes.map(m => (
              <button
                key={m.id}
                onClick={() => onChange({ mode: m.id })}
                className={`py-1.5 px-2 rounded-lg font-mono text-[10px] transition-all truncate ${
                  config.mode === m.id
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-medium'
                    : 'bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800'
                }`}
              >
                {m.label.split(' ')[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Shape Selector */}
        <div>
          <label className="block text-neutral-400 font-mono text-[10px] uppercase tracking-wider mb-1.5">
            Dot Geometry
          </label>
          <div className="grid grid-cols-4 gap-1">
            {shapes.map(s => (
              <button
                key={s.id}
                onClick={() => onChange({ shape: s.id })}
                className={`py-1.5 px-1 rounded-lg font-mono text-[10px] capitalize transition-all ${
                  config.shape === s.id
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-medium'
                    : 'bg-neutral-900 text-neutral-400 hover:text-white border border-neutral-800'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Reveal Radius Slider */}
        <div>
          <div className="flex justify-between text-[11px] font-mono mb-1">
            <span className="text-neutral-400">Reveal Aperture</span>
            <span className="text-emerald-400">{Math.round(config.revealRadius * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.15"
            max="0.65"
            step="0.01"
            value={config.revealRadius}
            onChange={e => onChange({ revealRadius: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
          />
        </div>

        {/* Dot Density Slider */}
        <div>
          <div className="flex justify-between text-[11px] font-mono mb-1">
            <span className="text-neutral-400">Halftone Density</span>
            <span className="text-emerald-400">{config.dotDensity} cells</span>
          </div>
          <input
            type="range"
            min="35"
            max="130"
            step="1"
            value={config.dotDensity}
            onChange={e => onChange({ dotDensity: parseInt(e.target.value) })}
            className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
          />
        </div>

        {/* Edge Hardness */}
        <div>
          <div className="flex justify-between text-[11px] font-mono mb-1">
            <span className="text-neutral-400">Loupe Edge Softness</span>
            <span className="text-emerald-400">{Math.round((1 - config.edge) * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.3"
            max="0.95"
            step="0.05"
            value={config.edge}
            onChange={e => onChange({ edge: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
          />
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-[10px] text-neutral-400 font-mono">
        <span className="flex items-center gap-1">
          <Eye className="w-3 h-3 text-cyan-400" />
          WebGL 2.0 Shader Engine
        </span>
        <span className="text-emerald-400/80">Active</span>
      </div>
    </div>
  );
}
