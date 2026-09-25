import { X, Cpu, Layers, Eye } from 'lucide-react';

export default function TechSpecsModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-3xl bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl p-6 sm:p-8 text-neutral-200 max-h-[90vh] overflow-y-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-neutral-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-display text-lg font-bold text-white tracking-tight">
                DepthWizard Architecture & Technical Specs
              </h3>
              <p className="text-xs font-mono text-neutral-400">
                Single-View Monocular Optical Elevation Ingestion (SIH-26175)
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-neutral-800 text-neutral-400 hover:text-white hover:bg-neutral-700 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="mt-6 space-y-6 text-sm">
          
          {/* Spec Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 space-y-1.5">
              <div className="text-xs font-mono text-emerald-400 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                <span>NEURAL INFERENCE BACKBONE</span>
              </div>
              <p className="text-xs text-neutral-300">
                Hierarchical Vision Transformer with multi-scale cross-attention for high-frequency terrain slope and building geometry extraction.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 space-y-1.5">
              <div className="text-xs font-mono text-cyan-400 flex items-center gap-1.5">
                <Eye className="w-3.5 h-3.5" />
                <span>INTERACTIVE HALFTONE ENGINE</span>
              </div>
              <p className="text-xs text-neutral-300">
                Custom WebGL 2.0 fragment shader powered by <code className="text-emerald-300">OGL</code>. Renders CMYK/duotone screening with dynamic cursor loupe reveal.
              </p>
            </div>
          </div>

          {/* Details Table */}
          <div className="space-y-2">
            <h4 className="font-mono text-xs uppercase tracking-wider text-neutral-400 font-semibold">
              Pipeline Performance Benchmarks
            </h4>
            <div className="border border-neutral-800 rounded-xl overflow-hidden text-xs font-mono">
              <div className="grid grid-cols-3 bg-neutral-950 p-3 text-neutral-400 border-b border-neutral-800">
                <span>Metric</span>
                <span>Specification</span>
                <span>Standard Baseline</span>
              </div>
              <div className="grid grid-cols-3 p-3 border-b border-neutral-800/60 bg-neutral-900/40">
                <span className="text-neutral-300">Vertical Accuracy (RMSE)</span>
                <span className="text-emerald-400 font-semibold">&lt; 0.94 m</span>
                <span className="text-neutral-500">2.80 m (Stereo)</span>
              </div>
              <div className="grid grid-cols-3 p-3 border-b border-neutral-800/60 bg-neutral-900/40">
                <span className="text-neutral-300">Inference Latency</span>
                <span className="text-emerald-400 font-semibold">142 ms (4K UHD)</span>
                <span className="text-neutral-500">30+ sec (Photogrammetry)</span>
              </div>
              <div className="grid grid-cols-3 p-3 bg-neutral-900/40">
                <span className="text-neutral-300">Input Modality</span>
                <span className="text-emerald-400 font-semibold">Single-view Optical RGB</span>
                <span className="text-neutral-500">Multi-view Stereo Pair</span>
              </div>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-neutral-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-mono text-xs font-semibold cursor-pointer"
          >
            Close Specs
          </button>
        </div>

      </div>
    </div>
  );
}
