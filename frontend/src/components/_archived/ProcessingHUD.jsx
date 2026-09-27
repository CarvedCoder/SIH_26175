import { useState, useEffect } from 'react';
import { 
  Cpu, 
  Terminal, 
  Activity, 
  Scan
} from 'lucide-react';

export default function ProcessingHUD({ 
  fileName = 'satellite_optical_tile_01.tif', 
  fileSize = '14.2 MB', 
  previewUrl = '/chris-grant-wVfgzs0oxRk-unsplash.jpg',
  onComplete 
}) {
  const [progress, setProgress] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const pipelineSteps = [
    {
      label: 'Raster Ingestion & GDAL Tile Decoding',
      log: 'INGEST: Decompressing 3-channel optical RGB raster (3840×2160)...'
    },
    {
      label: 'Feature Extraction & Scale Space',
      log: 'FEATURE_MAP: Constructing multi-scale pyramid representations...'
    },
    {
      label: 'Extracting geometric representations...',
      log: 'Extracting geometric representations... Generating Relative Digital Surface Model (rDSM)...'
    },
    {
      label: 'Generating Relative Digital Surface Model (rDSM)...',
      log: 'INFERENCE: Deep monocular depth neural estimator converged (RMSE: 0.88m)...'
    },
    {
      label: 'Delaunay Triangulation & Mesh Synthesis',
      log: 'SURFACE: Generating 3D point cloud & normal vectors...'
    },
    {
      label: 'Exporting 32-bit Float GeoTIFF & Metadata',
      log: 'COMPLETE: Elevation raster synthesized in 2.8s.'
    }
  ];

  useEffect(() => {
    const startTime = Date.now();
    const duration = 3600; // 3.6 seconds for realistic feeling

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, Math.floor((elapsed / duration) * 100));
      setProgress(pct);

      const stepIdx = Math.min(
        pipelineSteps.length - 1,
        Math.floor((pct / 100) * pipelineSteps.length)
      );
      setCurrentStepIndex(stepIdx);

      if (pct >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          onComplete();
        }, 500);
      }
    }, 40);

    return () => clearInterval(interval);
  }, [onComplete, pipelineSteps.length]);

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-in fade-in duration-300">
      
      {/* Main Processing Box */}
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/80 backdrop-blur-xl p-8 sm:p-10 shadow-2xl relative overflow-hidden">
        
        {/* Top Status Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-neutral-800">
          <div className="flex items-center gap-3">
            <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </div>
            <div>
              <span className="font-mono text-xs uppercase tracking-wider text-neutral-300 font-semibold">
                NEURAL ELEVATION INFERENCE PIPELINE
              </span>
              <p className="text-[11px] font-mono text-neutral-500">
                Target: {fileName} ({fileSize})
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5 text-neutral-400">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>GPU: NVIDIA A100G (94% Core)</span>
            </div>
            <div className="text-emerald-400 font-bold text-sm">
              {progress}%
            </div>
          </div>
        </div>

        {/* Center Scanner Preview & Visualizer */}
        <div className="my-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          
          {/* Raster Preview with Live Laser Scanline */}
          <div className="relative rounded-xl overflow-hidden border border-neutral-700/80 aspect-video md:aspect-4/3 bg-neutral-950 shadow-inner group">
            <img 
              src={previewUrl} 
              alt="Processing Raster"
              className="w-full h-full object-cover filter brightness-90 contrast-110"
            />
            {/* Laser scanning beam */}
            <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_15px_#34d399] animate-scan pointer-events-none" />
            
            {/* Overlay Grid */}
            <div className="absolute inset-0 bg-tech-grid opacity-30 pointer-events-none" />
            
            <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md text-[10px] font-mono text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
              <Scan className="w-3 h-3 text-emerald-400 animate-spin" />
              <span>Scanning Disparity</span>
            </div>
          </div>

          {/* Primary Loading Spinner & Current Task Indicator */}
          <div className="md:col-span-2 space-y-4">
            
            {/* Modern Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-neutral-300 font-medium">Processing Status</span>
                <span className="text-emerald-400 font-semibold">{progress}%</span>
              </div>
              <div className="w-full h-2.5 bg-neutral-800/90 rounded-full overflow-hidden p-0.5 border border-neutral-700/60">
                <div 
                  className="h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 rounded-full transition-all duration-100 ease-out shadow-[0_0_12px_rgba(16,185,129,0.5)]"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Prompt Required Prominent Text Banner */}
            <div className="p-4 rounded-xl bg-neutral-950/80 border border-emerald-500/20 text-left font-mono">
              <div className="flex items-center gap-2 text-xs text-emerald-400 mb-1">
                <Activity className="w-3.5 h-3.5 animate-pulse" />
                <span className="uppercase tracking-wide font-semibold">Active Compute Phase:</span>
              </div>
              <p className="text-sm font-semibold text-white tracking-tight">
                Extracting geometric representations... Generating Relative Digital Surface Model (rDSM)...
              </p>
            </div>

            {/* Stage Indicators */}
            <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-neutral-400">
              <div className={`p-2 rounded-lg border ${progress > 20 ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300' : 'border-neutral-800 bg-neutral-950/50'}`}>
                1. Feature Tensor
              </div>
              <div className={`p-2 rounded-lg border ${progress > 50 ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300' : 'border-neutral-800 bg-neutral-950/50'}`}>
                2. rDSM Inference
              </div>
              <div className={`p-2 rounded-lg border ${progress > 85 ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300' : 'border-neutral-800 bg-neutral-950/50'}`}>
                3. TIN Mesh Export
              </div>
            </div>

          </div>
        </div>

        {/* Terminal Log Console */}
        <div className="rounded-xl bg-neutral-950 border border-neutral-800 p-4 font-mono text-xs text-neutral-300">
          <div className="flex items-center justify-between pb-2 border-b border-neutral-850 mb-3 text-neutral-500 text-[11px]">
            <div className="flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-emerald-400" />
              <span>GEOSPATIAL INFERENCE TERMINAL STREAM</span>
            </div>
            <span>LIVE OUTPUT</span>
          </div>

          <div className="space-y-1.5 max-h-32 overflow-y-auto text-left font-mono text-[11px]">
            {pipelineSteps.slice(0, currentStepIndex + 1).map((step, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <span className="text-emerald-500 select-none">&gt;</span>
                <span className={idx === currentStepIndex ? 'text-emerald-300 font-semibold' : 'text-neutral-400'}>
                  {step.log}
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>

    </div>
  );
}
