import { useState, useRef, useEffect } from 'react';
import { 
  Download, 
  RotateCcw, 
  Sliders, 
  Activity, 
  FileCheck2,
  Check
} from 'lucide-react';

export default function ResultDashboard({ 
  fileName = 'satellite_optical_tile_01.tif', 
  onReset 
}) {
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [colorRamp, setColorRamp] = useState('turbo');
  const [cursorPos, setCursorPos] = useState({ x: 50, y: 50 });
  const [isHoveringImg, setIsHoveringImg] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(null);
  const [containerWidth, setContainerWidth] = useState(800);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    updateWidth();
    const ro = new ResizeObserver(updateWidth);
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    
    setCursorPos({
      x: Math.round((x / rect.width) * 100),
      y: Math.round((y / rect.height) * 100)
    });

    if (isDragging) {
      const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPos(pct);
    }
  };

  const handleDownload = (format) => {
    setDownloadSuccess(format);
    setTimeout(() => {
      setDownloadSuccess(null);
    }, 2500);
  };

  // Simulated elevation based on cursor position
  const estimatedElevation = Math.round(1840 + (cursorPos.x * 12.4) + (cursorPos.y * 8.6));
  const estimatedSlope = (8.4 + (cursorPos.x % 15) * 1.8).toFixed(1);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 animate-in fade-in duration-500">
      
      {/* Top Banner & Action Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-neutral-900/80 backdrop-blur-xl border border-neutral-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <FileCheck2 className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display font-bold text-base text-white tracking-tight">
                Digital Surface Model (rDSM) Generated
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                100% CONVERGED
              </span>
            </div>
            <p className="text-xs font-mono text-neutral-400">
              Source: {fileName} &bull; Output: 3840×2160 Float32 GeoTIFF
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={onReset}
            className="text-xs font-mono px-4 py-2.5 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 flex items-center gap-2 transition-all cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Process Another Tile</span>
          </button>

          <button
            onClick={() => handleDownload('GeoTIFF')}
            className="text-xs font-mono px-4 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold flex items-center gap-2 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] cursor-pointer"
          >
            {downloadSuccess === 'GeoTIFF' ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Exported!</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Download GeoTIFF</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Dual-Layer Visualizer & Split View */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Interactive Split View Canvas (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div 
            ref={containerRef}
            onPointerMove={handlePointerMove}
            onPointerEnter={() => setIsHoveringImg(true)}
            onPointerLeave={() => {
              setIsHoveringImg(false);
              setIsDragging(false);
            }}
            onPointerDown={() => setIsDragging(true)}
            onPointerUp={() => setIsDragging(false)}
            className="relative w-full aspect-16/10 rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-950 shadow-2xl cursor-ew-resize select-none"
          >
            {/* Background: Elevation DSM (Layer 2) */}
            <img 
              src="/Gemini_Generated_Image_final.png" 
              alt="Elevation DSM Model"
              className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Foreground: Optical Base (Layer 1) clipped to slider position */}
            <div 
              className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-white shadow-[0_0_15px_rgba(255,255,255,0.7)]"
              style={{ width: `${sliderPos}%` }}
            >
              <img 
                src="/chris-grant-wVfgzs0oxRk-unsplash.jpg" 
                alt="Optical RGB Input"
                className="absolute inset-y-0 left-0 h-full object-cover max-w-none"
                style={{ width: `${containerWidth}px` }}
              />
              <div className="absolute top-4 left-4 px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-md text-[11px] font-mono text-white border border-white/20">
                OPTICAL RGB BASE
              </div>
            </div>

            {/* Right Tag for DSM */}
            <div className="absolute top-4 right-4 px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-md text-[11px] font-mono text-emerald-300 border border-emerald-500/30">
              ELEVATION DSM REVEAL
            </div>

            {/* Interactive Split Handle */}
            <div 
              className="absolute inset-y-0 w-8 -ml-4 flex items-center justify-center pointer-events-none"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="w-8 h-8 rounded-full bg-white text-neutral-900 shadow-xl flex items-center justify-center text-xs font-bold border-2 border-neutral-900">
                ↔
              </div>
            </div>

            {/* Dynamic Cursor Inspector Tooltip */}
            {isHoveringImg && (
              <div 
                className="absolute pointer-events-none z-30 bg-neutral-950/90 border border-emerald-500/40 rounded-lg p-2 text-[10px] font-mono text-neutral-200 shadow-2xl backdrop-blur-md"
                style={{ 
                  left: `${Math.min(80, Math.max(5, cursorPos.x))}%`, 
                  top: `${Math.min(80, Math.max(10, cursorPos.y))}%` 
                }}
              >
                <div className="text-emerald-400 font-semibold">ELEV: +{estimatedElevation}m</div>
                <div className="text-neutral-400">Slope: {estimatedSlope}° &bull; GSD: 0.25m</div>
              </div>
            )}
          </div>

          {/* Slider Position Legend & Instructions */}
          <div className="flex items-center justify-between text-xs font-mono text-neutral-400 px-2">
            <span>&larr; Drag slider left/right to compare Optical RGB vs. Elevation Surface Model &rarr;</span>
            <span className="text-emerald-400">Ratio: {Math.round(sliderPos)}% RGB</span>
          </div>

          {/* Elevation Cross-Section Curve Chart */}
          <div className="bg-neutral-900/60 border border-neutral-800/80 rounded-xl p-4">
            <div className="flex items-center justify-between text-xs font-mono mb-3">
              <span className="text-neutral-300 flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                Cross-Section Elevation Profile (West &rarr; East Transect)
              </span>
              <span className="text-neutral-500">Z-Range: 1,840m - 3,420m</span>
            </div>
            
            {/* SVG Elevation Profile Wave */}
            <div className="h-20 w-full relative">
              <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 400 80">
                <defs>
                  <linearGradient id="elevGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                  </linearGradient>
                </defs>
                <path
                  d="M 0,65 Q 40,55 80,42 T 160,20 T 240,45 T 320,12 T 400,35 L 400,80 L 0,80 Z"
                  fill="url(#elevGrad)"
                />
                <path
                  d="M 0,65 Q 40,55 80,42 T 160,20 T 240,45 T 320,12 T 400,35"
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="2.5"
                />
                {/* Active marker on graph */}
                <circle 
                  cx={`${(sliderPos / 100) * 400}`} 
                  cy="30" 
                  r="4" 
                  fill="#ffffff" 
                  stroke="#10b981" 
                  strokeWidth="2" 
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Right Metric & Telemetry Panel (1 col) */}
        <div className="space-y-4">
          
          {/* Geospatial Metrics */}
          <div className="bg-neutral-900/70 border border-neutral-800 rounded-xl p-5 space-y-4">
            <h4 className="font-mono text-xs uppercase tracking-wider text-neutral-400 font-semibold border-b border-neutral-800 pb-2 flex items-center justify-between">
              <span>Telemetry Analysis</span>
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
            </h4>

            <div className="space-y-3 font-mono text-xs">
              <div className="flex justify-between items-center">
                <span className="text-neutral-400">Peak Elevation:</span>
                <span className="text-white font-bold">+3,428.6 m</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400">Base Elevation:</span>
                <span className="text-white font-bold">+1,842.1 m</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400">Relative Relief:</span>
                <span className="text-emerald-400 font-bold">1,586.5 m</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400">Vertical RMSE:</span>
                <span className="text-cyan-400 font-bold">0.82 m</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-400">Point Cloud Count:</span>
                <span className="text-neutral-200">4,280,000 pts</span>
              </div>
            </div>
          </div>

          {/* Color Ramp Palette */}
          <div className="bg-neutral-900/70 border border-neutral-800 rounded-xl p-5 space-y-3">
            <h4 className="font-mono text-xs uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-emerald-400" />
              <span>Elevation Shading</span>
            </h4>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {[
                { id: 'turbo', label: 'Turbo / Thermal' },
                { id: 'viridis', label: 'Viridis' },
                { id: 'topo', label: 'Topo Relief' },
                { id: 'hillshade', label: 'Analytical Hillshade' }
              ].map(ramp => (
                <button
                  key={ramp.id}
                  onClick={() => setColorRamp(ramp.id)}
                  className={`p-2 rounded-lg text-left text-[11px] transition-all border ${
                    colorRamp === ramp.id
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-semibold'
                      : 'bg-neutral-950 text-neutral-400 border-neutral-800 hover:text-white'
                  }`}
                >
                  {ramp.label}
                </button>
              ))}
            </div>
          </div>

          {/* Export Artifacts List */}
          <div className="bg-neutral-900/70 border border-neutral-800 rounded-xl p-5 space-y-3">
            <h4 className="font-mono text-xs uppercase tracking-wider text-neutral-400 font-semibold flex items-center gap-2">
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export Pipelines</span>
            </h4>

            <div className="space-y-2 text-xs font-mono">
              <button
                onClick={() => handleDownload('GeoTIFF')}
                className="w-full p-2.5 rounded-lg bg-neutral-950 hover:bg-neutral-800 border border-neutral-800 text-left flex items-center justify-between text-neutral-300 hover:text-white transition-colors"
              >
                <span>32-bit Float GeoTIFF</span>
                <span className="text-[10px] text-emerald-400">14.2 MB</span>
              </button>

              <button
                onClick={() => handleDownload('LAS')}
                className="w-full p-2.5 rounded-lg bg-neutral-950 hover:bg-neutral-800 border border-neutral-800 text-left flex items-center justify-between text-neutral-300 hover:text-white transition-colors"
              >
                <span>LAS 1.4 Point Cloud</span>
                <span className="text-[10px] text-cyan-400">68.5 MB</span>
              </button>

              <button
                onClick={() => handleDownload('OBJ')}
                className="w-full p-2.5 rounded-lg bg-neutral-950 hover:bg-neutral-800 border border-neutral-800 text-left flex items-center justify-between text-neutral-300 hover:text-white transition-colors"
              >
                <span>3D OBJ Surface Mesh</span>
                <span className="text-[10px] text-teal-400">42.1 MB</span>
              </button>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
