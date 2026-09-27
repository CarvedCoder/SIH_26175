import { useState, useRef, useEffect } from 'react';
import { Upload, ArrowLeft, Download, RotateCcw } from 'lucide-react';

export default function UploadProcessingView({ onBackToHome }) {
  const [status, setStatus] = useState('idle'); // 'idle' | 'processing' | 'completed'
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [containerWidth, setContainerWidth] = useState(800);
  const fileInputRef = useRef(null);
  const sliderContainerRef = useRef(null);

  // ResizeObserver for clean split-slider responsiveness
  useEffect(() => {
    if (!sliderContainerRef.current) return;
    const updateWidth = () => {
      if (sliderContainerRef.current) {
        setContainerWidth(sliderContainerRef.current.clientWidth);
      }
    };
    updateWidth();
    const ro = new ResizeObserver(updateWidth);
    ro.observe(sliderContainerRef.current);
    return () => ro.disconnect();
  }, [status]);

  const startProcessing = () => {
    setStatus('processing');
    setProgress(0);

    const startTime = Date.now();
    const duration = 3200; // 3.2s smooth processing

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, Math.floor((elapsed / duration) * 100));
      setProgress(pct);

      if (pct >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          setStatus('completed');
        }, 400);
      }
    }, 40);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      startProcessing();
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      startProcessing();
    }
  };

  const handlePointerMove = (e) => {
    if (!isDraggingSlider || !sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    setSliderPos(Math.max(0, Math.min(100, (x / rect.width) * 100)));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 pt-28 pb-20 px-6 md:px-12 flex flex-col justify-center items-center">
      <div className="w-full max-w-3xl mx-auto flex flex-col items-center">
        
        {/* Top Back Navigation */}
        <div className="w-full flex items-center justify-between mb-8">
          <button
            onClick={onBackToHome}
            className="flex items-center gap-2 text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors duration-300 cursor-pointer focus:outline-none"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Overview</span>
          </button>

          <span className="text-xs text-slate-500 font-medium tracking-wide uppercase">
            Elevation Ingestion Pipeline
          </span>
        </div>

        {/* State 1: Upload Drop Zone */}
        {status === 'idle' && (
          <div className="w-full space-y-6">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`w-full rounded-3xl border-2 border-dashed p-14 sm:p-20 text-center cursor-pointer transition-all duration-300 bg-slate-900/40 flex flex-col items-center justify-center ${
                isDragging
                  ? 'border-slate-400 bg-slate-900/80 scale-[1.005]'
                  : 'border-slate-700/60 hover:border-slate-400'
              }`}
              style={{ WebkitBackdropFilter: 'blur(12px)', backdropFilter: 'blur(12px)' }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff"
                className="hidden"
                onChange={handleFileInputChange}
              />

              {/* Clean Icon */}
              <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-300 mb-6">
                <Upload className="w-5 h-5" />
              </div>

              {/* Required Exact Text from Brief */}
              <h3 className="font-display text-xl sm:text-2xl font-medium text-slate-100 tracking-tight">
                Drag and drop RGB imagery or GeoTIFF
              </h3>

              <p className="mt-2 text-sm text-slate-400">
                or click to select a file from your device
              </p>
            </div>

            {/* Quick Sample Option */}
            <div className="flex items-center justify-center gap-2 pt-2">
              <span className="text-xs text-slate-500">Need a test raster?</span>
              <button
                onClick={startProcessing}
                className="text-xs font-medium text-slate-300 hover:text-white underline underline-offset-4 transition-colors duration-300 cursor-pointer focus:outline-none"
              >
                Use sample optical satellite tile
              </button>
            </div>
          </div>
        )}

        {/* State 2: Simulated Minimal Processing */}
        {status === 'processing' && (
          <div className="w-full rounded-3xl border border-slate-800 bg-slate-900/60 p-12 sm:p-16 text-center space-y-8" style={{ animation: 'fadeIn 0.3s ease-out both', WebkitBackdropFilter: 'blur(12px)', backdropFilter: 'blur(12px)' }}>
            
            {/* Required Status Text from Brief */}
            <div className="space-y-2">
              <h3 className="font-display text-lg sm:text-xl font-medium text-slate-100 tracking-tight">
                Extracting geometric representations... Generating Relative Digital Surface Model (rDSM)...
              </h3>
              <p className="text-xs text-slate-400">
                Resolving depth gradients and surface curvature
              </p>
            </div>

            {/* Smooth Minimal Progress Bar */}
            <div className="max-w-md mx-auto space-y-3">
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-slate-200 rounded-full transition-all duration-100 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="text-right text-xs font-medium text-slate-400">
                {progress}%
              </div>
            </div>

          </div>
        )}

        {/* State 3: Completed Surface Model Preview */}
        {status === 'completed' && (
          <div className="w-full space-y-6" style={{ animation: 'fadeIn 0.3s ease-out both' }}>
            
            {/* Split Comparison Viewer */}
            <div 
              ref={sliderContainerRef}
              onPointerMove={handlePointerMove}
              onPointerDown={() => setIsDraggingSlider(true)}
              onPointerUp={() => setIsDraggingSlider(false)}
              onPointerLeave={() => setIsDraggingSlider(false)}
              className="relative w-full aspect-[16/10] rounded-3xl overflow-hidden border border-slate-800 bg-slate-900 shadow-2xl cursor-ew-resize select-none"
            >
              {/* Background: Elevation DSM */}
              <img 
                src="/Gemini_Generated_Image_final.png" 
                alt="Elevation DSM"
                className="absolute inset-0 w-full h-full object-cover"
              />

              {/* Foreground: Optical Base */}
              <div 
                className="absolute inset-y-0 left-0 overflow-hidden border-r border-white/80"
                style={{ width: `${sliderPos}%` }}
              >
                <img 
                  src="/chris-grant-wVfgzs0oxRk-unsplash.jpg" 
                  alt="Optical Base"
                  className="absolute inset-y-0 left-0 h-full object-cover max-w-none"
                  style={{ width: `${containerWidth}px` }}
                />
                <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-slate-950/70 backdrop-blur-md text-xs text-slate-200">
                  Optical RGB
                </div>
              </div>

              <div className="absolute top-4 right-4 px-3 py-1 rounded-full bg-slate-950/70 backdrop-blur-md text-xs text-slate-200">
                Elevation DSM
              </div>

              {/* Minimal Slider Handle */}
              <div 
                className="absolute inset-y-0 w-6 -ml-3 flex items-center justify-center pointer-events-none"
                style={{ left: `${sliderPos}%` }}
              >
                <div className="w-5 h-5 rounded-full bg-white shadow-lg border border-slate-900" />
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
              <button
                onClick={() => setStatus('idle')}
                className="px-5 py-2.5 rounded-full bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs font-medium transition-colors duration-300 flex items-center gap-2 cursor-pointer focus:outline-none"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Process Another Image</span>
              </button>

              <button
                onClick={() => alert('Exporting 32-bit Float GeoTIFF...')}
                className="px-6 py-2.5 rounded-full bg-slate-100 hover:bg-white text-slate-950 text-xs font-medium transition-all duration-300 shadow-md flex items-center gap-2 cursor-pointer focus:outline-none"
              >
                <Download className="w-3.5 h-3.5 text-slate-700" />
                <span>Download GeoTIFF</span>
              </button>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
