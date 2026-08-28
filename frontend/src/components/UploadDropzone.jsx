import { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  Sparkles, 
  ArrowUpRight, 
  Sliders, 
  ChevronDown
} from 'lucide-react';

export default function UploadDropzone({ 
  onFileSelected, 
  onSelectPreset 
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef(null);

  const samplePresets = [
    {
      id: 'rainier',
      name: 'Alpine Topography (Cascade Range)',
      resolution: '3840 × 2160',
      gsd: '0.25m/px',
      type: 'Single-View RGB',
      preview: '/chris-grant-wVfgzs0oxRk-unsplash.jpg'
    },
    {
      id: 'canyon',
      name: 'Sedimentary Escarpment & Ridgeline',
      resolution: '2048 × 1536',
      gsd: '0.50m/px',
      type: 'UAV Orthomosaic',
      preview: '/chris-grant-wVfgzs0oxRk-unsplash.jpg'
    },
    {
      id: 'urban',
      name: 'Coastal Foothills & Volcanic Cone',
      resolution: '4096 × 2160',
      gsd: '0.15m/px',
      type: 'Satellite GeoTIFF',
      preview: '/chris-grant-wVfgzs0oxRk-unsplash.jpg'
    }
  ];

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
      const file = e.dataTransfer.files[0];
      onFileSelected(file);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      onFileSelected(file);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in zoom-in-95 duration-500">
      
      {/* Upload Zone Card */}
      <div 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative group rounded-2xl border-2 border-dashed p-10 sm:p-14 text-center cursor-pointer transition-all duration-300 backdrop-blur-xl ${
          isDragging 
            ? 'border-emerald-400 bg-emerald-950/20 shadow-[0_0_40px_rgba(16,185,129,0.2)] scale-[1.01]' 
            : 'border-neutral-800 hover:border-neutral-700 bg-neutral-900/50 hover:bg-neutral-900/70 shadow-2xl'
        }`}
      >
        {/* Technical Corner Crosshairs */}
        <div className="absolute top-4 left-4 w-3 h-3 border-t-2 border-l-2 border-neutral-700 group-hover:border-emerald-500/60 transition-colors" />
        <div className="absolute top-4 right-4 w-3 h-3 border-t-2 border-r-2 border-neutral-700 group-hover:border-emerald-500/60 transition-colors" />
        <div className="absolute bottom-4 left-4 w-3 h-3 border-b-2 border-l-2 border-neutral-700 group-hover:border-emerald-500/60 transition-colors" />
        <div className="absolute bottom-4 right-4 w-3 h-3 border-b-2 border-r-2 border-neutral-700 group-hover:border-emerald-500/60 transition-colors" />

        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff"
          className="hidden"
          onChange={handleFileInputChange}
        />

        {/* Center Icon */}
        <div className="mx-auto w-16 h-16 rounded-2xl bg-neutral-800/80 border border-neutral-700/80 flex items-center justify-center text-emerald-400 mb-6 group-hover:scale-110 group-hover:border-emerald-500/50 transition-all duration-300 shadow-lg">
          <UploadCloud className="w-8 h-8" />
        </div>

        {/* Required Primary Text from Brief */}
        <h3 className="font-display text-xl sm:text-2xl font-bold text-white tracking-tight">
          Upload non-georeferenced RGB or georeferenced GeoTIFF imagery.
        </h3>

        {/* Subtitle instructions */}
        <p className="mt-3 text-sm text-neutral-400 font-sans max-w-lg mx-auto leading-relaxed">
          Drag and drop optical satellite or UAV imagery here, or browse from local filesystem.
        </p>

        {/* Format Badges */}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          {['GeoTIFF (.tif)', 'RGB PNG', 'High-Res JPEG', 'Auto EPSG'].map((format) => (
            <span 
              key={format}
              className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-neutral-800/60 border border-neutral-700/60 text-neutral-300"
            >
              {format}
            </span>
          ))}
        </div>

        {/* Action Button */}
        <div className="mt-8">
          <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-white font-medium text-xs font-mono border border-neutral-700 group-hover:border-emerald-500/50 transition-all">
            <FileImage className="w-4 h-4 text-emerald-400" />
            <span>Select Imagery File</span>
          </span>
        </div>
      </div>

      {/* Instant Sample Tiles Selection */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono text-xs text-neutral-400">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span className="uppercase tracking-wider">Or Test with Built-in Optical Benchmarks</span>
          </div>
          <span className="text-[11px] font-mono text-neutral-500">1-Click Neural Ingestion</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          {samplePresets.map((preset, index) => (
            <div
              key={preset.id}
              onClick={() => onSelectPreset(preset)}
              className="group p-4 rounded-xl bg-neutral-900/60 hover:bg-neutral-900 border border-neutral-800/80 hover:border-emerald-500/40 cursor-pointer transition-all duration-200 text-left relative overflow-hidden"
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-neutral-700">
                  Preset 0{index + 1}
                </span>
                <ArrowUpRight className="w-4 h-4 text-neutral-500 group-hover:text-emerald-400 transition-colors" />
              </div>
              <h4 className="font-semibold text-sm text-neutral-200 group-hover:text-white transition-colors">
                {preset.name}
              </h4>
              <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-neutral-400 border-t border-neutral-800/60 pt-2">
                <span>{preset.resolution}</span>
                <span className="text-emerald-400/90">{preset.gsd}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Collapsible Advanced Parameters */}
      <div className="bg-neutral-900/30 border border-neutral-800/60 rounded-xl overflow-hidden transition-all">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-xs font-mono text-neutral-400 hover:text-white transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span>PIPELINE HYPERPARAMETERS & METRIC CALIBRATION</span>
          </span>
          <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${showAdvanced ? 'rotate-180' : ''}`} />
        </button>

        {showAdvanced && (
          <div className="px-5 pb-5 pt-2 border-t border-neutral-800/40 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div className="space-y-1.5">
              <label className="text-neutral-400 text-[10px] uppercase">Neural Backbone</label>
              <select className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-200 focus:outline-none focus:border-emerald-500">
                <option>TerraDepth-v2 (Swin-Large Transformer)</option>
                <option>GeoTIN-Dense (Edge-Preserving)</option>
                <option>FastDepth (Real-Time 60fps)</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-neutral-400 text-[10px] uppercase">Metric Alignment</label>
              <select className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-200 focus:outline-none focus:border-emerald-500">
                <option>Auto-Affine Scale & Shift (NAVD88)</option>
                <option>Relative Normalized rDSM [0, 1]</option>
                <option>Ground Control Point (GCP) Anchor</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-neutral-400 text-[10px] uppercase">Export Artifacts</label>
              <select className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-200 focus:outline-none focus:border-emerald-500">
                <option>GeoTIFF Float32 + LAS 1.4 Point Cloud</option>
                <option>GeoTIFF Only</option>
                <option>3D OBJ Surface Mesh</option>
              </select>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
