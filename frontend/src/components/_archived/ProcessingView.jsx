import { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import UploadDropzone from './UploadDropzone';
import ProcessingHUD from './ProcessingHUD';
import ResultDashboard from './ResultDashboard';

export default function ProcessingView({ onBackToLanding }) {
  const [stage, setStage] = useState('upload'); // 'upload' | 'processing' | 'result'
  const [, setSelectedFile] = useState(null);
  const [fileDetails, setFileDetails] = useState({
    name: 'chris-grant-wVfgzs0oxRk-unsplash.jpg',
    size: '14.8 MB',
    preview: '/chris-grant-wVfgzs0oxRk-unsplash.jpg'
  });

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    const previewUrl = URL.createObjectURL(file);
    setFileDetails({
      name: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      preview: previewUrl
    });
    setStage('processing');
  };

  const handleSelectPreset = (preset) => {
    setFileDetails({
      name: `${preset.id}_optical_ortho.tif`,
      size: '18.4 MB',
      preview: preset.preview
    });
    setStage('processing');
  };

  const handleProcessingComplete = () => {
    setStage('result');
  };

  const handleReset = () => {
    setStage('upload');
    setSelectedFile(null);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 pt-24 pb-16 px-4 sm:px-6 lg:px-8 relative bg-tech-grid">
      
      {/* Subtle Background Glows */}
      <div className="fixed top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[500px] bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-10 right-10 w-[400px] h-[400px] bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto space-y-8 relative z-10">
        
        {/* Breadcrumb & Navigation Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-neutral-800/80 pb-5">
          <div className="flex items-center gap-3">
            <button
              onClick={onBackToLanding}
              className="p-2 rounded-xl bg-neutral-900 border border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700 transition-colors flex items-center gap-1.5 text-xs font-mono cursor-pointer"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Overview</span>
            </button>

            <div className="h-4 w-px bg-neutral-800" />

            <div className="flex items-center gap-2 font-mono text-xs text-neutral-400">
              <span className="text-neutral-500">Workspace</span>
              <span>/</span>
              <span className="text-white font-medium">Neural Elevation Processing (rDSM)</span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-neutral-400">
            <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-emerald-400">
              ● GPU CLUSTER: ONLINE
            </span>
            <span className="hidden sm:inline text-neutral-500">
              LATENCY: &lt; 140ms
            </span>
          </div>
        </div>

        {/* View States */}
        {stage === 'upload' && (
          <UploadDropzone
            onFileSelected={handleFileSelected}
            onSelectPreset={handleSelectPreset}
          />
        )}

        {stage === 'processing' && (
          <ProcessingHUD
            fileName={fileDetails.name}
            fileSize={fileDetails.size}
            previewUrl={fileDetails.preview}
            onComplete={handleProcessingComplete}
          />
        )}

        {stage === 'result' && (
          <ResultDashboard
            fileName={fileDetails.name}
            onReset={handleReset}
          />
        )}

      </div>
    </div>
  );
}
