import React from 'react';
import { 
  Eye, 
  Radio, 
  History, 
  CalendarCheck, 
  Upload, 
  X, 
  CheckCircle2, 
  FileCode2,
  Layers
} from 'lucide-react';

const SLOTS = [
  {
    id: 'optical',
    name: 'Optical Imagery',
    tag: 'VIS / RGB / VNIR',
    description: 'High-res optical sensor (e.g., Sentinel-2, Landsat, Cartosat)',
    icon: Eye,
    borderColor: 'hover:border-cyan-500/60 focus-within:border-cyan-500',
    activeBadge: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
  },
  {
    id: 'sar',
    name: 'SAR Imagery',
    tag: 'C-Band / L-Band Radar',
    description: 'Synthetic Aperture Radar (e.g., Sentinel-1 GRD, NISAR, RISAT)',
    icon: Radio,
    borderColor: 'hover:border-emerald-500/60 focus-within:border-emerald-500',
    activeBadge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  },
  {
    id: 'before',
    name: 'Before Image (T₀)',
    tag: 'Bi-Temporal Baseline',
    description: 'Pre-event or historical reference image for change analysis',
    icon: History,
    borderColor: 'hover:border-amber-500/60 focus-within:border-amber-500',
    activeBadge: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  },
  {
    id: 'after',
    name: 'After Image (T₁)',
    tag: 'Bi-Temporal Target',
    description: 'Post-event or current observation image for change analysis',
    icon: CalendarCheck,
    borderColor: 'hover:border-purple-500/60 focus-within:border-purple-500',
    activeBadge: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
  },
];

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function UploadPanel({
  files,
  setFiles,
  uploadManifest,
  isUploading,
  onUploadAll,
}) {
  const handleFileChange = (slotId, e) => {
    const file = e.target.files?.[0];
    if (file) {
      setFiles((prev) => ({
        ...prev,
        [slotId]: file,
      }));
    }
  };

  const handleClearSlot = (slotId, e) => {
    e.stopPropagation();
    setFiles((prev) => {
      const next = { ...prev };
      delete next[slotId];
      return next;
    });
  };

  const totalSelected = Object.keys(files).length;

  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl relative overflow-hidden">
      {/* Background radial accent */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-satcyan-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 bg-space-800 border border-space-700 rounded-xl text-satcyan-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              Sensor Imagery Input Slots
              <span className="text-xs font-mono font-normal text-slate-400 px-2 py-0.5 rounded-full bg-space-800 border border-space-700">
                {totalSelected}/4 Selected
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Upload GeoTIFF/TIFF rasters or benchmark images for multi-modal remote sensing
            </p>
          </div>
        </div>

        {uploadManifest && (
          <div className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full font-mono">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Uploaded: {uploadManifest.upload_id.slice(0, 8)}...</span>
          </div>
        )}
      </div>

      {/* 4 Named Image Slots Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
        {SLOTS.map((slot) => {
          const Icon = slot.icon;
          const selectedFile = files[slot.id];
          const isUploaded = uploadManifest?.files?.[slot.id];

          return (
            <div
              key={slot.id}
              className={`relative group border transition-all duration-200 rounded-xl p-3.5 bg-space-950/60 ${
                selectedFile
                  ? 'border-satcyan-500/50 bg-space-850/60 shadow-lg shadow-satcyan-500/5'
                  : `border-space-700/50 ${slot.borderColor}`
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <div className={`p-1.5 rounded-lg border bg-space-800/80 ${selectedFile ? 'text-satcyan-400 border-satcyan-500/40' : 'text-slate-400 border-space-700'}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-slate-200">{slot.name}</h3>
                    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border inline-block ${slot.activeBadge}`}>
                      {slot.tag}
                    </span>
                  </div>
                </div>

                {selectedFile && (
                  <button
                    onClick={(e) => handleClearSlot(slot.id, e)}
                    className="p-1 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-md transition-colors"
                    title="Remove file"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              <p className="text-[11px] text-slate-400 mb-3 line-clamp-1">
                {slot.description}
              </p>

              {/* Upload Drop/Picker Area */}
              <label className="cursor-pointer block">
                <input
                  type="file"
                  id={`slot-${slot.id}`}
                  className="hidden"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  onChange={(e) => handleFileChange(slot.id, e)}
                />
                {selectedFile ? (
                  <div className="flex items-center justify-between p-2 rounded-lg bg-space-900 border border-space-700 text-xs">
                    <div className="flex items-center space-x-2 min-w-0">
                      <FileCode2 className="w-4 h-4 text-satcyan-400 shrink-0" />
                      <div className="min-w-0 truncate">
                        <p className="font-mono text-slate-200 truncate">{selectedFile.name}</p>
                        <p className="text-[10px] text-slate-400">{formatBytes(selectedFile.size)}</p>
                      </div>
                    </div>
                    {isUploaded && (
                      <span className="shrink-0 text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 font-mono">
                        Saved
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="border border-dashed border-space-700 group-hover:border-slate-500 rounded-lg py-2.5 px-3 flex items-center justify-center space-x-2 text-xs text-slate-400 hover:text-slate-200 transition-colors">
                    <Upload className="w-3.5 h-3.5" />
                    <span>Choose file or drag here</span>
                  </div>
                )}
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
}
