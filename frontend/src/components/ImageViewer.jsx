import React from 'react';
import { Image, Crosshair } from 'lucide-react';

export default function ImageViewer({ activeFileUrl, activeSlotName }) {
  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl flex flex-col h-full min-h-[300px]">
      <div className="flex items-center justify-between pb-3 border-b border-space-800 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
            <Image className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Satellite Imagery Viewer</h3>
            <p className="text-[11px] text-slate-400">
              {activeSlotName ? `Active Slot: ${activeSlotName.toUpperCase()}` : 'No raster active'}
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-space-800 border border-space-700 text-slate-500">
          Overlay Grounding: Phase 4
        </span>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-space-800 rounded-xl bg-space-950/40 relative">
        {activeFileUrl ? (
          <div className="relative max-h-72 w-full flex items-center justify-center overflow-hidden rounded-lg">
            <img
              src={activeFileUrl}
              alt="Active raster preview"
              className="max-h-72 object-contain rounded border border-space-700"
            />
          </div>
        ) : (
          <>
            <div className="w-9 h-9 rounded-full bg-space-800 border border-space-700 flex items-center justify-center text-slate-500 mb-2">
              <Crosshair className="w-4 h-4" />
            </div>
            <p className="text-xs font-medium text-slate-400 mb-1">Raster & Grounding Viewer</p>
            <p className="text-[11px] text-slate-500 max-w-xs">
              Visual grounding overlays (Phase 4) and bi-temporal change maps (Phase 6) will render here.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
