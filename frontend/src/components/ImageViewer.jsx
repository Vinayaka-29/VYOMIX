import React, { useState } from 'react';
import { Image, Crosshair, ZoomIn, ZoomOut, RotateCcw, Layers, Eye } from 'lucide-react';

export default function ImageViewer({ activeFileUrl, activeSlotName, visualArtifacts }) {
  const [showOverlay, setShowOverlay] = useState(true);

  const bbox = visualArtifacts?.normalized_bbox; // [xmin, ymin, xmax, ymax] in 0.0 - 1.0
  const isFound = visualArtifacts?.grounding_found !== false;
  const changeOverlayUrl = visualArtifacts?.change_overlay_path;

  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl flex flex-col h-full min-h-[340px]">
      <div className="flex items-center justify-between pb-3 border-b border-space-800 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
            <Crosshair className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              Earth Observation Raster Viewer
              {bbox && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-satcyan-500/20 text-satcyan-400 border border-satcyan-500/30">
                  Grounding Active
                </span>
              )}
            </h3>
            <p className="text-[11px] text-slate-400">
              {activeSlotName ? `Active Slot: ${activeSlotName.toUpperCase()}` : 'Select an image slot to inspect'}
            </p>
          </div>
        </div>

        {bbox && (
          <button
            type="button"
            onClick={() => setShowOverlay(!showOverlay)}
            className={`text-xs px-2.5 py-1 rounded-lg border font-mono flex items-center gap-1.5 transition-colors ${
              showOverlay
                ? 'bg-satcyan-500/20 text-satcyan-300 border-satcyan-500/40'
                : 'bg-space-800 text-slate-400 border-space-700'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>{showOverlay ? 'Overlay On' : 'Overlay Off'}</span>
          </button>
        )}
      </div>

      {/* Main Raster Visualizer */}
      <div className="flex-1 flex flex-col items-center justify-center text-center p-4 border border-dashed border-space-800 rounded-xl bg-space-950/60 relative overflow-hidden min-h-[260px]">
        {activeFileUrl ? (
          <div className="relative max-h-80 w-full flex items-center justify-center overflow-hidden rounded-lg group">
            {/* Base Image */}
            <img
              src={activeFileUrl}
              alt="Satellite Raster"
              className="max-h-80 object-contain rounded border border-space-700 shadow-xl"
            />

            {/* SVG Grounding Bounding Box Overlay (Phase 4) */}
            {bbox && showOverlay && isFound && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
              >
                {/* Glowing Bounding Box */}
                <rect
                  x={`${bbox[0] * 100}%`}
                  y={`${bbox[1] * 100}%`}
                  width={`${(bbox[2] - bbox[0]) * 100}%`}
                  height={`${(bbox[3] - bbox[1]) * 100}%`}
                  fill="rgba(6, 182, 212, 0.2)"
                  stroke="#00f2fe"
                  strokeWidth="0.8"
                  strokeDasharray="2 1"
                  className="animate-pulse"
                />
              </svg>
            )}

            {/* Bounding Box Label Pill */}
            {bbox && showOverlay && isFound && (
              <div
                className="absolute text-[10px] font-mono bg-space-950/90 text-satcyan-300 border border-satcyan-500 px-2 py-0.5 rounded shadow-lg pointer-events-none"
                style={{
                  left: `${bbox[0] * 100}%`,
                  top: `${Math.max(2, bbox[1] * 100 - 6)}%`,
                }}
              >
                Grounded Target
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <div className="w-10 h-10 rounded-full bg-space-800 border border-space-700 flex items-center justify-center text-slate-500 mx-auto">
              <Crosshair className="w-5 h-5 text-satcyan-400" />
            </div>
            <p className="text-xs font-medium text-slate-300">No Raster Active</p>
            <p className="text-[11px] text-slate-500 max-w-xs">
              Upload imagery in the slots to the left to inspect spatial resolution, CRS, and visual grounding.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
