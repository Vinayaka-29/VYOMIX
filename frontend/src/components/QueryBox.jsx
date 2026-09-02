import React from 'react';
import { Send, Sparkles, HelpCircle, Loader2, Compass, Layers } from 'lucide-react';

const PRESET_QUERIES = [
  {
    label: 'Single-Image VQA (Phase 3)',
    text: 'What is the dominant land cover type in this satellite image?',
    task: 'vqa',
  },
  {
    label: 'Dense Captioning (Phase 3)',
    text: 'Provide a detailed scene description of the land cover and water bodies.',
    task: 'caption',
  },
  {
    label: 'Text Grounding (Phase 4)',
    text: 'Highlight the water body in this image',
    task: 'ground',
  },
  {
    label: 'Bi-Temporal Change (Phase 6)',
    text: 'What changed between the before and after dates?',
    task: 'change',
  },
  {
    label: 'Optical + SAR Fusion (Phase 7)',
    text: 'Use optical and SAR sensors together to extract land cover and water boundaries.',
    task: 'fusion',
  },
];

export default function QueryBox({
  queryText,
  setQueryText,
  onAnalyze,
  isLoading,
  canAnalyze,
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canAnalyze && !isLoading) {
        onAnalyze();
      }
    }
  };

  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl relative">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-satcyan-500/10 border border-satcyan-500/30 rounded-lg text-satcyan-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <label htmlFor="query-input" className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            Natural Language Geospatial Query
            <span className="text-[10px] font-mono text-satcyan-400 px-2 py-0.5 rounded bg-space-800 border border-space-700">
              Agentic Auto-Routing
            </span>
          </label>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Press Enter to run
        </span>
      </div>

      {/* Textarea */}
      <div className="relative mb-3">
        <textarea
          id="query-input"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the imagery (e.g., 'What land cover is visible?', 'Highlight the water body', 'Detect changes between before and after', 'Extract complementary water bodies from Optical+SAR')..."
          rows={3}
          className="w-full bg-space-950/80 border border-space-700 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-satcyan-500 focus:ring-1 focus:ring-satcyan-500/50 resize-none transition-all duration-150"
        />
      </div>

      {/* Preset Chips (Phase 10 Demo Readiness) */}
      <div className="space-y-1.5 mb-4">
        <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1">
          <Compass className="w-3.5 h-3.5 text-satcyan-400" /> Pipeline Preset Prompts:
        </span>
        <div className="flex flex-wrap gap-1.5">
          {PRESET_QUERIES.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQueryText(preset.text)}
              className="text-[11px] text-slate-300 hover:text-satcyan-300 bg-space-800 hover:bg-space-750 border border-space-700 hover:border-satcyan-500/40 rounded-lg px-2.5 py-1 transition-all"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Submit Button */}
      <div className="flex items-center justify-between pt-2 border-t border-space-800">
        <p className="text-xs text-slate-400 font-mono">
          {!canAnalyze ? 'Select at least 1 image slot to analyze' : 'Auto-Routing ready'}
        </p>

        <button
          id="btn-analyze"
          type="button"
          onClick={onAnalyze}
          disabled={!canAnalyze || isLoading}
          className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 shadow-lg ${
            !canAnalyze || isLoading
              ? 'bg-space-800 text-slate-500 border border-space-700 cursor-not-allowed shadow-none'
              : 'bg-gradient-to-r from-satcyan-500 to-teal-500 text-space-950 hover:from-satcyan-400 hover:to-teal-400 font-semibold shadow-satcyan-500/25 active:scale-[0.99]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Orchestrating Specialist Models...</span>
            </>
          ) : (
            <>
              <span>Execute Geospatial Query</span>
              <Send className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
