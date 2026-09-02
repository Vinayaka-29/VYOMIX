import React from 'react';
import { 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  FileDown, 
  Radio, 
  Eye, 
  Clock, 
  Terminal,
  ShieldCheck,
  Zap
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function ResultsDashboard({ queryResponse, uploadManifest, error }) {
  if (error) {
    return (
      <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl">
        <div className="flex items-center space-x-2 text-rose-400 font-semibold mb-2 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Execution / Validation Error</span>
        </div>
        <p className="text-xs text-rose-300 font-mono bg-rose-950/40 p-3 rounded-xl border border-rose-900/60 leading-relaxed">
          {error}
        </p>
      </div>
    );
  }

  if (!queryResponse) {
    return (
      <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl flex flex-col items-center justify-center text-center min-h-[260px]">
        <div className="w-10 h-10 rounded-full bg-space-800 border border-space-700 flex items-center justify-center text-slate-500 mb-3">
          <Terminal className="w-5 h-5" />
        </div>
        <p className="text-xs font-medium text-slate-300 mb-1">Awaiting Query Execution</p>
        <p className="text-[11px] text-slate-500 max-w-xs">
          Select imagery and run a geospatial query. The Agentic Controller will auto-route to the optimal specialist models.
        </p>
      </div>
    );
  }

  const { 
    task, 
    answer, 
    confidence, 
    disagreement_flagged, 
    visual_artifacts, 
    query_id 
  } = queryResponse;

  const crossModalEvidence = visual_artifacts?.cross_modal_evidence;
  const isConflict = disagreement_flagged;

  const handleDownloadPdf = () => {
    if (query_id) {
      window.open(`${API_BASE_URL}/report/${query_id}`, '_blank');
    }
  };

  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-space-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Intelligence Synthesis</h3>
            <p className="text-[11px] font-mono text-slate-400 uppercase">Task: {task}</p>
          </div>
        </div>

        {/* Confidence Badge */}
        <div className="flex items-center space-x-2">
          <div className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold border flex items-center gap-1.5 ${
            confidence >= 0.85
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : confidence >= 0.70
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
          }`}>
            <Zap className="w-3.5 h-3.5" />
            <span>{Math.round(confidence * 100)}% Confidence</span>
          </div>
        </div>
      </div>

      {/* Disagreement Warning Banner (Phase 9) */}
      {isConflict && (
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
          <div>
            <p className="font-semibold mb-0.5">Model Disagreement Flagged</p>
            <p className="text-[11px] text-amber-200/80">
              One or more specialist models produced divergent outputs. Confidence score has been penalized, and both findings are surfaced in the trace.
            </p>
          </div>
        </div>
      )}

      {/* Final Synthesized Answer Box */}
      <div className="p-4 rounded-xl bg-space-950/80 border border-satcyan-500/30 relative">
        <span className="text-[10px] font-mono text-satcyan-400 uppercase tracking-wider block mb-1.5 font-semibold">
          Final Synthesized Answer
        </span>
        <p className="text-xs text-slate-200 leading-relaxed font-sans font-medium">
          {answer}
        </p>
      </div>

      {/* Cross-Modal Dual-Branch Breakdown (Phase 7) */}
      {crossModalEvidence && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
          {/* Optical Evidence */}
          <div className="p-3 rounded-xl bg-space-950 border border-cyan-500/20 text-xs">
            <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold mb-1 font-mono">
              <Eye className="w-3.5 h-3.5" />
              <span>Optical Branch Contribution</span>
            </div>
            <p className="text-[11px] text-slate-300 line-clamp-3">
              {crossModalEvidence.optical?.findings || 'Spectral color and vegetation reflectance.'}
            </p>
          </div>

          {/* SAR Evidence */}
          <div className="p-3 rounded-xl bg-space-950 border border-emerald-500/20 text-xs">
            <div className="flex items-center space-x-1.5 text-emerald-400 font-semibold mb-1 font-mono">
              <Radio className="w-3.5 h-3.5" />
              <span>SAR Radar Branch Contribution</span>
            </div>
            <p className="text-[11px] text-slate-300 line-clamp-3">
              {crossModalEvidence.sar?.findings || 'Microwave backscatter, surface roughness & double-bounce.'}
            </p>
          </div>
        </div>
      )}

      {/* PDF Download Button (Phase 9) */}
      <div className="pt-2 border-t border-space-800/80 flex items-center justify-between">
        <span className="text-[11px] font-mono text-slate-400">
          Trace Ledger: Active
        </span>

        <button
          type="button"
          onClick={handleDownloadPdf}
          className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-space-800 hover:bg-space-700 text-slate-200 hover:text-white border border-space-700 text-xs font-medium transition-colors shadow"
        >
          <FileDown className="w-3.5 h-3.5 text-satcyan-400" />
          <span>Download Mission Report (PDF)</span>
        </button>
      </div>
    </div>
  );
}
