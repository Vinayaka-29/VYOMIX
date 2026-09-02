import React, { useState } from 'react';
import { 
  GitBranch, 
  Clock, 
  ShieldCheck, 
  ChevronDown, 
  ChevronUp, 
  Cpu, 
  AlertCircle,
  Terminal,
  CheckCircle2
} from 'lucide-react';

export default function ExecutionTracePanel({ executionTrace }) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!executionTrace) {
    return (
      <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl flex flex-col">
        <div className="flex items-center justify-between pb-3 border-b border-space-800 mb-3">
          <div className="flex items-center space-x-2">
            <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
              <GitBranch className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Auditable Execution Trace</h3>
              <p className="text-[11px] text-slate-400">Deterministic model call ledger (PS 26167)</p>
            </div>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-space-800 border border-space-700 text-slate-400">
            Awaiting Query
          </span>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-space-800 rounded-xl">
          <Clock className="w-5 h-5 text-slate-500 mb-2" />
          <p className="text-xs font-medium text-slate-400 mb-1">Trace Ledger Idle</p>
          <p className="text-[11px] text-slate-500 max-w-xs">
            Observable routing decisions, model latencies, and parameter ledgers will stream here upon query execution.
          </p>
        </div>
      </div>
    );
  }

  const {
    task,
    models_called = [],
    steps = [],
    parameters = {},
    final_confidence,
    disagreement_flagged,
    conflicts_detected = [],
  } = executionTrace;

  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl">
      {/* Header with expand toggle */}
      <div 
        className="flex items-center justify-between pb-3 border-b border-space-800 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
            <GitBranch className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              Auditable Execution Trace
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-space-800 border border-space-700 text-slate-300">
                {steps.length} Steps
              </span>
            </h3>
            <p className="text-[11px] font-mono text-slate-400">
              Task: {task} • {parameters.total_latency_ms || 0}ms total
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {disagreement_flagged ? (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" /> Disagreement Flagged
            </span>
          ) : (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" /> Consensus
            </span>
          )}
          <button className="text-slate-400 hover:text-slate-200">
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded Trace Body */}
      {isExpanded && (
        <div className="mt-4 space-y-3">
          {/* Step-by-Step Chronological Ledger */}
          <div className="space-y-2.5">
            {steps.map((step, idx) => (
              <div 
                key={step.step_id || idx}
                className="p-3 rounded-xl bg-space-950 border border-space-800/80 hover:border-space-700 transition-colors"
              >
                <div className="flex items-center justify-between mb-1 text-xs">
                  <span className="font-semibold text-slate-200 flex items-center gap-1.5 font-mono">
                    <span className="w-4 h-4 rounded-full bg-satcyan-500/10 text-satcyan-400 flex items-center justify-center text-[10px] border border-satcyan-500/30">
                      {idx + 1}
                    </span>
                    {step.model}
                  </span>
                  <div className="flex items-center space-x-2 font-mono text-[10px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-500" />
                      {step.latency_ms}ms
                    </span>
                    <span className="text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                      {Math.round(step.confidence * 100)}%
                    </span>
                  </div>
                </div>

                <p className="text-[11px] text-slate-400 font-mono line-clamp-2 bg-space-900/60 p-2 rounded border border-space-850">
                  {step.output_summary}
                </p>
              </div>
            ))}
          </div>

          {/* Compliance & Audit Footer */}
          <div className="pt-2 flex items-center justify-between text-[10px] font-mono text-slate-500 border-t border-space-800">
            <span>Standard: ISRO/SAC PS 26167 Observable Execution</span>
            <span>Raw CoT Suppressed (Strict Compliance)</span>
          </div>
        </div>
      )}
    </div>
  );
}
