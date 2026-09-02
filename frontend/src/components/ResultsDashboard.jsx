import React from 'react';
import { Terminal, Activity, FileCheck, CheckCircle, AlertCircle } from 'lucide-react';

export default function ResultsDashboard({ queryResponse, uploadManifest, error }) {
  return (
    <div className="bg-space-900/90 border border-space-700/60 rounded-2xl p-5 backdrop-blur-xl shadow-2xl flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 border-b border-space-800 mb-4">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-space-800 border border-space-700 rounded-lg text-satcyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Inference & Results Dashboard</h3>
            <p className="text-[11px] text-slate-400">Phase 1: Backend plumbing verification</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-space-800 border border-space-700 text-slate-400">
          VLM Baseline: Phase 3
        </span>
      </div>

      {error ? (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <p className="font-semibold mb-1">Communication Error</p>
            <p className="font-mono">{error}</p>
          </div>
        </div>
      ) : queryResponse ? (
        <div className="space-y-3">
          <div className="p-3.5 rounded-xl bg-space-950 border border-satcyan-500/30">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-satcyan-400 font-medium flex items-center gap-1.5">
                <CheckCircle className="w-3.5 h-3.5" /> Query Acknowledged
              </span>
              <span className="font-mono text-[10px] text-slate-500">
                Task: {queryResponse.task}
              </span>
            </div>
            <p className="text-xs text-slate-300 font-mono bg-space-900/90 p-2.5 rounded-lg border border-space-800">
              {queryResponse.message || "Phase 1: Query received and stub executed successfully."}
            </p>
          </div>

          <div className="bg-space-950 border border-space-800 rounded-xl p-3">
            <div className="flex items-center space-x-2 text-xs text-slate-400 mb-2 font-mono">
              <Terminal className="w-3.5 h-3.5 text-satcyan-400" />
              <span>Response Payload</span>
            </div>
            <pre className="text-[11px] font-mono text-slate-300 overflow-x-auto p-2 bg-space-900 rounded border border-space-800/80 max-h-44">
              {JSON.stringify(queryResponse, null, 2)}
            </pre>
          </div>
        </div>
      ) : uploadManifest ? (
        <div className="p-4 rounded-xl bg-space-950/80 border border-space-800">
          <div className="flex items-center space-x-2 text-xs text-emerald-400 font-mono mb-2">
            <FileCheck className="w-4 h-4" />
            <span>Files Stored to Backend: {uploadManifest.upload_id}</span>
          </div>
          <p className="text-xs text-slate-400 mb-3">
            Manifest received. Click "Analyze Satellite Imagery" to verify /query echo endpoint.
          </p>
          <pre className="text-[11px] font-mono text-slate-300 overflow-x-auto p-2.5 bg-space-900 rounded border border-space-800 max-h-40">
            {JSON.stringify(uploadManifest, null, 2)}
          </pre>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-space-800 rounded-xl">
          <div className="w-10 h-10 rounded-full bg-space-800 border border-space-700 flex items-center justify-center text-slate-500 mb-3">
            <Terminal className="w-5 h-5" />
          </div>
          <p className="text-xs font-medium text-slate-300 mb-1">Awaiting Upload & Execution</p>
          <p className="text-[11px] text-slate-500 max-w-xs">
            Select files in the upload slots and click Analyze to verify end-to-end backend plumbing.
          </p>
        </div>
      )}
    </div>
  );
}
