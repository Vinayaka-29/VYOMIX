import React, { useState, useEffect } from 'react';
import { 
  Satellite, 
  Cpu, 
  Activity, 
  Wifi, 
  WifiOff, 
  RefreshCw,
  ExternalLink,
  Info
} from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import QueryBox from './components/QueryBox';
import ResultsDashboard from './components/ResultsDashboard';
import ExecutionTracePanel from './components/ExecutionTracePanel';
import ImageViewer from './components/ImageViewer';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [files, setFiles] = useState({});
  const [queryText, setQueryText] = useState('What is the dominant land cover type in this satellite image?');
  const [uploadManifest, setUploadManifest] = useState(null);
  const [queryResponse, setQueryResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState({ online: false, checking: true, data: null });
  const [activeSlot, setActiveSlot] = useState(null);
  const [activeFilePreviewUrl, setActiveFilePreviewUrl] = useState(null);

  // Check backend health on mount and periodically
  const checkHealth = async () => {
    try {
      setBackendHealth((prev) => ({ ...prev, checking: true }));
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({ online: true, checking: false, data });
      } else {
        setBackendHealth({ online: false, checking: false, data: null });
      }
    } catch (err) {
      setBackendHealth({ online: false, checking: false, data: null });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Update image preview when files change
  useEffect(() => {
    const slots = Object.keys(files);
    if (slots.length > 0) {
      const firstSlot = slots[0];
      setActiveSlot(firstSlot);
      const file = files[firstSlot];
      if (file && (file.type.startsWith('image/') || file.name.match(/\.(png|jpe?g|webp)$/i))) {
        const url = URL.createObjectURL(file);
        setActiveFilePreviewUrl(url);
        return () => URL.revokeObjectURL(url);
      } else {
        setActiveFilePreviewUrl(null);
      }
    } else {
      setActiveSlot(null);
      setActiveFilePreviewUrl(null);
    }
  }, [files]);

  // Upload and execute query pipeline
  const handleAnalyze = async () => {
    const fileKeys = Object.keys(files);
    if (fileKeys.length === 0) {
      setError('Please select at least one image file to analyze.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setQueryResponse(null);

    try {
      // 1. Upload files
      const formData = new FormData();
      for (const slot of fileKeys) {
        formData.append(slot, files[slot]);
      }

      const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        const errData = await uploadRes.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${uploadRes.status}`);
      }

      const manifest = await uploadRes.json();
      setUploadManifest(manifest);

      // 2. Call query endpoint with upload_id
      const queryRes = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          upload_id: manifest.upload_id,
          query_text: queryText,
        }),
      });

      if (!queryRes.ok) {
        const errData = await queryRes.json().catch(() => ({}));
        throw new Error(errData.detail || `Query processing failed with status ${queryRes.status}`);
      }

      const qResponse = await queryRes.json();
      setQueryResponse(qResponse);
    } catch (err) {
      console.error('Workflow error:', err);
      setError(err.message || 'An unexpected error occurred during execution.');
    } finally {
      setIsLoading(false);
    }
  };

  const totalFilesSelected = Object.keys(files).length;

  return (
    <div className="min-h-screen bg-space-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header Navigation */}
      <header className="border-b border-space-800/80 bg-space-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-satcyan-500 to-teal-500 flex items-center justify-center shadow-lg shadow-satcyan-500/20 text-space-950 font-bold">
              <Satellite className="w-5 h-5 text-space-950" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  SatQuery AI
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-satcyan-500/10 border border-satcyan-500/30 text-satcyan-400">
                  SIH 2026 • PS 26167
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-800 border border-space-700 text-slate-300">
                  Team Vyomix
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Agentic Multi-Modal Remote Sensing VLM Platform (ISRO / SAC)
              </p>
            </div>
          </div>

          {/* Backend Status indicator */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-full bg-space-950 border border-space-800">
              {backendHealth.online ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-emerald-400">API Connected :8000</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-rose-400" />
                  <span className="text-rose-400">API Offline</span>
                </>
              )}
              <button
                onClick={checkHealth}
                title="Refresh backend status"
                className="text-slate-500 hover:text-slate-300 ml-1 transition-colors"
              >
                <RefreshCw className={`w-3 h-3 ${backendHealth.checking ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <div className="hidden sm:flex items-center space-x-2 text-xs text-slate-400 bg-space-800/80 px-3 py-1.5 rounded-lg border border-space-700">
              <Cpu className="w-3.5 h-3.5 text-satcyan-400" />
              <span>Phase 1: Scaffold & Plumbing</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Banner notice */}
        <div className="mb-6 p-3 rounded-xl bg-space-900/60 border border-satcyan-500/20 flex items-center justify-between text-xs text-slate-300">
          <div className="flex items-center space-x-2">
            <Info className="w-4 h-4 text-satcyan-400 shrink-0" />
            <span>
              <strong>Phase 1 Active:</strong> Upload plumbing, multi-slot staging, and backend query echo are active.
              Docker-ready with Postgres+PostGIS configuration.
            </span>
          </div>
          <span className="font-mono text-[11px] text-satcyan-400 hidden md:inline">
            FastAPI + React + Vite + Tailwind
          </span>
        </div>

        {/* Two-Column Responsive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Upload Panel + Query Box (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            <UploadPanel
              files={files}
              setFiles={setFiles}
              uploadManifest={uploadManifest}
              isUploading={isLoading}
            />

            <QueryBox
              queryText={queryText}
              setQueryText={setQueryText}
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              canAnalyze={totalFilesSelected > 0}
            />
          </div>

          {/* Right Column: Imagery Viewer + Results Dashboard + Execution Trace (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <ImageViewer
              activeFileUrl={activeFilePreviewUrl}
              activeSlotName={activeSlot}
            />

            <ResultsDashboard
              queryResponse={queryResponse}
              uploadManifest={uploadManifest}
              error={error}
            />

            <ExecutionTracePanel />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-space-800/80 bg-space-900/40 py-4 text-center text-xs text-slate-500">
        <p>SatQuery AI • Built for Smart India Hackathon (SIH) 2026 • Problem Statement 26167 (ISRO/SAC) • Team Vyomix</p>
      </footer>
    </div>
  );
}
