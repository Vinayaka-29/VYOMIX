import { useEffect, useState } from "react";
import { Radio, RadioTower, Cloud, Cpu, CheckCircle2, Sparkles } from "lucide-react";
import { getApiBaseUrl, setApiBaseUrl, checkHealth, getVlmStatus, configureVlm, type VLMStatus } from "@/lib/satquery";

export function BackendSettings({ onChange }: { onChange?: (url: string) => void }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [remoteUrl, setRemoteUrl] = useState("");
  const [online, setOnline] = useState<boolean | null>(null);
  const [vlmStatus, setVlmStatus] = useState<VLMStatus | null>(null);
  const [saving, setSaving] = useState(false);

  const refreshStatus = async () => {
    const isUp = await checkHealth();
    setOnline(isUp);
    if (isUp) {
      const vlm = await getVlmStatus();
      setVlmStatus(vlm);
      if (vlm?.remote_url) {
        setRemoteUrl(vlm.remote_url);
      }
    }
  };

  useEffect(() => {
    setValue(getApiBaseUrl());
    void refreshStatus();
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      setApiBaseUrl(value);
      onChange?.(value);
      const isUp = await checkHealth();
      setOnline(isUp);
      if (isUp) {
        const updated = await configureVlm(remoteUrl.trim());
        setVlmStatus(updated);
      }
      setOpen(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <button
          onClick={() => {
            setOpen((o) => !o);
            if (!open) void refreshStatus();
          }}
          className="flex items-center gap-2.5 rounded-full border border-slate-300 bg-white px-4 py-2 text-xs sm:text-sm font-semibold text-slate-800 shadow-2xs transition-all hover:bg-slate-50"
        >
          {online ? (
            <RadioTower className="size-4 text-emerald-600 font-bold" />
          ) : (
            <Radio className="size-4 text-amber-600 font-bold" />
          )}
          <span className="font-mono uppercase tracking-wider">
            {online === null ? "Checking..." : online ? "API Online" : "API Offline"}
          </span>

          {online && vlmStatus && (
            <span
              className={`ml-1.5 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-bold ${
                vlmStatus.is_remote_configured
                  ? "bg-purple-100 text-purple-800 border border-purple-300"
                  : "bg-sky-100 text-sky-800 border border-sky-300"
              }`}
            >
              {vlmStatus.is_remote_configured ? (
                <>
                  <Cloud className="size-3 text-purple-600" />
                  GeoChat Cloud
                </>
              ) : (
                <>
                  <Cpu className="size-3 text-sky-600" />
                  Local RS-VLM
                </>
              )}
            </span>
          )}
        </button>
      </div>

      {open && (
        <div className="panel absolute right-0 z-30 mt-2.5 w-96 p-5 shadow-2xl rounded-2xl border border-slate-200 bg-white">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
              <Sparkles className="size-4 text-sky-600" />
              Engine & Connection Settings
            </h3>
            <span className="text-[11px] font-mono text-slate-400">SIH 26167</span>
          </div>

          <div className="mt-4 space-y-4">
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-slate-600 block mb-1">
                Backend API Address
              </label>
              <input
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="http://localhost:8000"
                className="w-full rounded-xl border border-slate-300 bg-slate-50/50 p-2.5 font-mono text-xs font-medium text-slate-900 outline-none focus:border-slate-900 focus:bg-white shadow-2xs"
              />
              <p className="mt-1 text-[11px] text-slate-500">
                Primary API server running locally on <span className="font-mono font-semibold">http://localhost:8000</span>
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-bold uppercase tracking-wider text-slate-600">
                  Cloud GeoChat-7B URL (Kaggle / Gradio Live)
                </label>
                {remoteUrl && (
                  <button
                    onClick={() => setRemoteUrl("")}
                    className="text-[10px] text-rose-600 hover:underline font-semibold"
                  >
                    Clear (Switch to Local)
                  </button>
                )}
              </div>
              <input
                value={remoteUrl}
                onChange={(e) => setRemoteUrl(e.target.value)}
                placeholder="https://xxxx.gradio.live or MBZUAI/geochat-7B"
                className="w-full rounded-xl border border-slate-300 bg-slate-50/50 p-2.5 font-mono text-xs font-medium text-slate-900 outline-none focus:border-purple-600 focus:bg-white shadow-2xs"
              />
              <p className="mt-1 text-[11px] text-slate-500">
                Paste your active Kaggle Gradio Live link (<span className="font-mono text-slate-700">*.gradio.live</span>) or Hugging Face Space ID.
              </p>
            </div>

            {/* Current Telemetry Box */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-slate-500 font-medium">Active Engine:</span>
                <span className="font-semibold text-slate-800">
                  {vlmStatus?.active_engine || (online ? "SatQuery Local RS-VLM" : "Unavailable")}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500 font-medium">Hardware / Mode:</span>
                <span className="font-semibold text-slate-700">
                  {vlmStatus ? `${vlmStatus.device.toUpperCase()} (LoRA: ${vlmStatus.is_lora_adapted ? "Active" : "Off"})` : "Local Host"}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-500 font-medium">Fallback Ready:</span>
                <span className="font-semibold text-emerald-700 flex items-center gap-1">
                  <CheckCircle2 className="size-3 text-emerald-600" />
                  Rich EO Reasoning
                </span>
              </div>
            </div>
          </div>

          <div className="mt-5 flex justify-end gap-2.5 border-t border-slate-100 pt-3">
            <button
              onClick={() => setOpen(false)}
              className="rounded-xl px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-900 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => void save()}
              disabled={saving}
              className="rounded-xl bg-slate-900 px-4 py-2 text-xs font-semibold text-white shadow-md hover:bg-slate-800 disabled:opacity-50"
            >
              {saving ? "Applying..." : "Save & Connect"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


