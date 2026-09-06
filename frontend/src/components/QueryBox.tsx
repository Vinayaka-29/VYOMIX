import { Send, Loader2 } from "lucide-react";

const EXAMPLES = [
  "Describe this scene",
  "Where are the buildings?",
  "Is there a water body?",
  "What changed between these images?",
  "Compare the optical and SAR imagery",
];

export function QueryBox({
  value,
  onChange,
  onSubmit,
  busy,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  disabled: boolean;
}) {
  return (
    <section className="panel p-6 sm:p-7">
      <h2 className="text-base sm:text-lg font-bold text-slate-900">Your question</h2>
      <div className="mt-4 flex flex-col sm:flex-row items-stretch sm:items-end gap-3">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!disabled && !busy) onSubmit();
            }
          }}
          rows={2}
          placeholder="Ask Earth Query Lens… e.g., What features are present in this satellite imagery?"
          className="min-h-[84px] flex-1 resize-none rounded-xl border border-slate-300 bg-white p-3.5 sm:p-4 text-base font-medium text-slate-900 outline-none placeholder:text-slate-400 focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-xs"
        />
        <button
          onClick={onSubmit}
          disabled={disabled || busy}
          className="flex h-12 items-center justify-center gap-2 rounded-xl bg-slate-900 px-7 text-sm font-semibold tracking-wide text-white shadow-md transition-all hover:bg-slate-800 disabled:opacity-40 shrink-0"
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
          {busy ? "Analyzing..." : "Ask Question"}
        </button>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2.5">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-500 mr-1">
          Suggestions:
        </span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => onChange(ex)}
            className="rounded-full border border-slate-200 bg-slate-50/90 px-3.5 py-1.5 text-xs sm:text-sm font-semibold text-slate-700 transition-all hover:border-slate-900 hover:bg-slate-900 hover:text-white shadow-2xs"
          >
            {ex}
          </button>
        ))}
      </div>
      <p className="mt-4 text-xs sm:text-sm font-medium text-slate-600">
        No task selection needed — the assistant automatically detects whether scene VQA, grounding, or change detection is required.
      </p>
    </section>
  );
}

