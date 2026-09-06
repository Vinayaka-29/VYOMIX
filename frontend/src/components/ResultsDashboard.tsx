import { AlertTriangle, Check, Download } from "lucide-react";
import type { AnalysisResponse, EvidenceItem } from "@/lib/satquery";
import { confidenceRatio, formatConfidence, getApiBaseUrl } from "@/lib/satquery";

export function ResultsDashboard({ result }: { result: AnalysisResponse }) {
  const answer = result.answer ?? result.caption;
  const conf = formatConfidence(result.confidence);
  const ratio = confidenceRatio(result.confidence);
  const evidence = normalizeEvidence(result.evidence);
  const disagreement = result.execution_trace?.disagreement;
  const reportHref = result.report_url
    ? result.report_url
    : result.report_id
      ? `${getApiBaseUrl()}/report/${result.report_id}`
      : null;

  return (
    <section className="panel overflow-hidden">
      <header className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-6 py-4">
        <h2 className="text-base sm:text-lg font-bold tracking-wider uppercase text-slate-900">
          Analysis Result
        </h2>
        {result.task && <span className="label-mono font-semibold">{result.task}</span>}
      </header>

      <div className="space-y-6 p-6 sm:p-7">
        <div>
          <p className="label-mono">Answer</p>
          <p className="mt-2 font-serif text-2xl sm:text-3xl leading-snug text-slate-950 font-normal">
            {answer ?? <span className="text-slate-400 italic">Not available</span>}
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <p className="label-mono">Confidence Score</p>
            {conf && ratio !== null ? (
              <div className="mt-2.5">
                <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                  <div className="h-full rounded-full bg-sky-700 transition-all duration-500" style={{ width: `${ratio * 100}%` }} />
                </div>
                <p className="mt-2 font-mono text-base font-bold text-slate-900">{conf}</p>
              </div>
            ) : (
              <p className="mt-2 text-sm sm:text-base font-medium text-slate-500">Not available</p>
            )}
          </div>
          <div>
            <p className="label-mono">Specialist Model</p>
            <p className="mt-2 font-mono text-sm sm:text-base font-semibold text-slate-900">
              {result.model ?? result.execution_trace?.model ?? (
                <span className="text-slate-400 font-normal">Not available</span>
              )}
            </p>
          </div>
        </div>

        {disagreement && (
          <p className="flex items-start gap-2.5 rounded-xl border border-amber-300 bg-amber-50 p-4 text-xs sm:text-sm font-medium text-amber-900">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600" />
            {typeof disagreement === "string" ? disagreement : "Evidence disagreement detected."}
          </p>
        )}

        <div>
          <p className="label-mono">Evidence & Findings</p>
          {evidence.length > 0 ? (
            <ul className="mt-3 space-y-2.5">
              {evidence.map((e, i) => (
                <li key={i} className="flex items-start gap-3 text-sm sm:text-base font-medium">
                  <Check className="mt-1 size-4 shrink-0 text-emerald-600 font-bold" />
                  <span>
                    <span className="font-semibold text-slate-900">{e.label ?? e.type ?? "Evidence"}</span>
                    {(e.detail ?? e.description) && (
                      <span className="block text-xs sm:text-sm text-slate-600 font-normal mt-0.5">
                        {e.detail ?? e.description}
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm sm:text-base font-medium text-slate-500">Not available</p>
          )}
        </div>

        {result.warnings && result.warnings.length > 0 && (
          <ul className="space-y-1.5">
            {result.warnings.map((w, i) => (
              <li key={i} className="text-xs sm:text-sm font-medium text-amber-800">
                ⚠️ {w}
              </li>
            ))}
          </ul>
        )}

        {reportHref && (
          <a
            href={reportHref}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2.5 rounded-xl border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-900 shadow-sm transition-all hover:bg-slate-50"
          >
            <Download className="size-4 text-sky-700" />
            Download Detailed Report
          </a>
        )}
      </div>
    </section>
  );
}


function normalizeEvidence(evidence: AnalysisResponse["evidence"]): EvidenceItem[] {
  if (!evidence) return [];
  return evidence.map((e) => (typeof e === "string" ? { label: e } : e));
}
