import { Check, CircleDashed, Loader2, XCircle } from "lucide-react";
import type { ExecutionStep, ExecutionTrace } from "@/lib/satquery";
import { formatConfidence } from "@/lib/satquery";

/**
 * Observable execution summary — shows only what the backend reports.
 * No hidden reasoning, no invented steps.
 */
export function ExecutionTracePanel({
  trace,
  running,
}: {
  trace?: ExecutionTrace | undefined;
  running?: boolean | undefined;
}) {
  const hasTrace = trace && Object.keys(trace).length > 0;

  return (
    <section className="panel p-6 sm:p-7">
      <div className="flex items-center justify-between">
        <h2 className="text-base sm:text-lg font-bold text-slate-900">Execution Summary</h2>
        {running && (
          <span className="flex items-center gap-2 label-mono text-sky-700 font-bold">
            <Loader2 className="size-3.5 animate-spin text-sky-700" /> Running
          </span>
        )}
      </div>

      {!hasTrace ? (
        <p className="mt-4 text-xs sm:text-sm font-medium text-slate-600">
          {running
            ? "Waiting for the backend to report execution step progress..."
            : "The backend will populate an observable execution summary here once a query is executed."}
        </p>
      ) : (
        <div className="mt-5 space-y-4 text-xs sm:text-sm">
          <Field label="Query" value={trace!.query} />
          <Field label="Input" value={trace!.inputs} />
          <Field label="Detected task" value={trace!.detected_task} />
          <Field label="Validation" value={trace!.validation} />
          <Field label="Specialist" value={trace!.specialist} />
          <Field label="Model" value={trace!.model} />

          {trace!.steps && trace!.steps.length > 0 && (
            <div>
              <p className="label-mono">Processing Steps</p>
              <ul className="mt-2.5 space-y-2">
                {trace!.steps.map((s, i) => (
                  <StepRow key={i} step={s} />
                ))}
              </ul>
            </div>
          )}

          <Field label="Result" value={trace!.result} />
          <Field label="Confidence" value={formatConfidence(trace!.confidence) ?? undefined} />

          {trace!.errors && trace!.errors.length > 0 && (
            <div>
              <p className="label-mono text-rose-600">Errors</p>
              <ul className="mt-2 space-y-1.5">
                {trace!.errors.map((e, i) => (
                  <li key={i} className="text-sm font-semibold text-rose-700">
                    {e}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function Field({ label, value }: { label: string; value?: string | undefined }) {
  if (!value) return null;
  return (
    <div>
      <p className="label-mono">{label}</p>
      <p className="mt-1 text-sm sm:text-base font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function StepRow({ step }: { step: ExecutionStep }) {
  const icon =
    step.status === "done" ? (
      <Check className="size-4 text-emerald-600 font-bold" />
    ) : step.status === "running" ? (
      <Loader2 className="size-4 animate-spin text-sky-700" />
    ) : step.status === "failed" ? (
      <XCircle className="size-4 text-rose-600" />
    ) : (
      <CircleDashed className="size-4 text-slate-400" />
    );

  return (
    <li className="flex items-start gap-2.5 text-sm sm:text-base font-medium">
      <span className="mt-0.5">{icon}</span>
      <span>
        <span className="font-semibold text-slate-900">{step.name}</span>
        {step.detail && <span className="block text-xs sm:text-sm text-slate-600 font-normal mt-0.5">{step.detail}</span>}
      </span>
    </li>
  );
}

