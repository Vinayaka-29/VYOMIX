/**
 * SatQuery AI backend contract + client.
 *
 * Every field is optional: the UI renders only what the backend actually
 * returns and shows "Not available" otherwise. Nothing is invented here.
 */

export type Modality = "optical" | "sar" | "unknown";

export interface ImageMetadata {
  filename?: string;
  modality?: string;
  width?: number;
  height?: number;
  bands?: number | string[];
  crs?: string;
  resolution?: string | number;
  file_type?: string;
  validation_status?: string;
  [key: string]: unknown;
}

export interface BoundingBox {
  bbox: [number, number, number, number];
  label?: string;
  confidence?: number;
}

export interface EvidenceItem {
  type?: string;
  label?: string;
  detail?: string;
  description?: string;
}

export interface ExecutionStep {
  name: string;
  status?: "pending" | "running" | "done" | "failed";
  detail?: string;
}

export interface ExecutionTrace {
  query?: string;
  inputs?: string;
  detected_task?: string;
  validation?: string;
  specialist?: string;
  model?: string;
  steps?: ExecutionStep[];
  result?: string;
  confidence?: number;
  errors?: string[];
  disagreement?: string | boolean;
}

export interface ChangeResult {
  change_detected?: boolean;
  changed_area_percent?: number;
  description?: string;
  change_map_url?: string;
  regions?: BoundingBox[];
  confidence?: number;
}

export interface OpticalSarResult {
  optical_evidence?: string | string[];
  sar_evidence?: string | string[];
  complementary?: string | string[];
  confidence?: number;
}

export interface AnalysisResponse {
  answer?: string;
  caption?: string;
  confidence?: number;
  model?: string;
  task?: string;
  evidence?: EvidenceItem[] | string[];
  grounding?: BoundingBox[];
  change?: ChangeResult;
  optical_sar?: OpticalSarResult;
  metadata?: ImageMetadata[];
  execution_trace?: ExecutionTrace;
  report_url?: string;
  report_id?: string;
  warnings?: string[];
}

export class SatQueryError extends Error {
  status: number | undefined;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

const STORAGE_KEY = "satquery.api_base_url";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) return stored.replace(/\/$/, "");
  }
  const env = import.meta.env["VITE_SATQUERY_API_URL"] as string | undefined;
  return (env ?? "").replace(/\/$/, "");
}

export function setApiBaseUrl(url: string) {
  if (typeof window === "undefined") return;
  const trimmed = url.trim().replace(/\/$/, "");
  if (trimmed) window.localStorage.setItem(STORAGE_KEY, trimmed);
  else window.localStorage.removeItem(STORAGE_KEY);
}

/** Turn any backend/network failure into a sentence a human can act on. */
export function humanizeError(err: unknown): string {
  if (err instanceof SatQueryError) {
    if (err.status === 0)
      return "Could not reach the SatQuery backend. Check that it is running and that the address in Backend settings is correct.";
    if (err.status === 413) return "That file is too large for the backend to accept.";
    if (err.status === 415) return "That file format isn't supported by the backend.";
    if (err.status === 422 || err.status === 400) return err.message;
    if (err.status === 503)
      return err.message || "The requested model is currently unavailable on the backend.";
    if (err.status && err.status >= 500)
      return err.message || "The backend failed while processing this request.";
    return err.message;
  }
  return err instanceof Error ? err.message : "Something went wrong.";
}

async function readError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  if (!text) return `Backend returned ${res.status}.`;
  try {
    const json = JSON.parse(text) as Record<string, unknown>;
    const msg = json["detail"] ?? json["message"] ?? json["error"];
    if (typeof msg === "string") return msg;
    if (Array.isArray(msg) && typeof msg[0] === "string") return msg[0] as string;
  } catch {
    /* plain text body */
  }
  return text.slice(0, 300);
}

export async function analyze(input: {
  query: string;
  files: File[];
  signal?: AbortSignal;
}): Promise<AnalysisResponse> {
  const base = getApiBaseUrl();
  if (!base)
    throw new SatQueryError(
      "No backend address is configured yet. Open Backend settings and paste your SatQuery API address.",
      0,
    );

  const form = new FormData();
  form.append("query", input.query);
  input.files.forEach((file, i) => {
    form.append("images", file);
    form.append(`image_${i + 1}`, file);
  });

  let res: Response;
  try {
    res = await fetch(`${base}/analyze`, {
      method: "POST",
      body: form,
      signal: input.signal ?? null,
    });
  } catch {
    throw new SatQueryError("network", 0);
  }

  if (!res.ok) throw new SatQueryError(await readError(res), res.status);
  return (await res.json()) as AnalysisResponse;
}

export async function checkHealth(): Promise<boolean> {
  const base = getApiBaseUrl();
  if (!base) return false;
  try {
    const res = await fetch(`${base}/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export function formatConfidence(value?: number): string | null {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  const pct = value <= 1 ? value * 100 : value;
  return `${Math.round(pct)}%`;
}

export function confidenceRatio(value?: number): number | null {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  const ratio = value <= 1 ? value : value / 100;
  return Math.max(0, Math.min(1, ratio));
}
