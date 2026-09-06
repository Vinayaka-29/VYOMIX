import type { ImageMetadata } from "@/lib/satquery";

const FIELDS: Array<[keyof ImageMetadata & string, string]> = [
  ["modality", "Sensor type"],
  ["file_type", "File type"],
  ["crs", "Coordinate system"],
  ["resolution", "Resolution"],
  ["validation_status", "Validation"],
];

export function MetadataPanel({ metadata }: { metadata?: ImageMetadata[] | undefined }) {
  if (!metadata || metadata.length === 0) return null;

  return (
    <section className="panel p-6 sm:p-7">
      <h2 className="text-base sm:text-lg font-bold text-slate-900">Image Details & Metadata</h2>
      <div className={`mt-4 grid gap-4 ${metadata.length > 1 ? "sm:grid-cols-2" : ""}`}>
        {metadata.map((m, i) => (
          <div key={i} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
            <p className="label-mono text-slate-700">{m.filename ?? `Image ${i + 1}`}</p>
            <dl className="mt-3 space-y-2 text-xs sm:text-sm font-medium">
              {m.width && m.height && (
                <Row label="Dimensions" value={`${m.width} × ${m.height} px`} />
              )}
              {m.bands !== undefined && (
                <Row
                  label="Bands"
                  value={Array.isArray(m.bands) ? m.bands.join(", ") : String(m.bands)}
                />
              )}
              {FIELDS.map(([key, label]) =>
                m[key] !== undefined && m[key] !== null ? (
                  <Row key={key} label={label} value={formatMetaValue(key, m[key])} />
                ) : null,
              )}
            </dl>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatMetaValue(key: string, val: unknown): string {
  if (val === undefined || val === null) return "Not available";
  if (key === "resolution" && typeof val === "object" && val !== null) {
    const res = val as { x?: number; y?: number; unit?: string };
    if (res.x !== undefined && res.y !== undefined) {
      return `${res.x} × ${res.y} ${res.unit ?? "m"}`;
    }
  }
  if (typeof val === "object") {
    return JSON.stringify(val);
  }
  return String(val);
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-slate-600">{label}</dt>
      <dd className="text-right font-mono text-xs sm:text-sm font-semibold text-slate-900">{value}</dd>
    </div>
  );
}

