import { FileImage } from "lucide-react";
import type { UploadedImage } from "./UploadPanel";
import type { BoundingBox } from "@/lib/satquery";
import { formatConfidence } from "@/lib/satquery";

/**
 * Shows the loaded imagery. Bounding boxes are drawn only from backend
 * coordinates — never generated here.
 */
export function ImageViewer({
  images,
  boxes,
}: {
  images: UploadedImage[];
  boxes?: BoundingBox[] | undefined;
}) {
  if (images.length === 0) return null;

  return (
    <section className="panel p-6 sm:p-7">
      <div className="flex items-center justify-between">
        <h2 className="text-base sm:text-lg font-bold text-slate-900">Image view</h2>
        {boxes && boxes.length > 0 && (
          <span className="label-mono">{boxes.length} region(s) returned</span>
        )}
      </div>

      <div className={`mt-4 grid gap-4 ${images.length > 1 ? "sm:grid-cols-2" : ""}`}>
        {images.map((img, idx) => (
          <figure key={img.id} className="overflow-hidden rounded-2xl border border-slate-200 shadow-sm">
            <div className="relative bg-slate-950">
              {img.previewUrl ? (
                <>
                  <img
                    src={img.previewUrl}
                    alt={`Uploaded imagery ${idx + 1}: ${img.file.name}`}
                    className="block w-full object-contain"
                  />
                  {idx === 0 &&
                    boxes?.map((b, i) => <BoxOverlay key={i} box={b} imageUrl={img.previewUrl!} />)}
                </>
              ) : (
                <div className="flex aspect-square flex-col items-center justify-center gap-2.5 px-4 text-center">
                  <FileImage className="size-8 text-slate-400" />
                  <p className="text-xs sm:text-sm font-medium text-slate-400">
                    This format cannot be displayed natively in browser. The backend processes it directly.
                  </p>
                </div>
              )}
            </div>
            <figcaption className="flex items-center justify-between gap-2 border-t border-slate-200 bg-slate-50 px-4 py-2.5">
              <span className="truncate text-xs sm:text-sm font-medium text-slate-700">{img.file.name}</span>
              {images.length > 1 && <span className="label-mono text-slate-600">IMG {idx + 1}</span>}
            </figcaption>
          </figure>
        ))}
      </div>

      {boxes && boxes.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {boxes.map((b, i) => (
            <li key={i} className="flex items-center gap-3 font-mono text-xs sm:text-sm font-medium">
              <span className="font-bold text-sky-700">{b.label ?? `region ${i + 1}`}</span>
              <span className="text-slate-600">[{b.bbox.join(", ")}]</span>
              {formatConfidence(b.confidence) && (
                <span className="text-slate-600">({formatConfidence(b.confidence)})</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}


/** Boxes are assumed pixel coordinates unless all values are <= 1 (normalized). */
function BoxOverlay({ box, imageUrl }: { box: BoundingBox; imageUrl: string }) {
  void imageUrl;
  const [x1, y1, x2, y2] = box.bbox;
  const normalized = [x1, y1, x2, y2].every((v) => v >= 0 && v <= 1);
  if (!normalized) return null;

  return (
    <div
      className="absolute rounded-sm border-2 border-primary"
      style={{
        left: `${x1 * 100}%`,
        top: `${y1 * 100}%`,
        width: `${(x2 - x1) * 100}%`,
        height: `${(y2 - y1) * 100}%`,
      }}
    >
      {box.label && (
        <span className="absolute -top-5 left-0 rounded bg-primary px-1 font-mono text-[10px] text-primary-foreground">
          {box.label}
        </span>
      )}
    </div>
  );
}
