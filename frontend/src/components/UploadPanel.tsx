import { useRef, useState } from "react";
import { ImagePlus, X, Layers } from "lucide-react";

export interface UploadedImage {
  id: string;
  file: File;
  previewUrl: string | null;
}

const RASTER_EXT = /\.(tif|tiff|geotiff|png|jpg|jpeg|jp2|img)$/i;

export function UploadPanel({
  images,
  onChange,
}: {
  images: UploadedImage[];
  onChange: (next: UploadedImage[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const accepted: UploadedImage[] = [];
    const rejected: string[] = [];
    Array.from(files).forEach((file) => {
      if (!RASTER_EXT.test(file.name)) {
        rejected.push(file.name);
        return;
      }
      const previewable = /\.(png|jpg|jpeg)$/i.test(file.name);
      accepted.push({
        id: `${file.name}-${file.size}-${Math.random().toString(36).slice(2, 7)}`,
        file,
        previewUrl: previewable ? URL.createObjectURL(file) : null,
      });
    });

    const next = [...images, ...accepted].slice(0, 2);
    onChange(next);

    if (rejected.length)
      setNotice(
        `Not a supported image: ${rejected.join(", ")}. Use GeoTIFF/TIFF, PNG or JPEG imagery.`,
      );
    else if (images.length + accepted.length > 2)
      setNotice("Only two images can be compared at a time. Extra files were left out.");
    else setNotice(null);
  };

  const remove = (id: string) => {
    const target = images.find((i) => i.id === id);
    if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);
    onChange(images.filter((i) => i.id !== id));
    setNotice(null);
  };

  return (
    <section className="panel p-6 sm:p-7">
      <div className="flex items-center justify-between">
        <h2 className="text-base sm:text-lg font-bold text-slate-900">Imagery Upload</h2>
        <span className="label-mono">{images.length}/2 loaded</span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          addFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`mt-4 cursor-pointer rounded-2xl border-2 border-dashed p-6 sm:p-8 text-center transition-all ${
          dragging
            ? "border-sky-600 bg-sky-50/80 shadow-md"
            : "border-slate-300 bg-slate-50/60 hover:border-slate-400 hover:bg-slate-100/50"
        }`}
      >
        <ImagePlus className="mx-auto size-8 text-sky-700" />
        <p className="mt-3 text-base font-semibold text-slate-900">Drop imagery here or click to browse</p>
        <p className="mt-1.5 text-xs sm:text-sm font-medium text-slate-600">
          One image for scene questions · two images for a before/after or optical + radar pair
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.jp2,.img"
          className="hidden"
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {notice && (
        <p className="mt-4 rounded-xl border border-amber-300 bg-amber-50 p-3.5 text-xs sm:text-sm font-medium text-amber-900">
          {notice}
        </p>
      )}

      {images.length > 0 && (
        <ul className="mt-4 space-y-2.5">
          {images.map((img, idx) => (
            <li
              key={img.id}
              className="flex items-center gap-3.5 rounded-xl border border-slate-200 bg-white p-3.5 shadow-2xs"
            >
              <span className="label-mono w-16 shrink-0 text-slate-600">
                {images.length === 2 ? `IMG ${idx + 1}` : "INPUT"}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-slate-900">{img.file.name}</p>
                <p className="text-xs font-medium text-slate-500">
                  {(img.file.size / 1024 / 1024).toFixed(2)} MB
                  {img.previewUrl ? "" : " · preview rendered via backend"}
                </p>
              </div>
              <button
                onClick={() => remove(img.id)}
                aria-label={`Remove ${img.file.name}`}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-rose-600 transition-colors"
              >
                <X className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {images.length === 2 && (
        <p className="mt-4 flex items-center gap-2.5 text-xs sm:text-sm font-medium text-slate-600">
          <Layers className="size-4 text-sky-700 shrink-0" />
          Two images loaded — the assistant determines if this is a temporal comparison or a cross-sensor pair.
        </p>
      )}
    </section>
  );
}

