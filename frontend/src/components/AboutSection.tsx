import { Globe, ShieldCheck, Eye, Layers } from "lucide-react";

export function AboutSection() {
  return (
    <section className="panel mt-12 p-6 sm:p-10 bg-white w-full">
      <div className="w-full">
        <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3.5 py-1 text-xs font-semibold text-sky-800">
          <Globe className="size-3.5 text-sky-700" />
          <span>Product Overview</span>
        </div>

        <h2 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-slate-950 mt-4">
          About Earth Query Lens
        </h2>

        <p className="mt-4 text-base sm:text-lg leading-relaxed text-slate-700 font-medium max-w-4xl">
          Earth Query Lens is a multimodal satellite-imagery analysis platform designed to help users interact with optical and SAR imagery using natural-language queries.
        </p>

        <div className="mt-8 grid gap-6 sm:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-6">
            <div className="flex size-10 items-center justify-center rounded-xl bg-sky-100 text-sky-800 mb-4">
              <Layers className="size-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900">Multimodal Fusion</h3>
            <p className="mt-2 text-sm text-slate-600 font-medium leading-relaxed">
              Analyze both optical imagery and Synthetic Aperture Radar (SAR) for cloud-invariant insights.
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-6">
            <div className="flex size-10 items-center justify-center rounded-xl bg-teal-100 text-teal-800 mb-4">
              <Eye className="size-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900">Spatial Grounding</h3>
            <p className="mt-2 text-sm text-slate-600 font-medium leading-relaxed">
              Receive bounding-box coordinates for detected objects directly on satellite scenes.
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-6">
            <div className="flex size-10 items-center justify-center rounded-xl bg-indigo-100 text-indigo-800 mb-4">
              <ShieldCheck className="size-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900">Transparent Tracing</h3>
            <p className="mt-2 text-sm text-slate-600 font-medium leading-relaxed">
              Inspect observable step-by-step execution summaries for full model transparency.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

