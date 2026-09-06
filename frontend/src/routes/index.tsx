import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { AlertCircle, Satellite } from "lucide-react";
import { Hero } from "@/components/Hero";
import { UploadPanel, type UploadedImage } from "@/components/UploadPanel";
import { QueryBox } from "@/components/QueryBox";
import { ImageViewer } from "@/components/ImageViewer";
import { MetadataPanel } from "@/components/MetadataPanel";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { ExecutionTracePanel } from "@/components/ExecutionTracePanel";
import { BackendSettings } from "@/components/BackendSettings";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { AboutSection } from "@/components/AboutSection";
import { useAuth } from "@/lib/auth-context";
import { analyze, humanizeError, type AnalysisResponse } from "@/lib/satquery";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Earth Query Lens — VYOMIX" },
      {
        name: "description",
        content:
          "Earth Query Lens by VYOMIX is an interactive vision-language assistant for multimodal remote sensing analysis.",
      },
      { property: "og:title", content: "Earth Query Lens — VYOMIX" },
      {
        property: "og:description",
        content:
          "Upload optical, SAR or bi-temporal imagery, ask a question in plain language, and inspect the answer, evidence and execution trace.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

  const [images, setImages] = useState<UploadedImage[]>([]);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const submit = async () => {
    setError(null);
    setResult(null);

    if (!isLoggedIn) {
      void navigate({ to: "/login", search: { redirect: "#workspace" } });
      return;
    }

    if (images.length === 0) {
      setError("Add at least one image before asking a question.");
      return;
    }
    if (!query.trim()) {
      setError("Type a question first.");
      return;
    }
    setBusy(true);
    try {
      const res = await analyze({ query: query.trim(), files: images.map((i) => i.file) });
      setResult(res);
    } catch (err) {
      setError(humanizeError(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-6 sm:px-8 lg:px-12 lg:py-8 w-full">
        <Hero />

        <header
          id="workspace"
          className="mt-8 sm:mt-12 flex scroll-mt-6 flex-wrap items-center justify-between gap-4 border-b border-slate-200/80 pb-4"
        >
          <div className="flex items-center gap-3.5">
            <span className="flex size-11 items-center justify-center rounded-2xl border border-slate-200 bg-white shadow-sm">
              <Satellite className="size-6 text-sky-700" />
            </span>
            <div>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
                Interactive Workspace
              </h2>
              <p className="text-sm font-medium text-slate-600">
                Ask natural-language questions about remote sensing optical & SAR imagery
              </p>
            </div>
          </div>
          <BackendSettings />
        </header>

        <div className="mt-8 grid gap-8 lg:grid-cols-12">
          {/* Left Column: Imagery Upload, Query Input, Results & Metadata */}
          <div className="space-y-6 lg:col-span-7">
            <UploadPanel images={images} onChange={setImages} />
            <QueryBox
              value={query}
              onChange={setQuery}
              onSubmit={() => void submit()}
              busy={busy}
              disabled={false}
            />

            {error && (
              <p className="flex items-start gap-2.5 rounded-2xl border border-rose-300 bg-rose-50 p-4 text-sm font-medium text-rose-800 shadow-sm">
                <AlertCircle className="mt-0.5 size-5 shrink-0 text-rose-600" />
                {error}
              </p>
            )}

            <ImageViewer images={images} boxes={result?.grounding} />
            {result && <ResultsDashboard result={result} />}
            <MetadataPanel metadata={result?.metadata} />
          </div>

          {/* Right Column: Execution Summary & How This Works */}
          <div className="space-y-6 lg:col-span-5">
            <ExecutionTracePanel trace={result?.execution_trace} running={busy} />
            {!result && !busy && (
              <section className="panel p-6 sm:p-7">
                <h2 className="text-base sm:text-lg font-bold text-slate-900">How this works</h2>
                <ol className="mt-4 space-y-3 text-sm sm:text-base font-medium text-slate-700">
                  <li className="flex items-start gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-900">
                      1
                    </span>
                    <span>Load one image, or two for a before/after or optical + radar pair.</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-900">
                      2
                    </span>
                    <span>Ask your question in plain language.</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-900">
                      3
                    </span>
                    <span>The assistant picks the right analysis specialist model.</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-900">
                      4
                    </span>
                    <span>View the answer, confidence score, and supporting evidence.</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-900">
                      5
                    </span>
                    <span>Open the execution summary to check every processing step.</span>
                  </li>
                </ol>
                <div className="mt-6 rounded-xl border border-slate-200/80 bg-slate-50/80 p-3.5 text-xs sm:text-sm font-medium text-slate-600">
                  Nothing on this screen is filled with mock numbers — any metric not returned by the backend is safely shown as “Not available”.
                </div>
              </section>
            )}
          </div>
        </div>

        {/* Product Information Section */}
        <AboutSection />
      </main>

      <Footer />
    </div>
  );
}


