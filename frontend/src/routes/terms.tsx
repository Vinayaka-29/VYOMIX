import { createFileRoute } from "@tanstack/react-router";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { FileText } from "lucide-react";

export const Route = createFileRoute("/terms")({
  head: () => ({
    meta: [
      { title: "Terms & Conditions — Earth Query Lens — VYOMIX" },
      { name: "description", content: "Terms & Conditions for Earth Query Lens by VYOMIX." },
    ],
  }),
  component: TermsPage,
});

function TermsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-8 sm:px-8 lg:px-12 w-full">
        <div className="panel p-6 sm:p-10 bg-white w-full">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3.5 py-1 text-xs font-semibold text-sky-800">
            <FileText className="size-3.5 text-sky-700" />
            <span>Terms of Service</span>
          </div>

          <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-slate-950 mt-4">
            Terms & Conditions
          </h1>

          <p className="mt-2 text-xs sm:text-sm font-semibold uppercase tracking-wider text-slate-500">
            Effective Date: September 2026 · VYOMIX
          </p>

          <div className="mt-6 space-y-6 text-sm sm:text-base font-medium text-slate-700 leading-relaxed">
            <section>
              <h2 className="text-lg font-bold text-slate-900">1. Acceptance of Terms</h2>
              <p className="mt-2">
                By accessing Earth Query Lens, users agree to abide by these terms of service. Earth Query Lens provides vision-language intelligence tools for satellite imagery research and observation.
              </p>
            </section>

            <section>
              <h2 className="text-lg font-bold text-slate-900">2. Appropriate Usage</h2>
              <p className="mt-2">
                Users are responsible for ensuring they possess legitimate rights or public clearance to upload satellite scenes for processing. Reverse-engineering of model pipelines or automated flooding of backend endpoints is prohibited.
              </p>
            </section>

            <section>
              <h2 className="text-lg font-bold text-slate-900">3. Limitation of Model Outputs</h2>
              <p className="mt-2">
                Confidence metrics, grounding coordinates, and answer outputs provided by model specialists are observational assistance tools and should be validated for mission-critical operations.
              </p>
            </section>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs sm:text-sm font-medium text-slate-600">
              Note: Earth Query Lens is an official VYOMIX project.
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
