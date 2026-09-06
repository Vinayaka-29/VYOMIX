import { createFileRoute } from "@tanstack/react-router";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { Shield } from "lucide-react";

export const Route = createFileRoute("/privacy")({
  head: () => ({
    meta: [
      { title: "Privacy Policy — Earth Query Lens — VYOMIX" },
      { name: "description", content: "Privacy Policy for Earth Query Lens by VYOMIX." },
    ],
  }),
  component: PrivacyPage,
});

function PrivacyPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-8 sm:px-8 lg:px-12 w-full">
        <div className="panel p-6 sm:p-10 bg-white w-full">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3.5 py-1 text-xs font-semibold text-sky-800">
            <Shield className="size-3.5 text-sky-700" />
            <span>Legal Documentation</span>
          </div>

          <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-slate-950 mt-4">
            Privacy Policy
          </h1>

          <p className="mt-2 text-xs sm:text-sm font-semibold uppercase tracking-wider text-slate-500">
            Last Updated: September 2026 · VYOMIX
          </p>

          <div className="mt-6 space-y-6 text-sm sm:text-base font-medium text-slate-700 leading-relaxed">
            <section>
              <h2 className="text-lg font-bold text-slate-900">1. Data Collection Overview</h2>
              <p className="mt-2">
                Earth Query Lens processes satellite imagery uploaded directly by users to perform visual question answering, spatial grounding, and change detection analysis. Uploaded files are processed in-memory or securely streamed to backend analysis workers.
              </p>
            </section>

            <section>
              <h2 className="text-lg font-bold text-slate-900">2. Usage of Remote Sensing Data</h2>
              <p className="mt-2">
                User-provided GeoTIFF, Optical, and SAR imagery is strictly used for fulfilling user analysis queries and generating execution summary traces. Data is never shared with third-party advertising networks.
              </p>
            </section>

            <section>
              <h2 className="text-lg font-bold text-slate-900">3. User Security & Privacy</h2>
              <p className="mt-2">
                Earth Query Lens enforces standard data encryption in transit. Session configuration preferences are maintained locally in browser storage.
              </p>
            </section>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs sm:text-sm font-medium text-slate-600">
              Note: This document serves as a product privacy disclosure for Earth Query Lens, a VYOMIX project.
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
