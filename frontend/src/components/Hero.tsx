import { ArrowDown, Sparkles, ShieldCheck, Layers } from "lucide-react";
import earthBg from "@/assets/earth_orbital_bg.png";
import { useAuth } from "@/lib/auth-context";
import { useNavigate } from "@tanstack/react-router";

const NAV = [
  { label: "Analyse", href: "#workspace", active: true },
  { label: "Evidence", href: "#workspace" },
  { label: "Execution", href: "#workspace" },
  { label: "Report", href: "#workspace" },
];

export function Hero() {
  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

  const handleAuthGatedClick = (e: React.MouseEvent, targetHash: string = "#workspace") => {
    if (!isLoggedIn) {
      e.preventDefault();
      void navigate({ to: "/login", search: { redirect: targetHash } });
    }
  };

  return (
    <section className="relative overflow-hidden rounded-3xl border border-border/80 bg-white shadow-lg">
      {/* FULL EARTH BACKGROUND */}
      <div className="absolute inset-0 pointer-events-none">
        <img
          src={earthBg}
          alt="Planet Earth seen from space"
          className="size-full object-cover object-center opacity-80"
        />
        {/* Soft Powder-Blue Light Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-white/95 via-white/50 to-slate-100/95" />
        <div className="absolute inset-0 bg-radial from-transparent via-cyan-900/5 to-slate-900/15 mix-blend-multiply" />
      </div>

      {/* Hero Content Overlay */}
      <div className="relative z-10 p-5 sm:p-8 lg:p-10">
        {/* Top Navbar */}
        <nav className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-900/10 pb-4">
          <a href="#workspace" onClick={(e) => handleAuthGatedClick(e, "#workspace")} className="flex items-center gap-2.5 font-serif text-2xl font-bold tracking-tight text-slate-900">
            <span className="flex size-9 items-center justify-center rounded-full bg-slate-900 text-xs font-sans text-white shadow">
              E
            </span>
            earth<span className="font-sans font-normal text-slate-600">query</span>
          </a>
          <div className="flex items-center gap-6 sm:gap-8">
            <div className="hidden items-center gap-6 sm:flex sm:gap-8">
              {NAV.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  onClick={(e) => handleAuthGatedClick(e, "#workspace")}
                  className={item.active ? "nav-link nav-link-active font-semibold text-slate-900" : "nav-link text-slate-700"}
                >
                  {item.label}
                </a>
              ))}
            </div>
            <a
              href="#workspace"
              onClick={(e) => handleAuthGatedClick(e, "#workspace")}
              className="rounded-full bg-slate-900 px-6 py-2 text-xs sm:text-sm font-semibold tracking-wide text-white shadow-md transition-all hover:bg-slate-800"
            >
              Start Analysis
            </a>
          </div>
        </nav>

        {/* Main Hero Header - Compact Layout */}
        <div className="relative mx-auto my-6 sm:my-8 max-w-4xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-300/80 bg-white/90 px-4 py-1.5 text-xs sm:text-sm font-semibold text-slate-800 backdrop-blur shadow-sm">
            <Sparkles className="size-4 text-sky-600" />
            <span className="font-bold tracking-wider uppercase text-sky-700 text-[11px] sm:text-xs">VYOMIX PRESENTS</span>
          </div>

          <h1 className="hero-title mt-4 text-4xl sm:text-5xl lg:text-6xl font-normal text-slate-950">
            Earth Query Lens
          </h1>

          <p className="text-xs sm:text-sm font-bold uppercase tracking-widest text-slate-600 mt-2">
            Multimodal Vision-Language Satellite Intelligence
          </p>

          <p className="mx-auto mt-3 max-w-2xl text-sm sm:text-base leading-relaxed font-medium text-slate-700">
            Upload optical or SAR satellite imagery, ask natural-language questions, and inspect detailed confidence metrics, grounding boxes, and execution traces.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
            <a
              href="#workspace"
              onClick={(e) => handleAuthGatedClick(e, "#workspace")}
              className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-8 py-3 text-xs sm:text-sm font-semibold uppercase tracking-widest text-white shadow-md transition-all hover:bg-slate-800 hover:scale-105"
            >
              Open Workspace
            </a>
            <a
              href="#workspace"
              onClick={(e) => handleAuthGatedClick(e, "#workspace")}
              className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white/90 px-6 py-3 text-xs sm:text-sm font-semibold uppercase tracking-widest text-slate-800 shadow-sm backdrop-blur transition-all hover:bg-slate-50"
            >
              <Layers className="size-4 text-sky-600" />
              Cross-Sensor VQA
            </a>
          </div>

          {/* Floating Feature Pills Bar */}
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <div className="inline-flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/90 px-4 py-2 text-xs sm:text-sm shadow-sm backdrop-blur">
              <span className="flex size-7 items-center justify-center rounded-lg bg-sky-100 text-sky-800">
                <ShieldCheck className="size-4" />
              </span>
              <div className="text-left">
                <p className="font-semibold text-slate-900">SAR Radar Invariant</p>
                <p className="text-xs text-slate-500">All-weather cloud penetration</p>
              </div>
            </div>

            <div className="inline-flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/90 px-4 py-2 text-xs sm:text-sm shadow-sm backdrop-blur">
              <span className="flex size-7 items-center justify-center rounded-lg bg-teal-100 text-teal-800">
                <Sparkles className="size-4" />
              </span>
              <div className="text-left">
                <p className="font-semibold text-slate-900">Bi-Temporal VQA</p>
                <p className="text-xs text-slate-500">Sub-meter change detection</p>
              </div>
            </div>
          </div>
        </div>

        {/* Compact Scroll Arrow */}
        <div className="flex justify-center mt-4">
          <a
            href="#workspace"
            onClick={(e) => handleAuthGatedClick(e, "#workspace")}
            aria-label="Scroll to workspace"
            className="flex size-9 items-center justify-center rounded-full border border-slate-300 bg-white/90 text-slate-900 shadow-sm backdrop-blur transition-all hover:bg-slate-50 hover:scale-110"
          >
            <ArrowDown className="size-4" />
          </a>
        </div>
      </div>
    </section>
  );
}
