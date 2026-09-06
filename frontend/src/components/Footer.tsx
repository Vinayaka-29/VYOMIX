import { Link } from "@tanstack/react-router";
import { Sparkles, Globe2 } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-16 border-t border-slate-200 bg-white/80 backdrop-blur-md">
      <div className="mx-auto max-w-[1440px] px-4 py-12 sm:px-8 lg:px-12">
        <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-5">
          {/* Brand Col (2 cols wide on desktop) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <span className="flex size-10 items-center justify-center rounded-2xl bg-slate-900 text-white shadow-md">
                <Sparkles className="size-5 text-sky-400" />
              </span>
              <div>
                <span className="block text-[10px] font-bold uppercase tracking-widest text-sky-700 leading-none">
                  VYOMIX PRESENTS
                </span>
                <span className="font-serif text-2xl font-bold tracking-tight text-slate-950">
                  Earth Query Lens
                </span>
              </div>
            </div>

            <p className="text-sm font-medium leading-relaxed text-slate-600 max-w-sm">
              Multimodal Vision-Language Satellite Intelligence. Interact with optical and SAR remote sensing imagery using plain language.
            </p>

            <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3.5 py-1.5 text-xs font-semibold text-slate-700">
              <Globe2 className="size-3.5 text-sky-700" />
              <span>VYOMIX Earth Intelligence Suite</span>
            </div>
          </div>

          {/* Navigation Col */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-4">
              Navigation
            </h3>
            <ul className="space-y-2.5 text-sm font-medium text-slate-600">
              <li>
                <Link to="/" className="hover:text-slate-950 transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <a href="/#workspace" className="hover:text-slate-950 transition-colors">
                  Analyse
                </a>
              </li>
              <li>
                <a href="/#workspace" className="hover:text-slate-950 transition-colors">
                  Evidence
                </a>
              </li>
              <li>
                <a href="/#workspace" className="hover:text-slate-950 transition-colors">
                  Execution
                </a>
              </li>
              <li>
                <a href="/#workspace" className="hover:text-slate-950 transition-colors">
                  Report
                </a>
              </li>
            </ul>
          </div>

          {/* Account Col */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-4">
              Account
            </h3>
            <ul className="space-y-2.5 text-sm font-medium text-slate-600">
              <li>
                <Link to="/login" className="hover:text-slate-950 transition-colors">
                  Login
                </Link>
              </li>
              <li>
                <Link to="/signup" className="hover:text-slate-950 transition-colors">
                  Sign Up
                </Link>
              </li>
              <li>
                <Link to="/profile" className="hover:text-slate-950 transition-colors">
                  Profile
                </Link>
              </li>
            </ul>
          </div>

          {/* Information Col */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-4">
              Information
            </h3>
            <ul className="space-y-2.5 text-sm font-medium text-slate-600">
              <li>
                <Link to="/about" className="hover:text-slate-950 transition-colors">
                  About
                </Link>
              </li>
              <li>
                <Link to="/privacy" className="hover:text-slate-950 transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/terms" className="hover:text-slate-950 transition-colors">
                  Terms & Conditions
                </Link>
              </li>
              <li>
                <Link to="/contact" className="hover:text-slate-950 transition-colors">
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-200/80 pt-6 text-xs font-medium text-slate-500">
          <p>© 2026 VYOMIX. All rights reserved.</p>
          <p className="font-semibold text-slate-700">Earth Query Lens is a VYOMIX project.</p>
        </div>
      </div>
    </footer>
  );
}
