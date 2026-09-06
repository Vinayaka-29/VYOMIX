import { useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useAuth } from "@/lib/auth-context";
import { LogOut, User as UserIcon, Menu, X, Sparkles } from "lucide-react";

export function Navbar() {
  const { user, isLoggedIn, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    void navigate({ to: "/" });
  };

  const handleAuthGatedClick = (e: React.MouseEvent, targetHash: string = "#workspace") => {
    if (!isLoggedIn) {
      e.preventDefault();
      void navigate({ to: "/login", search: { redirect: targetHash } });
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200/80 bg-white/90 backdrop-blur-md shadow-2xs">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between px-4 py-3 sm:px-8 lg:px-12">
        {/* Brand Hierarchy */}
        <Link to="/" className="flex items-center gap-3 group">
          <span className="flex size-10 items-center justify-center rounded-2xl bg-slate-900 text-white shadow-md transition-transform group-hover:scale-105">
            <Sparkles className="size-5 text-sky-400" />
          </span>
          <div className="text-left">
            <span className="block text-[10px] font-bold uppercase tracking-widest text-sky-700 leading-none">
              VYOMIX presents
            </span>
            <span className="font-serif text-xl sm:text-2xl font-bold tracking-tight text-slate-950 leading-tight">
              Earth Query Lens
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden lg:flex items-center gap-8 text-sm font-semibold text-slate-700">
          <a
            href="/#workspace"
            onClick={(e) => handleAuthGatedClick(e, "#workspace")}
            className="hover:text-slate-950 transition-colors"
          >
            Analyse
          </a>
          <a
            href="/#workspace"
            onClick={(e) => handleAuthGatedClick(e, "#workspace")}
            className="hover:text-slate-950 transition-colors"
          >
            Evidence
          </a>
          <a
            href="/#workspace"
            onClick={(e) => handleAuthGatedClick(e, "#workspace")}
            className="hover:text-slate-950 transition-colors"
          >
            Execution
          </a>
          <a
            href="/#workspace"
            onClick={(e) => handleAuthGatedClick(e, "#workspace")}
            className="hover:text-slate-950 transition-colors"
          >
            Report
          </a>
          <Link to="/about" className="hover:text-slate-950 transition-colors">
            About
          </Link>
        </nav>

        {/* Right Controls / Auth */}
        <div className="hidden sm:flex items-center gap-4">
          {isLoggedIn ? (
            <div className="flex items-center gap-3">
              <Link
                to="/profile"
                className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3.5 py-1.5 text-xs sm:text-sm font-semibold text-slate-800 hover:bg-slate-100 transition-all shadow-2xs"
              >
                <UserIcon className="size-4 text-sky-700" />
                <span>{user?.name}</span>
              </Link>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-xs sm:text-sm font-semibold text-slate-700 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 transition-all shadow-2xs"
                title="Log out"
              >
                <LogOut className="size-4" />
                <span>Logout</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                to="/login"
                className="text-xs sm:text-sm font-semibold text-slate-700 hover:text-slate-950 transition-colors px-3 py-1.5"
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="rounded-full border border-slate-300 bg-white px-4 py-2 text-xs sm:text-sm font-semibold text-slate-950 shadow-2xs hover:bg-slate-50 transition-all"
              >
                Sign Up
              </Link>
              <a
                href="/#workspace"
                onClick={(e) => handleAuthGatedClick(e, "#workspace")}
                className="rounded-full bg-slate-900 px-5 py-2 text-xs sm:text-sm font-semibold text-white shadow-md hover:bg-slate-800 transition-all"
              >
                Start Analysis
              </a>
            </div>
          )}
        </div>

        {/* Mobile Menu Toggle */}
        <button
          onClick={() => setMobileMenuOpen((o) => !o)}
          className="flex sm:hidden p-2 rounded-xl text-slate-700 hover:bg-slate-100"
          aria-label="Toggle Navigation Menu"
        >
          {mobileMenuOpen ? <X className="size-6" /> : <Menu className="size-6" />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="sm:hidden border-t border-slate-200 bg-white px-6 py-4 space-y-3 shadow-lg">
          <a
            href="/#workspace"
            onClick={(e) => {
              setMobileMenuOpen(false);
              handleAuthGatedClick(e, "#workspace");
            }}
            className="block text-sm font-semibold text-slate-800 py-1.5"
          >
            Analyse
          </a>
          <a
            href="/#workspace"
            onClick={(e) => {
              setMobileMenuOpen(false);
              handleAuthGatedClick(e, "#workspace");
            }}
            className="block text-sm font-semibold text-slate-800 py-1.5"
          >
            Evidence
          </a>
          <a
            href="/#workspace"
            onClick={(e) => {
              setMobileMenuOpen(false);
              handleAuthGatedClick(e, "#workspace");
            }}
            className="block text-sm font-semibold text-slate-800 py-1.5"
          >
            Execution
          </a>
          <a
            href="/#workspace"
            onClick={(e) => {
              setMobileMenuOpen(false);
              handleAuthGatedClick(e, "#workspace");
            }}
            className="block text-sm font-semibold text-slate-800 py-1.5"
          >
            Report
          </a>
          <Link
            to="/about"
            onClick={() => setMobileMenuOpen(false)}
            className="block text-sm font-semibold text-slate-800 py-1.5"
          >
            About
          </Link>

          <div className="pt-3 border-t border-slate-200 space-y-2">
            {isLoggedIn ? (
              <>
                <Link
                  to="/profile"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-2 text-sm font-semibold text-slate-900 py-1.5"
                >
                  <UserIcon className="size-4 text-sky-700" />
                  <span>Profile ({user?.name})</span>
                </Link>
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    handleLogout();
                  }}
                  className="flex w-full items-center gap-2 text-sm font-semibold text-rose-600 py-1.5"
                >
                  <LogOut className="size-4" />
                  <span>Logout</span>
                </button>
              </>
            ) : (
              <div className="flex flex-col gap-2 pt-1">
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="w-full text-center rounded-xl border border-slate-300 bg-white py-2.5 text-sm font-semibold text-slate-900"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  onClick={() => setMobileMenuOpen(false)}
                  className="w-full text-center rounded-xl bg-slate-900 py-2.5 text-sm font-semibold text-white shadow-md"
                >
                  Sign Up
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </header>
  );
}

