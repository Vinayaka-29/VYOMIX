import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { Eye, EyeOff, Lock, Mail, Sparkles, AlertCircle } from "lucide-react";

function GoogleIcon() {
  return (
    <svg className="size-5" viewBox="0 0 24 24">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
      />
    </svg>
  );
}

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Login — Earth Query Lens — VYOMIX" },
      { name: "description", content: "Log in to your Earth Query Lens account." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const { login, loginWithGoogle } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleSuccessRedirect = () => {
    void navigate({ to: "/" });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!email.trim()) {
      setError("Please enter your email address.");
      return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      setError("Please enter a valid email address.");
      return;
    }
    if (!password) {
      setError("Please enter your password.");
      return;
    }

    setBusy(true);
    try {
      login(email.trim());
      handleSuccessRedirect();
    } catch (err) {
      setError("Failed to sign in. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const handleGoogleSignIn = () => {
    loginWithGoogle();
    handleSuccessRedirect();
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 flex items-center justify-center px-4 py-12 sm:px-6">
        <div className="w-full max-w-md">
          {/* Card */}
          <div className="panel p-8 sm:p-10 bg-white/95 shadow-xl backdrop-blur-md">
            {/* Header Branding */}
            <div className="text-center">
              <div className="inline-flex items-center justify-center size-12 rounded-2xl bg-slate-900 text-white shadow-md mb-4">
                <Sparkles className="size-6 text-sky-400" />
              </div>
              <p className="text-xs font-bold uppercase tracking-widest text-sky-700">
                VYOMIX presents
              </p>
              <h1 className="font-serif text-3xl font-bold tracking-tight text-slate-950 mt-1">
                Earth Query Lens
              </h1>
              <h2 className="text-lg font-semibold text-slate-700 mt-2">Welcome back</h2>
              <p className="text-xs sm:text-sm text-slate-500 font-medium mt-1">
                Enter your credentials to access your workspace
              </p>
            </div>

            {error && (
              <div className="mt-6 flex items-start gap-2.5 rounded-2xl border border-rose-300 bg-rose-50 p-3.5 text-xs sm:text-sm font-medium text-rose-800">
                <AlertCircle className="mt-0.5 size-4 shrink-0 text-rose-600" />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              {/* Email */}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-slate-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="user@vyomix.com"
                    className="w-full rounded-xl border border-slate-300 bg-white pl-10 pr-4 py-3 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-slate-400" />
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full rounded-xl border border-slate-300 bg-white pl-10 pr-10 py-3 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((s) => !s)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700"
                    aria-label="Toggle password visibility"
                  >
                    {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                  </button>
                </div>
              </div>

              {/* Remember me & Forgot Password */}
              <div className="flex items-center justify-between text-xs sm:text-sm font-medium">
                <label className="flex items-center gap-2 cursor-pointer text-slate-700">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="size-4 rounded border-slate-300 text-slate-900 focus:ring-slate-900"
                  />
                  <span>Remember me</span>
                </label>
                <a href="#forgot" className="text-sky-700 font-semibold hover:underline">
                  Forgot password?
                </a>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-xl bg-slate-900 py-3.5 text-sm font-semibold text-white shadow-md hover:bg-slate-800 transition-all disabled:opacity-50 mt-2"
              >
                {busy ? "Signing in..." : "Login"}
              </button>
            </form>

            {/* OR Divider */}
            <div className="my-6 flex items-center gap-3">
              <div className="h-px flex-1 bg-slate-200" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">OR</span>
              <div className="h-px flex-1 bg-slate-200" />
            </div>

            {/* Google Authentication Button */}
            <button
              type="button"
              onClick={handleGoogleSignIn}
              className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-300 bg-white py-3 text-sm font-semibold text-slate-800 shadow-xs hover:bg-slate-50 transition-all"
            >
              <GoogleIcon />
              <span>Continue with Google</span>
            </button>

            <div className="mt-6 text-center text-xs sm:text-sm font-medium text-slate-600">
              Don't have an account?{" "}
              <Link to="/signup" className="font-semibold text-slate-950 hover:underline">
                Sign up
              </Link>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

