import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { User as UserIcon, Mail, Shield, LogOut, Sparkles } from "lucide-react";

export const Route = createFileRoute("/profile")({
  head: () => ({
    meta: [
      { title: "Profile — Earth Query Lens — VYOMIX" },
      { name: "description", content: "User profile details for Earth Query Lens." },
    ],
  }),
  component: ProfilePage,
});

function ProfilePage() {
  const { user, isLoggedIn, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    void navigate({ to: "/" });
  };

  if (!isLoggedIn) {
    return (
      <div className="flex min-h-screen flex-col bg-background">
        <Navbar />
        <main className="flex-1 flex items-center justify-center px-4 py-12">
          <div className="panel p-8 text-center max-w-md bg-white">
            <h1 className="text-xl font-bold text-slate-900">Access Restricted</h1>
            <p className="mt-2 text-sm text-slate-600 font-medium">
              Please log in to view your user profile details.
            </p>
            <Link
              to="/login"
              className="mt-6 inline-block rounded-xl bg-slate-900 px-6 py-2.5 text-sm font-semibold text-white shadow-md hover:bg-slate-800"
            >
              Go to Login
            </Link>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-8 sm:px-8 lg:px-12 w-full flex justify-center">
        <div className="panel p-6 sm:p-10 bg-white w-full max-w-3xl">
          <div className="flex items-center gap-4 border-b border-slate-200 pb-6">
            <div className="flex size-14 items-center justify-center rounded-2xl bg-slate-900 text-white shadow-md font-serif text-2xl font-bold">
              {user?.name?.[0]?.toUpperCase() || "U"}
            </div>
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-sky-700">
                Registered Account
              </span>
              <h1 className="text-2xl font-bold text-slate-950">{user?.name}</h1>
              <p className="text-sm font-medium text-slate-500">{user?.email}</p>
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex items-center gap-3.5 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <UserIcon className="size-5 text-slate-600" />
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Full Name</p>
                <p className="text-sm sm:text-base font-semibold text-slate-900">{user?.name}</p>
              </div>
            </div>

            <div className="flex items-center gap-3.5 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <Mail className="size-5 text-slate-600" />
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Email Address</p>
                <p className="text-sm sm:text-base font-semibold text-slate-900">{user?.email}</p>
              </div>
            </div>

            <div className="flex items-center gap-3.5 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <Shield className="size-5 text-slate-600" />
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Product Ecosystem</p>
                <p className="text-sm sm:text-base font-semibold text-slate-900">
                  Earth Query Lens (VYOMIX)
                </p>
              </div>
            </div>
          </div>

          <div className="mt-8 flex justify-between items-center border-t border-slate-200 pt-6">
            <Link
              to="/"
              className="inline-flex items-center gap-2 text-sm font-semibold text-sky-700 hover:underline"
            >
              <Sparkles className="size-4" />
              Back to Workspace
            </Link>
            <button
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-5 py-2.5 text-sm font-semibold text-rose-700 hover:bg-rose-100 transition-colors"
            >
              <LogOut className="size-4" />
              Log Out
            </button>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
