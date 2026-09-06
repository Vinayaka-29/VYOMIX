import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { Mail, Globe, Send, CheckCircle2, Sparkles, MessageSquare } from "lucide-react";

export const Route = createFileRoute("/contact")({
  head: () => ({
    meta: [
      { title: "Contact Us — Earth Query Lens — VYOMIX" },
      { name: "description", content: "Contact the Earth Query Lens team at VYOMIX." },
    ],
  }),
  component: ContactPage,
});

function ContactPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [sent, setSent] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !email || !message) return;
    setSent(true);
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-8 sm:px-8 lg:px-12 w-full">
        {/* Full Widescreen 2-Column Grid */}
        <div className="grid gap-8 lg:grid-cols-12 items-start">
          {/* Left Column: Info & Details */}
          <div className="lg:col-span-5 space-y-6">
            <div className="panel p-6 sm:p-8 bg-white">
              <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3.5 py-1 text-xs font-semibold text-sky-800">
                <Mail className="size-3.5 text-sky-700" />
                <span>Get in Touch</span>
              </div>

              <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-slate-950 mt-4">
                Contact VYOMIX
              </h1>

              <p className="mt-3 text-sm sm:text-base leading-relaxed font-medium text-slate-700">
                Have questions about Earth Query Lens, dataset integration, or remote sensing analysis features? Send us a message and our team will get back to you.
              </p>

              <div className="mt-8 space-y-4">
                <div className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="flex size-10 items-center justify-center rounded-xl bg-sky-100 text-sky-800 shrink-0">
                    <Mail className="size-5" />
                  </div>
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Contact Inquiry</p>
                    <p className="text-sm sm:text-base font-semibold text-slate-900">contact@vyomix.org</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="flex size-10 items-center justify-center rounded-xl bg-teal-100 text-teal-800 shrink-0">
                    <Globe className="size-5" />
                  </div>
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Project Initiative</p>
                    <p className="text-sm sm:text-base font-semibold text-slate-900">Earth Query Lens (VYOMIX)</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="flex size-10 items-center justify-center rounded-xl bg-indigo-100 text-indigo-800 shrink-0">
                    <Sparkles className="size-5" />
                  </div>
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Specialization</p>
                    <p className="text-sm sm:text-base font-semibold text-slate-900">Multimodal Remote Sensing Intelligence</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Contact Form */}
          <div className="lg:col-span-7">
            <div className="panel p-6 sm:p-8 bg-white">
              <div className="flex items-center gap-3 border-b border-slate-200 pb-4 mb-6">
                <span className="flex size-9 items-center justify-center rounded-xl bg-slate-900 text-white">
                  <MessageSquare className="size-5 text-sky-400" />
                </span>
                <div>
                  <h2 className="text-lg font-bold text-slate-900">Send Us a Message</h2>
                  <p className="text-xs text-slate-500 font-medium">Fill out the form below to connect with our team</p>
                </div>
              </div>

              {sent ? (
                <div className="rounded-2xl border border-emerald-300 bg-emerald-50 p-6 text-center space-y-3">
                  <CheckCircle2 className="mx-auto size-10 text-emerald-600" />
                  <h3 className="text-lg font-bold text-emerald-950">Message Received!</h3>
                  <p className="text-sm font-medium text-emerald-800">
                    Thank you for reaching out to Earth Query Lens. Your message has been recorded successfully.
                  </p>
                  <button
                    onClick={() => {
                      setSent(false);
                      setName("");
                      setEmail("");
                      setSubject("");
                      setMessage("");
                    }}
                    className="mt-2 inline-block rounded-xl bg-emerald-900 px-5 py-2 text-xs font-semibold text-white"
                  >
                    Send Another Message
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                        Your Name
                      </label>
                      <input
                        type="text"
                        required
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Alex Morgan"
                        className="w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                        Email Address
                      </label>
                      <input
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="alex@example.com"
                        className="w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                      Subject
                    </label>
                    <input
                      type="text"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="Question about optical & SAR dataset analysis"
                      className="w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                      Message
                    </label>
                    <textarea
                      required
                      rows={5}
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Write your inquiry or feedback here..."
                      className="w-full resize-none rounded-xl border border-slate-300 bg-white p-3.5 text-sm font-medium text-slate-900 outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10 shadow-2xs"
                    />
                  </div>

                  <button
                    type="submit"
                    className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-7 py-3 text-sm font-semibold text-white shadow-md hover:bg-slate-800 transition-all"
                  >
                    <Send className="size-4" />
                    <span>Send Message</span>
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

