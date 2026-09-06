import { createFileRoute } from "@tanstack/react-router";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { AboutSection } from "@/components/AboutSection";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — Earth Query Lens — VYOMIX" },
      { name: "description", content: "Learn about Earth Query Lens, a VYOMIX project." },
    ],
  }),
  component: AboutPage,
});

function AboutPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 mx-auto max-w-[1440px] px-4 py-8 sm:px-8 lg:px-12 w-full">
        <AboutSection />
      </main>

      <Footer />
    </div>
  );
}
