import { ArrowRight } from 'lucide-react';
import HalftoneReveal from './HalftoneReveal';

export default function LandingHero({ onOpenAuth, onOpenApp, user }) {
  const handleAction = () => {
    if (user && onOpenApp) {
      onOpenApp();
    } else {
      onOpenAuth?.();
    }
  };

  return (
    <section className="relative w-full h-screen min-h-[640px] overflow-hidden flex flex-col justify-center items-center select-none bg-[var(--dw-void)]">

      {/* Background Interactive Halftone WebGL Canvas */}
      <div className="absolute inset-0 z-0 pointer-events-auto">
        <HalftoneReveal
          src="/chris-grant-wVfgzs0oxRk-unsplash.jpg"
          // Mouse-reveal circle shows the full-colour RGB aerial (the same
          // photo the halftone is built from) — halftone → real image.
          // (The old path pointed at an image never committed, so the
          // circle rendered black.)
          revealSrc="/chris-grant-wVfgzs0oxRk-unsplash.jpg"
          mode="color"
          dotSize={1.2}
          dotDensity={80}
          angle={45}
          shape="circle"
          contrast={1.35}
          revealRadius={0.34}
          edge={0.78}
          follow={0.28}
          trigger="hover"
          inkColor="#0b0f19"
          paperColor="#f8fafc"
          className="w-full h-full"
        />
      </div>

      {/* Subtle Dark Tint Overlay for Legibility */}
      <div className="absolute inset-0 z-10 pointer-events-none bg-[#0c1220]/25" />
      <div className="absolute inset-0 z-10 pointer-events-none bg-gradient-to-t from-[#0c1220]/70 via-transparent to-[#0c1220]/35" />

      {/* Center Hero: Unboxed, Centered Typography (Google Earth Studio Style) */}
      <div className="relative z-20 max-w-4xl mx-auto px-6 text-center flex flex-col items-center pointer-events-none">

        {/* Headline */}
        <h1 className="font-display text-4xl sm:text-6xl md:text-7xl font-semibold text-slate-100 tracking-tight leading-[1.08]">
          Transform RGB Imagery into Precision Elevation.
        </h1>

        {/* Subhead */}
        <p className="mt-6 text-base sm:text-lg md:text-xl text-slate-300 max-w-2xl font-normal leading-relaxed">
          Extract high-fidelity Digital Surface Models from single-view optical data.
        </p>

        {/* Minimal CTA Button */}
        <div className="mt-10 pointer-events-auto">
          <button
            onClick={handleAction}
            className="group px-7 py-3.5 rounded-full bg-[var(--dw-accent)] hover:bg-[var(--dw-accent-strong)] text-[var(--dw-fg-invert)] font-medium text-sm transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-2.5 cursor-pointer focus:outline-none"
          >
            <span>{user ? 'Open Workspace' : 'Try Prototype'}</span>
            <ArrowRight className="w-4 h-4 text-slate-700 group-hover:text-slate-950 transition-transform duration-300 group-hover:translate-x-0.5" />
          </button>
        </div>

      </div>

    </section>
  );
}
