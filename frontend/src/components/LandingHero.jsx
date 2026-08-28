import { ArrowRight } from 'lucide-react';
import HalftoneReveal from './HalftoneReveal';

export default function LandingHero({ onOpenAuth }) {
  return (
    <section className="relative w-full h-screen min-h-[640px] overflow-hidden flex flex-col justify-center items-center select-none bg-slate-950">

      {/* Background Interactive Halftone WebGL Canvas */}
      <div className="absolute inset-0 z-0 pointer-events-auto">
        <HalftoneReveal
          src="/chris-grant-wVfgzs0oxRk-unsplash.jpg"
          revealSrc="/Gemini_Generated_Image_final.png"
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
      <div className="absolute inset-0 z-10 pointer-events-none bg-slate-950/25" />
      <div className="absolute inset-0 z-10 pointer-events-none bg-gradient-to-t from-slate-950/60 via-transparent to-slate-950/30" />

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
            onClick={onOpenAuth}
            className="group px-7 py-3.5 rounded-full bg-slate-100 hover:bg-white text-slate-950 font-medium text-sm transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-2.5 cursor-pointer focus:outline-none"
          >
            <span>Try Prototype</span>
            <ArrowRight className="w-4 h-4 text-slate-700 group-hover:text-slate-950 transition-transform duration-300 group-hover:translate-x-0.5" />
          </button>
        </div>

      </div>

    </section>
  );
}
