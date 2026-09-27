# DepthWizard UI Redesign — "Monochrome Ivory"

**Goal:** Replace the blue-everywhere look with a professional, top-company-grade monochrome system (Vercel/Linear chrome): near-black neutral surfaces, white primary actions, color reserved strictly for status. Full sweep of every page and component.

## Frozen — will not be touched
- `frontend/src/components/HalftoneReveal.jsx` + `HalftoneReveal.css` — the WebGL shader, dot pattern, cursor loupe reveal, animation loop, fallback: byte-for-byte untouched.
- `frontend/src/components/LandingHero.jsx` — hero layout, halftone props (dotSize/density/angle/revealRadius/ink/paper colors), headline copy, revealing image (`chris-grant-...unsplash.jpg`): untouched.
- 3D terrain engine visuals — GLSL material colors, colormaps, semantic-class hexes, fog/contours (data, not chrome).
- All API wiring, state machines, auth logic, localStorage keys, the `assetFetch`/JWT rules.

**One flagged exception:** the landing **Navbar** currently has a `bg-blue-600` "Try Prototype" button — the only blue left on the landing view. I'll recolor just that button to the same monochrome style used elsewhere (white pill / neutral ghost). No layout or structural change to the landing. If you want even that left alone, say so and I'll skip it.

## New design system (in `frontend/src/index.css` tokens)
| Token | Old (blue-tinted) | New (neutral zinc) |
|---|---|---|
| `--dw-void` | `#07090e` | `#09090b` |
| `--dw-panel` | `#0d1117` | `#111113` |
| `--dw-surface` | `#131923` | `#18181b` |
| `--dw-rim` | `#1c2636` | `#27272a` |
| `--dw-fg / muted / ghost` | slate blues | `#fafafa / #a1a1aa / #71717a` |
| `--dw-accent` | `#3b82f6` blue | `#fafafa` ivory (interactive states) |
| `--dw-fov / probe` | blue rgba | neutral white rgba |
| status colors | confirm/live/fault | unchanged (green/amber/red) |

Plus: white focus rings, zinc scrollbars & selection, shadcn oklch `.dark` tokens re-mapped to zinc, hover = one-step surface lightening (no glows), consistent radii (10px panels / 6px controls), Geist stays as the typeface. I'll follow the project's `impeccable`/`frontend-design` skills and the DESIGN.md banned patterns (no gradient text, no glass, no neon) — and update DESIGN.md's palette section to match the new system.

## Work items (in order)
1. **Token overhaul** — `src/index.css` (dw tokens, selection, scrollbars, focus, shadcn oklch blocks), `index.html` (remove unused Google Fonts: Outfit / Plus Jakarta Sans — loaded but never used; update theme-color).
2. **shadcn primitives check** — `src/components/ui/*` (button/input/field/card): verify variants render correctly on the new zinc tokens; primary button = white bg / black text.
3. **Auth page redesign** — `AuthPage.jsx`, `login-form.jsx`, `signup-form.jsx`: remove blue glow blob, clean centered zinc card, white primary button, same Supabase wiring and form logic.
4. **Accent sweep of the workspace** (~35 files using `var(--dw-accent)` / hardcoded `rgba(59,130,246,…)` / `blue-*` classes): Header.jsx, Toolbar.jsx, AnalysisPanel.jsx, DetailMode.jsx, ViewModeBar.jsx, RecentProjects.jsx, Minimap FOV cone, etc. Special care: anywhere that had "blue background + white text" becomes "white background + black text" (primary); rgba glows become neutral or removed.
5. **Page-by-page polish** — Home (pipeline diagram, upload zone), Processing (progress list), ResultDashboard (layer cards, metrics, export), TerrainWorkspace chrome (panels, HUDs, StatusStrip), FailedPage. Consistent spacing rhythm, 11px uppercase section labels, mono for data readouts, skeleton/empty states in zinc.
6. **Dead code archive** — move verified-unreferenced files into `src/components/_archived/`: `ProcessingView.jsx`, `ProcessingHUD.jsx`, `UploadProcessingView.jsx`, `UploadDropzone.jsx`, old `components/ResultDashboard.jsx`, `HalftoneControls.jsx`, `TechSpecsModal.jsx`, `pages/Validation.jsx` (the `Validation/` component folder stays — it's live). Re-verified by grep before moving; reversible.
7. **Docs** — update `frontend/DESIGN.md` palette/motion sections to the new system so the team's design bible stays true.

## Verification
- `npm run build` (catches syntax) + engine node tests still pass.
- Run the Vite dev server and screenshot landing, auth, home, and whatever workspace state renders without the backend — confirming: halftone effect + hover reveal animation work exactly as before; zero blue remnants (`grep` audit for `#3b82f6`, `59,130,246`, `blue-`, `cyan-` across live files, excluding frozen shader CMY inks and data colormaps); console clean.
- Fix anything the screenshots surface, then re-verify.

## Out of scope
Backend, ML pipeline, API contracts, routing/state logic, terrain engine rendering, the halftone effect itself.