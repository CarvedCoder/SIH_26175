# Design

<!-- impeccable:design-schema 1 -->

## Visual World

**Instrument panel for terrain that cannot be trusted to look beautiful on its own.**

The brief pins the aesthetic: scientific / geospatial / mission-control. The subject's world — remote sensing, field operations, topographic analysis — has its own visual culture: the muted greens and amber of tactical displays, the dense numeric readouts of LANDSAT imagery products, the thin-ruled grid overlays of survey maps, the monochrome precision of an oscilloscope. DepthWizard borrows that grammar, not as costume but as structure: data lives in labelled fields, not floating cards; elevation speaks in numbers, not gauges; the terrain occupies the screen the way it occupies reality.

The signature element is the **minimap as live mirror**: the uploaded photograph — the one thing the user already trusts — becomes the navigation overlay for the reconstructed world. Every design decision bends around this: panels are narrow and dark so the terrain breathes, the minimap sits in a privileged corner, colours earn their presence through information.

---

## Palette

| Role | Name | Value | Usage |
|------|------|-------|-------|
| Ground | `--dw-void` | `#0a0e17` | Page / canvas background — deep blue-black field |
| Panel | `--dw-panel` | `#0f1522` | Side panels, bottom toolbar, header |
| Surface | `--dw-surface` | `#151d2e` | Cards, input fields, popovers |
| Hover | `--dw-hover` | `#1c2639` | One-step hover lightening on surfaces |
| Rim | `--dw-rim` | `#273650` | Thin 1px borders, dividers |
| Rim / Strong | `--dw-rim-strong` | `#3d5375` | Hover borders, secondary control outlines |
| Text / Primary | `--dw-fg` | `#f1f5f9` | Body copy, panel labels |
| Text / Secondary | `--dw-fg-muted` | `#a9b7cd` | Metadata, secondary labels |
| Text / Tertiary | `--dw-fg-ghost` | `#7f90ab` | Disabled, placeholder |
| Text / Inverted | `--dw-fg-invert` | `#06121f` | Text on cyan primary surfaces |
| Accent / Instrument | `--dw-accent` | `#38bdf8` | Primary buttons, active states, selection, progress |
| Accent / Wash | `--dw-accent-soft` | `rgba(56,189,248,0.14)` | Subtle active backgrounds, hover fills |
| Accent / Dim | `--dw-accent-dim` | `rgba(56,189,248,0.45)` | Active borders, secondary highlights |
| Accent / Confirm | `--dw-confirm` | `#34d399` | Processing complete, validation pass |
| Accent / Live | `--dw-live` | `#fbbf24` | Active processing stage, warnings |
| Accent / Fault | `--dw-fault` | `#f87171` | Errors, failed stages |
| Terrain / Probe | `--dw-probe` | `#38bdf8` | Elevation probe crosshair, measurement markers |
| Minimap / Cone | `--dw-fov` | `rgba(56,189,248,0.18)` | FOV cone fill on minimap |
| Minimap / Marker | `--dw-marker` | `#fbbf24` | Camera position dot |

Colour strategy: **Deep field + instrument cyan**. Surfaces are blue-tinted slate — never pure black — so panels read as depth, not void. Interactive states carry HUE: the cyan accent (`--dw-accent`) fills primary buttons (dark `--dw-fg-invert` text on top), rims active/selected controls, and colors focus rings, so a pressed or focused control is always unambiguously visible. Confirm (green), live (amber), and fault (red) remain status-only. The remaining colour on screen belongs to the data itself (imagery, colormaps, semantic classes).

Data colormaps are exempt from the chrome accent rule: the diverging error map keeps its blue→grey→red ramp, viridis keeps its purples, semantic classes keep their assignment hexes — legends must stay truthful to the GLSL they mirror.

---

## Typography

| Role | Face | Weight | Size | Tracking | Usage |
|------|------|--------|------|----------|-------|
| Display | `Geist Variable` | 600 | clamp(2.5rem, 5vw, 4rem) | -0.03em | Hero headline only |
| UI Label | `Geist Variable` | 500 | 0.6875rem (11px) | 0.06em uppercase | Section eyebrows, panel group labels |
| Body | `Geist Variable` | 400 | 0.875rem (14px) | 0em | Panel content, descriptions |
| Data | `ui-monospace, SFMono-Regular, Menlo` | 400 | 0.8125rem (13px) | 0em | Elevation values, coordinates, metrics |
| Data / Large | `ui-monospace` | 500 | 1.125rem (18px) | -0.01em | Primary readout values (altitude, RMSE) |
| Caption | `Geist Variable` | 400 | 0.75rem (12px) | 0.02em | Units, axis labels, supplementary info |

Faces chosen from the subject's world: Geist is the face of precision tooling and instrument firmware — compact, engineered, zero warmth, exactly right for a terrain analysis tool. Monospace for numbers because numbers are data, not prose.

No Fraunces, Playfair, Cormorant, Space Grotesk, or any of the category defaults.

---

## Spacing & Layout

```
Base unit: 4px (0.25rem)
Panel padding: 12px (3 units)
Section gap inside panel: 16px (4 units)  
Group gap (label + fields): 8px (2 units)
Terrain viewport: 70-80% of usable screen width
Minimap: 200×200px, top-left of terrain viewport
Side panel: 280px collapsed width, collapsible
Bottom toolbar: 48px height, full width
Header: 48px height
```

Structure is functional: the sidebar labels its own sections with a compact uppercase group label (8px above group, 4px below), then fields. No decorative dividers except a single 1px `--dw-rim` line between major panel sections.

---

## Component Character

- **Panels:** `background: --dw-panel`, `border: 1px solid --dw-rim`, `border-radius: 6px`. No shadow — depth comes from the rim, not the shadow.
- **Buttons / toolbar items:** `border-radius: 4px`, `height: 32px`, `padding: 0 10px`. Active state: `background: --dw-surface`, `border: 1px solid --dw-accent`.
- **Input fields:** `border-radius: 4px`, `border: 1px solid --dw-rim`, `background: --dw-surface`. Focus: `border-color: --dw-accent`.
- **Sliders:** Custom track 2px high, `--dw-rim` unfilled, `--dw-accent` filled. Thumb 12px, `--dw-fg`, no border-radius on the track container.
- **Status dots:** 6px circle: `--dw-confirm` (online), `--dw-live` (processing), `--dw-fault` (error), `--dw-fg-ghost` (unknown).
- **Pipeline stages:** monospace label, status icon left (✓ / ● / ○), no card chrome — a list, not a set of cards. A 1px `--dw-rim` connector rail runs through the icon column.
- **Progress bar:** 4px track, `--dw-rim` unfilled, status-coloured fill (`--dw-live` while processing, `--dw-confirm` complete, `--dw-fault` failed). Width is driven ONLY by backend-reported `job.progress` — never animated synthetically; the bar never moves backwards.
- **Minimap:** `border: 1px solid --dw-rim`, `border-radius: 4px`, canvas overlay for FOV cone + markers drawn directly in 2D canvas context.
- **Layer control:** radio group, no cards — a labelled list with a left indicator dot, `--dw-accent` when active.
- **Measurement readouts:** two-column table, left column `--dw-fg-muted` (label), right column data face (value + unit).

---

## Motion

Single authored moment per surface:

- **Processing view:** the progress bar eases to each newly reported backend percentage over 600ms `ease-out`; stage items animate in as each stage completes (slide + fade, 200ms `ease-out`). The tile counter counts up numerically from the backend's live message.
- **Terrain load:** mesh fades in from flat plane to full displacement over 600ms on first load (vertex shader lerp).
- **Layer switch:** active texture cross-fades on the terrain mesh over 250ms.
- **Minimap marker:** position interpolated with lerp(0.15) every frame — smooth, instrument-needle feel.
- **Panel collapse/expand:** 200ms `ease-out` height transition.
- **No entrance animations on panels or toolbar items** — they are not a reveal, they are infrastructure.
- `prefers-reduced-motion`: all transitions collapse to instant swaps.

---

## Icons

lucide-react, `stroke-width: 1.5`, consistent 16px in toolbar, 14px in panels. No filled icons. No emoji. No Unicode substitutes.

---

## Anti-patterns (banned for this project)

- Gradient text or gradient fills on UI chrome
- Glass/blur as decoration — blur only on the minimap overlay (functional transparency)
- Kicker/eyebrow labels above headings (craft-floor absolute ban)
- Section numbers (01/02/03) — they are not a sequence the user navigates
- `border-left` accents on cards or list items
- Hard-offset box shadows (neobrutalism costume — not this world)
- Rounded-pill buttons anywhere except the landing hero CTA
- Any neon accent, bloom, HDR glow
- Decorative statistics (big number + small label) with no data behind them
- Progress rings or sparklines as decoration
- Generic AI spinner — always show the actual pipeline stage

---

## Mode

**Operate** — the user is completing analytical tasks. Expression lives in precise details: the minimap, the data-face readouts, the ivory active state. The terrain viewport earns all the screen space it can get.

---

## Surface briefs

See `.impeccable/surfaces/` for per-page direction contracts as they are written.

