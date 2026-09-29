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
| Ground | `--dw-void` | `#0c1220` | Page / canvas background |
| Panel | `--dw-panel` | `#111a2c` | Side panels, bottom toolbar, header |
| Surface | `--dw-surface` | `#17223a` | Cards, input fields, popovers |
| Hover | `--dw-hover` | `#1d2b47` | One-step hover lightening on surfaces |
| Rim | `--dw-rim` | `#2a3a58` | Thin 1px borders, dividers |
| Rim / Strong | `--dw-rim-strong` | `#43587c` | Hover borders, secondary control outlines |
| Glass | `--dw-glass` | `rgba(15,23,41,0.92)` | Translucent in-viewport bars over terrain |
| Text / Primary | `--dw-fg` | `#eef2f9` | Body copy, panel labels |
| Text / Secondary | `--dw-fg-muted` | `#9db0cc` | Metadata, secondary labels |
| Text / Tertiary | `--dw-fg-ghost` | `#64748b` | Disabled, placeholder |
| Text / Inverted | `--dw-fg-invert` | `#071224` | Text on sky primary surfaces |
| Accent / Instrument | `--dw-accent` | `#4cc3ff` | Active states, selected elements, progress indicators |
| Accent / Strong | `--dw-accent-strong` | `#7dd3fc` | Hover on accent surfaces |
| Accent / Wash | `--dw-accent-soft` | `rgba(76,195,255,0.12)` | Subtle active backgrounds, hover fills |
| Accent / Confirm | `--dw-confirm` | `#34d399` | Processing complete, validation pass |
| Accent / Live | `--dw-live` | `#fbbf24` | Active processing stage, warnings |
| Accent / Fault | `--dw-fault` | `#f87171` | Errors, failed stages |
| Terrain / Probe | `--dw-probe` | `#e2e8f0` | Elevation probe crosshair, measurement markers |
| Minimap / Cone | `--dw-fov` | `rgba(76,195,255,0.16)` | FOV cone fill on minimap |
| Minimap / Marker | `--dw-marker` | `#fbbf24` | Camera position dot |

Colour strategy: **Slate instrument (single-hue accent)**. Deep navy-slate surfaces carry the UI — softer than pure black, cool enough to read as mission-control hardware rather than a consumer dark theme. Interactive states use the sky instrument accent (`--dw-accent`): a sky-filled primary button, a sky border, a sky progress bar — always paired with `--dw-fg-invert` (deep navy) text so labels stay legible on every fill. Status hues (confirm/live/fault) remain reserved strictly for status; the remaining colour on screen belongs to the data itself (imagery, colormaps, semantic classes).

Data colormaps are exempt from the accent rule: the diverging error map keeps its blue→grey→red ramp, viridis keeps its purples, semantic classes keep their assignment hexes — legends must stay truthful to the GLSL they mirror.

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
- **Pipeline stages:** monospace label, status icon left (✓ / ● / ○), no card chrome — a list, not a set of cards.
- **Minimap:** `border: 1px solid --dw-rim`, `border-radius: 4px`, canvas overlay for FOV cone + markers drawn directly in 2D canvas context.
- **Layer control:** radio group, no cards — a labelled list with a left indicator dot, `--dw-accent` when active.
- **Measurement readouts:** two-column table, left column `--dw-fg-muted` (label), right column data face (value + unit).

---

## Motion

Single authored moment per surface:

- **Processing view:** stage items animate in as each stage completes (slide + fade, 200ms `ease-out`). Tile progress counter counts up numerically — no progress bar, just the number.
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

