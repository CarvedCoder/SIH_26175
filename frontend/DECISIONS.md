# DepthWizard Frontend — Architecture & Design Decisions

> Running log of every significant choice. Read this before touching code — nothing gets re-derived.

---

## D01 — Framework & toolchain

| Slot | Choice | Rationale |
|------|--------|-----------|
| Framework | React 19 | Already in scaffold; no reason to change |
| Build | Vite 8 | Already in scaffold |
| Styling | Tailwind CSS v4 | Already in scaffold; CSS custom props for design tokens |
| Component base | shadcn/ui + @base-ui/react | Already installed; used for accessible primitives only, not for visual identity |
| Icons | lucide-react | Already installed; `stroke-width: 1.5`, 16px toolbar / 14px panel |
| 3D renderer | OGL | Already installed; low-level WebGL gives full control over vertex shader (exaggeration, colormaps, LOD) |

---

## D02 — State management

**Pattern:** React Context + useReducer (explicit named state machine)  
**Location:** `src/store/appStore.jsx`

States (in order):
```
NO_SCENE → UPLOADING → SCENE_READY → PROCESSING →
RESULTS_READY → TERRAIN_LOADING → TERRAIN_READY → ANALYSIS
                    ↓
                  FAILED → (RETRY | FALLBACK)
```

Rationale: spec §37 and §73 demand explicit states. A minimal Context + useReducer is readable, has no external dependency, and makes illegal state transitions obvious. Measurement tools render only when state ≥ `TERRAIN_READY` (§73 guard).

---

## D03 — API client architecture

**Single client:** `src/api/client.js` — one `apiFetch(path, options)` wrapper  
**Base URL:** `import.meta.env.VITE_API_BASE_URL` (`.env.example`: `http://localhost:8000/api/v1`)  
**Error contract:** always `{ error: { code, message, details, recoverable } }` (spec §69)  
**Behaviour on `error.recoverable = true`:** show retry action; `false`: show fallback/abort action

API modules: `upload.js` / `processing.js` / `results.js` / `terrain.js` / `validation.js` / `export.js`  
Named exports: `uploadScene()`, `startProcessing()`, `getJobStatus()`, `getScene()`, `getResults()`, `getTerrain()`, `getValidation()`, `exportDSM()` etc.

Rationale: spec §72 forbids direct `fetch()` from components.

---

## D04 — Folder structure

```
src/
├── api/              # client.js + domain modules
├── components/
│   ├── common/       # Header, Toolbar, AnalysisPanel, LayerImageCard, RecentProjects, StatusIndicator
│   ├── Upload/       # UploadZone, FileInfo, ProcessingPath
│   ├── Processing/   # PipelineProgress, ProcessingStatus
│   ├── TerrainViewer/# TerrainCanvas, CameraHUD, Minimap, LayerControl, TerrainControls, MeasurementOverlay
│   ├── Analysis/     # ElevationProbe, HeightMeasurement, DistanceMeasurement, SlopeMeasurement, StructureInspector
│   ├── Validation/   # MetricsPanel, ReferenceComparison, ErrorMap, ComparisonView
│   └── Export/       # ExportPanel
├── hooks/            # useUpload, useProcessing, useTerrain, useValidation
├── pages/            # Home, Processing, ResultDashboard, TerrainWorkspace, Validation
├── store/            # appStore.jsx
└── types/            # api.js (JSDoc schemas)
```

Existing flat components directory will be reorganised into this structure. Auth components (AuthPage, login-form, signup-form) archived but not deleted.

---

## D05 — Visual design (derived from DESIGN.md)

The brief pins the world: **scientific / geospatial / mission-control** (spec §30). This is not chosen — it is required. Decisions downstream must not dilute it.

Key choices not re-derived each time:

- **Dark mode only.** Use scene: field operations, disaster assessment, reconnaissance. No light/dark toggle.
- **Accent: instrument ivory `#fafafa`.** Interactive states are luminance, not hue — white borders, white fills, dark text on primary surfaces. Fault/confirm/live are the only chrome hues, status-only.
- **Face: Geist Variable.** Already installed. Precision tooling register. Data values in system monospace.
- **No cards for panels.** Lists with labelled fields. Thin `--dw-rim` borders, no shadows.
- **Minimap is the signature.** 200×200 canvas overlay; camera position in amber; FOV cone in translucent white.
- **Terrain is primary.** 70–80% of usable screen. Panels collapse, toolbar is 48px, header is 48px.

---

## D06 — 3D renderer (React Three Fiber + Three.js)

| Concern | Decision |
|---------|----------|
| Framework | `@react-three/fiber` (R3F) + `@react-three/drei` + `three` |
| Terrain geometry | Multi-tile `THREE.BufferGeometry` with vertex displacement from heightmap |
| Heightmap decode | `OffscreenCanvas` + `ImageBitmap` -> `THREE.DataTexture`; read pixel values as elevation |
| Exaggeration | Vertex shader `uniform float uExaggeration`; visual only |
| Colormaps | Fragment shader `uniform float uColormapMode` (RGB, Greyscale, Viridis, Diverging) |
| Camera controls | Drei `<OrbitControls>` with smooth damping, custom top and first-person fly modes |
| Progressive | Render low-res 64×64 mesh first; swap to high-res 256×256 when ready |
| Texturing | Diffuse `uTexture`; switch texture URL on layer change |
| Disposal | `geometry.dispose()`, `material.dispose()`, `texture.dispose()` on scene change/unmount |

---

## D07 — Minimap coordinate mapping

- **Georeferenced:** `3D world pos → CRS geographic coords (via terrain bounds) → image pixel (via image bounds)`
- **Non-georeferenced:** `normalised terrain [0,1]² → normalised image [0,1]² → pixel`
- Backend provides `bounds` on both terrain and minimap endpoints; frontend must not assume WGS84 (spec §76)
- Camera update is purely frontend — no HTTP call per frame (spec §58)

---

## D08 — Polling

Interval: 2500ms while job `status` is `queued | preprocessing | depth_estimation | geospatial_alignment | scale_calibration | refinement | dsm_generation | validation | terrain_generation`  
Stop: immediately on `completed | failed | cancelled`  
Future: upgrade to SSE when backend supports it

---

## D09 — Landing & Auth Integration

The initial hero section with interactive WebGL visualizer (`LandingHero`) and top navigation (`Navbar`) serves as the entry landing experience. Users can click to authenticate via `AuthPage` (incorporating `LoginForm` and `SignupForm` with social providers and back-navigation). Upon authentication, the app directs straight to the full DepthWizard workspace (`AppRoutes` with `Home`, `Processing`, `ResultDashboard`, and `TerrainWorkspace`), with session persistence and header profile/sign-out controls.

---

## D10 — Units discipline

- Metric elevation values always labelled `m`
- Relative elevation values labelled `scene units` or `relative`; never `m`
- No measurement shown without its unit
- `elevation_mode` from `/results` drives which label appears throughout the session

---

## D11 — Naming conventions

| What | Convention |
|------|-----------|
| React components | PascalCase `.jsx` |
| Hooks | `use` prefix, camelCase `.js` |
| API functions | camelCase (`uploadScene`, `getJobStatus`) |
| State actions | `SCREAMING_SNAKE_CASE` |
| CSS custom props | `--dw-*` prefix |
| Tailwind utilities | primary; custom CSS only for OGL canvas + 2D overlays |

---

*Last updated: 2026-09-07 — impeccable init*
