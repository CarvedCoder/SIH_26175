# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary:** Geospatial analysts, disaster management teams, and military reconnaissance officers who need to derive terrain intelligence from single optical images — without specialized photogrammetry equipment or dense point clouds. They are working under time pressure, often in field-adjacent conditions, and need answers fast: "How high is that structure?" "How steep is that slope?" "How does my estimate compare to reference data?"

**Secondary (SIH jury):** Evaluators assessing an end-to-end AI terrain reconstruction pipeline for technical merit, visualisation quality, and practical applicability to disaster management.

## Product Purpose

DepthWizard converts a single optical remote-sensing image into an interactive 3D terrain you can navigate, measure, and validate. It runs a two-path pipeline: if the input is a GeoTIFF with georeferencing, it produces an Absolute DSM with metric elevation; if it is a plain PNG/JPG, it produces a Relative DSM in scene units. The frontend makes this journey explicit and navigable without requiring technical knowledge.

Success means a non-specialist can upload an image, watch it become terrain, walk through that terrain, measure structures and slopes, compare against reference elevation, and export the result — all within a single cohesive session, without reading a manual.

## Positioning

The only product mechanism that cannot be copied by a neighbouring product: **a single optical image becomes a navigable, measurable, validatable 3D terrain in one browser session**, with a minimap that turns the source photograph into a live navigation overlay — a direct, tactile link between the uploaded picture and the reconstructed world.

## Operating Context

- Evaluated in a hackathon demo setting (SIH 2026), 60–120 second live jury walkthrough
- Desktop browser primary; tablet and mobile must degrade gracefully
- Backend: FastAPI running locally at `localhost:8000`; WebGL renderer in-browser via OGL
- Input images: GeoTIFF (georeferenced, up to 4096×4096), PNG/JPG (non-georeferenced)
- Processing model: Depth Anything V2; reference elevation: SRTM where available
- The jury gives 50% weight to DSM accuracy/validation and 50% to visualisation/rendering quality and UX

## Capabilities and Constraints

**Confirmed capabilities:**
- Upload: PNG, JPG, GeoTIFF (with optional reference DEM / GCP file)
- Dual DSM pipeline: Absolute (georeferenced) and Relative (non-georeferenced)
- 3D terrain viewer with OGL/WebGL; tile-based loading; LOD
- Camera modes: First-person (WASD), Orbit, Top View
- Minimap: source image as live camera-position overlay
- Layer system: RGB, Depth, Relative DSM, Absolute DSM, Reference DEM, Error Map, Slope
- Analysis tools: Elevation probe, Height measurement, Distance measurement, Slope measurement, Structure inspector
- Validation: RMSE, MAE, Correlation vs. reference DEM (only when reference is available)
- Export: DSM, Depth map, Validation report, 3D scene (only outputs that actually exist)
- Terrain exaggeration (1×–5×), contour lines, wireframe toggle
- Detail refinement: high-resolution local tile re-processing
- Scenario presets: Terrain Exploration / Disaster Assessment

**Confirmed constraints:**
- No authentication, no user profiles, no social features, no payments
- No fake accuracy: measured ≠ target; relative values must never be labelled as metric
- No invented error messages — always: what happened / why / what to do next
- API polling only (no WebSocket yet); 2–3 s intervals
- Frontend never reproduces model/geospatial calculations; all numeric results come from backend
- Undecided: exact backend endpoint paths until backend is inspected

## Brand Commitments

- Name: **DepthWizard**
- No military targeting imagery, no weapon-adjacent UI
- Presented as "terrain intelligence / preliminary terrain assessment support"
- Professional register; not consumer, not gaming

## Evidence on Hand

- Full UX spec: `DepthWizard_Frontend_3D_UX_Design.md` (83 sections, 2742 lines)
- Incumbent frontend scaffold: React 19, Vite 8, Tailwind CSS v4, OGL, shadcn/ui, Geist Variable
- Existing components: halftone WebGL hero, drag-and-drop upload zone, processing HUD, result dashboard (rough)
- No real terrain data or backend running yet at init time

## Product Principles

1. **Terrain first.** The 3D viewer occupies 70–80% of the screen; every panel exists to serve the terrain, not compete with it.
2. **No fake answers.** Every value on screen is either measured from real data, labelled as relative/estimated, or absent. The UI never invents accuracy.
3. **The source image is the anchor.** The minimap keeps the uploaded photograph visible and spatially meaningful throughout the entire session — users never lose the connection between input and output.
4. **State is always visible.** The user knows whether the system is idle, uploading, processing (which stage), or ready. No blank screens, no unexplained spinners.
5. **Capability-driven controls.** Every tool, layer, and export button is enabled only when the backend confirms the underlying data exists. Nothing is offered that cannot be delivered.

## Accessibility & Inclusion

- Keyboard-navigable controls for all primary actions (click/tap equivalents)
- `prefers-reduced-motion` respected — animations suppress cleanly
- Contrast: body text ≥4.5:1, large text ≥3:1 against dark terrain workspace
- Labels for all measurement values include units; relative values are never unlabelled
