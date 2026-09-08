# DepthWizard Frontend — Progress

## Current Status

> **Done:** Phase 0 + 1 + 2 + 3 + 4 + 5 + 6 complete.
> **In Progress:** Task 7.1 — CameraHUD navigation readouts
> **Next:** Task 8.1 — LayerControl

---

## Phase 0 — Foundation (no visual output yet; build infra)

- [x] **0.1** Folder structure: create `src/api/`, `src/hooks/`, `src/store/`, `src/types/`, `src/pages/`, `src/components/{common,Upload,Processing,TerrainViewer,Analysis,Validation,Export}/`
- [x] **0.2** Design tokens: inject DESIGN.md palette + type scale as CSS custom properties into `index.css`
- [x] **0.3** API client: `src/api/client.js` — one `apiFetch()` wrapper with base URL from `VITE_API_BASE_URL`, normalised error shape `{ error: { code, message, recoverable } }`
- [x] **0.4** API modules: `upload.js`, `processing.js`, `results.js`, `terrain.js`, `validation.js`, `export.js`
- [x] **0.5** Type definitions: `src/types/api.js` — JSDoc schemas for all backend response shapes (Scene, Job, Results, Terrain, Validation, Error)
- [x] **0.6** App state machine: `src/store/appStore.jsx` — React Context + useReducer; states: `NO_SCENE / UPLOADING / SCENE_READY / PROCESSING / RESULTS_READY / TERRAIN_LOADING / TERRAIN_READY / ANALYSIS / FAILED`
- [x] **0.7** Rewrite `App.jsx` — remove auth flow; drive routing entirely from state machine
- [x] **0.8** `.env.example` with `VITE_API_BASE_URL=http://localhost:8000/api/v1`
- [x] **0.9** Verify `npm run dev` starts with zero errors

---

## Phase 1 — Landing Page & Upload (§4, §5, §1.1–1.6)

- [x] **1.1** `Header` component (`src/components/common/Header.jsx`) — wordmark, nav links, backend status dot (polls `/health` on mount)
- [x] **1.2** Hero section with pipeline diagram — headline, subline, "Upload Image" + "View Demo" CTAs, `RGB→Depth→DSM→3D Terrain→Analysis` visual flow
- [x] **1.3** `UploadZone` (`src/components/Upload/UploadZone.jsx`) — drag-and-drop + browse; accepts PNG/JPG/GeoTIFF; calls `POST /scenes` on drop; transitions state to `UPLOADING → SCENE_READY`
- [x] **1.4** `FileInfo` (`src/components/Upload/FileInfo.jsx`) — shows filename, dimensions, format, CRS, georeferenced YES/NO, processing path label
- [x] **1.5** `ProcessingPath` (`src/components/Upload/ProcessingPath.jsx`) — explains Absolute vs Relative DSM pipeline based on `processing_path` from upload response
- [x] **1.6** `Home` page (`src/pages/Home.jsx`) — assembles Hero + UploadZone + FileInfo; shown in `NO_SCENE` + `SCENE_READY` states

---

## Phase 2 — Processing Experience (§6, §49, §50)

- [x] **2.1** `PipelineProgress` (`src/components/Processing/PipelineProgress.jsx`) — stage checklist with ✓ / ● / ○ icons; maps backend stage names to human labels per §49
- [x] **2.2** `ProcessingStatus` (`src/components/Processing/ProcessingStatus.jsx`) — "Processing scene… Tile N / M" footer; no invented ETA
- [x] **2.3** `useProcessing` hook (`src/hooks/useProcessing.js`) — starts job, polls every 2.5s, updates state machine; stops on `completed / failed / cancelled`
- [x] **2.4** Failure panel — "DSM GENERATION INTERRUPTED" with reason, `[Retry]` and `[Continue with Relative DSM]` buttons
- [x] **2.5** Cancel button — confirmation dialog; calls `POST /jobs/{id}/cancel`
- [x] **2.6** `Processing` page (`src/pages/Processing.jsx`) — assembles PipelineProgress + ProcessingStatus + cancel; shown in `PROCESSING` state

---

## Phase 3 — 2D Result Previews (§33 Scene 3, §53, §54)

- [x] **3.1** `LayerImageCard` (`src/components/common/LayerImageCard.jsx`) — image display + label + min/max metadata + units
- [x] **3.2** `ResultDashboard` (`src/pages/ResultDashboard.jsx`) — three cards: RGB / Depth / DSM; fetches `/depth` and `/dsm` metadata; capability-driven (hides Absolute DSM when `elevation_mode: relative`)
- [x] **3.3** "Enter 3D Terrain" CTA — transitions state to `TERRAIN_LOADING`

---

## Phase 4 — 3D Terrain Core (§7, §20, §32, §75)

- [x] **4.1** `TerrainCanvas` (`src/components/TerrainViewer/TerrainCanvas.jsx`) — OGL renderer; full-bleed canvas; initialises Renderer, Camera, Scene, GL context
- [x] **4.2** Heightmap mesh — fetch `/terrain/heightmap` PNG; decode via `OffscreenCanvas`; build `PlaneGeometry` with vertex Y displacement; transitions to `TERRAIN_READY`
- [x] **4.3** RGB texture projection — fetch `/terrain/texture`; apply as diffuse map on terrain mesh
- [x] **4.4** Lighting — directional (sun) + ambient; no bloom, no neon; professional geospatial look (§20)
- [x] **4.5** Terrain exaggeration slider — 1×–5× range; changes `uniform float uExaggeration` in vertex shader only; label: "Visual exaggeration — measured elevation values remain unchanged"
- [x] **4.6** Wireframe toggle — flips `mesh.mode`
- [x] **4.7** Progressive rendering — render low-resolution mesh immediately, swap to full-res when loaded

---

## Phase 5 — Camera System (§9)

- [x] **5.1** Orbit camera controller — mouse drag rotates, scroll zooms, right-drag pans
- [x] **5.2** First-person camera controller — WASD movement, mouse look; terrain-collision height clamp
- [x] **5.3** Top-view camera — orthographic overhead; syncs with minimap view
- [x] **5.4** Camera mode switcher UI — `[ First Person ] [ Orbit ] [ Top View ]` button group in toolbar
- [x] **5.5** Camera reset — returns to default orbit position/target

---

## Phase 6 — Minimap — Signature UX Feature (§8, §57, §58)

- [x] **6.1** `Minimap` component (`src/components/TerrainViewer/Minimap.jsx`) — 200×200 canvas; draws source image as background; sits top-left over terrain viewport
- [x] **6.2** Camera position marker — amber dot at projected 2D position, updated every frame
- [x] **6.3** Camera heading arrow — rotates with camera yaw
- [x] **6.4** FOV cone — filled `--dw-fov` triangle showing camera view frustum
- [x] **6.5** Coordinate mapping — georeferenced: world→CRS bounds→image pixel; non-georeferenced: normalised terrain→normalised image
- [x] **6.6** Selected point marker — secondary dot at last terrain click; preserved across layer switches
- [x] **6.7** Navigation trail — breadcrumb path drawn in first-person mode; max 200 points, oldest culled

---

## Phase 7 — Navigation HUD (§22)

- [ ] **7.1** `CameraHUD` (`src/components/TerrainViewer/CameraHUD.jsx`) — four data-face readouts: ALTITUDE / HEADING / SLOPE / POSITION; shown only in first-person mode
- [ ] **7.2** HUD values from camera + terrain state, no API calls; SLOPE computed from local heightmap gradient at cursor position

---

## Phase 8 — Layer System (§10, §52)

- [ ] **8.1** `LayerControl` (`src/components/TerrainViewer/LayerControl.jsx`) — radio list; pulls `available_layers` from `/results`; disables unavailable layers with tooltip explaining why
- [ ] **8.2** Layer switch handler — swaps active texture/colormap on terrain mesh; camera and minimap state preserved (§31 Rule 5)
- [ ] **8.3** Colormap rendering — greyscale for depth, viridis for DSM, diverging red-blue for error map, applied as fragment shader uniform
- [ ] **8.4** Contour lines toggle + interval input — metric when `elevation_mode: absolute`, scene-units otherwise (§21)

---

## Phase 9 — Analysis Tools (§11–§15)

- [ ] **9.1** `ElevationProbe` (`src/components/Analysis/ElevationProbe.jsx`) — hover crosshair; reads heightmap pixel at cursor; throttled to 60fps; labels "m" or "scene units" per mode (§14)
- [ ] **9.2** `HeightMeasurement` (`src/components/Analysis/HeightMeasurement.jsx`) — click ground → click top → shows Ground Elev / Top Elev / Estimated Height panel; vertical line drawn in 3D (§11)
- [ ] **9.3** `DistanceMeasurement` (`src/components/Analysis/DistanceMeasurement.jsx`) — two-point click; shows Horizontal + 3D distance; units labelled correctly (§12)
- [ ] **9.4** `SlopeMeasurement` (`src/components/Analysis/SlopeMeasurement.jsx`) — two-point click; shows Elevation Difference / Horizontal Distance / Slope° (§13)
- [ ] **9.5** `StructureInspector` (`src/components/Analysis/StructureInspector.jsx`) — click terrain; highlights local area; shows "Selected Area" panel (§15)
- [ ] **9.6** Tool enable guard — measurement tools render but show "Terrain not ready" when state < `TERRAIN_READY`

---

## Phase 10 — Side Analysis Panel (§25)

- [ ] **10.1** `AnalysisPanel` (`src/components/common/AnalysisPanel.jsx`) — collapsible right panel; default: Scene info + Model info
- [ ] **10.2** Context switching — point selected → "Selected Location" (elevation, slope); area clicked → "Selected Structure"
- [ ] **10.3** Collapse/expand with 200ms height transition

---

## Phase 11 — Bottom Toolbar (§26)

- [ ] **11.1** `Toolbar` (`src/components/common/Toolbar.jsx`) — `[Layers] [Measure] [Compare] [Terrain] [Camera] [Reset]`; 48px height, full width, `--dw-panel` background
- [ ] **11.2** Toolbar popover submenus — each button opens a compact popover with its sub-options
- [ ] **11.3** Active tool/layer state styled with `--dw-accent` border

---

## Phase 12 — Reference Comparison & Validation (§16, §17, §63, §64)

- [ ] **12.1** `ReferenceComparison` (`src/components/Validation/ReferenceComparison.jsx`) — Estimated DSM / Reference DEM / Difference; fetches `/reference`; disabled if `available: false`
- [ ] **12.2** `MetricsPanel` (`src/components/Validation/MetricsPanel.jsx`) — RMSE / MAE / Correlation from `/validation`; shown only when `available: true`; values must be actual computed results
- [ ] **12.3** Error map layer — fetches `/validation/error-map`; diverging colormap; integrated with layer system
- [ ] **12.4** `ComparisonView` (`src/components/Validation/ComparisonView.jsx`) — slider or toggle between Estimated / Reference / Error overlays

---

## Phase 13 — Detail Refinement (§18, §65)

- [ ] **13.1** `DetailMode` UI — Standard↔High Resolution slider + "Refine Area" button
- [ ] **13.2** Region selection — rubber-band bounding box drawn over terrain/minimap
- [ ] **13.3** Submit refinement job — `POST /scenes/{id}/refine`; shows "Refining selected area…" progress
- [ ] **13.4** Local tile update — reload only the refined region when refinement job completes

---

## Phase 14 — Disaster Assessment Mode (§23)

- [ ] **14.1** Scenario switcher — `[ Terrain Exploration ] [ Disaster Assessment ]` in toolbar or side panel
- [ ] **14.2** Disaster Assessment preset — re-orders panels to surface elevation, slope, structure height, reference comparison; label: "Terrain intelligence / preliminary terrain assessment support"

---

## Phase 15 — Export (§27, §66)

- [ ] **15.1** `ExportPanel` (`src/components/Export/ExportPanel.jsx`) — buttons for DSM / Depth / Snapshot / Validation Report / 3D Scene
- [ ] **15.2** Capability-driven — only enable buttons for `outputs` that exist in scene summary
- [ ] **15.3** Download trigger — stream or redirect to backend URL

---

## Phase 16 — Recent Projects (§28)

- [ ] **16.1** `RecentProjects` (`src/components/common/RecentProjects.jsx`) — list from localStorage; name / mode / timestamp
- [ ] **16.2** Resume session — click restores scene state and navigates to correct page

---

## Phase 17 — Responsive Design (§29)

- [ ] **17.1** Tablet (≤1024px): side panels collapse by default; toggle to open
- [ ] **17.2** Mobile (≤640px): bottom sheet controls; terrain takes full screen
- [ ] **17.3** All actions reachable at every breakpoint

---

## Phase 18 — Error Handling & Empty States (§6, §31, §38, §69)

- [ ] **18.1** Global error boundary — catches render errors; shows structured recovery message
- [ ] **18.2** API error mapping — `error.code` → human message with what/why/what-next
- [ ] **18.3** Partial-result states — correct messaging for each input type (PNG/JPG / GeoTIFF without reference / GeoTIFF + DEM)
- [ ] **18.4** Backend offline — status dot turns red; show "Backend unreachable" inline; UI stays usable for demo replay

---

## Phase 19 — Polish & Performance (§20, §32)

- [ ] **19.1** Frustum culling — skip rendering tiles outside camera view frustum
- [ ] **19.2** GPU resource disposal — dispose geometry + textures on scene change or unmount
- [ ] **19.3** Throttled elevation reads — client-side heightmap sampling; no HTTP request per mouse event
- [ ] **19.4** Optional fog toggle for depth perception (§20)
- [ ] **19.5** `prefers-reduced-motion` — all transitions collapse to instant
- [ ] **19.6** Keyboard focus styles — visible on all interactive elements; test with Tab navigation

---

## Phase 20 — Demo Readiness (§33, §40, §82)

- [ ] **20.1** Full WOW demo flow works end-to-end without developer intervention
- [ ] **20.2** All 17 acceptance criteria from §40 pass
- [ ] **20.3** All integration checklist items from §82 pass

---

*Last updated: 2026-09-08 — Phase 6 complete; Phase 7 starting*
