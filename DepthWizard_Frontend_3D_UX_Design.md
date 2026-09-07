
# DepthWizard Frontend & 3D Terrain UX Specification

## 1. Purpose

DepthWizard should feel like a **professional geospatial
terrain-analysis application**, not a generic 3D model viewer.

The frontend must make the complete journey obvious:

**Upload optical image → process → inspect depth/DSM → explore 3D
terrain → measure → compare with reference → export/share results**

The Problem Statement requires an end-to-end system for single-view
optical imagery, support for georeferenced and non-georeferenced inputs,
DSM generation, reference-based scale calibration, and an interactive 3D
terrain environment. The evaluation gives **50% weight to DSM
estimation/accuracy/validation and 50% to visualization/rendering
quality and user experience**, including projection accuracy, visual
fidelity, navigability, interface intuitiveness, stability and
standalone deployment.

Therefore, UI/UX is not decoration. It is a core part of the solution.

------------------------------------------------------------------------

# 2. Product Experience

## Primary User Goal

A user should be able to answer these questions without technical
knowledge:

-   What image did I upload?
-   Is it georeferenced?
-   What did the model predict?
-   Where am I in the reconstructed world?
-   What is the elevation here?
-   How high is this structure?
-   How steep is this terrain?
-   How reliable is the prediction?
-   How does the result compare with reference elevation?
-   Can I inspect the same area from different viewpoints?
-   Can I export the resulting DSM/terrain?

## Design Principle

**Every important visual feature must help the user understand the
terrain or validate the reconstruction.**

Avoid adding decorative UI that does not support the PS.

------------------------------------------------------------------------

# 3. Recommended Application Structure

The website should have four main experiences:

1.  **Home / Upload**
2.  **Processing / Results**
3.  **3D Terrain Workspace**
4.  **Validation & Export**

Recommended navigation:

``` text
DEPTHWIZARD
├── Home
├── New Reconstruction
├── Recent Projects
└── Help / About
```

The main 3D workspace should remain the dominant screen after
processing.

------------------------------------------------------------------------

# 4. Home / Landing Page

The landing page should immediately explain the product.

## Hero

**DepthWizard**

### From a Single Image to an Interactive 3D Terrain

Short supporting line:

> Turn optical remote-sensing imagery into depth, elevation and an
> explorable 3D terrain.

Primary action:

**Upload Image**

Secondary action:

**View Demo**

## Show a simple visual pipeline

``` text
RGB IMAGE
   ↓
DEPTH
   ↓
DSM
   ↓
3D TERRAIN
   ↓
ANALYSIS
```

Do not overload the landing page with technical paragraphs.

## Upload Area

Large drag-and-drop area:

``` text
┌──────────────────────────────────────────────┐
│                                              │
│        Drop your image here                  │
│                                              │
│        or                                    │
│        [ Browse Files ]                      │
│                                              │
│        PNG • JPG • GeoTIFF                   │
│                                              │
└──────────────────────────────────────────────┘
```

Show file name, dimensions, file type and georeferencing status after
selection.

------------------------------------------------------------------------

# 5. Upload Intelligence

Immediately after upload, show a compact file-information panel.

``` text
IMAGE INFORMATION

File:         scene_042.tif
Dimensions:   4096 × 4096
Format:       GeoTIFF
CRS:          Available
Georeferenced: YES

Processing Path:
Absolute DSM
```

For PNG/JPG:

``` text
Format:       JPG
Georeferenced: NO

Processing Path:
Relative DSM
```

This makes the Dual DSM Pipeline understandable without requiring the
user to understand the backend.

------------------------------------------------------------------------

# 6. Processing Experience

Do not show a generic spinner.

Show actual pipeline progress.

``` text
PROCESSING

✓ Image loaded
✓ Remote-sensing preprocessing
✓ Depth estimation
● Geospatial alignment
○ DEM/GCP calibration
○ DSM generation
○ Terrain reconstruction
○ Texture projection
```

At the bottom:

``` text
Processing scene...
Tile 12 / 36
Estimated remaining: --
```

If exact remaining time is unavailable, do not invent it.

## Failure States

If something fails:

``` text
DSM GENERATION INTERRUPTED

We could not complete this stage.

Reason:
Reference elevation data unavailable.

[Retry] [Continue with Relative DSM]
```

Never show a blank screen or unexplained error.

------------------------------------------------------------------------

# 7. Main 3D Terrain Workspace

This is the most important screen in the entire product.

The 3D terrain should occupy roughly **70--80% of the usable screen**.

Do not let panels overpower the terrain.

Recommended structure:

``` text
┌────────────────────────────────────────────────────────────────────┐
│ DEPTHWIZARD          Project: Scene_042        [Export] [Settings] │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ ┌──────────────┐                                                   │
│ │    MINIMAP   │                         3D TERRAIN                │
│ │              │                                                   │
│ │   ┌─────┐    │                       /\                          │
│ │   │  ▲  │    │                  ____/  \____                    │
│ │   │ YOU │    │              ___/              \___               │
│ │   └─────┘    │                                                   │
│ │   FOV ◢      │                                                   │
│ └──────────────┘                                                   │
│                                                                    │
│                                                                    │
│                    [−]  1× ─────●───── 5×  [+]                  │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ [RGB] [DEPTH] [DSM] [REFERENCE] [SLOPE]     [Measure] [Layers]   │
└────────────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 8. Minimap --- Signature UX Feature

The uploaded optical image should become the minimap.

This creates a direct relationship between the source image and
reconstructed 3D world.

## Minimap must show

-   Original uploaded image
-   Current camera position
-   Camera direction
-   Field-of-view cone
-   Current selected point
-   Optional path/trail when navigation is active

Example:

``` text
┌─────────────────────┐
│       SOURCE        │
│       IMAGE         │
│                     │
│        ◢            │
│       /             │
│      ▲              │
│    YOU ARE HERE     │
│                     │
└─────────────────────┘
```

## Behavior

When the camera moves in 3D:

**3D camera position → corresponding source-image position → minimap
marker moves**

When the camera rotates:

**camera heading → minimap direction rotates**

This should be one of the most polished interactions in the application.

------------------------------------------------------------------------

# 9. Camera Modes

Provide three camera modes:

### First Person

For immersive terrain exploration.

### Orbit

For inspecting the entire reconstructed surface.

### Top View

For comparing the terrain with the source image/minimap.

UI:

``` text
CAMERA
[ First Person ] [ Orbit ] [ Top View ]
```

Keyboard shortcuts can be added later, but all important actions must
also be available through visible controls.

------------------------------------------------------------------------

# 10. Layer System

Use a single layer control rather than cluttering the screen.

``` text
LAYERS

● RGB Texture
○ Depth
○ Relative DSM
○ Absolute DSM
○ Reference DEM
○ Error Map
○ Slope
```

## Layer meanings

### RGB Texture

Original optical image projected onto the terrain.

### Depth

Model-predicted relative depth.

### Relative DSM

Relative elevation for non-georeferenced imagery.

### Absolute DSM

Metric elevation after reference-based scale recovery.

### Reference DEM

Available reference elevation.

### Error Map

Difference between estimated and reference elevation.

### Slope

Terrain slope visualization.

------------------------------------------------------------------------

# 11. Height Measurement

This should be a primary tool.

Button:

**Measure Height**

Workflow:

``` text
Click Ground
      ↓
Click Structure Top
      ↓
Calculate Difference
```

Display:

``` text
HEIGHT MEASUREMENT

Ground Elevation     126.2 m
Top Elevation        144.6 m
Estimated Height      18.4 m
```

The selected measurement should remain visible until cleared.

------------------------------------------------------------------------

# 12. Distance Measurement

Add a simple two-point distance tool.

``` text
DISTANCE

Horizontal Distance: 42.1 m
3D Distance:         43.0 m
```

Only display metric units when the dataset is calibrated to a metric
coordinate/elevation reference. For relative imagery, clearly label
values as relative or scene units.

------------------------------------------------------------------------

# 13. Slope Analysis

Button:

**Measure Slope**

User selects two points.

Show:

``` text
SLOPE ANALYSIS

Elevation Difference   8.6 m
Horizontal Distance   42.1 m
Slope                  11.5°
```

Also provide an optional terrain-wide slope layer.

------------------------------------------------------------------------

# 14. Elevation Probe

A lightweight interaction:

``` text
CURSOR LOCATION

Elevation
142.63 m
```

When the user moves/clicks on the terrain, show the elevation at that
location.

For relative DSMs:

``` text
Relative Elevation
+16.4 units
```

Do not label relative depth as meters.

------------------------------------------------------------------------

# 15. Structure Inspection

Clicking a visible structure should highlight the local area.

Example:

``` text
STRUCTURE INSPECTION

Structure ID        BLD-042
Estimated Height    18.4 m
Ground Elevation    126.2 m
Top Elevation       144.6 m
```

If the system cannot reliably identify a building, do not pretend that
it has semantic building detection. A generic **Selected Area** panel is
acceptable.

------------------------------------------------------------------------

# 16. Reference Comparison

For georeferenced imagery with reference elevation:

``` text
REFERENCE COMPARISON

Estimated DSM
Reference DEM
Difference

RMSE          6.8 m
MAE           4.9 m
Correlation   0.91
```

The exact values must come from actual evaluation results.

Never display target values as achieved results.

------------------------------------------------------------------------

# 17. Error Map

Provide a visual comparison mode:

``` text
ESTIMATED DSM
        ↕
REFERENCE DEM
        ↓
ELEVATION ERROR
```

The user should be able to inspect where the model performs well or
poorly.

This is especially useful for:

-   buildings
-   vegetation
-   steep terrain
-   scene boundaries
-   low-detail regions

------------------------------------------------------------------------

# 18. Small-Structure / Detail Mode

A major known challenge is loss of small structures during depth
estimation.

Provide a controlled local refinement workflow.

``` text
DETAIL MODE

Standard ─────────●── High Resolution

Selected area:
[ Refine Area ]
```

Possible flow:

``` text
Select Region
      ↓
High-Resolution Local Tile
      ↓
Depth Refinement
      ↓
Local DSM Update
```

The UI should clearly indicate when local refinement is being processed.

This is better than silently changing the result.

------------------------------------------------------------------------

# 19. Terrain Exaggeration

Add:

``` text
TERRAIN EXAGGERATION

1× ─────────●──────── 5×
```

This changes only the visual vertical exaggeration.

Important label:

> Visual exaggeration --- measured elevation values remain unchanged.

This is particularly useful for flat terrain where elevation differences
are hard to see.

------------------------------------------------------------------------

# 20. Lighting & Visual Quality

The 3D viewer should provide visually strong but controlled rendering.

Recommended:

-   directional lighting
-   ambient lighting
-   soft shadows where performance permits
-   optional wireframe
-   optional contour lines
-   terrain texture
-   subtle fog only if it improves depth perception
-   level-of-detail for large scenes

Avoid excessive bloom, neon effects and gaming-style visual clutter.

The target is:

**professional geospatial visualization**, not a video-game UI.

------------------------------------------------------------------------

# 21. Contours

Optional toggle:

``` text
[✓] Contours
Interval: 5 m
```

For relative DSMs:

``` text
Contour Interval: 5 scene-units
```

Only use metric contour labels when the terrain is metrically
calibrated.

------------------------------------------------------------------------

# 22. Navigation HUD

For first-person mode, provide a minimal HUD:

``` text
ALTITUDE       142.63 m
HEADING        084°
SLOPE          12.4°
POSITION       X / Y
```

Keep this subtle.

The terrain must remain the focus.

------------------------------------------------------------------------

# 23. Disaster-Management Analysis Mode

DepthWizard is under the Disaster Management theme.

Add an optional analysis preset:

``` text
SCENARIO

[Terrain Exploration]
[Disaster Assessment]
```

For Disaster Assessment, emphasize:

-   elevation
-   slope
-   low/high terrain
-   terrain accessibility
-   structure height
-   reference comparison

Do not claim that DepthWizard itself predicts flood or landslide
probability unless a separate validated disaster model is implemented.

The system should be presented as **terrain intelligence / preliminary
terrain assessment support**.

------------------------------------------------------------------------

# 24. Reconnaissance / Terrain Awareness

For the PS's military reconnaissance use case, provide the same
underlying terrain-analysis tools:

-   first-person navigation
-   top view
-   elevation
-   slope
-   structure height
-   source-image minimap
-   terrain comparison

Avoid adding weapon/targeting functionality.

The value is **rapid terrain awareness from optical imagery**.

------------------------------------------------------------------------

# 25. Side Analysis Panel

A collapsible right panel should contain context-sensitive information.

Default:

``` text
SCENE

Image: scene_042.tif
Resolution: 4096 × 4096
Mode: Absolute DSM
Reference: SRTM

MODEL

Depth Model: Depth Anything V2
Status: Complete
```

When the user selects a point:

``` text
SELECTED LOCATION

Elevation: 142.63 m
Slope: 12.4°
```

When the user selects a structure:

``` text
SELECTED STRUCTURE

Height: 18.4 m
Ground: 126.2 m
Top: 144.6 m
```

------------------------------------------------------------------------

# 26. Bottom Tool Bar

Recommended:

``` text
[Layers] [Measure] [Compare] [Terrain] [Camera] [Reset]
```

### Layers

RGB / Depth / DSM / DEM / Slope / Error

### Measure

Height / Distance / Slope / Elevation Probe

### Compare

Estimated DSM / Reference / Error

### Terrain

Exaggeration / Contours / Wireframe

### Camera

First Person / Orbit / Top View

### Reset

Return to default camera and visual settings.

------------------------------------------------------------------------

# 27. Export

The user should be able to export useful results.

``` text
EXPORT

[ DSM ]
[ Relative DSM ]
[ Depth Map ]
[ Terrain Snapshot ]
[ Validation Report ]
[ 3D Scene ]
```

Where supported by the backend.

Do not show export formats that the implementation cannot actually
produce.

------------------------------------------------------------------------

# 28. Project / Session System

A user should not lose a processed scene accidentally.

Example:

``` text
RECENT PROJECTS

Scene_042
Absolute DSM
Processed 2 min ago

Hill_Area
Relative DSM
Processed yesterday

Urban_Block
Absolute DSM
Processed yesterday
```

For the hackathon prototype, this can initially be
in-memory/session-based if persistent project storage is not
implemented.

------------------------------------------------------------------------

# 29. Responsive Design

The primary demo will probably be desktop, but the application should
still degrade gracefully.

Desktop:

**3D terrain + minimap + analysis panel**

Tablet:

**3D terrain + collapsible panels**

Mobile:

**3D terrain + bottom sheet controls**

Do not attempt to display every control simultaneously on small screens.

------------------------------------------------------------------------

# 30. Visual Design Direction

## Overall Style

Use a restrained **scientific / geospatial / mission-control
aesthetic**.

Recommended characteristics:

-   dark terrain workspace
-   high contrast
-   neutral panels
-   one accent color for active states
-   compact typography
-   clean icons
-   thin borders
-   minimal shadows
-   strong spacing
-   large terrain viewport

Avoid:

-   excessive gradients
-   glassmorphism everywhere
-   huge rounded cards
-   excessive animations
-   neon gaming aesthetics
-   decorative statistics with no purpose

------------------------------------------------------------------------

# 31. Important UX Rules

## Rule 1 --- Terrain comes first

The 3D terrain should always be the largest visual element.

## Rule 2 --- Never hide essential information behind hover

Every important action must work with click/tap.

## Rule 3 --- Always show processing state

Never leave the user wondering whether the system is working.

## Rule 4 --- Never fake accuracy

Clearly distinguish:

**Measured Result** vs. **Target** vs. **Reference** vs. **Relative
Value**

## Rule 5 --- Preserve user context

When switching layers, camera position and minimap state should remain
synchronized.

## Rule 6 --- Errors should be understandable

Use:

``` text
What happened
Why it happened
What can I do next?
```

------------------------------------------------------------------------

# 32. Performance Requirements

The UI must remain responsive when working with large terrain scenes.

Implement where appropriate:

-   tile-based loading
-   lazy texture loading
-   level-of-detail meshes
-   frustum culling
-   texture compression where appropriate
-   progressive rendering
-   mesh simplification
-   efficient disposal of GPU resources
-   limited simultaneous high-resolution tiles

Do not load an entire enormous terrain mesh and every texture at maximum
resolution at once.

------------------------------------------------------------------------

# 33. The "WOW" Demo Sequence

For the SIH jury, the demo should take approximately 60--120 seconds.

## Scene 1 --- Upload

``` text
Upload GeoTIFF
```

Immediately show:

``` text
Georeferenced: YES
Processing Path: Absolute DSM
```

## Scene 2 --- Processing

Show the pipeline completing.

## Scene 3 --- Result

Show:

``` text
RGB → Depth → DSM → 3D
```

## Scene 4 --- Enter 3D

Automatically transition into the terrain.

## Scene 5 --- Move

Walk through the terrain.

The minimap marker moves.

The field-of-view cone rotates.

## Scene 6 --- Inspect

Click a structure.

Show height.

## Scene 7 --- Slope

Switch to slope mode.

Show slope.

## Scene 8 --- Validate

Open reference comparison.

Show actual RMSE / MAE / correlation.

## Scene 9 --- Compare

Toggle:

``` text
RGB ↔ DSM ↔ Reference ↔ Error
```

## Scene 10 --- Final

Return to beautiful full-screen 3D terrain.

This sequence tells the complete story without requiring a long verbal
explanation.

------------------------------------------------------------------------

# 34. The Three Screenshots We Must Have

For the PPT and final demo, capture these:

## Screenshot A --- Pipeline

``` text
RGB → Depth → DSM
```

## Screenshot B --- 3D Terrain

A strong terrain view with:

-   minimap
-   camera marker
-   elevation panel
-   clean UI

## Screenshot C --- Validation

``` text
Estimated DSM
Reference DEM
Error
RMSE
MAE
Correlation
```

These three screenshots communicate the majority of the product.

------------------------------------------------------------------------

# 35. Frontend Component Plan

Suggested component structure:

``` text
src/
├── components/
│   ├── Upload/
│   │   ├── UploadZone
│   │   ├── FileInfo
│   │   └── ProcessingPath
│   │
│   ├── Processing/
│   │   ├── PipelineProgress
│   │   └── ProcessingStatus
│   │
│   ├── TerrainViewer/
│   │   ├── TerrainCanvas
│   │   ├── CameraController
│   │   ├── Minimap
│   │   ├── CameraHUD
│   │   ├── LayerControl
│   │   ├── TerrainControls
│   │   └── MeasurementOverlay
│   │
│   ├── Analysis/
│   │   ├── ElevationProbe
│   │   ├── HeightMeasurement
│   │   ├── DistanceMeasurement
│   │   ├── SlopeMeasurement
│   │   ├── StructureInspector
│   │   └── ReferenceComparison
│   │
│   ├── Validation/
│   │   ├── MetricsPanel
│   │   ├── ErrorMap
│   │   └── ComparisonView
│   │
│   ├── Export/
│   │   └── ExportPanel
│   │
│   └── common/
│       ├── Header
│       ├── Toolbar
│       ├── Tooltip
│       └── StatusIndicator
│
├── pages/
│   ├── Home
│   ├── Processing
│   ├── TerrainWorkspace
│   └── Validation
│
└── services/
    └── api
```

Adapt this to the existing repository rather than blindly recreating the
project structure.

------------------------------------------------------------------------

# 36. Backend Contract Expected by the UI

The frontend should be designed around clear backend contracts.

Conceptually:

``` text
POST /upload
POST /process
GET  /process/{id}/status
GET  /result/{id}
GET  /result/{id}/depth
GET  /result/{id}/dsm
GET  /result/{id}/reference
GET  /result/{id}/validation
GET  /result/{id}/terrain
```

The exact endpoints must match the actual backend implementation.

Do not invent frontend API calls that do not exist.

------------------------------------------------------------------------

# 37. State Model

The frontend should have explicit states:

``` text
IDLE
 ↓
UPLOADING
 ↓
UPLOADED
 ↓
PROCESSING
 ↓
DEPTH_READY
 ↓
DSM_READY
 ↓
TERRAIN_READY
 ↓
ANALYSIS
 ↓
VALIDATED
```

Failure:

``` text
PROCESSING
 ↓
ERROR
 ↓
RETRY / FALLBACK
```

This prevents inconsistent UI states.

------------------------------------------------------------------------

# 38. Empty / Partial Results

The UI must gracefully support:

### PNG/JPG

``` text
Absolute elevation unavailable
Relative DSM available
```

### GeoTIFF without reference

``` text
Georeferencing detected
Reference elevation unavailable
Relative DSM available
```

### GeoTIFF + DEM/GCP

``` text
Absolute DSM available
Metric analysis enabled
```

This is a core part of the Dual DSM workflow and should be visible to
the user.

------------------------------------------------------------------------

# 39. What NOT to Build

Do not spend hackathon time on:

-   unnecessary authentication
-   complex user profiles
-   social features
-   generic dashboards
-   payment systems
-   chat systems
-   decorative animations
-   unrelated AI assistants

Every development hour should improve:

**DSM → validation → terrain → navigation → analysis → UX**

------------------------------------------------------------------------

# 40. Definition of a Strong Final Prototype

The prototype should be considered presentation-ready when a judge can:

1.  Upload an image.
2.  Understand which processing path is being used.
3.  See meaningful processing progress.
4.  View the original image.
5.  View the depth result.
6.  View the DSM.
7.  Enter an interactive 3D terrain.
8.  Navigate in first-person mode.
9.  See their position on the source-image minimap.
10. Switch terrain layers.
11. Probe elevation.
12. Measure height.
13. Inspect slope.
14. Compare against reference data when available.
15. View actual validation metrics.
16. Export the resulting output.
17. Recover gracefully from an error.

------------------------------------------------------------------------

# 41. Priority Order for Development

If time is limited, implement in this order:

### P0 --- Must Have

-   Upload
-   Processing state
-   Depth output
-   DSM output
-   3D terrain
-   Texture projection
-   First-person/orbit navigation
-   Minimap synchronization

### P1 --- High Value

-   Elevation probe
-   Height measurement
-   Slope measurement
-   Layer switching
-   Reference DEM comparison
-   Validation metrics
-   Good loading/error states

### P2 --- Strong Polish

-   Terrain exaggeration
-   Contours
-   Error map
-   Structure selection
-   Detail refinement
-   LOD optimization
-   Export workflow

### P3 --- Only if time remains

-   Saved projects
-   Advanced scenario presets
-   Additional visual effects

------------------------------------------------------------------------

# 42. Final Product Philosophy

The application should communicate one idea instantly:

> **"Give us an optical image. We turn it into terrain you can
> understand, measure and explore."**

The strongest UX is not the one with the most buttons.

It is the one where the judge can immediately understand:

**where the image came from → what the model predicted → how elevation
was recovered → what the terrain looks like → how accurate it is → what
the user can do with it.**

That should be the standard for every frontend decision in DepthWizard.

# 43. Backend ↔ Frontend Integration Contract

This section is the implementation contract for the frontend team. The frontend must be built against explicit backend contracts rather than assuming that a result, file, coordinate system, or processing stage exists.

## Integration Principle

The backend owns:

- file validation
- image metadata extraction
- georeferencing detection
- preprocessing
- depth inference
- DEM/GCP retrieval or ingestion
- scale/offset estimation
- DSM generation
- refinement
- validation
- result file generation

The frontend owns:

- upload experience
- processing progress
- visualization
- camera state
- minimap synchronization
- measurements
- layer controls
- comparison UI
- error presentation
- export controls

The frontend should never reproduce model or geospatial calculations that belong to the backend unless the calculation is purely visual.

---

# 44. API Base Contract

Use one configurable API base URL.

Example:

```text
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Production should use the deployed backend URL without hard-coding it into components.

Recommended client structure:

```text
src/
├── api/
│   ├── client.ts
│   ├── upload.ts
│   ├── processing.ts
│   ├── results.ts
│   ├── terrain.ts
│   ├── validation.ts
│   └── export.ts
│
├── hooks/
│   ├── useUpload.ts
│   ├── useProcessing.ts
│   ├── useTerrain.ts
│   └── useValidation.ts
│
└── types/
    └── api.ts
```

If the project uses JavaScript instead of TypeScript, keep equivalent JSDoc/schema definitions so API responses remain predictable.

---

# 45. Required Backend Endpoints

The following are the recommended contracts needed to make the complete UI work.

## 45.1 Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true
}
```

Frontend use:

- backend availability indicator
- startup check
- debugging
- deployment verification

Do not block the entire UI forever if this endpoint is temporarily unavailable. Show a clear connection error.

---

# 46. Upload Endpoint

```http
POST /api/v1/scenes
Content-Type: multipart/form-data
```

Form field:

```text
file
```

Optional fields:

```text
reference_type
reference_file
gcp_file
```

Recommended response:

```json
{
  "scene_id": "scene_042",
  "filename": "scene_042.tif",
  "format": "GeoTIFF",
  "width": 4096,
  "height": 4096,
  "channels": 3,
  "georeferenced": true,
  "crs": "EPSG:4326",
  "bounds": {
    "min_x": 0,
    "min_y": 0,
    "max_x": 1,
    "max_y": 1
  },
  "processing_path": "absolute_dsm",
  "reference_available": false
}
```

For PNG/JPG:

```json
{
  "georeferenced": false,
  "processing_path": "relative_dsm"
}
```

The backend should determine whether the file is actually georeferenced. The frontend should not infer this from the extension alone.

---

# 47. Input Validation Endpoint

Optional but strongly recommended:

```http
POST /api/v1/scenes/{scene_id}/validate
```

Response:

```json
{
  "valid": true,
  "warnings": [
    "Image resolution is relatively low."
  ],
  "errors": [],
  "recommendations": [
    "Local high-resolution refinement may improve small-structure preservation."
  ]
}
```

This lets the UI warn users before expensive processing begins.

---

# 48. Start Processing Endpoint

```http
POST /api/v1/scenes/{scene_id}/process
```

Request:

```json
{
  "model": "depth-anything-v2",
  "tile_size": 1024,
  "overlap": 0.15,
  "enable_refinement": true,
  "enable_reference_calibration": true
}
```

Backend should return immediately rather than keeping a long HTTP request open.

Response:

```json
{
  "job_id": "job_123",
  "scene_id": "scene_042",
  "status": "queued"
}
```

The frontend then follows the job status.

---

# 49. Processing Status Endpoint

```http
GET /api/v1/jobs/{job_id}
```

Response:

```json
{
  "job_id": "job_123",
  "scene_id": "scene_042",
  "status": "processing",
  "stage": "depth_estimation",
  "progress": 47,
  "current_tile": 17,
  "total_tiles": 36,
  "message": "Running depth inference"
}
```

Possible status values:

```text
queued
preprocessing
depth_estimation
geospatial_alignment
scale_calibration
refinement
dsm_generation
validation
terrain_generation
completed
failed
cancelled
```

Frontend mapping:

```text
queued              → Waiting
preprocessing       → Preparing image
depth_estimation    → Estimating depth
geospatial_alignment → Aligning geospatial data
scale_calibration   → Recovering metric scale
refinement          → Refining structures
dsm_generation      → Generating DSM
validation          → Validating result
terrain_generation  → Building 3D terrain
completed           → Ready
failed              → Error
```

If the backend cannot provide exact percentage progress, use stage-based progress rather than inventing a percentage.

---

# 50. Cancel Processing

```http
POST /api/v1/jobs/{job_id}/cancel
```

Response:

```json
{
  "job_id": "job_123",
  "status": "cancelled"
}
```

The UI should provide:

```text
[ Cancel Processing ]
```

and ask for confirmation only if cancellation is destructive.

---

# 51. Scene Summary Endpoint

```http
GET /api/v1/scenes/{scene_id}
```

Response:

```json
{
  "scene_id": "scene_042",
  "status": "completed",
  "input": {
    "filename": "scene_042.tif",
    "width": 4096,
    "height": 4096,
    "format": "GeoTIFF",
    "georeferenced": true,
    "crs": "EPSG:4326"
  },
  "processing": {
    "model": "Depth Anything V2",
    "tile_size": 1024,
    "overlap": 0.15
  },
  "outputs": {
    "depth": true,
    "dsm": true,
    "terrain": true,
    "validation": true
  }
}
```

This powers the scene information panel.

---

# 52. Result Metadata Endpoint

```http
GET /api/v1/scenes/{scene_id}/results
```

Response:

```json
{
  "scene_id": "scene_042",
  "available_layers": [
    "rgb",
    "depth",
    "dsm",
    "reference_dem",
    "error",
    "slope"
  ],
  "elevation_mode": "absolute",
  "units": "meters",
  "bounds": {},
  "resolution": {
    "width": 4096,
    "height": 4096
  }
}
```

For a non-georeferenced image:

```json
{
  "available_layers": [
    "rgb",
    "depth",
    "relative_dsm",
    "slope"
  ],
  "elevation_mode": "relative",
  "units": "scene_units"
}
```

The frontend must use this metadata to decide which controls are available.

---

# 53. Depth Result

```http
GET /api/v1/scenes/{scene_id}/depth
```

Recommended response options:

- image URL
- tiled image URL
- binary raster download
- metadata

Example JSON metadata:

```json
{
  "scene_id": "scene_042",
  "url": "/api/v1/scenes/scene_042/depth/image",
  "min": 0.02,
  "max": 0.98,
  "format": "PNG"
}
```

The frontend should render the returned result and should not attempt to reconstruct the depth map itself.

---

# 54. DSM Result

```http
GET /api/v1/scenes/{scene_id}/dsm
```

Metadata response:

```json
{
  "scene_id": "scene_042",
  "mode": "absolute",
  "units": "meters",
  "format": "GeoTIFF",
  "download_url": "/api/v1/scenes/scene_042/dsm/download",
  "min_elevation": 121.2,
  "max_elevation": 187.6
}
```

For web visualization, the backend should ideally expose a web-optimized representation rather than forcing the browser to parse a massive GeoTIFF.

---

# 55. Web Terrain Endpoint

The 3D viewer needs a browser-friendly representation.

Recommended:

```http
GET /api/v1/scenes/{scene_id}/terrain
```

Response:

```json
{
  "scene_id": "scene_042",
  "terrain_type": "heightfield",
  "width": 1024,
  "height": 1024,
  "bounds": {},
  "height_scale": 1.0,
  "min_elevation": 121.2,
  "max_elevation": 187.6,
  "heightmap_url": "/api/v1/scenes/scene_042/terrain/heightmap",
  "texture_url": "/api/v1/scenes/scene_042/terrain/texture"
}
```

For very large scenes, support tiles:

```http
GET /api/v1/scenes/{scene_id}/terrain/tiles/{z}/{x}/{y}
```

or an equivalent backend-specific terrain-tile scheme.

The frontend must not assume a specific tiling scheme until the backend implements it.

---

# 56. Terrain Tile Endpoint

For scalable rendering:

```http
GET /api/v1/scenes/{scene_id}/terrain/tiles
```

Recommended response:

```json
{
  "tile_size": 256,
  "tiles": [
    {
      "id": "0_0",
      "x": 0,
      "y": 0,
      "resolution": 256,
      "height_url": "...",
      "texture_url": "..."
    }
  ]
}
```

The viewer should load high-resolution tiles near the camera and lower-resolution tiles farther away.

This supports the Level-of-Detail strategy in the UX specification.

---

# 57. Minimap Data Endpoint

The minimap should use the same coordinate system as the 3D terrain.

```http
GET /api/v1/scenes/{scene_id}/minimap
```

Response:

```json
{
  "image_url": "/api/v1/scenes/scene_042/minimap/image",
  "width": 4096,
  "height": 4096,
  "bounds": {
    "min_x": 78.12,
    "min_y": 20.14,
    "max_x": 78.19,
    "max_y": 20.21
  },
  "coordinate_system": "EPSG:4326"
}
```

For non-georeferenced imagery, use normalized image coordinates:

```json
{
  "coordinate_system": "image",
  "width": 4096,
  "height": 4096
}
```

---

# 58. Camera ↔ Minimap Synchronization Contract

This is primarily frontend logic, but the backend must provide enough metadata.

The frontend maintains:

```text
camera_position_3d
camera_heading
camera_fov
terrain_bounds
source_image_bounds
```

Then maps the camera into source-image coordinates.

For georeferenced terrain:

```text
3D world coordinate
        ↓
CRS / geographic coordinate
        ↓
image coordinate
        ↓
minimap marker
```

For non-georeferenced terrain:

```text
normalized terrain coordinate
        ↓
normalized image coordinate
        ↓
minimap marker
```

The backend does not need to receive every camera movement.

Only send camera state to the backend if a future feature requires server-side session recording.

---

# 59. Point Elevation Endpoint

For accurate terrain probing, provide:

```http
GET /api/v1/scenes/{scene_id}/elevation?x={x}&y={y}
```

Response:

```json
{
  "x": 78.154,
  "y": 20.173,
  "elevation": 142.63,
  "units": "meters",
  "source": "estimated_dsm"
}
```

For relative terrain:

```json
{
  "elevation": 0.64,
  "units": "relative"
}
```

The UI must display the correct unit.

For high-frequency mouse movement, do NOT make an HTTP request for every mouse event. Prefer client-side sampling from the loaded heightmap or throttle requests.

---

# 60. Height Measurement Endpoint

If the frontend has access to the terrain heightmap, simple measurements can be computed locally.

For authoritative values, support:

```http
POST /api/v1/scenes/{scene_id}/measure/height
```

Request:

```json
{
  "ground": {
    "x": 78.154,
    "y": 20.172
  },
  "top": {
    "x": 78.154,
    "y": 20.173
  }
}
```

Response:

```json
{
  "ground_elevation": 126.2,
  "top_elevation": 144.6,
  "height": 18.4,
  "units": "meters"
}
```

---

# 61. Slope Endpoint

```http
POST /api/v1/scenes/{scene_id}/measure/slope
```

Request:

```json
{
  "point_a": {},
  "point_b": {}
}
```

Response:

```json
{
  "elevation_difference": 8.6,
  "horizontal_distance": 42.1,
  "slope_degrees": 11.5
}
```

If slope is calculated entirely from the local DSM in the browser, the endpoint is optional.

---

# 62. Reference DEM Endpoint

For scenes where reference data is available:

```http
GET /api/v1/scenes/{scene_id}/reference
```

Response:

```json
{
  "available": true,
  "source": "SRTM",
  "resolution": "30m",
  "crs": "EPSG:4326",
  "download_url": "...",
  "visualization_url": "..."
}
```

If unavailable:

```json
{
  "available": false,
  "reason": "No reference elevation data available for this scene."
}
```

The frontend should then disable or hide reference-specific controls rather than displaying an empty panel.

---

# 63. Validation Endpoint

```http
GET /api/v1/scenes/{scene_id}/validation
```

Response:

```json
{
  "available": true,
  "metrics": {
    "rmse": 6.8,
    "mae": 4.9,
    "correlation": 0.91
  },
  "units": "meters",
  "reference": "SRTM",
  "evaluated_area": {}
}
```

Optional per-region results:

```json
{
  "scene_types": {
    "urban": {},
    "sparse": {},
    "hilly": {},
    "forested": {}
  }
}
```

Only display validation metrics when they have actually been calculated.

---

# 64. Error Map Endpoint

```http
GET /api/v1/scenes/{scene_id}/validation/error-map
```

Response:

```json
{
  "url": "/api/v1/scenes/scene_042/validation/error-map",
  "units": "meters",
  "min_error": -12.2,
  "max_error": 18.4
}
```

Frontend uses this for the **Error Map** layer.

---

# 65. Local Refinement Endpoint

For the small-structure refinement workflow:

```http
POST /api/v1/scenes/{scene_id}/refine
```

Request:

```json
{
  "bbox": {
    "x_min": 0.42,
    "y_min": 0.38,
    "x_max": 0.55,
    "y_max": 0.51
  },
  "resolution": "high"
}
```

Response:

```json
{
  "job_id": "refine_001",
  "status": "queued"
}
```

The frontend should show:

```text
Refining selected area...
```

and update only the affected terrain region when complete.

---

# 66. Export Endpoints

## Export DSM

```http
GET /api/v1/scenes/{scene_id}/export/dsm
```

## Export Depth

```http
GET /api/v1/scenes/{scene_id}/export/depth
```

## Export Validation Report

```http
GET /api/v1/scenes/{scene_id}/export/validation
```

## Export Terrain

```http
GET /api/v1/scenes/{scene_id}/export/terrain
```

Response should either stream the file or return a temporary download URL.

Frontend:

```text
[ Export DSM ]
[ Export Depth ]
[ Export Validation Report ]
[ Export 3D Terrain ]
```

Only enable options for outputs that actually exist.

---

# 67. Scene Delete / Cleanup

Recommended:

```http
DELETE /api/v1/scenes/{scene_id}
```

Use confirmation before deletion.

The backend should clean up temporary files, generated tiles and GPU/processing artifacts.

---

# 68. Optional Recent Scenes Endpoint

If the application has project history:

```http
GET /api/v1/scenes
```

Response:

```json
{
  "items": [
    {
      "scene_id": "scene_042",
      "filename": "urban_block.tif",
      "status": "completed",
      "created_at": "2026-09-06T15:30:00Z"
    }
  ]
}
```

This is optional for the hackathon. It should not delay the core prototype.

---

# 69. API Error Contract

All backend errors should use one predictable structure.

```json
{
  "error": {
    "code": "REFERENCE_DATA_UNAVAILABLE",
    "message": "No reference elevation data is available for this scene.",
    "details": {},
    "recoverable": true
  }
}
```

Suggested error codes:

```text
INVALID_FILE
UNSUPPORTED_FORMAT
FILE_TOO_LARGE
INVALID_GEOTIFF
NO_GEOREFERENCE
REFERENCE_DATA_UNAVAILABLE
DEPTH_MODEL_ERROR
DSM_GENERATION_ERROR
TERRAIN_GENERATION_ERROR
VALIDATION_ERROR
JOB_NOT_FOUND
SCENE_NOT_FOUND
RESOURCE_LIMIT
INTERNAL_ERROR
```

Frontend behavior should be based on `code` and `recoverable`, not string matching arbitrary error messages.

---

# 70. API Timeout & Polling Rules

Do not keep the upload or processing request open for the entire model execution.

Recommended:

```text
POST /process
        ↓
job_id
        ↓
poll /jobs/{job_id}
        ↓
completed
```

Polling:

```text
2–3 seconds during active processing
```

Stop polling immediately when:

```text
completed
failed
cancelled
```

For future production use, WebSocket or Server-Sent Events can replace polling.

For the hackathon, polling is simpler and sufficiently reliable.

---

# 71. CORS

The FastAPI backend must allow the frontend origin during development.

Example development origins:

```text
http://localhost:5173
http://127.0.0.1:5173
```

Production should use an explicit allowed-origin list.

Do not use unrestricted CORS in the final deployment unless there is a specific reason.

---

# 72. API Client Rules for Frontend

Do not call `fetch()` directly from every component.

Use one API client:

```text
api/client
```

Then:

```text
uploadScene()
startProcessing()
getJobStatus()
getScene()
getResults()
getTerrain()
getValidation()
exportDSM()
```

Benefits:

- one base URL
- consistent headers
- consistent error handling
- easier backend changes
- easier testing

---

# 73. Frontend State ↔ Backend State

Use a clear state machine:

```text
NO_SCENE
   ↓
UPLOADING
   ↓
SCENE_READY
   ↓
PROCESSING
   ↓
RESULTS_READY
   ↓
TERRAIN_LOADING
   ↓
TERRAIN_READY
   ↓
ANALYSIS
```

Backend failures:

```text
PROCESSING
   ↓
FAILED
   ↓
RETRY / FALLBACK
```

Do not allow the user to open measurement tools before terrain data is ready.

---

# 74. Capability-Driven UI

The frontend should derive available features from backend result metadata.

Example:

```json
{
  "capabilities": {
    "absolute_elevation": true,
    "relative_elevation": true,
    "reference_comparison": true,
    "slope": true,
    "height_measurement": true,
    "error_map": true,
    "local_refinement": true
  }
}
```

Then the UI automatically enables the correct tools.

For a JPG:

```text
Absolute DSM       disabled
Reference DEM      disabled
Metric Height      disabled
Relative DSM       enabled
3D Terrain         enabled
```

For a calibrated GeoTIFF:

```text
Absolute DSM       enabled
Reference DEM      enabled
Metric Height      enabled
Slope              enabled
Validation         enabled
```

This prevents misleading controls.

---

# 75. 3D Renderer Data Contract

The renderer should receive a normalized scene object:

```text
TerrainScene
├── dimensions
├── bounds
├── coordinateSystem
├── elevationMode
├── units
├── heightmap
├── texture
├── reference
├── layers
└── capabilities
```

The renderer should not know how Depth Anything V2, SRTM or scale calibration works.

It should only consume the generated outputs.

---

# 76. Coordinate System Rules

This is critical for the minimap and measurement tools.

Backend must explicitly return:

```text
CRS
bounds
pixel dimensions
geotransform / equivalent mapping
elevation units
```

Never assume:

```text
latitude = x
longitude = y
```

and never assume every image uses WGS84.

For non-georeferenced imagery, use a clearly documented normalized/image coordinate system.

---

# 77. Image-to-Terrain Mapping

For texture projection, the backend should provide the mapping metadata required by the renderer.

Minimum information:

```text
image width
image height
terrain width
terrain height
bounds
coordinate transform
```

If the terrain is generated directly from the same image grid, document that relationship so the frontend can preserve exact texture alignment.

The UI should expose a **projection alignment check** in development/debug mode:

```text
[✓] Texture aligned with source image
```

This is especially important because the PS evaluates projection accuracy and visual fidelity.

---

# 78. Recommended Backend Deliverables Before Frontend Integration

Backend team should provide:

1. OpenAPI specification.
2. Running `/health` endpoint.
3. Upload endpoint.
4. Processing job endpoint.
5. Job status endpoint.
6. Scene metadata endpoint.
7. Depth result.
8. DSM result.
9. Browser-friendly terrain/heightmap.
10. Original image/minimap asset.
11. Reference DEM metadata where available.
12. Validation metrics.
13. Error map.
14. Export endpoints.
15. Consistent error schema.

FastAPI should expose Swagger/OpenAPI documentation automatically.

The frontend team should use that contract as the source of truth.

---

# 79. Frontend Development Order

Build in this exact order.

## Phase 1 — API Connectivity

```text
Health
↓
Upload
↓
Scene Metadata
↓
Process
↓
Job Status
```

## Phase 2 — Result Viewer

```text
Original RGB
↓
Depth
↓
DSM
```

## Phase 3 — 3D

```text
Heightmap
+
Texture
↓
Terrain Mesh
↓
Camera
```

## Phase 4 — Signature UX

```text
3D Camera
↕
Minimap
```

## Phase 5 — Analysis

```text
Elevation Probe
Height
Slope
Layers
```

## Phase 6 — Validation

```text
Reference DEM
↓
Error Map
↓
RMSE / MAE / Correlation
```

## Phase 7 — Polish

```text
LOD
Progressive Loading
Transitions
Error States
Export
Responsive UI
```

---

# 80. Minimum Viable Backend for the Jury Demo

If time becomes critical, the minimum backend contract is:

```text
GET  /health

POST /scenes
POST /scenes/{id}/process
GET  /jobs/{job_id}

GET  /scenes/{id}
GET  /scenes/{id}/results
GET  /scenes/{id}/depth
GET  /scenes/{id}/dsm
GET  /scenes/{id}/terrain
GET  /scenes/{id}/minimap

GET  /scenes/{id}/validation
GET  /scenes/{id}/export/dsm
```

With these endpoints the frontend can already demonstrate:

**Upload → Processing → Depth → DSM → 3D Terrain → Minimap → Validation → Export**

Everything else can be layered on top.

---

# 81. Important Implementation Constraint

Do not let Claude/Codex or another coding agent invent endpoints that conflict with the existing FastAPI backend.

Before implementation:

1. Inspect the existing backend.
2. List current routes.
3. Identify existing request/response schemas.
4. Reuse existing endpoints where possible.
5. Add only missing endpoints.
6. Update the OpenAPI schema.
7. Connect frontend against the actual contract.
8. Test every endpoint independently.
9. Test the complete upload-to-render flow.

If an endpoint already exists with a different name, adapt the frontend client instead of creating a duplicate endpoint.

---

# 82. Final Integration Acceptance Test

The frontend/backend integration is complete only when this test passes:

```text
[ ] Backend health check works
[ ] Upload GeoTIFF works
[ ] Upload JPG/PNG works
[ ] Georeference status is correctly detected
[ ] Correct Absolute/Relative path is selected
[ ] Processing job starts
[ ] Progress updates correctly
[ ] Processing failure is visible
[ ] Depth result loads
[ ] DSM result loads
[ ] Terrain loads
[ ] RGB texture aligns with terrain
[ ] Camera navigation works
[ ] Minimap marker follows camera
[ ] Camera heading updates minimap
[ ] Layer switching works
[ ] Elevation probe works
[ ] Height measurement works
[ ] Slope analysis works
[ ] Reference DEM loads when available
[ ] Validation metrics display actual values
[ ] Error map loads when available
[ ] Unsupported capabilities are disabled gracefully
[ ] DSM export works
[ ] Page refresh does not corrupt completed-result state
[ ] Large scenes do not freeze the UI
[ ] Backend errors produce understandable UI messages
```

---

# 83. Definition of Done

The final frontend should not be considered complete merely because:

**“The 3D model renders.”**

It is complete when the jury can perform this complete workflow without developer intervention:

```text
UPLOAD
  ↓
UNDERSTAND INPUT
  ↓
PROCESS
  ↓
VIEW DEPTH
  ↓
VIEW DSM
  ↓
ENTER 3D TERRAIN
  ↓
SEE POSITION ON MINIMAP
  ↓
MOVE THROUGH TERRAIN
  ↓
INSPECT ELEVATION
  ↓
MEASURE STRUCTURE HEIGHT
  ↓
CHECK SLOPE
  ↓
COMPARE WITH REFERENCE
  ↓
SEE VALIDATION
  ↓
EXPORT RESULT
```

That is the experience the frontend should be engineered around.
