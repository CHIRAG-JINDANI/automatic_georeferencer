# GeoRef Studio

**Automatic georeferencing of raster images using AI-assisted feature matching.**

GeoRef Studio lets you take any photograph, scanned map, or historical aerial image and register it precisely onto real-world satellite imagery — producing a GeoTIFF with accurate WGS84 coordinates. The pipeline runs entirely locally: a FastAPI backend handles the heavy computer-vision work, and a Next.js frontend provides a live, step-by-step dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Backend — `main.py`](#backend--mainpy)
  - [The Georeferencing Pipeline](#the-georeferencing-pipeline)
  - [Function Reference](#function-reference)
  - [API Endpoint](#api-endpoint)
- [Frontend](#frontend)
  - [App Stages (State Machine)](#app-stages-state-machine)
  - [Component Reference](#component-reference)
  - [Key Data Types](#key-data-types)
- [Configuration Files](#configuration-files)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Output Format](#output-format)

---

## Overview

The core idea is **image-to-map registration**:

1. The user navigates an interactive satellite map to the approximate region covered by their reference image.
2. A tile screenshot ("proxy image") is captured from the map at that location.
3. The user uploads their reference image (photo, scan, historical aerial, etc.).
4. The backend performs CLAHE enhancement → SIFT feature extraction → FLANN matching → RANSAC affine estimation → perspective warp → GeoTIFF generation.
5. The warped result is overlaid on the live map and can be exported as a georeferenced GeoTIFF (EPSG:4326).

The entire processing pipeline is **streamed** back to the frontend as newline-delimited JSON (NDJSON), so the UI updates in real time — showing intermediate visualisations for each pipeline stage as they complete.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Computer Vision | OpenCV (`cv2`), NumPy, scikit-image |
| Geospatial I/O | Rasterio (GeoTIFF read/write) |
| HTTP Client | HTTPX (async tile fetching) |
| Frontend | Next.js 14, React 18, TypeScript |
| Mapping | Leaflet + react-leaflet |
| Styling | Tailwind CSS, global CSS variables |
| Fonts | Plus Jakarta Sans (Google Fonts) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser (Next.js)                  │
│                                                         │
│  ControlPanel ──► captures proxy URL + uploads ref img  │
│  MapPanel     ──► Leaflet map, ImageOverlay result      │
│  LogPanel     ──► live NDJSON stream log                │
│  ResultPanel  ──► match stats, validate, GeoTIFF DL     │
│  PipelineViewer ► 5-step visual inspector               │
└───────────────────────┬─────────────────────────────────┘
                        │  POST /api/py/process  (multipart form)
                        │  ◄── NDJSON stream (log / step_img / result)
┌───────────────────────▼─────────────────────────────────┐
│                  FastAPI (localhost:8000)                │
│                                                         │
│  /process  StreamingResponse ──► event_generator()      │
│    │                                                     │
│    ├── prep_img()    CLAHE + bilateral filter            │
│    ├── get_kps()     grid-tiled SIFT keypoint detection  │
│    ├── get_matrix()  FLANN + RANSAC affine estimation    │
│    ├── warpAffine()  perspective warp + alpha mask       │
│    └── make_tiff()   rasterio GeoTIFF with EPSG:4326    │
└─────────────────────────────────────────────────────────┘
```

The Next.js `next.config.js` rewrites all `/api/py/*` requests to `http://localhost:8000/*`, so the frontend never needs to know the backend port directly.

---

## Backend — `main.py`

### The Georeferencing Pipeline

The pipeline runs inside a single async generator (`event_generator`) that yields NDJSON events as each stage completes. This is what drives the live log and step images in the UI.

#### Stage 1 — Image Ingestion & Pre-flight

The reference image is read from the multipart upload. Optional horizontal/vertical flips are applied (useful when the source image has an unknown orientation). The proxy map tile is fetched asynchronously via HTTPX from the Esri World Imagery URL captured by the frontend, with up to 3 retries using exponential back-off. Both images are stacked side-by-side and emitted as **Step 1** preview.

#### Stage 2 — CLAHE Enhancement + SIFT Feature Extraction

Both images pass through `prep_img()`:
- Converted to LAB colour space.
- Lightness channel enhanced with CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) to normalise uneven exposure.
- Bilateral filter applied to reduce noise while preserving edges.

Keypoints are extracted via `get_kps()` using a **grid-tiled strategy** (4×4 grid by default): each cell runs Shi-Tomasi first; if too few keypoints are found, SIFT with a lower contrast threshold fills in. This prevents keypoints from clustering in high-contrast regions only. SIFT descriptors are then computed on all detected keypoints. Keypoints are visualised and emitted as **Step 2** preview.

#### Stage 3 — FLANN Matching + RANSAC Consensus

Descriptors are matched with FLANN (`KDTree`, 5 trees, 50 checks). Lowe's ratio test (threshold 0.80) filters ambiguous matches. `cv2.estimateAffinePartial2D` with RANSAC (reprojection threshold 20 px) estimates the best-fit 2D affine transform (translation, rotation, uniform scale — no shear). Inlier matches are drawn with a hue gradient and emitted as **Step 3** preview.

A **GCP residual table** is computed for every inlier: the affine matrix is applied to each source keypoint and the distance to the actual destination point is recorded (`dx`, `dy`, `residual`). This data is returned to the frontend for quality inspection.

#### Stage 4 — Warp, Alpha Mask & Crop

The reference image is warped into proxy-map pixel space with `cv2.warpAffine` (Lanczos4 interpolation). A full-white mask image is warped identically (nearest-neighbour) to track which output pixels contain real data. The transform is padded to a canvas large enough to hold the entire warped image, then tight-cropped to the non-zero mask bounding box. An RGBA image is produced (BGR + alpha mask). The warped result is emitted as **Step 4** preview.

#### Stage 5 — Geo-coordinate Calculation & GeoTIFF Export

The pixel extents of the warped crop are converted to geographic coordinates using the map centre, zoom level, and the Web Mercator metres-per-pixel formula:

```
mpp = 156543.03392 × cos(lat_rad) / 2^zoom
```

`rasterio` writes a GeoTIFF with an affine transform derived from these bounds and CRS `EPSG:4326`. The RGBA image data is laid into 4 bands (R, G, B, α). The GeoTIFF is base64-encoded and returned in the result payload. A composite "stitched" preview (reference warped over proxy) is emitted as **Step 5** preview.

---

### Function Reference

| Function | One-liner |
|---|---|
| `get_mpp(lat, zoom)` | Returns metres-per-pixel at a given latitude and zoom using the Web Mercator scale formula. |
| `get_bounds(lat, lng, zoom, w, h)` | Converts a map centre + zoom + pixel dimensions into geographic bounding box (N, S, E, W). |
| `prep_img(img)` | Applies CLAHE on the L channel (LAB space) + bilateral filter; returns a contrast-normalised BGR image. |
| `get_kps(img, shi, slo, r, c, mkp)` | Grid-tiles the image into `r×c` cells, runs Shi-Tomasi first and falls back to low-threshold SIFT if fewer than `mkp` keypoints are found per cell. |
| `make_preview(img, max_dim)` | Downscales the image to fit within `max_dim` and encodes it as a base64 WebP data-URI for streaming to the frontend. |
| `get_matrix(proxy, ref, img_type)` | Full feature pipeline: prep → grid keypoints → SIFT descriptors → FLANN → ratio test → RANSAC affine. Returns the matrix, inlier list, match score, keypoint debug images, inlier match image, and GCP residual list. |
| `make_tiff(img, n, s, e, w)` | Writes a 4-band RGBA GeoTIFF with `rasterio` using bounds-derived affine transform and EPSG:4326 CRS; returns raw bytes. |
| `event_generator()` | Async generator driving the entire pipeline; yields NDJSON events (`log`, `step_img`, `result`, `error`) consumed by the frontend's `ReadableStream`. |

---

### API Endpoint

#### `POST /process`

Multipart form data. Streams NDJSON back.

| Field | Type | Description |
|---|---|---|
| `reference_image` | `File` | The image to be georeferenced. |
| `proxy_url` | `str` | Full tile URL of the captured satellite map snapshot. |
| `center_lat` | `float` | Latitude of the map centre at capture time. |
| `center_lng` | `float` | Longitude of the map centre at capture time. |
| `zoom` | `float` | Leaflet zoom level at capture time. |
| `map_width` | `int` | Pixel width of the captured map area (default 640). |
| `map_height` | `int` | Pixel height of the captured map area (default 640). |
| `image_type` | `str` | One of `sharp`, `medium`, `blurry` — controls pre-blur applied to the proxy before SIFT. |
| `flip_h` | `str` | `"true"` to horizontally flip the reference image before processing. |
| `flip_v` | `str` | `"true"` to vertically flip the reference image before processing. |

**Stream event types:**

```jsonc
{ "type": "log",      "msg": "preprocessing images" }
{ "type": "step_img", "step": 1, "img": "data:image/webp;base64,..." }
{ "type": "result",   "data": { "stitchedUrl": "...", "geotiffUrl": "...", "overlayBounds": {...}, "inlierCount": 72, "matchScore": 0.9833, "gcpData": [...] } }
{ "type": "error",    "msg": "too few matches" }
```

---

## Frontend

### App Stages (State Machine)

The entire UI is driven by a linear `AppStage` type. Components read this value to decide what to render and whether controls are enabled.

```
idle ──► navigating ──► captured ──► ready ──► processing ──► preview ──► validated
```

| Stage | Meaning |
|---|---|
| `idle` | Initial state; map is shown with a navigation hint. |
| `navigating` | User is panning/zooming; crosshair is visible. |
| `captured` | Proxy tile URL has been locked; awaiting reference image upload. |
| `ready` | Reference image uploaded and quality type selected; pipeline can run. |
| `processing` | NDJSON stream is open; log and step images are updating. |
| `preview` | Pipeline complete; overlay is shown on map; user can validate. |
| `validated` | User accepted the result; GeoTIFF download is available. |

---

### Component Reference

#### `Dashboard` (page root, implicit)
The top-level page component owns all shared state: `stage`, `viewport`, `result`, `logs`, `stepImages`, `capturedProxy`, `referenceFile`, etc. It wires together all child panels and drives the NDJSON streaming fetch.

#### `MapPanel.tsx`
Renders the Leaflet map with two base layers (Esri Satellite, Esri Terrain) switchable via `LayersControl`. Displays:
- An animated crosshair overlay during `navigating` / `captured` / `ready` stages.
- An `ImageOverlay` with the warped result once `stage === 'preview' || 'validated'`.
- A HUD chip showing live resolution (m/px) and coverage (km²).
- A scan-line animation during `processing`.
- A "✓ georeferenced" badge once `validated`.

Emits `onViewportChange` on every `moveend` event so the parent always has the latest lat/lng/zoom for the capture request.

#### `ControlPanel.tsx`
Three-step wizard rendered as collapsible `StepSection` blocks:

1. **Capture proxy map** — shows live viewport coordinates, enables the capture button, displays the locked proxy thumbnail once captured.
2. **Upload reference** — drag-and-drop / file-picker zone, image quality selector (`sharp` / `medium` / `blurry`), optional flip toggles (horizontal / vertical), live preview with applied transforms.
3. **Run pipeline** — shows the fixed pipeline config table (CLAHE params, RANSAC threshold, etc.) and the "Run georeferencing" button.

#### `LogPanel.tsx`
Renders the live terminal log as a scrolling list of timestamped entries (`info`, `ok`, `warn`, `error`, `dim`). The header shows a pipeline inspector button with a step counter (`N/5`). Auto-scrolls to the latest entry on each update.

#### `ResultPanel.tsx`
Shown in the right sidebar after processing completes. Displays:
- A colour-coded match quality bar (poor / moderate / good / excellent) based on inlier count.
- A stats grid: inlier count, CRS, and the four geographic bounds.
- A **Validate result** button (`stage === 'preview'`) or a **Export GeoTIFF** download button (`stage === 'validated'`).

#### `PipelineViewer.tsx`
A modal/drawer that shows all five intermediate step images as a visual inspector. Opened via the arrow button in `LogPanel`. Each step has a label and the captured `step_img` data-URI from the stream. Also renders the **GCP residual table** from `result.gcpData` — showing per-point source pixel, destination pixel, predicted pixel, dx, dy, and residual error in pixels.

#### `MapSearchBox.tsx`
A floating search bar rendered inside the Leaflet map (using `useMap` from react-leaflet). Queries the Photon geocoder API (`photon.komoot.io`) with 300 ms debounce and flies the map to the selected result. All mouse/touch/wheel events are stopped from propagating to the map underneath.

---

### Key Data Types

```typescript
// Geographic viewport — synced from Leaflet on every moveend
type MapViewport = {
  lat: number
  lng: number
  zoom: number
}

// The full pipeline result returned in the NDJSON "result" event
type ProcessingResult = {
  stitchedUrl: string          // base64 WebP data-URI of the warped+cropped RGBA image
  geotiffUrl: string           // base64 GeoTIFF data-URI for download
  overlayBounds: {             // geographic bounds for ImageOverlay
    north: number
    south: number
    east: number
    west: number
  }
  inlierCount: number          // RANSAC inlier match count
  matchScore: number           // normalised score 0–1 (inliers / 30, capped at 1)
  gcpData: GCPPoint[]          // per-inlier residual data for the inspector table
}

// One ground control point residual entry
type GCPPoint = {
  id: number
  src: [number, number]        // keypoint in reference image (pixels)
  dst: [number, number]        // corresponding keypoint in proxy image (pixels)
  pred: [number, number]       // where the matrix maps src to (pixels)
  dx: number                   // horizontal residual
  dy: number                   // vertical residual
  residual: number             // Euclidean residual (pixels)
}

// Live log entry
type LogEntry = {
  ts: number                   // Unix timestamp (ms)
  msg: string
  type: 'info' | 'ok' | 'warn' | 'error' | 'dim'
}
```

---

## Configuration Files

| File | Purpose |
|---|---|
| `next.config.js` | Rewrites `/api/py/*` → `http://localhost:8000/*` so the frontend proxies to FastAPI without exposing the port. |
| `tailwind.config.js` | Sets `Plus Jakarta Sans` as both `font-mono` and `font-sans`; adds `slate-950` to the colour palette. Content paths cover `app/` and `components/`. |
| `globals.css` | Defines all CSS custom properties (`--bg-*`, `--text-*`, `--accent-*`, `--glow-*`) used throughout. Also defines utility classes: `.panel`, `.card`, `.btn-primary`, `.btn-ghost`, `.btn-validate`, `.btn-download`, `.badge-*`, `.progress-*`, `.log-terminal`, `.upload-zone`, `.section-label`, and keyframe animations (`blink`, `pulse-dot`, `fadeIn`). |
| `tsconfig.json` | Standard Next.js TS config with `bundler` module resolution and `@/*` path alias pointing to the project root. |
| `postcss.config.js` | Enables Tailwind + Autoprefixer. |
| `layout.tsx` | Root Next.js layout; sets page title to "GeoRef Studio" and imports global CSS. |
| `globals.d.ts` | Declares `*.css` module type so TypeScript doesn't complain about CSS imports. |
| `requirements.txt` | Python dependencies: `fastapi`, `uvicorn[standard]`, `httpx`, `numpy`, `opencv-python-headless`, `rasterio`, `scikit-image`, `python-multipart`. |

---

## Getting Started

### Backend

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
# Install Node dependencies
npm install

# Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Environment Variables

See `_env.local` for required variables. Copy to `.env.local` before running:

```bash
cp _env.local .env.local
```

---

## Output Format

The exported GeoTIFF is a **4-band RGBA image** in **EPSG:4326 (WGS84)** with a pixel-accurate affine transform. It can be loaded directly into QGIS, ArcGIS, GDAL, or any other GIS tool that supports georeferenced rasters.

```
Band 1 — Red
Band 2 — Green
Band 3 — Blue
Band 4 — Alpha (0 = no data, 255 = valid pixel)
CRS    — EPSG:4326
Driver — GTiff
```
