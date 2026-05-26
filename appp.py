import io
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import requests
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

st.set_page_config(
    page_title="AI Georeferencer",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a;
    color: #c8d6e5;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.03em;
}

[data-testid="stAppViewContainer"] h1 {
    font-size: 1.55rem !important;
    color: #e8f4fd;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 0.5rem;
    margin-bottom: 0.2rem;
}

[data-testid="stAppViewContainer"] h3 {
    font-size: 0.78rem !important;
    color: #4a90b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}

.stButton > button {
    background: linear-gradient(135deg, #1a6eaf 0%, #0d4a7a 100%);
    color: #e8f4fd;
    border: 1px solid #2a8fd4;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2184cc 0%, #1260a0 100%);
    border-color: #4ab8f5;
    box-shadow: 0 0 18px rgba(42, 143, 212, 0.35);
}

[data-testid="stFileUploader"] {
    border: 1px dashed #1e3a52;
    border-radius: 6px;
    background: #0f1927;
    padding: 0.5rem;
}

[data-testid="stFileUploader"] label {
    color: #6a9ab8 !important;
    font-size: 0.8rem;
}

.stAlert {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    border-radius: 4px;
}

[data-testid="stImage"] img {
    border: 1px solid #1e3a52;
    border-radius: 4px;
}

.block-container {
    max-width: 1400px;
    padding-left: 2rem;
    padding-right: 2rem;
}

.metric-pill {
    display: inline-block;
    background: #0f1e30;
    border: 1px solid #1e3a52;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4ab8f5;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #3a6a88;
    margin-bottom: 0.35rem;
}

.result-card {
    background: #0f1927;
    border: 1px solid #1e3a52;
    border-radius: 6px;
    padding: 0.8rem;
    margin-top: 0.5rem;
}

div[data-testid="stCaption"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3a6a88;
    text-align: center;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def _tile_xy(lat: float, lon: float, zoom: int):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def fetch_satellite_image(bbox: list, target_zoom: int = 15) -> Image.Image:
    south, west, north, east = bbox
    zoom = target_zoom
    headers = {"User-Agent": "AI-Georeferencer/1.0 (research)"}

    for _ in range(3):
        x_min, y_max = _tile_xy(south, west, zoom)
        x_max, y_min = _tile_xy(north, east, zoom)
        cols, rows = x_max - x_min + 1, y_max - y_min + 1
        if cols * 256 <= 2048 and rows * 256 <= 2048:
            break
        zoom -= 1

    canvas = Image.new("RGB", (cols * 256, rows * 256), (20, 25, 35))
    url_tpl = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    for tx in range(x_min, x_max + 1):
        for ty in range(y_min, y_max + 1):
            url = url_tpl.format(z=zoom, y=ty, x=tx)
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                canvas.paste(tile, ((tx - x_min) * 256, (ty - y_min) * 256))
            except Exception:
                pass
    return canvas


@st.cache_resource(show_spinner=False)
def load_dino():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    return processor, model


def patch_features(img: Image.Image, processor, model) -> torch.Tensor:
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    feats = out.last_hidden_state[:, 1:, :]          # drop CLS  (1, N, D)
    return F.normalize(feats, dim=-1)





def best_bbox_from_heatmap(heatmap: np.ndarray, top_pct: float = 0.15):
    threshold = np.percentile(heatmap, (1 - top_pct) * 100)
    mask = (heatmap >= threshold).astype(np.uint8)
    rows_hit = np.where(mask.any(axis=1))[0]
    cols_hit = np.where(mask.any(axis=0))[0]
    if rows_hit.size == 0 or cols_hit.size == 0:
        g = heatmap.shape[0]
        return 0, g - 1, 0, g - 1
    return rows_hit[0], rows_hit[-1], cols_hit[0], cols_hit[-1]


def heatmap_overlay(base: Image.Image, heatmap: np.ndarray,
                    bbox_grid, alpha: float = 0.55) -> Image.Image:
    g = heatmap.shape[0]
    W, H = base.size
    rmin, rmax, cmin, cmax = bbox_grid

    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colormap = plt.cm.plasma
    colored = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored).resize((W, H), Image.BILINEAR)
    heat_pil = heat_pil.filter(ImageFilter.GaussianBlur(radius=8))

    overlay = Image.blend(base.convert("RGB"), heat_pil, alpha=alpha)

    draw = ImageDraw.Draw(overlay)
    x1 = int(cmin / g * W)
    y1 = int(rmin / g * H)
    x2 = int((cmax + 1) / g * W)
    y2 = int((rmax + 1) / g * H)

    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=(0, 220, 255) if offset == 0 else (0, 150, 200),
        )
    return overlay


def crop_localized(base: Image.Image, bbox_grid, g: int,
                   ref_img: Image.Image = None) -> Image.Image:
    W, H = base.size
    rmin, rmax, cmin, cmax = bbox_grid

    cx = int(((cmin + cmax + 1) / 2) / g * W)
    cy = int(((rmin + rmax + 1) / 2) / g * H)

    if ref_img is not None:
        rw, rh = ref_img.size
        # crop is ~1.6x ref area in each axis — always bigger than ref, never the whole map
        linear_scale = 1.265  # sqrt(1.6) — makes crop area ~1.6x the ref area
        crop_w = int(min(rw * linear_scale, W * 0.55))
        crop_h = int(min(rh * linear_scale, H * 0.55))
    else:
        crop_w = W // 4
        crop_h = H // 4

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(W, x1 + crop_w)
    y2 = min(H, y1 + crop_h)
    if x2 == W:
        x1 = max(0, W - crop_w)
    if y2 == H:
        y1 = max(0, H - crop_h)
    return base.crop((x1, y1, x2, y2))


# ── session state ────────────────────────────────────────────────────────────

for key, default in [
    ("bbox", None),
    ("map_image", None),
    ("heatmap", None),
    ("overlay", None),
    ("cropped", None),
    ("ref_image", None),
    ("sim_score", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── header ───────────────────────────────────────────────────────────────────

st.markdown("# 🛰️ AI Georeferencer")
st.markdown(
    '<p style="color:#3a6a88;font-family:\'DM Sans\',sans-serif;font-size:0.85rem;margin-top:-0.4rem;">'
    "Zero-shot aerial image localization · DINOv2 dense feature matching</p>",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.35, 1], gap="large")


# ── LEFT: map panel ──────────────────────────────────────────────────────────

with col_left:
    st.markdown("### Map Panel")
    st.markdown(
        '<p class="section-label">Draw a rectangle to define the search region</p>',
        unsafe_allow_html=True,
    )

    m = folium.Map(
        location=[34.404, -118.534],
        zoom_start=13,
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="ESRI World Imagery",
    )
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark Base",
        overlay=False,
        control=True,
        opacity=0.0,
    ).add_to(m)
    draw_ctrl = Draw(
        draw_options={
            "rectangle": {"shapeOptions": {"color": "#00dcff", "weight": 2}},
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polyline": False,
        },
        edit_options={"edit": False, "remove": False},
    )
    draw_ctrl.add_to(m)

    # JS: after a rectangle is finished —
    #   1. remove every previously drawn shape so only one bbox exists at a time
    #   2. disable the active draw handler so no second rectangle starts automatically
    _dv  = draw_ctrl.get_name()           # e.g. "draw_control_abc123"
    _div = f"drawnItems_{_dv}"            # matching featureGroup name
    _fix = folium.Element(f"""
    <script>
    (function waitForDraw() {{
        var dc = window['{_dv}'];
        var di = window['{_div}'];
        if (!dc || !di || !di._map) {{ setTimeout(waitForDraw, 250); return; }}
        di._map.on('draw:created', function () {{
            // keep only the newest shape — remove everything drawn before it
            var layers = [];
            di.eachLayer(function (l) {{ layers.push(l); }});
            layers.slice(0, -1).forEach(function (l) {{ di.removeLayer(l); }});
            // exit draw mode so no second rectangle begins
            setTimeout(function () {{
                for (var k in dc._toolbars) {{
                    var tb = dc._toolbars[k];
                    if (tb && tb._activeMode) {{
                        tb._activeMode.handler.disable();
                    }}
                }}
            }}, 60);
        }});
    }})();
    </script>
    """)
    m.get_root().html.add_child(_fix)

    map_data = st_folium(m, width="100%", height=480, returned_objects=["last_active_drawing"])

    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            st.session_state.bbox = [min(lats), min(lons), max(lats), max(lons)]

    if st.session_state.bbox:
        s, w, n, e = [round(v, 5) for v in st.session_state.bbox]
        st.markdown(
            f'<span class="metric-pill">S {s}</span>'
            f'<span class="metric-pill">W {w}</span>'
            f'<span class="metric-pill">N {n}</span>'
            f'<span class="metric-pill">E {e}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#2a5070;font-size:0.75rem;font-family:\'Space Mono\',monospace;">'
            "↖ Draw a rectangle on the map to set search bounds</span>",
            unsafe_allow_html=True,
        )


# ── RIGHT: controls + results ────────────────────────────────────────────────

with col_right:
    st.markdown("### Reference Image")
    st.markdown('<p class="section-label">Upload a map fragment, aerial photo, or document</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PNG / JPEG reference", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    if uploaded:
        ref = Image.open(uploaded).convert("RGB")
        st.session_state.ref_image = ref
        st.image(ref, use_column_width=True, caption="Reference image")

    st.markdown("<br>", unsafe_allow_html=True)
    process = st.button("🔍 Process Match", type="primary")

    if process:
        if not st.session_state.bbox:
            st.error("Draw a bounding box on the map first.")
        elif not st.session_state.ref_image:
            st.error("Upload a reference image before processing.")
        else:
            bar   = st.progress(0)
            label = st.empty()
            timer = st.empty()

            def update(pct: int, msg: str, detail: str = ""):
                bar.progress(pct)
                label.markdown(
                    f'<p style="font-family:\'Space Mono\',monospace;font-size:0.72rem;'
                    f'color:#4ab8f5;margin:0;">{msg}'
                    + (f'<span style="color:#2a5070;"> — {detail}</span>' if detail else "")
                    + "</p>",
                    unsafe_allow_html=True,
                )

            import time
            t0 = time.time()

            def elapsed():
                return f"{time.time() - t0:.1f}s"

            try:
                update(5, "⬇  Fetching satellite tiles", "connecting to ESRI servers …")
                map_img = fetch_satellite_image(st.session_state.bbox)
                st.session_state.map_image = map_img
                w, h = map_img.size
                update(28, "✓  Tiles fetched", f"{w}×{h}px · {elapsed()}")
            except Exception as exc:
                bar.empty(); label.empty()
                st.error(f"Tile fetch failed: {exc}")
                st.stop()

            try:
                model_cached = "dinov2_model" in st.session_state
                update(
                    32,
                    "⚙  Loading DINOv2-base backbone",
                    "already cached" if model_cached else "downloading ~330 MB on first run …",
                )
                processor, model = load_dino()
                update(50, "✓  DINOv2 ready", f"86M params · {elapsed()}")
            except Exception as exc:
                bar.empty(); label.empty()
                st.error(f"Model load failed: {exc}")
                st.stop()

            try:
                update(55, "🔬  Encoding map area", "forward pass through ViT backbone …")
                map_feats = patch_features(st.session_state.map_image, processor, model)
                update(68, "✓  Map patches encoded", f"196 patch embeddings · {elapsed()}")

                update(72, "🔬  Encoding reference image", "forward pass through ViT backbone …")
                ref_feats = patch_features(st.session_state.ref_image, processor, model)
                update(82, "✓  Reference image encoded", f"{elapsed()}")

                update(85, "📐  Computing cosine similarity", "196 × 768-dim dot products …")
                ref_vec = ref_feats.mean(dim=1)
                sim = torch.einsum("bnd,bd->bn", map_feats, ref_vec).squeeze(0).numpy()
                g = int(round(math.sqrt(len(sim))))
                heatmap = sim[: g * g].reshape(g, g)
                st.session_state.heatmap = heatmap
                update(90, "✓  Similarity heatmap built", f"{g}×{g} grid · {elapsed()}")

                update(93, "📍  Localising best-match region", "thresholding top 12% patches …")
                bbox_grid = best_bbox_from_heatmap(heatmap, top_pct=0.12)
                st.session_state.sim_score = float(
                    heatmap[bbox_grid[0]: bbox_grid[1] + 1,
                            bbox_grid[2]: bbox_grid[3] + 1].mean()
                )

                update(96, "🎨  Rendering overlay & crop", "blending heatmap …")
                st.session_state.overlay = heatmap_overlay(
                    st.session_state.map_image, heatmap, bbox_grid
                )
                st.session_state.cropped = crop_localized(
                    st.session_state.map_image, bbox_grid, heatmap.shape[0],
                    ref_img=st.session_state.ref_image
                )

                update(100, f"✅  Done in {elapsed()}", "results below")
                time.sleep(0.6)
                bar.empty(); label.empty(); timer.empty()

            except Exception as exc:
                bar.empty(); label.empty()
                st.error(f"Feature matching failed: {exc}")
                st.stop()

    # ── results ──────────────────────────────────────────────────────────────

    if st.session_state.heatmap is not None:
        score = st.session_state.sim_score
        score_color = "#00ff99" if score > 0.6 else "#ffd166" if score > 0.4 else "#ef6c6c"

        st.markdown(
            f'<div class="result-card">'
            f'<span class="section-label">Match confidence</span><br>'
            f'<span style="font-family:\'Space Mono\',monospace;font-size:1.3rem;color:{score_color};">'
            f'{score:.3f}</span>'
            f'<span style="color:#3a6a88;font-size:0.7rem;"> cosine similarity</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.image(st.session_state.map_image, use_column_width=True, caption="Fetched area")

        with r2:
            st.image(st.session_state.overlay, use_column_width=True, caption="Similarity overlay")

        st.image(st.session_state.cropped, use_column_width=True, caption="Localized region (best match crop)")

        with st.expander("Raw similarity heatmap"):
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0b0f1a")
            ax.set_facecolor("#0b0f1a")
            im = ax.imshow(st.session_state.heatmap, cmap="plasma", interpolation="bilinear")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
            ax.set_title("Patch cosine similarity", color="#c8d6e5",
                         fontsize=9, fontname="monospace")
            ax.tick_params(colors="#3a6a88", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e3a52")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
