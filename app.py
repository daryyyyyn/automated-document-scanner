"""
app.py — Streamlit UI for the Automated Document Scanner & Enhancer
====================================================================
Wraps DocumentScanner in a polished drag-and-drop interface with:
  • Side-by-side Before / After visualisation
  • Interactive CLAHE and Canny parameter controls in the sidebar
  • Per-stage latency metrics displayed inline
  • Downloadable scanned PDF output

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from scanner import DocumentScanner, ScannerConfig, ScanResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocScan Pro",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean technical aesthetic, dark sidebar
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    code, pre, .metric-value { font-family: 'JetBrains Mono', monospace; }

    /* header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .app-header h1 { color: #f8fafc; font-size: 2.4rem; font-weight: 800; margin: 0; }
    .app-header p  { color: #94a3b8; font-size: 1rem; margin: .4rem 0 0; }

    /* metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #64748b; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: #38bdf8; font-size: 1.5rem; font-weight: 600; }

    /* status badges */
    .badge-ok  { background:#064e3b; color:#34d399; padding:.25rem .7rem; border-radius:999px; font-size:.8rem; }
    .badge-warn{ background:#451a03; color:#fb923c; padding:.25rem .7rem; border-radius:999px; font-size:.8rem; }

    /* stage table */
    .stage-row { display:flex; justify-content:space-between; padding:.4rem 0;
                 border-bottom:1px solid #1e293b; font-size:.9rem; }
    .stage-name{ color:#cbd5e1; }
    .stage-ms  { color:#38bdf8; font-family:'JetBrains Mono',monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>🗂️ DocScan Pro</h1>
        <p>Week 1 · Computer Vision Portfolio Sprint &nbsp;|&nbsp;
           Perspective warp · CLAHE · Adaptive threshold · MLOps benchmarking</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — interactive parameter controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Configuration")

    st.markdown("### Edge Detection")
    canny_auto = st.toggle("Auto Canny (Otsu-derived)", value=True)
    if not canny_auto:
        canny_low  = st.slider("Canny Low Threshold",  10, 200, 50)
        canny_high = st.slider("Canny High Threshold", 50, 400, 150)
    else:
        canny_low, canny_high = 50, 150   # ignored when auto=True

    st.divider()

    st.markdown("### Bilateral Filter")
    bil_d            = st.slider("Neighbourhood Diameter (d)",  3, 15, 9, step=2)
    bil_sigma_color  = st.slider("Sigma Color",  10, 150, 75)
    bil_sigma_space  = st.slider("Sigma Space",  10, 150, 75)

    st.divider()

    st.markdown("### CLAHE")
    clahe_clip  = st.slider("Clip Limit",      1.0, 8.0, 2.0, step=0.5)
    clahe_tiles = st.select_slider(
        "Tile Grid Size", options=[4, 8, 16, 32], value=8
    )

    st.divider()

    st.markdown("### Output")
    adaptive_thresh = st.toggle("Adaptive Binarisation", value=True)
    adaptive_block  = st.slider("Block Size (odd)", 7, 51, 21, step=2)
    adaptive_c      = st.slider("Subtract Constant C", 2, 30, 10)

    out_width  = st.number_input("Output Width px",  value=1240, step=10)
    out_height = st.number_input("Output Height px", value=1754, step=10)

# ---------------------------------------------------------------------------
# Build scanner from sidebar config
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_scanner(
    bil_d: int, bil_sc: float, bil_ss: float,
    c_auto: bool, c_low: float, c_high: float,
    clahe_clip: float, clahe_tiles: int,
    adaptive: bool, block: int, c_val: int,
    out_w: int, out_h: int,
) -> DocumentScanner:
    cfg = ScannerConfig(
        bilateral_d=bil_d,
        bilateral_sigma_color=bil_sc,
        bilateral_sigma_space=bil_ss,
        canny_auto=c_auto,
        canny_low=c_low,
        canny_high=c_high,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=(clahe_tiles, clahe_tiles),
        adaptive_threshold_fallback=adaptive,
        adaptive_block_size=block,
        adaptive_c=c_val,
        output_width=out_w,
        output_height=out_h,
    )
    return DocumentScanner(cfg)


scanner = build_scanner(
    bil_d, float(bil_sigma_color), float(bil_sigma_space),
    canny_auto, float(canny_low), float(canny_high),
    clahe_clip, int(clahe_tiles),
    adaptive_thresh, adaptive_block, adaptive_c,
    int(out_width), int(out_height),
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.markdown("### 📤 Upload Document Photo")
uploaded = st.file_uploader(
    "Drag & drop or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Demo image fallback
# ---------------------------------------------------------------------------

def _make_demo_image() -> np.ndarray:
    """Synthesise a skewed 'document' so the app works without a real upload."""
    canvas = np.ones((900, 700, 3), dtype=np.uint8) * 220
    # Draw ruled lines
    for y in range(80, 820, 30):
        cv2.line(canvas, (60, y), (640, y), (180, 180, 180), 1)
    # Draw some mock text blocks
    for y in range(90, 400, 30):
        w = np.random.randint(300, 560)
        cv2.line(canvas, (60, y), (60 + w, y), (40, 40, 40), 2)
    # Skew with a mild perspective transform
    src_pts = np.float32([[0, 0], [700, 0], [700, 900], [0, 900]])
    dst_pts = np.float32([[40, 80], [660, 20], [720, 880], [10, 940]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    skewed = cv2.warpPerspective(canvas, M, (740, 980))
    # Add synthetic shadow gradient
    shadow = np.zeros_like(skewed, dtype=np.float32)
    for x in range(skewed.shape[1]):
        alpha = 0.3 * (x / skewed.shape[1])
        shadow[:, x] = skewed[:, x] * (1 - alpha)
    return shadow.astype(np.uint8)


if uploaded is None:
    st.info("💡 No file uploaded — showing a synthesised demo document.")
    raw_bytes = None
else:
    raw_bytes = uploaded.read()


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

def load_image(raw: Optional[bytes]) -> np.ndarray:
    if raw is None:
        return _make_demo_image()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode the uploaded image.")
        st.stop()
    return img


col_scan, col_bench = st.columns([3, 1])

with col_scan:
    if st.button("🔍 Scan Document", type="primary", use_container_width=True):
        with st.spinner("Running pipeline…"):
            img_bgr = load_image(raw_bytes)
            result: ScanResult = scanner.scan(img_bgr)
        st.session_state["result"] = result
        st.session_state["img_bgr"] = img_bgr

# Quick-preview on upload without explicit button press
if uploaded and "result" not in st.session_state:
    img_bgr = load_image(raw_bytes)
    result: ScanResult = scanner.scan(img_bgr)
    st.session_state["result"] = result
    st.session_state["img_bgr"] = img_bgr

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "result" in st.session_state:
    result: ScanResult = st.session_state["result"]
    img_bgr: np.ndarray = st.session_state["img_bgr"]

    # ── Detection badge ──────────────────────────────────────────────────
    badge = (
        '<span class="badge-ok">✅ Document detected</span>'
        if result.document_found
        else '<span class="badge-warn">⚠️ No document polygon found — full-image fallback</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)

    st.markdown("---")

    # ── Before / After ──────────────────────────────────────────────────
    col_before, col_after = st.columns(2, gap="large")

    def bgr_to_pil(img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def gray_to_pil(img: np.ndarray) -> Image.Image:
        return Image.fromarray(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with col_before:
        st.markdown("#### Before — Original + Debug Overlay")
        overlay = scanner.draw_debug_overlay(img_bgr, result.corners)
        st.image(bgr_to_pil(overlay), use_container_width=True)

    with col_after:
        st.markdown("#### After — Scanned & Enhanced")
        st.image(gray_to_pil(result.enhanced), use_container_width=True)

    st.markdown("---")

    # ── Intermediate stages ─────────────────────────────────────────────
    with st.expander("🔬 Intermediate Pipeline Stages", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Edge Map (Canny)**")
            st.image(gray_to_pil(result.preprocessed), use_container_width=True)
        with c2:
            if result.warped is not None:
                st.markdown("**After Perspective Warp (pre-CLAHE)**")
                st.image(bgr_to_pil(result.warped), use_container_width=True)

    st.markdown("---")

    # ── Latency metrics ─────────────────────────────────────────────────
    st.markdown("### ⏱️ Pipeline Latency")

    total_ms = result.total_latency_ms
    target_ms = 200.0
    status_colour = "#34d399" if total_ms < target_ms else "#fb923c"

    metric_cols = st.columns(len(result.metrics) + 1)
    for i, m in enumerate(result.metrics):
        with metric_cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{m.stage_name}</div>
                    <div class="metric-value">{m.latency_ms:.1f}<span style="font-size:.8rem;color:#64748b"> ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with metric_cols[-1]:
        st.markdown(
            f"""
            <div class="metric-card" style="border-color:{status_colour}55">
                <div class="metric-label">Total</div>
                <div class="metric-value" style="color:{status_colour}">
                    {total_ms:.1f}<span style="font-size:.8rem;color:#64748b"> ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Download ────────────────────────────────────────────────────────
    st.markdown("### 💾 Export")

    # PNG download
    pil_out = gray_to_pil(result.enhanced)
    buf_png = io.BytesIO()
    pil_out.save(buf_png, format="PNG")

    # Single-page PDF download via PIL
    buf_pdf = io.BytesIO()
    pil_out.convert("RGB").save(buf_pdf, format="PDF", resolution=150)

    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        st.download_button(
            label="⬇️ Download PNG",
            data=buf_png.getvalue(),
            file_name="scanned_document.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Download PDF",
            data=buf_pdf.getvalue(),
            file_name="scanned_document.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    # Placeholder before any scan
    st.markdown(
        """
        <div style="text-align:center;padding:4rem;color:#475569;
                    border:2px dashed #334155;border-radius:12px;margin-top:1rem;">
            <div style="font-size:3rem">🗂️</div>
            <div style="font-size:1.2rem;font-weight:600;margin-top:.5rem;">
                Upload a document photo and press Scan
            </div>
            <div style="font-size:.9rem;margin-top:.3rem;">
                Supported: JPG · PNG · WEBP · BMP · TIFF
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
