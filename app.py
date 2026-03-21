"""
CT Scan Denoising — Interactive Web Application
================================================
A sleek Streamlit app for real-time CT / X-ray image denoising using the
trained U-Net model.

Run with:
    streamlit run app.py
"""

import os
import io
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_metric
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CT Scan Denoiser | U-Net",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d14;
    color: #e8e8f0;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    border: 1px solid #2a2a4a;
    box-shadow: 0 8px 32px rgba(124, 110, 253, 0.15);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c6efd, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub {
    color: #888898;
    font-size: 1.05rem;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #151520, #1e1e30);
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #2a2a3a;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(124,110,253,0.2);
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 4px 0 2px;
}
.metric-label {
    font-size: 0.82rem;
    color: #888898;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-delta {
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 4px;
}

/* Upload zone */
.upload-zone {
    background: #10101c;
    border: 2px dashed #2a2a4a;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    transition: border-color 0.2s;
}

/* Section headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #7c6efd;
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a3a;
}

/* Image labels */
.img-label {
    text-align: center;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 8px;
}
.label-noisy    { background: rgba(231,76,60,0.18);  color: #e74c3c; }
.label-denoised { background: rgba(46,204,113,0.18); color: #2ecc71; }
.label-clean    { background: rgba(52,152,219,0.18); color: #3498db; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10101c;
    border-right: 1px solid #1e1e2e;
}

/* Plotly chart bg */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* Badges */
.badge {
    display: inline-block;
    background: rgba(124,110,253,0.15);
    color: #7c6efd;
    border: 1px solid rgba(124,110,253,0.4);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #10101c;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1e1e2e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888898;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e1e2e !important;
    color: #7c6efd !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
IMG_H = IMG_W = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), "denoising_model.h5")
METRICS_CSV = os.path.join(os.path.dirname(__file__), "denoising_metrics.csv")
HISTORY_PNG = os.path.join(os.path.dirname(__file__), "training_history.png")

DARK_BG   = "#0d0d14"
CARD_BG   = "#151520"
ACCENT    = "#7c6efd"
GREEN     = "#2ecc71"
RED       = "#e74c3c"
AMBER     = "#f39c12"
BLUE      = "#3498db"
SUBTEXT   = "#888898"


# ─── Caching ───────────────────────────────────────────────────────────────────
class _Cast(tf.keras.layers.Layer):
    """Compatibility shim for mixed-precision Cast layers saved in older TF."""
    def __init__(self, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._target_dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._target_dtype or self.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg["dtype"] = self._target_dtype
        return cfg


@st.cache_resource(show_spinner=False)
def load_model():
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    # First attempt: standard load
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        pass
    # Second attempt: register Cast shim for mixed-precision models
    try:
        with tf.keras.utils.custom_object_scope({"Cast": _Cast}):
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        pass
    # Third attempt: safe_mode off (Keras 3+)
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_CSV):
        return None
    return pd.read_csv(METRICS_CSV)


# ─── Processing helpers ────────────────────────────────────────────────────────
def pil_to_gray_array(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    if CV2_AVAILABLE:
        return cv2.resize(arr, (IMG_W, IMG_H))
    # Fallback: PIL resize
    pil_gray = pil_img.convert("L").resize((IMG_W, IMG_H))
    return np.array(pil_gray, dtype=np.float32) / 255.0


def denoise_image(model, noisy: np.ndarray) -> np.ndarray:
    inp  = noisy.reshape(1, IMG_H, IMG_W, 1)
    pred = model.predict(inp, verbose=0)
    return pred[0].reshape(IMG_H, IMG_W)


def compute_metrics(clean, noisy, denoised):
    noisy_mse    = float(np.mean((clean - noisy)    ** 2))
    denoised_mse = float(np.mean((clean - denoised) ** 2))
    if SKIMAGE_AVAILABLE:
        return {
            "noisy_psnr":     psnr_metric(clean, noisy,    data_range=1.0),
            "denoised_psnr":  psnr_metric(clean, denoised, data_range=1.0),
            "noisy_ssim":     ssim_metric(clean, noisy,    data_range=1.0),
            "denoised_ssim":  ssim_metric(clean, denoised, data_range=1.0),
            "noisy_mse":      noisy_mse,
            "denoised_mse":   denoised_mse,
        }
    # Rough PSNR fallback if skimage not available
    def _psnr(a, b):
        mse = np.mean((a - b) ** 2)
        return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
    return {
        "noisy_psnr":     _psnr(clean, noisy),
        "denoised_psnr":  _psnr(clean, denoised),
        "noisy_ssim":     0.0,
        "denoised_ssim":  0.0,
        "noisy_mse":      noisy_mse,
        "denoised_mse":   denoised_mse,
    }


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


def gauge_chart(value: float, title: str, min_v: float, max_v: float, colour: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        title={"text": title, "font": {"color": "#e8e8f0", "size": 13}},
        gauge={
            "axis":  {"range": [min_v, max_v], "tickcolor": SUBTEXT},
            "bar":   {"color": colour},
            "bgcolor": CARD_BG,
            "borderwidth": 1,
            "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [min_v, (max_v - min_v) * 0.4 + min_v], "color": "#1e1e2e"},
                {"range": [(max_v - min_v) * 0.4 + min_v, max_v],  "color": "#1a1a2e"},
            ],
        },
        number={"font": {"color": colour, "size": 28}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
    )
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 12px 0 20px;'>
            <span style='font-size:3rem'>🫁</span><br>
            <span style='font-size:1.25rem; font-weight:700; color:#7c6efd;'>CT Denoiser</span><br>
            <span style='font-size:0.78rem; color:#888898;'>U-Net · Deep Learning</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        for item in [
            ("Model",    "Optimised U-Net"),
            ("Input",    "256×256 Grayscale"),
            ("Encoder",  "4 blocks (32→512 ch)"),
            ("Decoder",  "4 blocks + skip conn"),
            ("Loss",     "Mean Squared Error"),
            ("Optimizer","Adam (lr=1e-3)"),
            ("Mixed Prec","float16 ✓"),
        ]:
            st.markdown(
                f"<span style='color:{SUBTEXT}; font-size:0.82rem'>{item[0]}</span>&nbsp;"
                f"<span style='float:right; font-size:0.82rem; color:#e8e8f0'>{item[1]}</span><br>",
                unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Datasets Used")
        for ds in ["COVID-19 Radiography", "Montgomery & Shenzhen TB",
                   "LIDC-IDRI Lung Cancer", "NIH Chest X-ray"]:
            st.markdown(f"<span class='badge'>🗄 {ds}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Inference Settings")
        show_comparison = st.checkbox("Show side-by-side comparison", value=True)
        show_gauges     = st.checkbox("Show metric gauges",            value=True)
        return show_comparison, show_gauges


# ─── Hero ──────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🫁 CT Scan Denoising</p>
        <p class="hero-sub">
            Upload a noisy CT / X-ray image and watch our U-Net model restore it in real-time.<br>
            Powered by TensorFlow · Trained on 4 clinical datasets · 100% PSNR improvement success rate.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Tab 1 · Denoise ───────────────────────────────────────────────────────────
def tab_denoise(model, show_comparison, show_gauges):
    col_upload, col_ref = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="section-header">📤 Upload Noisy Image</p>', unsafe_allow_html=True)
        noisy_file = st.file_uploader(
            "Drag & drop a CT/X-ray image (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="noisy_uploader"
        )

    with col_ref:
        st.markdown('<p class="section-header">🔵 Upload Clean Reference <span style="font-weight:400; color:#888898; font-size:0.85rem">(optional — for metrics)</span></p>', unsafe_allow_html=True)
        clean_file = st.file_uploader(
            "Drop clean reference image for PSNR / SSIM evaluation",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="clean_uploader"
        )

    if noisy_file is None:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#888898;'>
            <div style='font-size:3rem'>⬆️</div>
            <p style='font-size:1.1rem; margin:8px 0 4px; color:#e8e8f0'>Upload a CT scan to get started</p>
            <p style='font-size:0.85rem'>Supported: PNG · JPG · BMP · TIFF</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not installed in this Python environment. Run: `pip install tensorflow`")
        return
    if model is None:
        st.error("❌ Model not found. Please train the model first: `python train.py`")
        return

    # ── Load & process ──────────────────────────────────────────────────────
    noisy_pil  = Image.open(noisy_file)
    noisy_arr  = pil_to_gray_array(noisy_pil)

    with st.spinner("🧠 Running U-Net inference …"):
        denoised_arr = denoise_image(model, noisy_arr)

    clean_arr = None
    if clean_file:
        clean_arr = pil_to_gray_array(Image.open(clean_file))

    # ── Image display ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🖼️ Results</p>', unsafe_allow_html=True)

    if clean_arr is not None:
        c1, c2, c3 = st.columns(3, gap="medium")
        img_cols = [(c1, noisy_arr,    "🔴  Noisy Input",         "label-noisy"),
                    (c2, denoised_arr, "🟢  Denoised Output",     "label-denoised"),
                    (c3, clean_arr,    "🔵  Clean Ground Truth",  "label-clean")]
    else:
        c1, c2 = st.columns(2, gap="large")
        img_cols = [(c1, noisy_arr,    "🔴  Noisy Input",     "label-noisy"),
                    (c2, denoised_arr, "🟢  Denoised Output", "label-denoised")]

    for col, arr, label, css_class in img_cols:
        with col:
            st.markdown(f'<div style="text-align:center"><span class="img-label {css_class}">{label}</span></div>', unsafe_allow_html=True)
            st.image(arr_to_pil(arr), use_container_width=True)

            # Download button
            buf = io.BytesIO()
            arr_to_pil(arr).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download",
                data=buf.getvalue(),
                file_name=f"{label.split()[-1].lower()}.png",
                mime="image/png",
                use_container_width=True,
            )

    # ── Metrics ─────────────────────────────────────────────────────────────
    if clean_arr is not None:
        m = compute_metrics(clean_arr, noisy_arr, denoised_arr)
        st.markdown('<p class="section-header">📊 Performance Metrics</p>', unsafe_allow_html=True)

        if show_gauges:
            g1, g2, g3 = st.columns(3, gap="medium")
            with g1:
                st.plotly_chart(gauge_chart(m["denoised_psnr"], "PSNR (dB)", 0, 40, GREEN),
                                use_container_width=True)
            with g2:
                st.plotly_chart(gauge_chart(m["denoised_ssim"], "SSIM", 0, 1, ACCENT),
                                use_container_width=True)
            with g3:
                mse_red = (1 - m["denoised_mse"] / m["noisy_mse"]) * 100
                st.plotly_chart(gauge_chart(mse_red, "MSE Reduction (%)", 0, 100, AMBER),
                                use_container_width=True)

        # Table
        metric_table = pd.DataFrame({
            "Metric": ["PSNR (dB)", "SSIM", "MSE"],
            "Noisy":    [f"{m['noisy_psnr']:.2f}",  f"{m['noisy_ssim']:.4f}",   f"{m['noisy_mse']:.6f}"],
            "Denoised": [f"{m['denoised_psnr']:.2f}", f"{m['denoised_ssim']:.4f}", f"{m['denoised_mse']:.6f}"],
            "Gain": [
                f"+{m['denoised_psnr'] - m['noisy_psnr']:.2f} dB",
                f"+{m['denoised_ssim'] - m['noisy_ssim']:.4f}",
                f"-{(1 - m['denoised_mse'] / m['noisy_mse']) * 100:.1f}%",
            ],
        })
        st.dataframe(metric_table, use_container_width=True, hide_index=True)

    else:
        st.info("ℹ️ Upload a clean reference image to see PSNR / SSIM / MSE metrics.")


# ─── Tab 2 · Analytics Dashboard ───────────────────────────────────────────────
def tab_analytics(df):
    if df is None:
        st.warning("No `denoising_metrics.csv` found. Run training to generate metrics.")
        return

    st.markdown('<p class="section-header">🏆 Model Performance Overview</p>', unsafe_allow_html=True)

    # ── KPI Cards ───────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
    kpis = [
        (kpi1, f"+{df['psnr_improvement'].mean():.2f} dB",  "Avg PSNR Gain",      GREEN),
        (kpi2, f"+{df['ssim_improvement'].mean():.4f}",      "Avg SSIM Gain",      ACCENT),
        (kpi3, f"{df['mse_reduction_percent'].mean():.1f}%", "Avg MSE Reduction",  AMBER),
        (kpi4, f"{len(df)} images",                          "Validation Set Size", BLUE),
    ]
    for col, val, label, colour in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{colour}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ──────────────────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2, gap="large")

    with r1c1:
        st.markdown('<p class="section-header">📈 PSNR Distribution</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["noisy_psnr"],    name="Noisy",    nbinsx=30,
                                   marker_color=RED,   opacity=0.75))
        fig.add_trace(go.Histogram(x=df["denoised_psnr"], name="Denoised", nbinsx=30,
                                   marker_color=GREEN, opacity=0.75))
        fig.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor=CARD_BG, font_color="#e8e8f0",
                          xaxis_title="PSNR (dB)", yaxis_title="Count",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<p class="section-header">📈 SSIM Distribution</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["noisy_ssim"],    name="Noisy",    nbinsx=30,
                                   marker_color=RED,   opacity=0.75))
        fig.add_trace(go.Histogram(x=df["denoised_ssim"], name="Denoised", nbinsx=30,
                                   marker_color=ACCENT, opacity=0.75))
        fig.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor=CARD_BG, font_color="#e8e8f0",
                          xaxis_title="SSIM", yaxis_title="Count",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">🔵 PSNR Improvement vs Starting Noisiness</p>', unsafe_allow_html=True)
    fig = px.scatter(df, x="noisy_psnr", y="psnr_improvement",
                     color="ssim_improvement",
                     color_continuous_scale="plasma",
                     labels={"noisy_psnr":     "Noisy Input PSNR (dB)",
                             "psnr_improvement":"PSNR Improvement (dB)",
                             "ssim_improvement":"SSIM Gain"},
                     hover_data=["image_id"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG,
                      font_color="#e8e8f0", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Box plots ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📦 PSNR Box Plot — Before vs After</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["noisy_psnr"],    name="Noisy",    marker_color=RED,
                         boxmean="sd"))
    fig.add_trace(go.Box(y=df["denoised_psnr"], name="Denoised", marker_color=GREEN,
                         boxmean="sd"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG,
                      font_color="#e8e8f0", yaxis_title="PSNR (dB)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data table ──────────────────────────────────────────────────────
    with st.expander("🗂 View Raw Metrics Table"):
        st.dataframe(df.round(4), use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, "denoising_metrics.csv",
                           "text/csv", use_container_width=True)


# ─── Tab 3 · Training History ──────────────────────────────────────────────────
def tab_training():
    st.markdown('<p class="section-header">📉 Training History</p>', unsafe_allow_html=True)
    if os.path.exists(HISTORY_PNG):
        st.image(HISTORY_PNG, caption="Model Loss & MAE across epochs",
                 use_container_width=True)
    else:
        st.warning("training_history.png not found. Run train.py to generate.")

    st.markdown("---")
    st.markdown('<p class="section-header">🏗️ U-Net Architecture</p>', unsafe_allow_html=True)

    arch_data = {
        "Layer / Block": [
            "Input", "Encoder Block 1", "Encoder Block 2", "Encoder Block 3",
            "Encoder Block 4", "Bottleneck", "Decoder Block 6", "Decoder Block 7",
            "Decoder Block 8", "Decoder Block 9", "Output"
        ],
        "Filters": ["-", 32, 64, 128, 256, 512, 256, 128, 64, 32, 1],
        "Output Shape": [
            "256×256×1", "256×256×32 → 128×128×32", "128×128×64 → 64×64×64",
            "64×64×128 → 32×32×128", "32×32×256 → 16×16×256", "16×16×512",
            "32×32×256", "64×64×128", "128×128×64", "256×256×32", "256×256×1"
        ],
        "Key Operation": [
            "—", "2×Conv + BN + ReLU + MaxPool", "2×Conv + BN + ReLU + MaxPool",
            "2×Conv + BN + ReLU + MaxPool", "2×Conv + BN + ReLU + MaxPool",
            "2×Conv + BN + ReLU", "UpSample + Skip-concat + 2×Conv",
            "UpSample + Skip-concat + 2×Conv", "UpSample + Skip-concat + 2×Conv",
            "UpSample + Skip-concat + 2×Conv", "Conv 1×1 + Sigmoid"
        ],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)


# ─── Tab 4 · How To Use ────────────────────────────────────────────────────────
def tab_howto():
    st.markdown('<p class="section-header">🚀 Quick Start</p>', unsafe_allow_html=True)
    st.markdown("""
```bash
# 1. Clone the repository
git clone https://github.com/yash-vks-chauhan/CT-Denoising-U-Net.git
cd CT-Denoising-U-Net

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Preprocess your own dataset
python preprocess.py --input_dir /path/to/raw_images --output_dir ./data --augment 4

# 4. Train the model (Kaggle / GPU recommended)
python train.py

# 5. Denoise a single image via CLI
python inference.py --input noisy.png --output denoised.png --compare

# 6. Batch denoise a folder
python inference.py --input_dir ./noisy/ --output_dir ./denoised/ --compare

# 7. Generate visualisation report
python visualize.py --metrics denoising_metrics.csv --output_dir ./results

# 8. Launch this web app
streamlit run app.py
```
    """)

    st.markdown('<p class="section-header">📂 Repository Structure</p>', unsafe_allow_html=True)
    st.markdown("""
| File | Purpose |
|------|---------|
| `train.py`          | Full training pipeline — data generator, U-Net build, callbacks, metrics |
| `inference.py`      | CLI tool for denoising single images or entire folders |
| `preprocess.py`     | Raw→paired dataset preparation with augmentation & noise injection |
| `visualize.py`      | Publication-quality metric charts & image comparison grids |
| `app.py`            | **This Streamlit web app** |
| `denoising_model.h5`| Trained U-Net weights (~89 MB) |
| `denoising_metrics.csv` | Per-image PSNR/SSIM/MSE evaluation on 321 validation images |
| `training_history.png` | Loss & MAE curves from training |
| `requirements.txt`  | Python package dependencies |
    """)

    st.markdown('<p class="section-header">📊 Datasets</p>', unsafe_allow_html=True)
    ds_data = {
        "Dataset": ["COVID-19 Radiography", "Montgomery & Shenzhen TB",
                    "LIDC-IDRI Lung Cancer", "NIH Chest X-ray"],
        "Modality": ["X-ray", "X-ray", "CT scan", "X-ray"],
        "Focus": ["COVID-19 lung opacity", "Tuberculosis", "Lung nodules", "14 thoracic diseases"],
    }
    st.dataframe(pd.DataFrame(ds_data), use_container_width=True, hide_index=True)


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    show_comparison, show_gauges = render_sidebar()
    render_hero()

    model = load_model()
    df    = load_metrics()

    # Status badge
    if model:
        st.success("✅ Model loaded — ready for inference", icon="🧠")
    else:
        st.warning("⚠️ Model not found. Train with `python train.py` first.", icon="⚠️")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🫁  Denoise Image",
        "📊  Analytics Dashboard",
        "📉  Training History",
        "📖  How To Use",
    ])

    with tab1:
        tab_denoise(model, show_comparison, show_gauges)

    with tab2:
        tab_analytics(df)

    with tab3:
        tab_training()

    with tab4:
        tab_howto()

    # Footer
    st.markdown("""
    <hr style='border-color:#1e1e2e; margin-top:40px'>
    <p style='text-align:center; color:#888898; font-size:0.8rem'>
        CT Denoising U-Net · Built with TensorFlow & Streamlit ·
        <a href='https://github.com/yash-vks-chauhan/CT-Denoising-U-Net'
           style='color:#7c6efd; text-decoration:none'>GitHub ↗</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
