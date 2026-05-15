from __future__ import annotations

import base64
import json
import mimetypes
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from science_explainer import (
    answer_science_question,
    build_ai_stack_cards,
    build_research_insight_cards,
    format_prediction_context,
)

CONF_THRESHOLDS = {"high": 0.75, "medium": 0.55}
MODEL = Path("artifacts/model.joblib")
DEEP_MODEL = Path("artifacts/deep_model.joblib")
FEATS = ["orbital_period", "transit_duration", "transit_depth", "planet_radius"]

SLIDER_LIMITS = {
    "orbital_period": {"min": 0.0, "max": 1000.0, "step": 0.1, "default": 17.0, "unit": "days"},
    "transit_duration": {"min": 0.0, "max": 30.0, "step": 0.05, "default": 9.0, "unit": "hours"},
    "transit_depth": {"min": 0.0, "max": 20000.0, "step": 10.0, "default": 1000.0, "unit": "ppm"},
    "planet_radius": {"min": 0.0, "max": 40.0, "step": 0.1, "default": 7.0, "unit": "R_earth"},
}

RESPONSIVE_CSS = """
<style>
[data-testid="stMetricValue"] {
  font-size: clamp(0.875rem, 2.2vw, 1.375rem) !important;
  line-height: 1.1 !important;
  white-space: nowrap;
}
[data-testid="stMetricLabel"] {
  font-size: clamp(0.70rem, 1.5vw, 0.875rem) !important;
  opacity: .9;
}
[data-testid="stMetricDelta"] {
  font-size: clamp(0.70rem, 1.5vw, 0.875rem) !important;
}
.metric-row { display: flex; gap: .6rem; flex-wrap: wrap; }
.metric-row > [data-testid="column"] { flex: 1 1 160px; }
@media (max-width: 900px) {
  .metric-row > [data-testid="column"] { flex: 1 1 45%; }
}
@media (max-width: 520px) {
  .metric-row > [data-testid="column"] { flex: 1 1 100%; }
}
.main {
  background: url("artifacts/backgrounds/bg.jpg") no-repeat center center fixed;
  background-size: cover;
  min-height: 100vh;
  padding-top: 1rem;
}
header, [data-testid="stHeader"] {
  background: transparent !important;
  box-shadow: none !important;
}
</style>
"""

def confidence_label(p: float) -> tuple[str, str]:
    if p >= CONF_THRESHOLDS["high"]:
        return "High", "✅"
    if p >= CONF_THRESHOLDS["medium"]:
        return "Medium", "🟡"
    return "Low", "⚠️"

def read_metrics(path="artifacts/metrics.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_resource
def load_model():
    if not MODEL.exists():
        st.error("Not found: artifacts/model.joblib. Train first: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL)

@st.cache_resource
def load_deep_model():
    if not DEEP_MODEL.exists():
        return None
    return joblib.load(DEEP_MODEL)

def predict_with_model(active_model, X: pd.DataFrame) -> tuple[list[str], np.ndarray, str, float]:
    if hasattr(active_model, "predict_proba"):
        proba = active_model.predict_proba(X)[0]
        classes = list(active_model.classes_)
    else:
        classes = list(active_model.classes_) if hasattr(active_model, "classes_") else ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
        proba = np.full(len(classes), 1 / len(classes))
    pred_idx = int(np.argmax(proba))
    return classes, proba, classes[pred_idx], float(proba[pred_idx])

def prob_bars(classes, proba):
    order = np.argsort(proba)
    for i in order:
        st.write(f"{classes[i]} - {proba[i]:.0%}")
        st.progress(min(max(float(proba[i]), 0.0), 1.0))

def _normalize_option_from_class(pred_cls: str) -> str:
    return pred_cls.strip().lower().replace(" ", "_")

def find_gif_path_for(pred_cls: str) -> Path | None:
    option = _normalize_option_from_class(pred_cls)
    base = Path("artifacts/gifs") / option
    if not base.exists():
        return None
    exts = (".gif", ".webp", ".mp4")
    for ext in exts:
        files = sorted(base.glob(f"*{ext}"))
        if files:
            return files[0]
    return None

def show_media_for(pred_cls: str):
    media = find_gif_path_for(pred_cls)
    if not media:
        st.info("No media found for this class.")
        return
    suffix = media.suffix.lower()
    if suffix == ".mp4":
        st.video(str(media.resolve()))
        return
    try:
        data = media.read_bytes()
        st.image(data, width="stretch")
    except Exception as e:
        st.error(f"Could not load media: {media.name}")
        st.exception(e)

def set_page_background(img_path: str | Path):
    p = Path(img_path)
    if not p.exists():
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"]{
          background: radial-gradient(circle at 30% 20%, #0b132b 0%, #000814 100%) fixed;
        }
        [data-testid="stHeader"]{ background: transparent !important; box-shadow: none !important; }
        </style>
        """, unsafe_allow_html=True)
        return
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
      background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
      background-size: cover;
    }}
    [data-testid="stHeader"] {{
      background: transparent !important;
      backdrop-filter: none !important;
      box-shadow: none !important;
    }}
    .block-container {{ padding-top: 1rem !important; }}
    </style>
    """, unsafe_allow_html=True)

RESPONSIVE_CSS += """
<style>
:root { --swai-blue: #007BFF; }
div[data-testid="stSlider"] > div > div > div { background: var(--swai-blue) !important; }
div[data-testid="stSlider"] [role="slider"] {
  background-color: var(--swai-blue) !important;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.3) !important;
}
div[data-testid="stSlider"] [role="slider"]:hover { background-color: #339CFF !important; }
button[kind="primary"], .stButton > button {
  background-color: var(--swai-blue) !important;
  border: none !important;
  color: white !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  transition: background-color 0.2s ease-in-out;
}
button[kind="primary"]:hover, .stButton > button:hover { background-color: #339CFF !important; }
input, textarea {
  border: 1px solid #aacdfd !important;
  border-radius: 6px !important;
}
.swai-floating-assistant {
  position: fixed;
  right: 1.25rem;
  bottom: 1.25rem;
  z-index: 9999;
  width: 4.25rem;
  height: 4.25rem;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #007BFF;
  color: white !important;
  font-size: 1.65rem;
  text-decoration: none !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.36);
  border: 1px solid rgba(255,255,255,.35);
}
.swai-floating-assistant:hover {
  background: #339CFF;
  color: white !important;
}
.swai-ai-card {
  min-height: 15.5rem;
  border: 1px solid rgba(160, 198, 255, .34);
  border-radius: 8px;
  padding: 1rem;
  background: rgba(0, 17, 45, .58);
}
.swai-ai-card h4 {
  margin-top: 0;
  margin-bottom: .45rem;
}
.swai-ai-card p {
  margin: .45rem 0;
}
.swai-card-label {
  color: #9cc8ff;
  font-weight: 700;
}
.swai-chat-panel {
  border: 1px solid rgba(160, 198, 255, .34);
  border-radius: 8px;
  padding: 1rem;
  background: rgba(0, 17, 45, .62);
}
.swai-assistant-note {
  font-size: .9rem;
  color: rgba(255,255,255,.82);
}
.swai-research-card {
  min-height: 18rem;
  border: 1px solid rgba(160, 198, 255, .34);
  border-radius: 8px;
  padding: 1rem;
  background: rgba(0, 17, 45, .58);
}
.swai-research-card h4 {
  margin-top: 0;
  margin-bottom: .35rem;
}
.swai-research-card h5 {
  margin-top: 0;
  margin-bottom: .75rem;
  color: #9cc8ff;
}
.swai-source-link {
  color: #9cc8ff !important;
  font-weight: 700;
}
</style>
"""

st.set_page_config(page_title="SWAI - Exoplanet Classifier", page_icon="🪐", layout="wide")
set_page_background("artifacts/backgrounds/bg.jpg")
st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)
st.markdown('<a class="swai-floating-assistant" href="#swai-science-explainer" title="Ask SWAI Science Explainer">?</a>', unsafe_allow_html=True)

st.title("Silent Watcher AI 🪐 Exoplanet Classifier")
st.caption("Categories: CONFIRMED / CANDIDATE / FALSE POSITIVE based on NASA's Kepler/TESS datasets")

model = load_model()
deep_model = load_deep_model()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "assistant_history" not in st.session_state:
    st.session_state.assistant_history = [
        (
            "assistant",
            "Hi, I am SWAI's Science Explainer. Ask me about the prediction, transit depth, false positives, or how AI fits into this project.",
        )
    ]

st.subheader("Manual entry")
with st.expander("What does each field mean?"):
    st.markdown("""
•  *Orbital period (days)*: how many days it takes the planet to orbit its star.
•  *Transit duration (hours)*: duration of the transit in hours.
•  *Transit depth (ppm)*: brightness drop during the transit (parts per million).
•  *Planet radius (R_earth)*: planet radius in Earth radii (Earth = 1).
""")

c1, c2 = st.columns(2)
with c1:
    pconf = SLIDER_LIMITS["orbital_period"]
    period = st.slider(
        "Orbital period (days)",
        min_value=float(pconf["min"]),
        max_value=float(pconf["max"]),
        value=float(pconf["default"]),
        step=float(pconf["step"]),
        help="Time for one full orbit around the star",
    )
    dconf = SLIDER_LIMITS["transit_depth"]
    depth = st.slider(
        "Transit depth (ppm)",
        min_value=float(dconf["min"]),
        max_value=float(dconf["max"]),
        value=float(dconf["default"]),
        step=float(dconf["step"]),
        help="Drop in starlight during a transit, in parts per million",
    )
with c2:
    tconf = SLIDER_LIMITS["transit_duration"]
    dur = st.slider(
        "Transit duration (hours)",
        min_value=float(tconf["min"]),
        max_value=float(tconf["max"]),
        value=float(tconf["default"]),
        step=float(tconf["step"]),
        help="Time the planet takes to cross the star",
    )
    rconf = SLIDER_LIMITS["planet_radius"]
    radius = st.slider(
        "Planet radius (R_earth)",
        min_value=float(rconf["min"]),
        max_value=float(rconf["max"]),
        value=float(rconf["default"]),
        step=float(rconf["step"]),
        help="Planet size in Earth radii proportion",
    )

if st.button("Predict (manual)", width="stretch"):
    X = pd.DataFrame([{
        "orbital_period": period,
        "transit_duration": dur,
        "transit_depth": depth,
        "planet_radius": radius,
    }])
    classes, proba, pred_cls, pred_conf = predict_with_model(model, X)
    deep_result = None
    if deep_model is not None:
        deep_classes, deep_proba, deep_pred_cls, deep_pred_conf = predict_with_model(deep_model, X)
        deep_result = {
            "classes": deep_classes,
            "proba": deep_proba,
            "class": deep_pred_cls,
            "confidence": deep_pred_conf,
        }
    st.session_state.last_prediction = {
        "class": pred_cls,
        "confidence": pred_conf,
        "deep_class": deep_result["class"] if deep_result else None,
        "deep_confidence": deep_result["confidence"] if deep_result else None,
        "inputs": {
            "orbital_period": period,
            "transit_duration": dur,
            "transit_depth": depth,
            "planet_radius": radius,
        },
    }
    label, icon = confidence_label(pred_conf)
    st.success(f"*Prediction: {pred_cls}*  | Trust: *{pred_conf:.0%}* {icon}  ({label})")
    if deep_result:
        dl_label, dl_icon = confidence_label(deep_result["confidence"])
        st.info(
            f"Deep Learning MVP (MLP neural network): *{deep_result['class']}* | "
            f"Trust: *{deep_result['confidence']:.0%}* {dl_icon} ({dl_label})"
        )
    else:
        st.warning("Deep Learning model not found yet. Train it with: python src/train_deep_model.py")
    col_stats, col_media = st.columns(2, gap="large")
    with col_stats:
        st.markdown("#### Machine Learning probability distribution")
        prob_bars(classes, proba)
        if deep_result:
            st.markdown("#### Deep Learning probability distribution")
            prob_bars(deep_result["classes"], deep_result["proba"])
        st.markdown("#### Input snapshot")
        with st.container():
            st.markdown('<div class="metric-row">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Period (d)", f"{period:.2f}")
            m2.metric("Dur (h)", f"{dur:.2f}")
            m3.metric("Depth (ppm)", f"{depth:.0f}")
            m4.metric("Radius (R⊕)", f"{radius:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("See numerical details"):
            st.json({c: round(float(p), 6) for c, p in zip(classes, proba)})
        mets = read_metrics()
        if mets:
            macro_f1 = mets.get("report", {}).get("macro avg", {}).get("f1-score")
            if macro_f1 is not None:
                st.caption(f"Model performance (hold-out): Macro-F1 ≈ *{macro_f1:.2f}*")
    with col_media:
        st.markdown("#### Visual simulation")
        show_media_for(pred_cls)

st.divider()

st.subheader("AI Roadmap for the Presentation")
st.caption("A compact MVP of how SWAI connects machine learning, deep learning, reinforcement learning, and LLMs.")

ai_cards = build_ai_stack_cards()
card_cols = st.columns(4)
for col, card in zip(card_cols, ai_cards):
    with col:
        st.markdown(
            f"""
            <div class="swai-ai-card">
              <h4>{card["title"]}</h4>
              <p><span class="swai-card-label">Role:</span> {card["role"]}</p>
              <p><span class="swai-card-label">MVP:</span> {card["mvp"]}</p>
              <p><span class="swai-card-label">Next:</span> {card["next_step"]}</p>
              <p><span class="swai-card-label">Status:</span> {card["maturity"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.info(
    "Presentation line: Machine Learning makes the prediction, Deep Learning can improve transit-signal detection, "
    "Reinforcement Learning can optimize follow-up observations, and LLMs make the science understandable and interactive."
)

st.markdown("#### Source-backed additions")
research_cards = build_research_insight_cards()
research_cols = st.columns(3)
for col, card in zip(research_cols, research_cards):
    with col:
        st.markdown(
            f"""
            <div class="swai-research-card">
              <h4>{card["title"]}</h4>
              <h5>{card["subtitle"]}</h5>
              <p>{card["body"]}</p>
              <p><span class="swai-card-label">Presentation takeaway:</span> {card["presentation_takeaway"]}</p>
              <p><a class="swai-source-link" href="{card["source_url"]}" target="_blank">{card["source_label"]}</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

st.subheader("Prediction by CSV")
st.caption("Columns: orbital_period, transit_duration, transit_depth, planet_radius")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        st.error(f"Columns are missing: {missing}")
    else:
        preds = model.predict(df[FEATS])
        out = df.copy()
        out["prediction"] = preds
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df[FEATS])
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    out[f"proba_{cls}"] = probas[:, i]
        st.dataframe(out.head(50), width="stretch")
        out.to_csv("artifacts/predictions.csv", index=False)
        st.success("Saved in artifacts/predictions.csv")

st.divider()
st.markdown('<span id="swai-science-explainer"></span>', unsafe_allow_html=True)
st.subheader("Ask SWAI - LLM Science Explainer")
st.markdown(
    """
    <div class="swai-chat-panel">
      <div class="swai-assistant-note">
        MVP note: this assistant is a local, grounded science explainer. It does not call an external LLM yet, so it is safe for live demos and can later be upgraded to RAG over NASA sources.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

last_prediction = st.session_state.last_prediction
if last_prediction:
    prediction_context = format_prediction_context(
        last_prediction["class"],
        last_prediction["confidence"],
        last_prediction["inputs"],
    )
else:
    prediction_context = format_prediction_context(None, None, None)

with st.expander("Current context used by the assistant", expanded=bool(last_prediction)):
    st.write(prediction_context)

prompt_cols = st.columns(4)
quick_prompts = [
    "Explain the current prediction",
    "What is transit depth?",
    "What is ExoMiner++?",
    "How can RL schedule follow-up observations?",
]
for col, prompt in zip(prompt_cols, quick_prompts):
    with col:
        if st.button(prompt, width="stretch"):
            answer = answer_science_question(prompt, prediction_context)
            st.session_state.assistant_history.extend([("user", prompt), ("assistant", answer)])

with st.form("science_explainer_form", clear_on_submit=True):
    question = st.text_input("Ask a science question", placeholder="Example: How does deep learning use light curves?")
    submitted = st.form_submit_button("Ask SWAI", width="stretch")
    if submitted:
        answer = answer_science_question(question, prediction_context)
        st.session_state.assistant_history.extend([("user", question), ("assistant", answer)])

for role, message in st.session_state.assistant_history[-8:]:
    with st.chat_message(role):
        st.write(message)
