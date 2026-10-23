"""Shared Streamlit helpers: theme CSS, KPI cards, cached loaders."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import config

NAVY, BLUE, LIGHT = "#0B1F3A", "#2E6BE6", "#EAF1FF"
DISCLAIMER = ("Predictions are experimental and intended for academic demonstration only, "
              "not for safety-critical driving decisions.")
GERMAN_WARNING = ("GTSRB contains German traffic signs. Predictions may be unreliable for Ethiopian "
                  "traffic signs or any signs outside the training distribution.")

CSS = f"""
<style>
.stApp {{ background: #F5F8FF; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] input {{ color: {NAVY} !important; }}
h1, h2, h3 {{ color: {NAVY}; font-family: 'Segoe UI', sans-serif; }}
.kpi {{ background:#fff; border-left:6px solid {BLUE}; border-radius:10px; padding:14px 16px;
        box-shadow:0 1px 4px rgba(11,31,58,.12); margin-bottom:8px; }}
.kpi .l {{ color:#5B6B82; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }}
.kpi .v {{ color:{NAVY}; font-size:1.7rem; font-weight:700; line-height:1.2; }}
.kpi .s {{ color:#5B6B82; font-size:.75rem; }}
.banner {{ background:{NAVY}; color:#fff; padding:18px 22px; border-radius:12px; margin-bottom:16px; }}
.banner h1 {{ color:#fff; margin:0; font-size:1.8rem; }}
.banner p {{ color:#BFD2F7; margin:4px 0 0 0; }}
</style>
"""


def setup(title: str, subtitle: str = "", icon: str = "🚦"):
    st.set_page_config(page_title=f"{title} | Traffic Sign Recognition", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="banner"><h1>{icon} {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
    sidebar_dataset_input()


def sidebar_dataset_input():
    st.sidebar.markdown("### Dataset folder")
    default = st.session_state.get("dataset_dir", str(config.DATASET_DIR))
    val = st.sidebar.text_input("Path", value=default, key="dataset_dir_input")
    st.session_state["dataset_dir"] = val


def get_dataset_dir() -> str:
    return st.session_state.get("dataset_dir", str(config.DATASET_DIR))


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def read_json(path: Path):
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading model...")
def cached_bundle(mtime: float):
    from src import predict
    return predict.load_bundle()


def get_bundle():
    """Returns (bundle|None, error|None). Cached per model-file modification time."""
    from src import predict
    if not predict.model_available():
        return None, None
    try:
        return cached_bundle(config.MODEL_PATH.stat().st_mtime), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def best_evaluation():
    """Prefer official test results; fall back to validation. Returns (source, summary) or (None, None)."""
    data = read_json(config.METRICS_PATH)
    if not data:
        return None, None
    ev = data.get("evaluations", {})
    for src in ("official_test", "validation"):
        if src in ev:
            return src, ev[src]
    return None, None


def source_label(src):
    return {"official_test": "official test set", "validation": "validation split"}.get(src, src or "-")
