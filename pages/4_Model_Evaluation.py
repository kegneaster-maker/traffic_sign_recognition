import json
import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import ui

ui.setup("Model Evaluation", "Metrics measured on validation or official test data", "📊")
data = ui.read_json(config.METRICS_PATH)
if not data or not data.get("evaluations"):
    st.info("No evaluation results exist yet. Generate them by training "
            "(`python -m src.train`) or, with a trained model, running `python -m src.evaluate`.")
    if config.MODEL_PATH.exists() and st.button("Run evaluation now"):
        with st.spinner("Evaluating..."):
            r = subprocess.run([sys.executable, "-m", "src.evaluate", "--data-dir", ui.get_dataset_dir()],
                               cwd=str(config.BASE_DIR), capture_output=True, text=True)
        (st.success if r.returncode == 0 else st.error)(r.stdout[-800:] or r.stderr[-800:])
        st.rerun()
    st.stop()

ev = data["evaluations"]
source = st.radio("Evaluation data", list(ev), format_func=ui.source_label, horizontal=True)
m = ev[source]
st.caption(f"Computed on the **{ui.source_label(source)}** ({m['n_samples']:,} images) at {m['evaluated_at']}. "
           "Macro scores average over classes present in the data.")
c = st.columns(4)
ui.kpi(c[0], "Accuracy", f"{m['accuracy']:.2%}")
ui.kpi(c[1], "Macro precision", f"{m['macro_precision']:.3f}")
ui.kpi(c[2], "Macro recall", f"{m['macro_recall']:.3f}")
ui.kpi(c[3], "Macro F1", f"{m['macro_f1']:.3f}")

cm_path = config.OUTPUTS_DIR / f"confusion_matrix_{source}.csv"
if cm_path.exists():
    cm = pd.read_csv(cm_path, index_col=0)
    st.subheader("Confusion matrix")
    norm = st.checkbox("Normalise by true class", True)
    z = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) if norm else cm
    fig = px.imshow(z.values, x=list(range(len(z))), y=list(range(len(z))), color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="True", color="Share" if norm else "Count"), aspect="auto")
    st.plotly_chart(fig.update_layout(height=620), use_container_width=True)

pc_path = config.OUTPUTS_DIR / f"per_class_metrics_{source}.csv"
if pc_path.exists():
    pc = pd.read_csv(pc_path)
    st.subheader("Per-class metrics")
    st.dataframe(pc.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download per-class CSV", pc.to_csv(index=False), f"per_class_{source}.csv", "text/csv")
    d2.download_button("Download metrics JSON", json.dumps(data, indent=2), "evaluation_metrics.json", "application/json")

hist = ui.read_json(config.HISTORY_PATH)
if hist:
    st.subheader("Learning curves")
    d = pd.DataFrame(hist)
    d.index += 1
    a, b = st.columns(2)
    a.plotly_chart(px.line(d[[k for k in ("accuracy", "val_accuracy") if k in d]]).update_layout(
        height=300, title="Accuracy", xaxis_title="Epoch"), use_container_width=True)
    b.plotly_chart(px.line(d[[k for k in ("loss", "val_loss") if k in d]]).update_layout(
        height=300, title="Loss", xaxis_title="Epoch"), use_container_width=True)

st.subheader("Example predictions")
for label, key in (("Correct", "correct"), ("Incorrect (most confident first)", "incorrect")):
    items = [e for e in m.get("examples", {}).get(key, []) if os.path.exists(e["path"])]
    st.markdown(f"**{label}**")
    if not items:
        st.caption("None available.")
        continue
    cols = st.columns(4)
    for i, e in enumerate(items[:8]):
        cols[i % 4].image(e["path"], caption=f"true {e['true']} / pred {e['pred']} ({e['confidence']:.0%})")
