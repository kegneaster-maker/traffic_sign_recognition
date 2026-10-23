import json
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import psutil
import streamlit as st

import config
from src import dataset as ds, ui

ui.setup("Model Training", "Configure and launch CNN training (runs as a separate process)", "🧠")


def current_status():
    s = ui.read_json(config.TRAIN_STATUS_PATH) or {}
    if s.get("state") == "running":
        pid = s.get("pid")
        alive = bool(pid) and psutil.pid_exists(int(pid))
        if not alive:  # process died without writing a final state
            s["state"] = "failed"
            s["message"] = "Training process is no longer running (interrupted)."
    return s


status = current_status()
running = status.get("state") == "running"

st.subheader("1. Dataset readiness")
ready = ds.readiness(ui.get_dataset_dir())
if ready["ready"]:
    st.success(f"Dataset found: {ready['n_train']:,} labelled images, {ready['n_classes']} classes, "
               f"layout `{ready['structure']}`. Official labelled test set: {'yes' if ready['official_test'] else 'no (validation split used)'}.")
    for n in ready.get("notes", []):
        st.warning(n)
else:
    st.error(ready["message"])

st.subheader("2. Settings")
c = st.columns(4)
epochs = c[0].number_input("Epochs", 1, 200, config.DEFAULT_EPOCHS)
batch = c[1].selectbox("Batch size", [16, 32, 64, 128, 256], index=2)
lr = c[2].number_input("Learning rate", 1e-5, 1e-1, config.DEFAULT_LR, format="%.5f")
size = c[3].selectbox("Image size", [32, 48, 64], index=0)
c = st.columns(4)
patience = c[0].number_input("Early-stopping patience", 2, 30, 6)
dropout = c[1].slider("Dropout", 0.0, 0.7, config.DEFAULT_DROPOUT, 0.05)
augment = c[2].checkbox("Data augmentation", True)
limit = c[3].number_input("Limit images per class (0 = all)", 0, 10000, 0)
st.caption("Time depends on your hardware. On a CPU, the full GTSRB training set (about 39,000 images, 32x32) "
           "typically takes several minutes per epoch; a compatible GPU is much faster. Early stopping may end it sooner.")

st.subheader("3. Start")
confirm = st.checkbox("I understand this will start a training run and overwrite the current model if validation improves.")
if st.button("▶ Start training", type="primary", disabled=running or not ready["ready"] or not confirm):
    cmd = [sys.executable, "-m", "src.train", "--data-dir", ui.get_dataset_dir(), "--epochs", str(int(epochs)),
           "--batch-size", str(int(batch)), "--learning-rate", str(lr), "--img-size", str(size),
           "--patience", str(int(patience)), "--dropout", str(dropout), "--limit-per-class", str(int(limit))]
    if not augment:
        cmd.append("--no-augment")
    p = subprocess.Popen(cmd, cwd=str(config.BASE_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    config.TRAIN_STATUS_PATH.write_text(json.dumps({
        "state": "running", "pid": p.pid, "epoch": 0, "epochs": int(epochs), "history": {},
        "started_at": datetime.now().isoformat(timespec="seconds"), "message": "Launching..."}))
    st.rerun()
if running:
    st.info("A training run is in progress; starting another is disabled.")


@st.fragment(run_every=4 if running else None)
def monitor():
    s = current_status()
    st.subheader("4. Progress")
    state = s.get("state", "idle")
    if state == "idle" or not s:
        st.info("No training run recorded yet.")
        return
    ep, tot = s.get("epoch", 0), max(s.get("epochs", 1), 1)
    st.progress(min(ep / tot, 1.0), text=f"Epoch {ep}/{tot} - {state}")
    {"completed": st.success, "failed": st.error}.get(state, st.info)(s.get("message", ""))
    h = s.get("history") or {}
    if h.get("loss"):
        d = pd.DataFrame(h)
        d.index += 1
        a, b = st.columns(2)
        a.plotly_chart(px.line(d[[k for k in ("accuracy", "val_accuracy") if k in d]], markers=True)
                       .update_layout(title="Accuracy", height=300, xaxis_title="Epoch"), use_container_width=True)
        b.plotly_chart(px.line(d[[k for k in ("loss", "val_loss") if k in d]], markers=True)
                       .update_layout(title="Loss", height=300, xaxis_title="Epoch"), use_container_width=True)
    if config.MODEL_PATH.exists():
        st.caption(f"Checkpoint: `{config.MODEL_PATH.name}` last saved "
                   f"{datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    else:
        st.caption("No checkpoint saved yet.")
    if config.TRAIN_LOG_PATH.exists():
        with st.expander("Training log (last 40 lines)", expanded=state == "running"):
            st.code("\n".join(config.TRAIN_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]))
    if st.button("Refresh status"):
        st.rerun()


monitor()
