import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import dataset as ds, ui

ui.setup("Dataset Explorer", "Inspect structure, class balance, image statistics and samples", "🗂️")
root = ui.get_dataset_dir()
st.caption(f"Dataset folder: `{root}`")

col_a, col_b = st.columns([1, 3])
sample = col_a.number_input("Images to check (sample)", 200, 60000, 3000, 500)
if col_b.button("🔄 Refresh dataset inspection", type="primary"):
    with st.spinner("Inspecting dataset..."):
        ds.inspect_dataset(root, sample=int(sample))
        st.session_state.pop("records", None)

report = ui.read_json(config.DATASET_REPORT_PATH)
if not report:
    st.info("No inspection has been run yet. Click **Refresh dataset inspection**.")
    st.stop()
if report.get("status") != "ok":
    st.error(report.get("message", "Dataset not found."))
    st.stop()
if report["dataset_root"] != str(root):
    st.warning("This report was generated for a different folder. Click refresh to inspect the current path.")

c = st.columns(4)
ui.kpi(c[0], "Total images", f"{report['n_images']:,}", f"layout: {report['structure']}")
ui.kpi(c[1], "Classes", report["n_classes"], f"expected {config.NUM_CLASSES}")
ui.kpi(c[2], "Official test images", f"{report['n_test_images']:,}",
       "labelled" if report["official_test_labels_available"] else "no labels")
ui.kpi(c[3], "Unreadable (sample)", report["unreadable_in_sample"], f"of {report['checked_sample']:,} checked")

for w in report["warnings"]:
    st.warning(w)
if not report["warnings"]:
    st.success("No dataset validation warnings.")

dist = pd.DataFrame({"class_id": [int(k) for k in report["class_distribution"]],
                     "images": list(report["class_distribution"].values())})
dist["name"] = dist["class_id"].map(config.CLASS_NAMES)
q = st.text_input("Search class (ID or name)")
sel = st.multiselect("Filter classes", dist["class_id"].tolist(),
                     format_func=lambda i: f"{i} - {config.CLASS_NAMES.get(i, '')}")
view = dist
if q:
    view = view[view["name"].str.contains(q, case=False, na=False) | (view["class_id"].astype(str) == q.strip())]
if sel:
    view = view[view["class_id"].isin(sel)]
st.plotly_chart(px.bar(view, x="class_id", y="images", hover_name="name", color_discrete_sequence=[ui.BLUE])
                .update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)

l, r = st.columns(2)
with l:
    st.subheader("Image formats")
    st.dataframe(pd.DataFrame({"format": list(report["formats"]), "count": list(report["formats"].values())}),
                 hide_index=True, use_container_width=True)
with r:
    st.subheader("Image dimensions (sample)")
    st.json(report["dimensions"])

st.subheader("Sample gallery")
if view.empty:
    st.info("No class matches the filter.")
else:
    cls = st.selectbox("Class", view["class_id"].tolist(), format_func=lambda i: f"{i} - {config.CLASS_NAMES.get(i, '')}")
    n = st.slider("Images", 4, 24, 12, 4)
    try:
        rec = ds.load_records(root)["train"]
        paths = rec[rec["class_id"] == cls]["path"].head(n).tolist()
        cols = st.columns(6)
        for i, p in enumerate(paths):  # only the selected images are read
            cols[i % 6].image(p, caption=f"class {cls}", use_container_width=True)
    except ds.DatasetNotFoundError as e:
        st.error(str(e))
