import streamlit as st

from src import history, ui

ui.setup("Prediction History", "Locally stored prediction records (no images are saved)", "🕘")
h = history.read_history()
if h.empty:
    st.info("No predictions recorded yet. Save one from the **Image Prediction** page.")
    st.stop()

st.dataframe(h[::-1], use_container_width=True, hide_index=True)
known = h.dropna(subset=["ground_truth_id"])
if len(known):
    acc = (known["ground_truth_id"].astype(int) == known["predicted_class_id"].astype(int)).mean()
    st.metric(f"Agreement with supplied ground truth ({len(known)} labelled records)", f"{acc:.1%}")
else:
    st.caption("No ground-truth labels were supplied, so correctness is not assessed.")
st.download_button("Download CSV", h.to_csv(index=False), "prediction_history.csv", "text/csv")
st.divider()
if st.checkbox("I want to permanently delete all prediction history") and st.button("Clear history", type="primary"):
    history.clear_history()
    st.rerun()
