import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import history, predict, ui
from src.preprocessing import InvalidImageError, validate_upload

ui.setup("Image Prediction", "Upload a traffic-sign image and classify it", "🖼️")
st.warning(ui.GERMAN_WARNING)
st.caption(ui.DISCLAIMER)

bundle, err = ui.get_bundle()
if err:
    st.error(f"The saved model could not be used: {err}. Retrain the model.")
    st.stop()
if bundle is None:
    st.info("The model has not been trained yet. Train it on the **Model Training** page or run `python -m src.train`.")
    st.stop()

up = st.file_uploader("Upload PNG, JPG or JPEG", type=["png", "jpg", "jpeg"])
if up is None:
    st.stop()
try:
    img = validate_upload(up.getvalue())
except InvalidImageError as e:
    st.error(f"Invalid image: {e}")
    st.stop()

c1, c2 = st.columns([1, 2])
c1.image(img, caption=f"{up.name} ({img.width}x{img.height})", use_container_width=True)
result = predict.predict_image(img, bundle)
with c2:
    st.subheader(result["class_name"])
    m = st.columns(2)
    ui.kpi(m[0], "Class ID", result["class_id"])
    ui.kpi(m[1], "Model confidence", f"{result['confidence']:.1%}")
    if result["uncertain"]:
        st.warning("The prediction is uncertain: confidence is low or the top two classes are close. "
                   "Treat this result with caution.")
    st.caption("Confidence is the model's own probability estimate, not a guarantee of real-world correctness.")

top = pd.DataFrame(result["top"])
st.subheader("Top 5 predictions")
l, r = st.columns(2)
l.dataframe(top.assign(probability=top["probability"].map("{:.2%}".format)), hide_index=True, use_container_width=True)
fig = px.bar(top.iloc[::-1], x="probability", y=top.iloc[::-1]["class_id"].astype(str) + " - " + top.iloc[::-1]["class_name"],
             orientation="h", color_discrete_sequence=[ui.BLUE])
fig.update_layout(height=300, yaxis_title="", margin=dict(l=0, r=0, t=10, b=0), xaxis_range=[0, 1])
r.plotly_chart(fig, use_container_width=True)

st.subheader("Save to history")
truth = st.selectbox("True class (optional, only if you know it)", [None] + list(range(config.NUM_CLASSES)),
                     format_func=lambda i: "Unknown" if i is None else f"{i} - {config.CLASS_NAMES[i]}")
if st.button("Save prediction"):
    history.add_record(up.name, result, truth)
    st.success("Saved. The image itself is not stored.")
