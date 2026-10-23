
"""Overview dashboard for Traffic Sign Recognition.
Run with: python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import history, ui
from src.predict import model_available


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
ui.setup(
    "Traffic Sign Recognition Dashboard",
    "CNN-based traffic sign classification using the GTSRB dataset",
)

st.title("🚦 Traffic Sign Recognition")
st.caption("Machine Learning Dashboard | GTSRB · 43 traffic-sign classes")

st.warning(ui.GERMAN_WARNING + " " + ui.DISCLAIMER)

# Consistent chart layout
CHART_HEIGHT = 340


# ---------------------------------------------------------
# Load project results
# ---------------------------------------------------------
report = ui.read_json(config.DATASET_REPORT_PATH) or {}
meta = ui.read_json(config.METADATA_PATH) or {}
training = ui.read_json(config.HISTORY_PATH) or {}
metrics = ui.read_json(config.METRICS_PATH) or {}

evaluations = metrics.get("evaluations", {})
validation = evaluations.get("validation", {})
official_test = evaluations.get("official_test", {})

trained = model_available()
has_report = report.get("status") == "ok"

# Prefer official-test results for the headline test metrics.
headline = official_test or validation
evaluation_source = (
    "Official test set" if official_test
    else "Validation set" if validation
    else "No evaluation available"
)


# ---------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------
st.subheader("Project Overview")

kpis = st.columns(5)

image_count = report.get("n_images")
class_count = report.get("n_classes")

ui.kpi(
    kpis[0],
    "Dataset Images",
    f"{image_count:,}" if has_report and image_count is not None else "-",
    "Images found during inspection",
)

ui.kpi(
    kpis[1],
    "Classes",
    class_count if has_report else "-",
    "Traffic-sign categories",
)

ui.kpi(
    kpis[2],
    "Test Accuracy",
    f"{official_test['accuracy']:.2%}" if "accuracy" in official_test else "-",
    "Official test set" if official_test else "No test results",
)

ui.kpi(
    kpis[3],
    "Macro F1",
    f"{headline['macro_f1']:.2%}" if "macro_f1" in headline else "-",
    evaluation_source,
)

ui.kpi(
    kpis[4],
    "Model Status",
    "Trained" if trained else "Not trained",
    f"Epochs: {meta['epochs_run']}" if meta.get("epochs_run") is not None else "",
)


# ---------------------------------------------------------
# Validation vs official test
# ---------------------------------------------------------
st.subheader("Evaluation Results")

if validation or official_test:
    rows = []

    for label, result in (
        ("Validation", validation),
        ("Official Test", official_test),
    ):
        if result:
            rows.append({
                "Dataset": label,
                "Accuracy": result.get("accuracy"),
                "Macro Precision": result.get("macro_precision"),
                "Macro Recall": result.get("macro_recall"),
                "Macro F1": result.get("macro_f1"),
                "Images Evaluated": result.get("n_samples"),
            })

    if rows:
        comparison = pd.DataFrame(rows)
        display = comparison.copy()

        for column in (
            "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
        ):
            display[column] = display[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else "-"
            )

        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Macro metrics average performance across classes, "
            "giving each class equal weight."
        )
else:
    st.info("No evaluation results found. Train and evaluate the model first.")


# ---------------------------------------------------------
# Dataset distribution
# ---------------------------------------------------------
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Dataset Class Distribution")

    distribution = report.get("class_distribution", {})

    if has_report and distribution:
        distribution_df = pd.DataFrame({
            "Class ID": [int(key) for key in distribution.keys()],
            "Images": [int(value) for value in distribution.values()],
        }).sort_values("Class ID")

        distribution_df["Traffic Sign"] = distribution_df["Class ID"].map(
            config.CLASS_NAMES
        ).fillna("Unknown class")

        fig = px.bar(
            distribution_df,
            x="Class ID",
            y="Images",
            hover_data=["Traffic Sign"],
            title="Images per traffic-sign class",
        )
        fig.update_layout(
            height=CHART_HEIGHT,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Class ID",
            yaxis_title="Number of images",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Dataset report is missing. Open Dataset Explorer and refresh "
            "the dataset inspection."
        )

with right:
    st.subheader("Training Accuracy")

    if training.get("accuracy"):
        curves = pd.DataFrame({
            "Epoch": range(1, len(training["accuracy"]) + 1),
            "Training Accuracy": training["accuracy"],
        })

        if training.get("val_accuracy"):
            curves["Validation Accuracy"] = training["val_accuracy"]

        fig = px.line(
            curves,
            x="Epoch",
            y=[c for c in curves.columns if c != "Epoch"],
            markers=True,
            title="Accuracy by epoch",
        )
        fig.update_layout(
            height=CHART_HEIGHT,
            margin=dict(l=10, r=10, t=45, b=10),
            yaxis_title="Accuracy",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training history is not available yet.")


# ---------------------------------------------------------
# Training loss
# ---------------------------------------------------------
st.subheader("Training Loss")

if training.get("loss"):
    loss_df = pd.DataFrame({
        "Epoch": range(1, len(training["loss"]) + 1),
        "Training Loss": training["loss"],
    })

    if training.get("val_loss"):
        loss_df["Validation Loss"] = training["val_loss"]

    fig = px.line(
        loss_df,
        x="Epoch",
        y=[c for c in loss_df.columns if c != "Epoch"],
        markers=True,
        title="Loss by epoch",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_title="Loss",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No training-loss history found.")


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
st.divider()
st.subheader("Confusion Matrix")

st.write(
    "Rows represent actual classes and columns represent predicted classes. "
    "Large values away from the main diagonal indicate misclassifications."
)

matrix_path = config.OUTPUTS_DIR / "confusion_matrix_official_test.png"

if not matrix_path.exists():
    matrix_path = config.OUTPUTS_DIR / "confusion_matrix_validation.png"

if matrix_path.exists():
    st.image(
        str(matrix_path),
        caption=f"Confusion matrix: {matrix_path.stem.replace('_', ' ')}",
        use_container_width=True,
    )
else:
    st.info("No confusion-matrix image was found in the outputs folder.")


# ---------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------
st.subheader("Per-Class Performance")

per_class_path = (
    config.OUTPUTS_DIR / "per_class_metrics_official_test.csv"
)

if not per_class_path.exists():
    per_class_path = (
        config.OUTPUTS_DIR / "per_class_metrics_validation.csv"
    )

if per_class_path.exists():
    per_class = pd.read_csv(per_class_path)

    # Handle common metric column naming variations.
    normalized = {
        str(column).strip().lower().replace("-", "_").replace(" ", "_"): column
        for column in per_class.columns
    }

    f1_column = next(
        (
            normalized[key]
            for key in ("f1", "f1_score", "f1score", "macro_f1")
            if key in normalized
        ),
        None,
    )

    class_column = next(
        (
            normalized[key]
            for key in ("class_id", "class", "label")
            if key in normalized
        ),
        None,
    )

    if class_column:
        per_class["Traffic Sign"] = per_class[class_column].map(
            lambda value: config.CLASS_NAMES.get(
                int(value), str(value)
            )
            if pd.notna(value) and str(value).replace(".", "", 1).isdigit()
            else str(value)
        )

    if f1_column:
        per_class[f1_column] = pd.to_numeric(
            per_class[f1_column], errors="coerce"
        )
        weakest = per_class.sort_values(
            f1_column, ascending=True, na_position="last"
        ).head(10)

        st.markdown("**10 weakest classes by F1 score**")
        st.dataframe(weakest, use_container_width=True, hide_index=True)

        plot_column = "Traffic Sign" if "Traffic Sign" in weakest else class_column

        if plot_column:
            fig = px.bar(
                weakest,
                x=f1_column,
                y=plot_column,
                orientation="h",
                title="Lowest-performing classes",
            )
            fig.update_layout(
                height=400,
                xaxis_title="F1 score",
                yaxis_title="Traffic-sign class",
                xaxis_tickformat=".0%",
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "The CSV was found, but its F1 column could not be identified. "
            "See the complete table below."
        )

    with st.expander("View all per-class metrics"):
        st.dataframe(per_class, use_container_width=True, hide_index=True)
else:
    st.info("No per-class metrics CSV was found.")


# ---------------------------------------------------------
# Model details and recent predictions
# ---------------------------------------------------------
st.divider()
details_left, details_right = st.columns(2)

with details_left:
    st.subheader("Model Information")

    if meta:
        fields = (
            "trained_at",
            "epochs_run",
            "batch_size",
            "learning_rate",
            "image_size",
            "train_images",
            "validation_images",
            "training_minutes",
            "tensorflow",
        )
        st.json({key: meta[key] for key in fields if key in meta})
    else:
        st.info("No model metadata found.")

    st.subheader("Project Status")
    st.write(f"{'✅' if has_report else '⚪'} Dataset inspected")
    st.write(f"{'✅' if trained else '⚪'} Trained model available")
    st.write(f"{'✅' if evaluations else '⚪'} Evaluation results available")

with details_right:
    st.subheader("Recent Predictions")

    prediction_history = history.read_history()

    if prediction_history.empty:
        st.info("No predictions recorded yet. Try the prediction page.")
    else:
        columns = [
            column for column in (
                "timestamp", "filename", "predicted_class", "confidence"
            )
            if column in prediction_history.columns
        ]

        st.dataframe(
            prediction_history.tail(10)[columns].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Traffic Sign Recognition | CNN trained on GTSRB | "
    "For educational and research use; predictions may be incorrect."
)
