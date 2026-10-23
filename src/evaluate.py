"""Evaluation: metrics, confusion matrix, example predictions, plots. Saves JSON/CSV/PNG to outputs/."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import config


def compute_metrics(y_true, y_pred, num_classes=config.NUM_CLASSES):
    """Macro scores average over classes present in y_true or y_pred (absent classes would be meaningless)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    all_labels = list(range(num_classes))
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    pp, rr, ff, ss = precision_recall_fscore_support(y_true, y_pred, labels=all_labels, zero_division=0)
    per_class = pd.DataFrame({"class_id": all_labels,
                              "class_name": [config.CLASS_NAMES.get(i, str(i)) for i in all_labels],
                              "precision": pp, "recall": rr, "f1": ff, "support": ss})
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
               "macro_recall": float(r), "macro_f1": float(f), "n_samples": int(len(y_true)),
               "classes_in_macro_average": len(present)}
    return summary, per_class, cm


def _examples(paths, y_true, y_pred, conf, correct, k=8):
    sel = np.where(y_true == y_pred)[0] if correct else np.where(y_true != y_pred)[0]
    if not correct:
        sel = sel[np.argsort(-conf[sel])]  # most confident mistakes first
    return [{"path": str(paths[i]), "true": int(y_true[i]), "pred": int(y_pred[i]),
             "confidence": float(conf[i])} for i in sel[:k]]


def evaluate_arrays(model, X, y, paths, source: str, num_classes=config.NUM_CLASSES, outdir=None, batch_size=128):
    """source: 'validation' or 'official_test'. Writes metrics JSON (merged), per-class CSV, CM CSV/PNG."""
    outdir = outdir or config.OUTPUTS_DIR
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred, conf = probs.argmax(1), probs.max(1)
    summary, per_class, cm = compute_metrics(y, y_pred, num_classes)
    summary["source"] = source
    summary["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["examples"] = {"correct": _examples(paths, y, y_pred, conf, True),
                           "incorrect": _examples(paths, y, y_pred, conf, False)}

    per_class.to_csv(outdir / f"per_class_metrics_{source}.csv", index=False)
    pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes)).to_csv(
        outdir / f"confusion_matrix_{source}.csv")
    plot_confusion_matrix(cm, outdir / f"confusion_matrix_{source}.png", f"Confusion matrix ({source})")

    metrics_file = outdir / config.METRICS_PATH.name
    data = json.loads(metrics_file.read_text()) if metrics_file.exists() else {"evaluations": {}}
    data.setdefault("evaluations", {})[source] = summary
    metrics_file.write_text(json.dumps(data, indent=2))
    return summary, per_class, cm


def plot_confusion_matrix(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history.get(key, []), label="train")
        ax.plot(history.get(f"val_{key}", []), label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    import argparse
    from src import dataset as ds, preprocessing as pp, predict
    ap = argparse.ArgumentParser(description="Evaluate the saved model")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--source", choices=["validation", "test", "both"], default="both")
    a = ap.parse_args()
    bundle = predict.load_bundle()
    if bundle is None:
        raise SystemExit("No trained model found. Run: python -m src.train")
    size = bundle.img_size
    rec = ds.load_records(a.data_dir)
    if a.source in ("validation", "both"):
        _, val = ds.split_records(rec["train"])
        X, y, kept, _ = pp.load_arrays(val, size)
        s, _, _ = evaluate_arrays(bundle.model, X, y, kept["path"].tolist(), "validation", bundle.num_classes)
        print("validation:", {k: round(v, 4) for k, v in s.items() if isinstance(v, float)})
    if a.source in ("test", "both"):
        if rec["test"] is None:
            print("No labelled official test set available; skipped.")
        else:
            X, y, kept, _ = pp.load_arrays(rec["test"], size)
            s, _, _ = evaluate_arrays(bundle.model, X, y, kept["path"].tolist(), "official_test", bundle.num_classes)
            print("official_test:", {k: round(v, 4) for k, v in s.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
