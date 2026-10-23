import json

import numpy as np

import config
from src import evaluate as ev


def test_metrics_with_missing_classes_does_not_crash():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])  # class 2 never predicted, 3..42 absent
    s, per_class, cm = ev.compute_metrics(y_true, y_pred)
    assert abs(s["accuracy"] - 0.6) < 1e-9 and cm.shape == (43, 43)
    assert len(per_class) == 43 and per_class.loc[2, "recall"] == 0.0
    assert 0 <= s["macro_f1"] <= 1 and s["classes_in_macro_average"] == 3


def test_evaluate_arrays_writes_outputs(tmp_config, tiny_model):
    X = np.random.default_rng(0).random((20, 32, 32, 3), dtype=np.float32)
    y = np.arange(20) % 4
    s, _, _ = ev.evaluate_arrays(tiny_model, X, y, [f"p{i}" for i in range(20)], "validation")
    out = config.OUTPUTS_DIR
    for f in ("per_class_metrics_validation.csv", "confusion_matrix_validation.csv",
              "confusion_matrix_validation.png"):
        assert (out / f).exists()
    data = json.loads(config.METRICS_PATH.read_text())
    assert data["evaluations"]["validation"]["source"] == "validation"
    assert s["n_samples"] == 20


def test_plot_history(tmp_path):
    ev.plot_history({"accuracy": [.1, .2], "val_accuracy": [.1, .3], "loss": [2, 1], "val_loss": [2, 1.5]},
                    tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
