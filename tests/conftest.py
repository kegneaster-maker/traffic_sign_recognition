import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect all output/model paths to a temp dir so tests never touch real results."""
    out, mod = tmp_path / "outputs", tmp_path / "models"
    out.mkdir(); mod.mkdir()
    for name, base in [("OUTPUTS_DIR", out), ("MODELS_DIR", mod)]:
        monkeypatch.setattr(config, name, base)
    monkeypatch.setattr(config, "MODEL_PATH", mod / "m.keras")
    monkeypatch.setattr(config, "CLASS_MAP_PATH", mod / "class_mapping.json")
    monkeypatch.setattr(config, "PREPROC_PATH", mod / "preprocessing.json")
    monkeypatch.setattr(config, "METADATA_PATH", mod / "meta.json")
    monkeypatch.setattr(config, "HISTORY_PATH", out / "history.json")
    monkeypatch.setattr(config, "METRICS_PATH", out / "metrics.json")
    monkeypatch.setattr(config, "DATASET_REPORT_PATH", out / "report.json")
    monkeypatch.setattr(config, "TRAIN_STATUS_PATH", out / "status.json")
    monkeypatch.setattr(config, "TRAIN_LOG_PATH", out / "train.log")
    monkeypatch.setattr(config, "PREDICTION_HISTORY_PATH", out / "pred.csv")
    return tmp_path


def make_img(path, color, size=(40, 40), mode="RGB"):
    Image.new(mode, size, color).save(path)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """SYNTHETIC fixture for testing code paths only - not real GTSRB data."""
    root = tmp_path / "ds"
    colors = {0: (200, 30, 30), 1: (30, 200, 30), 2: (30, 30, 200)}
    for cid, col in colors.items():
        d = root / "Train" / str(cid)
        d.mkdir(parents=True)
        for i in range(12):
            make_img(d / f"{i:05d}.png", tuple(min(255, c + i) for c in col))
    (root / "Test").mkdir()
    rows = ["Path,ClassId"]
    for i, cid in enumerate([0, 1, 2, 0, 1, 2]):
        make_img(root / "Test" / f"{i:05d}.png", colors[cid])
        rows.append(f"Test/{i:05d}.png,{cid}")
    (root / "Test.csv").write_text("\n".join(rows))
    return root


@pytest.fixture
def tiny_model():
    from src.model import build_model
    return build_model(32, config.NUM_CLASSES, filters=(4,), dense_units=8, dropout=0.1, augment=False)
