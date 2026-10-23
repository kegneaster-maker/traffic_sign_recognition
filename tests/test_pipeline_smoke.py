"""End-to-end smoke test on a SYNTHETIC dataset (verifies code paths, not real-world accuracy)."""
import json

import config
from src import predict, train


def test_train_and_evaluate_end_to_end(tmp_config, synthetic_dataset):
    train.main(["--data-dir", str(synthetic_dataset), "--epochs", "2", "--batch-size", "8",
                "--filters", "4,8", "--dense-units", "16", "--patience", "2"])
    status = json.loads(config.TRAIN_STATUS_PATH.read_text())
    assert status["state"] == "completed"
    assert config.MODEL_PATH.exists() and config.CLASS_MAP_PATH.exists() and config.METADATA_PATH.exists()
    metrics = json.loads(config.METRICS_PATH.read_text())["evaluations"]
    assert set(metrics) == {"validation", "official_test"}
    assert predict.load_bundle() is not None


def test_training_failure_reported(tmp_config, tmp_path):
    try:
        train.main(["--data-dir", str(tmp_path / "missing"), "--epochs", "1"])
    except SystemExit as e:
        assert e.code == 1
    assert json.loads(config.TRAIN_STATUS_PATH.read_text())["state"] == "failed"
