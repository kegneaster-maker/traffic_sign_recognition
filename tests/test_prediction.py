import io
import json

import numpy as np
import pytest
from PIL import Image

import config
from src import history, predict


def test_model_output_and_input_shape(tiny_model):
    assert tiny_model.input_shape == (None, 32, 32, 3)
    assert tiny_model.output_shape == (None, config.NUM_CLASSES)


def test_missing_model_returns_none(tmp_config):
    assert predict.model_available() is False
    assert predict.load_bundle() is None


def _save_bundle(model, n_names=config.NUM_CLASSES):
    model.save(config.MODEL_PATH)
    config.CLASS_MAP_PATH.write_text(json.dumps({"names": {str(i): f"c{i}" for i in range(n_names)}}))
    config.PREPROC_PATH.write_text(json.dumps({"image_size": 32}))


def test_probabilities_valid(tmp_config, tiny_model):
    _save_bundle(tiny_model)
    b = predict.load_bundle()
    r = predict.predict_image(Image.new("RGB", (60, 60), (200, 0, 0)), b)
    assert len(r["top"]) == 5
    ps = [t["probability"] for t in r["top"]]
    assert ps == sorted(ps, reverse=True) and 0 <= ps[0] <= 1
    assert r["confidence"] == ps[0] and isinstance(r["uncertain"], bool)
    assert r["class_name"] == f"c{r['class_id']}"


def test_class_mapping_mismatch_detected(tmp_config, tiny_model):
    _save_bundle(tiny_model, n_names=10)
    with pytest.raises(predict.ModelMismatchError):
        predict.load_bundle()


def test_history_roundtrip_and_clear(tmp_config):
    res = {"class_id": 14, "class_name": "Stop", "confidence": 0.93,
           "top": [{"class_id": 14, "class_name": "Stop", "probability": 0.93}]}
    history.add_record("a.png", res)
    history.add_record("b.png", res, ground_truth_id=14)
    h = history.read_history()
    assert len(h) == 2 and h["ground_truth_id"].notna().sum() == 1
    history.clear_history()
    assert history.read_history().empty
