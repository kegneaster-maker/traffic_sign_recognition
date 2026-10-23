"""Inference helpers: load/verify the saved model bundle and predict on a single image."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import config
from src.preprocessing import preprocess_image


class ModelNotTrainedError(Exception):
    pass


class ModelMismatchError(Exception):
    pass


@dataclass
class Bundle:
    model: object
    class_names: dict   # int id -> name
    img_size: int
    num_classes: int


def model_available() -> bool:
    return config.MODEL_PATH.exists() and config.CLASS_MAP_PATH.exists()


def load_bundle(model_path=None, class_map_path=None, preproc_path=None):
    """Return a verified Bundle, or None if no trained model exists. Raises ModelMismatchError on inconsistency."""
    model_path = model_path or config.MODEL_PATH
    class_map_path = class_map_path or config.CLASS_MAP_PATH
    preproc_path = preproc_path or config.PREPROC_PATH
    if not (model_path.exists() and class_map_path.exists()):
        return None
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    cmap = json.loads(class_map_path.read_text())
    names = {int(k): v for k, v in cmap["names"].items()}
    size = config.IMG_SIZE
    if preproc_path.exists():
        size = int(json.loads(preproc_path.read_text())["image_size"])
    n_out = int(model.output_shape[-1])
    if n_out != len(names):
        raise ModelMismatchError(f"Model outputs {n_out} classes but the class mapping has {len(names)}.")
    if tuple(model.input_shape[1:3]) != (size, size):
        raise ModelMismatchError(f"Model expects {model.input_shape[1:3]} but preprocessing size is {size}.")
    return Bundle(model, names, size, n_out)


def predict_image(img, bundle: Bundle, top_k: int = 5) -> dict:
    x = preprocess_image(img, bundle.img_size)[None]
    probs = np.asarray(bundle.model.predict(x, verbose=0)[0], dtype=np.float64)
    if probs.shape[0] != bundle.num_classes or not np.isfinite(probs).all():
        raise ModelMismatchError("Model returned an invalid probability vector.")
    order = np.argsort(-probs)[:top_k]
    top = [{"class_id": int(i), "class_name": bundle.class_names.get(int(i), str(i)),
            "probability": float(probs[i])} for i in order]
    margin = top[0]["probability"] - (top[1]["probability"] if len(top) > 1 else 0.0)
    uncertain = top[0]["probability"] < config.CONFIDENCE_THRESHOLD or margin < 0.2
    return {"class_id": top[0]["class_id"], "class_name": top[0]["class_name"],
            "confidence": top[0]["probability"], "top": top, "uncertain": bool(uncertain)}
