"""Shared preprocessing for training and inference (single code path => identical behaviour)."""
from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import config


class InvalidImageError(Exception):
    """Raised for corrupted, unsupported or oversized images."""


def preprocess_image(img, size: int = config.IMG_SIZE) -> np.ndarray:
    """PIL image or path -> float32 array (size, size, 3) in [0, 1]."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    arr = np.asarray(img.convert("RGB"))
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.float32) / 255.0


def validate_upload(data, max_mb: float = config.MAX_UPLOAD_MB) -> Image.Image:
    """Validate uploaded bytes/file-like and return a loaded PIL image."""
    raw = data if isinstance(data, (bytes, bytearray)) else data.read()
    if not raw:
        raise InvalidImageError("The file is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise InvalidImageError(f"The file is larger than {max_mb} MB.")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise InvalidImageError(f"The file is not a readable image ({e}).") from e
    if fmt not in ("PNG", "JPEG"):
        raise InvalidImageError(f"Unsupported format '{fmt}'. Use PNG, JPG or JPEG.")
    if min(img.size) < 8:
        raise InvalidImageError("The image is too small (minimum 8 x 8 pixels).")
    return img


def load_arrays(df: pd.DataFrame, size: int = config.IMG_SIZE, progress=None):
    """Load images listed in df[path, class_id]. Unreadable files are skipped and counted."""
    X, y, kept, skipped = [], [], [], 0
    n = len(df)
    for i, (p, c) in enumerate(zip(df["path"], df["class_id"])):
        try:
            X.append(preprocess_image(p, size))
            y.append(int(c))
            kept.append(i)
        except Exception:
            skipped += 1
        if progress and i % 2000 == 0:
            progress(i, n)
    if not X:
        return np.zeros((0, size, size, 3), np.float32), np.zeros((0,), np.int64), df.iloc[[]], skipped
    return np.stack(X), np.asarray(y, dtype=np.int64), df.iloc[kept].reset_index(drop=True), skipped


def preprocessing_config(size: int = config.IMG_SIZE) -> dict:
    return {"image_size": size, "color_mode": "RGB", "scale": "x/255.0 -> [0,1]",
            "resize": "cv2.INTER_AREA", "augmentation": "in-model (training only): rotation, zoom, translation; no flips"}


def save_preprocessing(size: int = config.IMG_SIZE):
    config.PREPROC_PATH.write_text(json.dumps(preprocessing_config(size), indent=2))
