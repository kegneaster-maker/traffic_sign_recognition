import io

import numpy as np
import pytest
from PIL import Image

import config
from src.preprocessing import InvalidImageError, load_arrays, preprocess_image, validate_upload


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_shape_dtype_range(mode):
    arr = preprocess_image(Image.new(mode, (50, 70)), 32)
    assert arr.shape == (32, 32, 3) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_custom_size_and_determinism():
    img = Image.new("RGB", (64, 64), (10, 200, 30))
    a, b = preprocess_image(img, 48), preprocess_image(img, 48)
    assert a.shape == (48, 48, 3) and np.array_equal(a, b)


def _bytes(fmt, size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, fmt)
    return buf.getvalue()


def test_validate_upload_ok():
    assert validate_upload(_bytes("PNG")).size == (32, 32)
    assert validate_upload(_bytes("JPEG")).size == (32, 32)


@pytest.mark.parametrize("bad", [b"", b"not an image", _bytes("GIF"), _bytes("PNG", (4, 4))])
def test_validate_upload_rejects(bad):
    with pytest.raises(InvalidImageError):
        validate_upload(bad)


def test_validate_upload_too_large():
    with pytest.raises(InvalidImageError):
        validate_upload(_bytes("PNG"), max_mb=0.00001)


def test_load_arrays_skips_corrupt(tmp_path):
    import pandas as pd
    good, bad = tmp_path / "g.png", tmp_path / "b.png"
    Image.new("RGB", (40, 40)).save(good)
    bad.write_bytes(b"corrupt")
    df = pd.DataFrame({"path": [str(good), str(bad)], "class_id": [3, 4]})
    X, y, kept, skipped = load_arrays(df, 32)
    assert X.shape == (1, 32, 32, 3) and y.tolist() == [3] and skipped == 1 and len(kept) == 1
