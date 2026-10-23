import json

import pytest

import config
from src import dataset as ds


def test_discovers_class_folders_and_test_csv(synthetic_dataset):
    rec = ds.load_records(synthetic_dataset)
    assert rec["structure"] == "class_folders"
    assert len(rec["train"]) == 36 and sorted(rec["train"]["class_id"].unique()) == [0, 1, 2]
    assert rec["test"] is not None and len(rec["test"]) == 6


def test_csv_layout(tmp_path):
    from PIL import Image
    (tmp_path / "Train").mkdir()
    rows = ["Width,Height,ClassId,Path"]
    for i in range(6):
        Image.new("RGB", (30, 30)).save(tmp_path / "Train" / f"{i}.png")
        rows.append(f"30,30,{i % 2},Train/{i}.png")
    rows.append("30,30,99,Train/0.png")  # invalid class id
    (tmp_path / "Train.csv").write_text("\n".join(rows))
    rec = ds.load_records(tmp_path)
    assert rec["structure"] == "csv" and len(rec["train"]) == 6 and rec["invalid_labels"] == 1


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(ds.DatasetNotFoundError) as e:
        ds.load_records(tmp_path / "nope")
    assert "config.py" in str(e.value)
    assert ds.readiness(tmp_path / "nope")["ready"] is False


def test_inspect_not_found_does_not_crash(tmp_config, tmp_path):
    r = ds.inspect_dataset(tmp_path / "nope")
    assert r["status"] == "not_found"


def test_inspect_report(tmp_config, synthetic_dataset):
    r = ds.inspect_dataset(synthetic_dataset)
    assert r["n_images"] == 36 and r["n_classes"] == 3 and r["unreadable_in_sample"] == 0
    assert r["official_test_labels_available"] and "PNG" in r["formats"]
    assert json.loads(config.DATASET_REPORT_PATH.read_text())["n_images"] == 36


def test_unreadable_detected(tmp_config, synthetic_dataset):
    (synthetic_dataset / "Train" / "0" / "99999.png").write_bytes(b"junk")
    r = ds.inspect_dataset(synthetic_dataset)
    assert r["unreadable_in_sample"] == 1 and any("unreadable" in w for w in r["warnings"])


def test_split_reproducible_and_disjoint(synthetic_dataset):
    df = ds.load_records(synthetic_dataset)["train"]
    a_tr, a_va = ds.split_records(df, 0.25, 1)
    b_tr, b_va = ds.split_records(df, 0.25, 1)
    assert a_va["path"].tolist() == b_va["path"].tolist()
    assert not set(a_tr["path"]) & set(a_va["path"])
