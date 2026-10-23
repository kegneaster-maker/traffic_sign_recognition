"""Dataset discovery, validation and inspection for GTSRB (folders and/or CSV layouts)."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
PATH_COLS = ("path", "filename", "file", "image")
CLASS_COLS = ("classid", "class_id", "class", "label")
TEST_CSV_NAMES = {"test.csv", "gt-final_test.csv", "gt-final_test.test.csv"}
TRAIN_CSV_NAMES = {"train.csv"}


class DatasetNotFoundError(Exception):
    """Raised when the dataset directory or usable images cannot be found."""


def not_found_message(root) -> str:
    return (f"Dataset not found at '{root}'. Set the folder in config.py (DATASET_DIR), or set the "
            "GTSRB_DATASET_DIR environment variable, or enter the path in the dashboard sidebar. "
            "The folder must contain GTSRB images (class folders 0-42 and/or Train.csv).")


def _is_img(name) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS


def _find_csv(root: Path, names: set):
    for dirpath, dirs, files in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) > 3:
            dirs[:] = []
            continue
        for f in files:
            if f.lower() in names:
                return Path(dirpath) / f
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _col(df, candidates):
    lowered = {c.lower().replace(" ", ""): c for c in df.columns}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _find_class_folders(root: Path):
    """Return (parent_dir, [class folder names]) for the best class-number folder group, or None."""
    best = None
    for dirpath, dirs, _files in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > 4 or any(p.lower().startswith("test") for p in rel.parts):
            dirs[:] = []
            continue
        digit = [d for d in dirs if d.isdigit()]
        if len(digit) < 2:
            continue
        ok = []
        for d in digit:
            try:
                if any(e.is_file() and _is_img(e.name) for e in os.scandir(Path(dirpath) / d)):
                    ok.append(d)
            except OSError:
                pass
        if len(ok) >= 2:
            score = (len(ok), "train" in dirpath.lower())
            if best is None or score > best[0]:
                best = (score, Path(dirpath), ok)
    return None if best is None else (best[1], best[2])


def _resolve(p, csv_dir: Path, root: Path) -> str:
    p = str(p).replace("\\", "/")
    for base in (csv_dir, root, root / "Test", root / "Train"):
        cand = base / p
        if cand.exists():
            return str(cand)
    return str(csv_dir / p)


def load_records(root=None) -> dict:
    """Locate training/test records without reading image pixels.

    Returns {"train": DataFrame[path,class_id], "test": DataFrame|None, "structure": str,
             "notes": [str], "invalid_labels": int, "missing_files": int}
    """
    root = Path(root or config.DATASET_DIR)
    if not root.exists() or not root.is_dir():
        raise DatasetNotFoundError(not_found_message(root))
    notes, invalid_labels, missing = [], 0, 0

    train, structure = None, "none"
    found = _find_class_folders(root)
    if found:
        parent, names = found
        rows = []
        for name in sorted(names, key=int):
            for e in os.scandir(parent / name):
                if e.is_file() and _is_img(e.name):
                    rows.append((e.path, int(name)))
        train = pd.DataFrame(rows, columns=["path", "class_id"])
        structure = "class_folders"
    else:
        csv = _find_csv(root, TRAIN_CSV_NAMES)
        if csv is not None:
            df = _read_csv(csv)
            pc, cc = _col(df, PATH_COLS), _col(df, CLASS_COLS)
            if pc and cc:
                train = pd.DataFrame({"path": [_resolve(p, csv.parent, root) for p in df[pc]],
                                      "class_id": pd.to_numeric(df[cc], errors="coerce")})
                structure = "csv"
            else:
                notes.append(f"{csv.name} has no recognisable path/class columns: {list(df.columns)}")
    if train is None or train.empty:
        raise DatasetNotFoundError(not_found_message(root) + " No training images/labels were found.")

    bad = train["class_id"].isna() | ~train["class_id"].between(0, config.NUM_CLASSES - 1)
    invalid_labels = int(bad.sum())
    train = train[~bad].copy()
    train["class_id"] = train["class_id"].astype(int)
    exists = train["path"].map(os.path.exists)
    missing = int((~exists).sum())
    train = train[exists].reset_index(drop=True)
    train = train.sort_values("path").reset_index(drop=True)  # deterministic order

    test = None
    tcsv = _find_csv(root, TEST_CSV_NAMES)
    if tcsv is not None:
        tdf = _read_csv(tcsv)
        pc, cc = _col(tdf, PATH_COLS), _col(tdf, CLASS_COLS)
        if pc and cc:
            t = pd.DataFrame({"path": [_resolve(p, tcsv.parent, root) for p in tdf[pc]],
                              "class_id": pd.to_numeric(tdf[cc], errors="coerce")})
            t = t[t["class_id"].between(0, config.NUM_CLASSES - 1)]
            t["class_id"] = t["class_id"].astype(int)
            t = t[t["path"].map(os.path.exists)].reset_index(drop=True)
            test = t if not t.empty else None
            if test is None:
                notes.append(f"{tcsv.name} has labels but its image files were not found.")
        else:
            notes.append(f"{tcsv.name} has no class labels: official test labels are unavailable, "
                         "so a validation split is used instead.")
    return {"train": train, "test": test, "structure": structure, "notes": notes,
            "invalid_labels": invalid_labels, "missing_files": missing}


def split_records(df: pd.DataFrame, val_split=config.VAL_SPLIT, seed=config.SEED):
    """Reproducible stratified train/validation split (falls back if a class is too small)."""
    from sklearn.model_selection import train_test_split
    try:
        tr, va = train_test_split(df.index, test_size=val_split, random_state=seed, stratify=df["class_id"])
    except ValueError:
        tr, va = train_test_split(df.index, test_size=val_split, random_state=seed)
    return df.loc[tr].reset_index(drop=True), df.loc[va].reset_index(drop=True)


def readiness(root=None) -> dict:
    """Quick check (no pixel reads) used by the training page."""
    try:
        rec = load_records(root)
    except DatasetNotFoundError as e:
        return {"ready": False, "message": str(e)}
    n_classes = rec["train"]["class_id"].nunique()
    return {"ready": True, "n_train": len(rec["train"]), "n_classes": int(n_classes),
            "structure": rec["structure"], "official_test": rec["test"] is not None,
            "message": "ok", "notes": rec["notes"]}


def inspect_dataset(root=None, sample=3000, seed=config.SEED, save=True) -> dict:
    """Build a dataset report. Pixel-level checks (readability/size) run on a reproducible sample."""
    root = Path(root or config.DATASET_DIR)
    report = {"dataset_root": str(root), "status": "ok", "warnings": []}
    try:
        rec = load_records(root)
    except DatasetNotFoundError as e:
        report.update(status="not_found", message=str(e))
        if save:
            config.DATASET_REPORT_PATH.write_text(json.dumps(report, indent=2))
        return report

    train, test = rec["train"], rec["test"]
    dist = train["class_id"].value_counts().sort_index()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train))
    if len(idx) > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    widths, heights, fmts, unreadable = [], [], Counter(), []
    for i in idx:
        p = train.at[i, "path"]
        try:
            with Image.open(p) as im:
                fmts[(im.format or "UNKNOWN")] += 1
                widths.append(im.width)
                heights.append(im.height)
                im.verify()
        except Exception:
            unreadable.append(p)

    missing_classes = [c for c in range(config.NUM_CLASSES) if c not in dist.index]
    report.update(
        structure=rec["structure"], n_images=int(len(train)), n_classes=int(dist.size),
        class_distribution={int(k): int(v) for k, v in dist.items()},
        missing_classes=missing_classes,
        n_test_images=int(len(test)) if test is not None else 0,
        official_test_labels_available=test is not None,
        checked_sample=int(len(idx)),
        formats=dict(fmts),
        dimensions={"width_min": int(min(widths)), "width_max": int(max(widths)),
                    "width_mean": round(float(np.mean(widths)), 1),
                    "height_min": int(min(heights)), "height_max": int(max(heights)),
                    "height_mean": round(float(np.mean(heights)), 1)} if widths else {},
        unreadable_in_sample=len(unreadable), unreadable_examples=unreadable[:20],
        invalid_labels=rec["invalid_labels"], missing_files=rec["missing_files"], notes=rec["notes"],
    )
    w = report["warnings"]
    if unreadable:
        w.append(f"{len(unreadable)} unreadable image(s) in the checked sample; they are skipped in training.")
    if rec["invalid_labels"]:
        w.append(f"{rec['invalid_labels']} record(s) have missing or invalid class IDs and are excluded.")
    if rec["missing_files"]:
        w.append(f"{rec['missing_files']} referenced file(s) are missing on disk.")
    if missing_classes:
        w.append(f"Classes with no images: {missing_classes}")
    if dist.size and dist.max() / max(dist.min(), 1) > 5:
        w.append("Strong class imbalance (largest class > 5x smallest).")
    if test is None:
        w.append("No labelled official test set found; results will be reported on a validation split.")
    w.extend(rec["notes"])
    if save:
        config.DATASET_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the GTSRB dataset")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sample", type=int, default=3000)
    r = inspect_dataset(ap.parse_args().data_dir, sample=ap.parse_args().sample)
    print(json.dumps({k: v for k, v in r.items() if k != "class_distribution"}, indent=2))
