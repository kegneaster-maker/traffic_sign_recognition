"""Local CSV prediction history. Uploaded images are never stored."""
from __future__ import annotations

import csv
import json
from datetime import datetime

import pandas as pd

import config

COLUMNS = ["timestamp", "filename", "predicted_class_id", "predicted_class", "confidence",
           "top_predictions", "ground_truth_id"]


def add_record(filename, result: dict, ground_truth_id=None, path=None):
    path = path or config.PREDICTION_HISTORY_PATH
    new = not path.exists()
    top = json.dumps([{"id": t["class_id"], "name": t["class_name"], "p": round(t["probability"], 4)}
                      for t in result["top"]])
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), filename, result["class_id"],
                    result["class_name"], round(result["confidence"], 4), top,
                    "" if ground_truth_id is None else int(ground_truth_id)])


def read_history(path=None) -> pd.DataFrame:
    path = path or config.PREDICTION_HISTORY_PATH
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def clear_history(path=None):
    path = path or config.PREDICTION_HISTORY_PATH
    if path.exists():
        path.unlink()
