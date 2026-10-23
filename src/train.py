"""Standalone training script:  python -m src.train --epochs 30 --batch-size 64"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import config

log = logging.getLogger("train")


def write_status(**kw):
    cur = {}
    if config.TRAIN_STATUS_PATH.exists():
        try:
            cur = json.loads(config.TRAIN_STATUS_PATH.read_text())
        except Exception:
            cur = {}
    cur.update(kw, updated_at=datetime.now().isoformat(timespec="seconds"))
    config.TRAIN_STATUS_PATH.write_text(json.dumps(cur, indent=2))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the traffic sign CNN")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    ap.add_argument("--learning-rate", type=float, default=config.DEFAULT_LR)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--filters", default=",".join(map(str, config.DEFAULT_FILTERS)))
    ap.add_argument("--dense-units", type=int, default=config.DEFAULT_DENSE)
    ap.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    ap.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=0, help="Use at most N images per class (quick tests)")
    return ap.parse_args(argv)


def run(a):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from src import dataset as ds, preprocessing as pp
    from src.evaluate import evaluate_arrays, plot_history
    from src.model import build_model

    tf.keras.utils.set_random_seed(config.SEED)
    gpus = tf.config.list_physical_devices("GPU")
    log.info("GPU detected: %s", [g.name for g in gpus] if gpus else "none, using CPU")

    rec = ds.load_records(a.data_dir)
    train_df = rec["train"]
    if a.limit_per_class:
        train_df = train_df.groupby("class_id", group_keys=False).head(a.limit_per_class).reset_index(drop=True)
    tr_df, va_df = ds.split_records(train_df, a.val_split, config.SEED)
    log.info("Records: train=%d validation=%d official_test=%s", len(tr_df), len(va_df),
             "available" if rec["test"] is not None else "not available")

    write_status(message="Loading images...")
    X_tr, y_tr, _, s1 = pp.load_arrays(tr_df, a.img_size, lambda i, n: log.info("loading train %d/%d", i, n))
    X_va, y_va, va_kept, s2 = pp.load_arrays(va_df, a.img_size)
    if s1 + s2:
        log.warning("Skipped %d unreadable image(s)", s1 + s2)
    if len(X_tr) == 0 or len(X_va) == 0:
        raise RuntimeError("No readable images were loaded.")

    filters = tuple(int(f) for f in a.filters.split(","))
    model = build_model(a.img_size, config.NUM_CLASSES, filters, a.dense_units, a.dropout,
                        a.learning_rate, not a.no_augment)
    log.info("Model parameters: %d", model.count_params())

    class StatusCB(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.hist = {}

        def on_epoch_end(self, epoch, logs=None):
            for k, v in (logs or {}).items():
                self.hist.setdefault(k, []).append(float(v))
            write_status(state="running", epoch=epoch + 1, epochs=a.epochs, history=self.hist,
                         message=f"Epoch {epoch + 1}/{a.epochs} finished")
            log.info("epoch %d/%d  %s", epoch + 1, a.epochs,
                     "  ".join(f"{k}={v:.4f}" for k, v in (logs or {}).items()))

    cb = StatusCB()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=a.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, a.patience // 2), min_lr=1e-6),
        cb,
    ]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=a.epochs,
                     batch_size=a.batch_size, callbacks=callbacks, verbose=0)
    minutes = (time.time() - t0) / 60

    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    config.HISTORY_PATH.write_text(json.dumps(history, indent=2))
    plot_history(history, config.OUTPUTS_DIR / "learning_curves.png")
    pp.save_preprocessing(a.img_size)
    config.CLASS_MAP_PATH.write_text(json.dumps({
        "class_ids": list(range(config.NUM_CLASSES)),
        "names": {str(i): n for i, n in config.CLASS_NAMES.items()}}, indent=2))
    config.METADATA_PATH.write_text(json.dumps({
        "trained_at": datetime.now().isoformat(timespec="seconds"), "epochs_requested": a.epochs,
        "epochs_run": len(history.get("loss", [])), "batch_size": a.batch_size,
        "learning_rate": a.learning_rate, "image_size": a.img_size, "filters": list(filters),
        "dense_units": a.dense_units, "dropout": a.dropout, "augmentation": not a.no_augment,
        "train_images": int(len(X_tr)), "validation_images": int(len(X_va)),
        "training_minutes": round(minutes, 2), "dataset_root": str(a.data_dir or config.DATASET_DIR),
        "tensorflow": tf.__version__, "seed": config.SEED}, indent=2))

    best = keras.models.load_model(config.MODEL_PATH)  # best checkpoint
    if config.METRICS_PATH.exists():
        config.METRICS_PATH.unlink()  # drop stale metrics from earlier runs
    s, _, _ = evaluate_arrays(best, X_va, y_va, va_kept["path"].tolist(), "validation")
    log.info("VALIDATION accuracy=%.4f macro_f1=%.4f", s["accuracy"], s["macro_f1"])
    if rec["test"] is not None:
        X_te, y_te, te_kept, _ = pp.load_arrays(rec["test"], a.img_size)
        if len(X_te):
            t, _, _ = evaluate_arrays(best, X_te, y_te, te_kept["path"].tolist(), "official_test")
            log.info("OFFICIAL TEST accuracy=%.4f macro_f1=%.4f", t["accuracy"], t["macro_f1"])
    write_status(state="completed", message="Training finished and model evaluated.")


def main(argv=None):
    a = parse_args(argv)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(config.TRAIN_LOG_PATH, mode="w", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
    write_status(state="running", pid=os.getpid(), started_at=datetime.now().isoformat(timespec="seconds"),
                 epoch=0, epochs=a.epochs, history={}, message="Starting...")
    try:
        run(a)
    except Exception as e:  # report honestly, never claim completion
        log.exception("Training failed")
        write_status(state="failed", message=f"{type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
