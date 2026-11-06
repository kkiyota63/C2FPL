"""Data adapter for a per-video directory layout (toyota2026 style).

Expected layout (all paths configurable from the CLI of run_cv.py)::

    <feat_dir>/<video>.npy      # I3D features, one file per video
    <label_dir>/<video>.npy     # exists ONLY for anomalous videos
    <split_dir>/splits_*.txt    # one video stem per line; K files == K folds

A video is labelled anomalous iff a file with the same basename exists under
``<label_dir>``; every other feature file is treated as normal.

IMPORTANT: these video level labels are used for EVALUATION and for diagnostics
only.  C2FPL is fully unsupervised -- neither the coarse stage, the fine stage
nor the BCE training of the detector ever sees them.
"""

import os
import pathlib
import random
import re

import numpy as np


# --------------------------------------------------------------------------- #
# small helpers (equivalents of read_txt / natural_keys)
# --------------------------------------------------------------------------- #
def natural_keys(text):
    """Sort key so that splits_2.txt comes before splits_10.txt."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(text))]


def read_txt(path):
    """One non-empty stripped line per entry."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def discover_dataset(feat_dir, label_dir, split_dir):
    """Return (file_to_label, splits, feat_dir).

    file_to_label : {"<video>.npy": 1 (anomaly) | 0 (normal)}
    splits        : list of lists of "<video>.npy", one list per splits_*.txt,
                    filtered down to the files that actually have features.
    """
    feat_dir = pathlib.Path(feat_dir)
    label_dir = pathlib.Path(label_dir)
    split_dir = pathlib.Path(split_dir)

    feature_files = sorted(
        os.path.basename(str(p)) for p in feat_dir.glob("*.npy") if p.is_file()
    )
    if not feature_files:
        raise FileNotFoundError(f"no *.npy features found under {feat_dir}")

    label_files = {
        os.path.basename(str(p)) for p in label_dir.glob("*.npy") if p.is_file()
    }

    file_to_label = {name: (1 if name in label_files else 0) for name in feature_files}

    available = set(feature_files)
    splits = []
    split_paths = sorted(
        (p for p in split_dir.glob("splits_*.txt") if p.is_file()), key=natural_keys
    )
    if not split_paths:
        raise FileNotFoundError(f"no splits_*.txt found under {split_dir}")

    for path in split_paths:
        names = [f"{stem}.npy" for stem in read_txt(path)]
        kept = [n for n in names if n in available]
        missing = len(names) - len(kept)
        if missing:
            print(f"[warn] {path.name}: {missing}/{len(names)} entries have no feature file")
        splits.append(kept)

    return file_to_label, splits, feat_dir


def limit_data(files, file_to_label, max_normal, max_anomaly, seed):
    """Cap the number of normal / anomalous videos. None or <0 means no cap."""
    rng = random.Random(seed)
    normal = sorted(f for f in files if file_to_label[f] == 0)
    anomaly = sorted(f for f in files if file_to_label[f] == 1)

    def take(pool, cap):
        if cap is None or cap < 0 or cap >= len(pool):
            return pool
        return rng.sample(pool, cap)

    picked = take(normal, max_normal) + take(anomaly, max_anomaly)
    rng.shuffle(picked)
    return picked


# --------------------------------------------------------------------------- #
# feature loading
# --------------------------------------------------------------------------- #
def load_feature(path, crop_axis="auto", max_crops=10):
    """Load one video's features and normalise them to ``(T, C, D)`` float32.

    ``T`` = number of 16-frame segments, ``C`` = number of crops (1 when the
    features are not multi-crop), ``D`` = feature dimension.

    Accepted on disk:
      (D,)        -> (1, 1, D)
      (T, D)      -> (T, 1, D)
      (T, C, D)   -> unchanged
      (C, T, D)   -> transposed to (T, C, D)

    For 3-D arrays the crop axis is guessed as the one whose length is <=
    ``max_crops`` while the other is larger; pass crop_axis=0 or 1 to force it.
    """
    arr = np.asarray(np.load(path), dtype=np.float32)

    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 1/2/3-D array, got shape {arr.shape}")

    if crop_axis == "auto":
        a, b = arr.shape[0], arr.shape[1]
        axis = 1 if (b <= max_crops and b <= a) else (0 if a <= max_crops else 1)
    else:
        axis = int(crop_axis)

    return arr if axis == 1 else arr.transpose(1, 0, 2)


def load_fold(files, feat_dir, crop_axis="auto"):
    """Load every video of a fold. Returns (list_of_(T,C,D)_arrays, kept_files)."""
    feats, kept = [], []
    for name in files:
        path = pathlib.Path(feat_dir) / name
        try:
            feats.append(load_feature(path, crop_axis=crop_axis))
            kept.append(name)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"[warn] skipping {name}: {exc}")
    if not feats:
        raise RuntimeError("no feature file of this fold could be loaded")

    dims = {f.shape[2] for f in feats}
    if len(dims) != 1:
        raise ValueError(f"inconsistent feature dimensions across videos: {sorted(dims)}")
    return feats, kept


def feature_dim(feats):
    return int(feats[0].shape[2])
