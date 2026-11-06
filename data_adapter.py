"""Data adapter for a per-video directory layout (toyota2026 style).

Expected layout (all paths configurable from the CLI)::

    <feat_dir>/<video>.npy      # I3D features, one file per video
    <label_dir>/<video>.npy     # exists ONLY for anomalous videos
    <split_dir>/splits_*.txt    # one video per line; K files == K folds

A video is anomalous iff a file with the same name exists under ``<label_dir>``;
every other feature file is treated as normal.

IMPORTANT: these video level labels are used for EVALUATION and diagnostics
only.  C2FPL is fully unsupervised -- neither the coarse stage, the fine stage
nor the BCE training of the detector ever sees them.

Two things vary between feature extractors and are handled here:

* **axis order.**  A 2-D feature file may be stored as ``(T, D)`` or ``(D, T)``.
  The layout is inferred by looking at several files at once: the axis whose
  length stays constant across videos is the feature dimension, the one that
  varies is time.  Pass ``layout="td"`` / ``"dt"`` to force it.
* **name spelling.**  Entries in ``splits_*.txt`` may carry a ``.npy``
  extension, a directory prefix, or neither.  Names are matched on their
  normalised stem, so all three spellings resolve to the same feature file.
"""

import os
import pathlib
import random
import re
from collections import Counter

import numpy as np

# feature widths that are almost certainly a channel count, not a time axis
COMMON_FEATURE_DIMS = {256, 512, 768, 1024, 1280, 2048, 4096}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def natural_keys(text):
    """Sort key so that splits_2.txt comes before splits_10.txt."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(text))]


def read_txt(path):
    """One non-empty stripped line per entry."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def normalize_name(entry):
    """'dir/x.npy' -> 'x', 'x.npy' -> 'x', 'x' -> 'x'.

    Only a trailing '.npy' is stripped, so a video called 'a.b' keeps its dot.
    """
    base = os.path.basename(str(entry).strip().replace("\\", "/"))
    if base.endswith(".npy"):
        base = base[:-4]
    return base


# --------------------------------------------------------------------------- #
# layout inference
# --------------------------------------------------------------------------- #
def infer_layout(feat_dir, names, probe=24):
    """Decide whether 2-D/3-D feature files are time-first or dim-first.

    Returns ("td" | "dt", evidence_string).
    """
    feat_dir = pathlib.Path(feat_dir)
    shapes = []
    for name in list(names)[:probe]:
        path = feat_dir / f"{name}.npy"
        if path.is_file():
            shapes.append(np.load(path, mmap_mode="r").shape)
    if not shapes:
        return "td", "no readable feature file; defaulting to (T, D)"

    ndims = Counter(len(s) for s in shapes)
    if len(ndims) > 1:
        return "td", f"mixed array ranks {dict(ndims)}; defaulting to (T, D)"

    first = Counter(s[0] for s in shapes)
    last = Counter(s[-1] for s in shapes)

    # the feature dimension is the axis that does not change between videos
    first_constant = len(first) == 1
    last_constant = len(last) == 1
    if first_constant and not last_constant:
        return "dt", (f"axis 0 is constant at {shapes[0][0]} across {len(shapes)} files "
                      f"while the last axis varies -> dim-first")
    if last_constant and not first_constant:
        return "td", (f"the last axis is constant at {shapes[0][-1]} across {len(shapes)} "
                      f"files while axis 0 varies -> time-first")

    # both constant (or both varying): fall back to "which looks like a channel count"
    if shapes[0][0] in COMMON_FEATURE_DIMS and shapes[0][-1] not in COMMON_FEATURE_DIMS:
        return "dt", f"axis 0 ({shapes[0][0]}) is a common feature width -> dim-first"
    if shapes[0][-1] in COMMON_FEATURE_DIMS and shapes[0][0] not in COMMON_FEATURE_DIMS:
        return "td", f"the last axis ({shapes[0][-1]}) is a common feature width -> time-first"
    return "td", f"inconclusive (shapes like {shapes[0]}); defaulting to (T, D)"


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def discover_dataset(feat_dir, label_dir, split_dir):
    """Return a dict describing the dataset.

    Keys: file_to_label, splits, split_names, feat_dir, unmatched, feature_stems
    """
    feat_dir = pathlib.Path(feat_dir)
    label_dir = pathlib.Path(label_dir)
    split_dir = pathlib.Path(split_dir)

    feature_files = sorted(
        os.path.basename(str(p)) for p in feat_dir.glob("*.npy") if p.is_file()
    )
    if not feature_files:
        raise FileNotFoundError(f"no *.npy features found under {feat_dir}")

    # stem -> stem, so every lookup goes through the same normalisation
    stems = {normalize_name(n) for n in feature_files}

    label_stems = {
        normalize_name(p.name) for p in label_dir.glob("*.npy") if p.is_file()
    }
    file_to_label = {s: (1 if s in label_stems else 0) for s in sorted(stems)}

    splits, split_names, unmatched = [], [], {}
    split_paths = sorted(
        (p for p in split_dir.glob("splits_*.txt") if p.is_file()), key=natural_keys
    )
    if not split_paths:
        raise FileNotFoundError(f"no splits_*.txt found under {split_dir}")

    for path in split_paths:
        raw = read_txt(path)
        kept, missed = [], []
        for entry in raw:
            stem = normalize_name(entry)
            (kept if stem in stems else missed).append(stem if stem in stems else entry)
        splits.append(kept)
        split_names.append(path.name)
        if missed:
            unmatched[path.name] = missed

    return {
        "file_to_label": file_to_label,
        "splits": splits,
        "split_names": split_names,
        "feat_dir": feat_dir,
        "label_dir": label_dir,
        "unmatched": unmatched,
        "feature_stems": sorted(stems),
        "label_stems": label_stems,
    }


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
def load_feature(path, layout="td", crop_axis="auto", max_crops=10):
    """Load one video's features, normalised to ``(T, C, D)`` float32.

    ``T`` = segments, ``C`` = crops (1 when not multi-crop), ``D`` = feature dim.

    layout "td": on disk (T, D) or (T, C, D)
    layout "dt": on disk (D, T) or (D, C, T) -- the time axis is last
    """
    arr = np.asarray(np.load(path), dtype=np.float32)

    if arr.ndim == 1:
        return arr[None, None, :]

    if arr.ndim == 2:
        if layout == "dt":
            arr = arr.T  # (D, T) -> (T, D)
        return arr[:, None, :]

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 1/2/3-D array, got shape {arr.shape}")

    if layout == "dt":
        arr = np.moveaxis(arr, -1, 0)  # (D, C, T) -> (T, D, C)
        arr = np.moveaxis(arr, -1, 1)  # -> (T, C, D)
        return arr

    if crop_axis == "auto":
        a, b = arr.shape[0], arr.shape[1]
        axis = 1 if (b <= max_crops and b <= a) else (0 if a <= max_crops else 1)
    else:
        axis = int(crop_axis)
    return arr if axis == 1 else arr.transpose(1, 0, 2)


def load_fold(stems, feat_dir, layout="td", crop_axis="auto"):
    """Load every video of a fold. Returns (list_of_(T,C,D)_arrays, kept_stems)."""
    feats, kept = [], []
    for stem in stems:
        path = pathlib.Path(feat_dir) / f"{stem}.npy"
        try:
            feats.append(load_feature(path, layout=layout, crop_axis=crop_axis))
            kept.append(stem)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"[warn] skipping {stem}: {exc}")
    if not feats:
        raise RuntimeError("no feature file of this fold could be loaded")

    dims = {f.shape[2] for f in feats}
    if len(dims) != 1:
        raise ValueError(
            f"inconsistent feature dimension across videos: {sorted(dims)}. "
            f"The axis order was probably misread -- pass --feature-layout explicitly."
        )
    return feats, kept


def feature_dim(feats):
    return int(feats[0].shape[2])
