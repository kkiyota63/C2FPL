"""Dry run: can this dataset be loaded, and is it shaped the way C2FPL needs?

Touches no GPU and trains nothing.  Run this first on the machine that holds
the data:

    uv run python check_data.py \
        --feat-dir /home/jovyan/work/toyota2026/tracking_src/output_i3d_feature \
        --label-dir /home/jovyan/work/toyota2026/data/label \
        --split-dir /home/jovyan/work/toyota2026/data/splits

It reports, and fails loudly on, everything run_cv.py depends on:
  - feature files found, and which of them count as anomalous
  - splits_*.txt parsed, coverage of / overlap between folds
  - feature array shapes, the inferred (T, C, D) interpretation and dtype
  - whether the feature dimension is consistent across videos
  - segment counts, and the fold sizes a K-fold run would produce
  - what the label .npy files look like (so frame level evaluation can be wired up)
"""

import argparse
import pathlib

import numpy as np

import data_adapter as da


def build_parser():
    p = argparse.ArgumentParser(description="validate the dataset layout for C2FPL")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--crop-axis", default="auto", choices=["auto", "0", "1"])
    p.add_argument("--probe", type=int, default=8,
                   help="how many feature files to actually load and inspect")
    p.add_argument("--full-scan", action="store_true",
                   help="load every feature file (slow) to verify all shapes")
    return p


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))


def main():
    args = build_parser().parse_args()
    problems, warnings = [], []

    section("directories")
    for name, path in [("feat", args.feat_dir), ("label", args.label_dir),
                       ("split", args.split_dir)]:
        p = pathlib.Path(path)
        print(f"{name:6s} {path}  exists={p.is_dir()}")
        if not p.is_dir():
            problems.append(f"{name}-dir does not exist: {path}")
    if problems:
        print("\nFAIL:\n  " + "\n  ".join(problems))
        return 1

    section("discovery")
    file_to_label, splits, feat_dir = da.discover_dataset(
        args.feat_dir, args.label_dir, args.split_dir
    )
    n_anomaly = sum(file_to_label.values())
    n_normal = len(file_to_label) - n_anomaly
    print(f"feature files : {len(file_to_label)}")
    print(f"  anomaly     : {n_anomaly}  (a matching file exists under label-dir)")
    print(f"  normal      : {n_normal}")
    if n_anomaly == 0:
        problems.append("no anomalous videos found -- label-dir basenames do not match "
                        "feature basenames, so every video looks normal")
    if n_normal == 0:
        problems.append("no normal videos found -- the coarse stage needs a normal bag")

    label_files = sorted(p.name for p in pathlib.Path(args.label_dir).glob("*.npy"))
    orphans = [n for n in label_files if n not in file_to_label]
    if orphans:
        warnings.append(f"{len(orphans)} label file(s) have no feature file, "
                        f"e.g. {orphans[:3]}")

    section("splits")
    covered, overlap = set(), set()
    for i, fold in enumerate(splits):
        labels = [file_to_label[f] for f in fold]
        print(f"splits_{i}: {len(fold):5d} videos  "
              f"({sum(labels)} anomaly / {len(labels) - sum(labels)} normal)")
        dup = covered & set(fold)
        overlap |= dup
        covered |= set(fold)
        if not fold:
            problems.append(f"splits_{i} is empty after filtering to existing features")
    if overlap:
        warnings.append(f"{len(overlap)} video(s) appear in more than one split, "
                        f"e.g. {sorted(overlap)[:3]}")
    missing = set(file_to_label) - covered
    if missing:
        warnings.append(f"{len(missing)} feature file(s) are in no split and will be "
                        f"ignored, e.g. {sorted(missing)[:3]}")
    print(f"coverage: {len(covered)}/{len(file_to_label)} feature files appear in a split")

    if len(splits) < 3:
        problems.append(f"only {len(splits)} split(s); a fold needs test + cal + train, "
                        f"so at least 3 are required")

    section("feature arrays")
    names = sorted(file_to_label)
    probe = names if args.full_scan else names[:max(args.probe, 1)]
    dims, crops, segment_counts, dtypes = {}, {}, [], set()
    for name in probe:
        path = pathlib.Path(feat_dir) / name
        raw = np.load(path, mmap_mode="r")
        dtypes.add(str(raw.dtype))
        try:
            feat = da.load_feature(path, crop_axis=args.crop_axis)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: {exc}")
            continue
        dims[feat.shape[2]] = dims.get(feat.shape[2], 0) + 1
        crops[feat.shape[1]] = crops.get(feat.shape[1], 0) + 1
        segment_counts.append(feat.shape[0])
        if len(segment_counts) <= 8:
            print(f"{name}")
            print(f"    on disk {tuple(raw.shape)} {raw.dtype}"
                  f"  ->  (T={feat.shape[0]}, C={feat.shape[1]}, D={feat.shape[2]})"
                  f"  finite={bool(np.isfinite(feat).all())}")

    print(f"\nprobed {len(probe)} file(s)"
          f"{' (full scan)' if args.full_scan else f' of {len(names)}; use --full-scan for all'}")
    print(f"dtypes on disk     : {sorted(dtypes)}")
    print(f"feature dim D      : {dims}")
    print(f"crop count C       : {crops}")
    if segment_counts:
        sc = np.array(segment_counts)
        print(f"segments T         : min {sc.min()}, median {int(np.median(sc))}, max {sc.max()}")
        if sc.min() < 5:
            warnings.append(f"some videos have very few segments (min {sc.min()}); the "
                            f"fine-stage window is int(0.2*T) and clamps to 1 there")
    if len(dims) > 1:
        problems.append(f"inconsistent feature dimension across videos: {dims}")
    if len(crops) > 1:
        warnings.append(f"inconsistent crop count across videos: {crops}; pass "
                        f"--crop-axis explicitly if the auto guess is wrong")

    section("label file format (for optional frame level evaluation)")
    if label_files:
        for name in label_files[:3]:
            arr = np.load(pathlib.Path(args.label_dir) / name, allow_pickle=True)
            arr = np.asarray(arr)
            print(f"{name}: shape {tuple(arr.shape)} dtype {arr.dtype} "
                  f"unique {np.unique(arr)[:6]}")
            if arr.ndim == 1 and name in file_to_label:
                feat = da.load_feature(pathlib.Path(feat_dir) / name, crop_axis=args.crop_axis)
                t = feat.shape[0]
                print(f"    feature T={t}  ->  len/T={len(arr) / t:.2f} "
                      f"(16.0 suggests frame level, 1.0 suggests segment level)")
    else:
        print("no label .npy files to inspect")

    section("what a K-fold run would look like")
    k = len(splits)
    for fold_idx in range(min(k, 3)):
        cal_idx = (fold_idx + 1) % k
        test_n = len(splits[fold_idx])
        cal_n = len(splits[cal_idx])
        train = [p for j, f in enumerate(splits) if j not in (fold_idx, cal_idx) for p in f]
        train_labels = [file_to_label[p] for p in train]
        print(f"fold {fold_idx}: train {len(train)} "
              f"({sum(train_labels)} anomaly / {len(train_labels) - sum(train_labels)} normal)"
              f"  cal {cal_n}  test {test_n}")
        if len(train) < 4:
            problems.append(f"fold {fold_idx}: only {len(train)} training videos")
        if sum(train_labels) == 0 or sum(train_labels) == len(train_labels):
            warnings.append(f"fold {fold_idx}: the training fold is single-class; C2FPL "
                            f"will still run (it is unsupervised) but the coarse split "
                            f"diagnostics will be meaningless")
        y_test = [file_to_label[p] for p in splits[fold_idx]]
        if len(set(y_test)) < 2:
            warnings.append(f"fold {fold_idx}: the test fold is single-class, so ROC-AUC "
                            f"and PR-AUC are undefined for it")
    if k > 3:
        print(f"... ({k} folds in total)")

    section("verdict")
    for w in warnings:
        print(f"[warn] {w}")
    if problems:
        print("\nFAIL -- these must be fixed before run_cv.py can work:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK -- the layout is loadable; run_cv.py should be able to consume it."
          + (" See the warnings above." if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
