"""Dry run: can this dataset be loaded, and is it shaped the way C2FPL needs?

Touches no GPU and trains nothing.  Run this first on the machine with the data:

    uv run python check_data.py \
        --feat-dir /home/jovyan/work/toyota2026/tracking_src/output_i3d_feature \
        --label-dir /home/jovyan/work/toyota2026/data/label \
        --split-dir /home/jovyan/work/toyota2026/data/splits

It reports, and fails loudly on, everything run_cv.py depends on: feature
discovery, split name matching, the feature axis order, dimension consistency,
segment counts, the fold sizes a K-fold run would produce, and how the label
files line up with the feature time axis.
"""

import argparse
import difflib
import pathlib

import numpy as np

import data_adapter as da


def build_parser():
    p = argparse.ArgumentParser(description="validate the dataset layout for C2FPL")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--feature-layout", default="auto", choices=["auto", "td", "dt"],
                   help="axis order on disk: td=(T,D), dt=(D,T); auto infers it")
    p.add_argument("--crop-axis", default="auto", choices=["auto", "0", "1"])
    p.add_argument("--probe", type=int, default=8,
                   help="how many feature files to load and print individually")
    p.add_argument("--shape-scan", type=int, default=200,
                   help="how many feature headers to read for the shape summary")
    p.add_argument("--full-scan", action="store_true",
                   help="read every feature header (slow) instead of --shape-scan")
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

    ds = da.discover_dataset(args.feat_dir, args.label_dir, args.split_dir)
    file_to_label = ds["file_to_label"]
    splits, split_names = ds["splits"], ds["split_names"]
    feat_dir, stems = ds["feat_dir"], ds["feature_stems"]

    section("discovery")
    n_anomaly = sum(file_to_label.values())
    n_normal = len(file_to_label) - n_anomaly
    print(f"feature files : {len(file_to_label)}")
    print(f"  anomaly     : {n_anomaly}  (a matching file exists under label-dir)")
    print(f"  normal      : {n_normal}")
    if n_anomaly == 0:
        problems.append("no anomalous videos found -- label basenames do not match "
                        "feature basenames")
    if n_normal == 0:
        problems.append("no normal videos found -- the coarse stage needs a normal bag")
    orphans = sorted(ds["label_stems"] - set(stems))
    if orphans:
        warnings.append(f"{len(orphans)} label file(s) have no feature file, "
                        f"e.g. {orphans[:3]}")

    section("split name matching")
    covered, overlap = set(), set()
    for name, fold in zip(split_names, splits):
        labels = [file_to_label[f] for f in fold]
        print(f"{name:16s} {len(fold):5d} matched  "
              f"({sum(labels)} anomaly / {len(labels) - sum(labels)} normal)")
        dup = covered & set(fold)
        overlap |= dup
        covered |= set(fold)
        if not fold:
            problems.append(f"{name}: no entry matched a feature file")

    if ds["unmatched"]:
        total_missed = sum(len(v) for v in ds["unmatched"].values())
        print(f"\n{total_missed} entr(y/ies) across "
              f"{len(ds['unmatched'])} file(s) matched nothing. Examples, with the "
              f"closest feature stem for comparison:")
        shown = 0
        for name, missed in ds["unmatched"].items():
            for entry in missed[:3]:
                near = difflib.get_close_matches(da.normalize_name(entry), stems, n=1,
                                                 cutoff=0.0)
                print(f"  {name}")
                print(f"    split entry   : {entry!r}")
                print(f"    normalised to : {da.normalize_name(entry)!r}")
                print(f"    closest feature stem: {near[0]!r}" if near else "    (no features)")
                shown += 1
                if shown >= 4:
                    break
            if shown >= 4:
                break
        problems.append(f"{total_missed} split entries do not match any feature file; "
                        f"compare the spellings printed above")

    if overlap:
        warnings.append(f"{len(overlap)} video(s) appear in more than one split, "
                        f"e.g. {sorted(overlap)[:3]}")
    missing = set(file_to_label) - covered
    if missing:
        warnings.append(f"{len(missing)} feature file(s) are in no split and will be "
                        f"ignored, e.g. {sorted(missing)[:3]}")
    print(f"\ncoverage: {len(covered)}/{len(file_to_label)} feature files appear in a split")
    if len(splits) < 3:
        problems.append(f"only {len(splits)} split(s); a fold needs test + cal + train, "
                        f"so at least 3 are required")

    section("feature axis order")
    layout, evidence = (args.feature_layout, "forced from the command line")
    if args.feature_layout == "auto":
        layout, evidence = da.infer_layout(feat_dir, stems)
    print(f"layout : {layout}  ({'(T, D)' if layout == 'td' else '(D, T) -- transposed'})")
    print(f"reason : {evidence}")

    section("feature arrays")
    scan = stems if args.full_scan else stems[:max(args.shape_scan, 1)]
    raw_shapes, dims, crops, segs, dtypes = [], {}, {}, [], set()
    for i, stem in enumerate(scan):
        path = pathlib.Path(feat_dir) / f"{stem}.npy"
        raw = np.load(path, mmap_mode="r")
        raw_shapes.append(tuple(raw.shape))
        dtypes.add(str(raw.dtype))
        try:
            feat = da.load_feature(path, layout=layout, crop_axis=args.crop_axis)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{stem}: {exc}")
            continue
        dims[feat.shape[2]] = dims.get(feat.shape[2], 0) + 1
        crops[feat.shape[1]] = crops.get(feat.shape[1], 0) + 1
        segs.append(feat.shape[0])
        if i < args.probe:
            print(f"{stem}.npy")
            print(f"    on disk {tuple(raw.shape)} {raw.dtype}"
                  f"  ->  (T={feat.shape[0]}, C={feat.shape[1]}, D={feat.shape[2]})"
                  f"  finite={bool(np.isfinite(feat).all())}")

    print(f"\nread {len(scan)} header(s)"
          f"{' (full scan)' if args.full_scan else f' of {len(stems)}; --full-scan for all'}")
    print(f"dtypes on disk     : {sorted(dtypes)}")
    print(f"feature dim D      : {dims if len(dims) <= 4 else f'{len(dims)} distinct values!'}")
    print(f"crop count C       : {crops}")
    if segs:
        sc = np.array(segs)
        print(f"segments T         : min {sc.min()}, median {int(np.median(sc))}, "
              f"max {sc.max()}, total {int(sc.sum())}")
        if sc.min() < 5:
            warnings.append(f"some videos have very few segments (min {sc.min()}); the "
                            f"fine-stage window is int(0.2*T) and clamps to 1 there")
    if len(dims) > 1:
        problems.append(f"inconsistent feature dimension across videos "
                        f"({len(dims)} distinct values) -- the axis order is probably "
                        f"wrong; try --feature-layout "
                        f"{'dt' if layout == 'td' else 'td'}")
    if len(crops) > 1:
        warnings.append(f"inconsistent crop count across videos: {crops}")

    section("label files vs the feature time axis")
    anomaly_stems = [s for s in stems if file_to_label[s] == 1]
    if anomaly_stems:
        ratios = []
        print(f"{'video':<52} {'len(label)':>10} {'T':>7} {'len/T':>7} {'diff':>6}")
        for stem in anomaly_stems[:12]:
            arr = np.asarray(np.load(pathlib.Path(ds["label_dir"]) / f"{stem}.npy",
                                     allow_pickle=True))
            try:
                feat = da.load_feature(pathlib.Path(feat_dir) / f"{stem}.npy",
                                       layout=layout, crop_axis=args.crop_axis)
            except Exception:  # noqa: BLE001
                continue
            t = feat.shape[0]
            n = int(arr.shape[0]) if arr.ndim >= 1 else 1
            ratios.append(n / t)
            print(f"{stem[:52]:<52} {n:>10} {t:>7} {n / t:>7.3f} {n - t:>6}")
        if ratios:
            r = np.array(ratios)
            print(f"\nlen(label)/T over {len(r)} file(s): "
                  f"min {r.min():.3f}, median {np.median(r):.3f}, max {r.max():.3f}")
            if abs(np.median(r) - 16.0) < 1.0:
                print("=> frame level labels (16 frames per segment)")
            elif abs(np.median(r) - 1.0) < 0.15:
                print("=> segment level labels, roughly one per feature segment")
            else:
                warnings.append(f"len(label)/T has median {np.median(r):.3f}, which is "
                                f"neither ~1 (segment level) nor ~16 (frame level); the "
                                f"label time base needs to be resolved before frame "
                                f"level evaluation can be wired up")
        sample = np.asarray(np.load(pathlib.Path(ds["label_dir"]) /
                                    f"{anomaly_stems[0]}.npy", allow_pickle=True))
        print(f"\nexample: {anomaly_stems[0]}.npy  shape {tuple(sample.shape)} "
              f"dtype {sample.dtype}  unique {np.unique(sample)[:6]}  "
              f"positive {float(np.mean(sample != 0)):.1%}")
    else:
        print("no anomalous videos to inspect")

    section("what a K-fold run would look like")
    k = len(splits)
    for fold_idx in range(min(k, 3)):
        cal_idx = (fold_idx + 1) % k
        train = [p for j, f in enumerate(splits) if j not in (fold_idx, cal_idx) for p in f]
        tl = [file_to_label[p] for p in train]
        print(f"fold {fold_idx}: train {len(train)} "
              f"({sum(tl)} anomaly / {len(tl) - sum(tl)} normal)"
              f"  cal {len(splits[cal_idx])}  test {len(splits[fold_idx])}")
        if len(train) < 4:
            problems.append(f"fold {fold_idx}: only {len(train)} training videos")
        y_test = [file_to_label[p] for p in splits[fold_idx]]
        if y_test and len(set(y_test)) < 2:
            warnings.append(f"fold {fold_idx}: the test fold is single-class, so ROC-AUC "
                            f"and PR-AUC are undefined for it")
    if k > 3:
        print(f"... ({k} folds in total)")

    section("verdict")
    for w in warnings:
        print(f"[warn] {w}")
    if problems:
        print("\nFAIL -- these must be fixed before run_cv.py can work:")
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
        return 1
    print("OK -- the layout is loadable; run_cv.py should be able to consume it."
          + (" See the warnings above." if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
