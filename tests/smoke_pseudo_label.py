"""Synthetic smoke test for the C2FPL pseudo label stage (numpy/sklearn only).

Builds fake per-video I3D features where "anomalous" videos contain a short
high-energy burst, then checks that the coarse stage separates the two
populations and that the fine stage localises the burst.

Run:  python tests/smoke_pseudo_label.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pseudo_label import generate_pseudo_labels  # noqa: E402


def make_dataset(n_normal=40, n_anomaly=20, dim=1024, crops=10, seed=0):
    rng = np.random.default_rng(seed)
    feats, truth, burst_spans = [], [], []

    for i in range(n_normal + n_anomaly):
        t = int(rng.integers(40, 120))
        feat = rng.normal(0.0, 1.0, size=(t, crops, dim)).astype(np.float32)
        is_anomaly = i >= n_normal
        span = None
        if is_anomaly:
            width = max(int(t * 0.2), 2)
            start = int(rng.integers(0, t - width + 1))
            feat[start:start + width] *= 3.0  # high-energy burst
            span = (start, start + width)
        feats.append(feat)
        truth.append(1 if is_anomaly else 0)
        burst_spans.append(span)

    order = rng.permutation(len(feats))
    return ([feats[i] for i in order],
            np.array([truth[i] for i in order]),
            [burst_spans[i] for i in order])


def main():
    feats, truth, spans = make_dataset()
    print(f"synthetic set: {len(feats)} videos, {int(truth.sum())} anomalous, "
          f"feature shape e.g. {feats[0].shape}")

    labels, info = generate_pseudo_labels(feats, nu=1.0, pct=3.0, length=0.2, seed=0)

    normal_idx, abnormal_idx = info["normal_idx"], info["abnormal_idx"]
    abnormal_precision = truth[abnormal_idx].mean() if abnormal_idx.size else 0.0
    normal_precision = 1.0 - truth[normal_idx].mean() if normal_idx.size else 0.0
    print(f"coarse: abnormal bag precision {abnormal_precision:.3f}, "
          f"normal bag precision {normal_precision:.3f}")

    # fine stage: overlap between the predicted window and the injected burst
    overlaps = []
    for i in abnormal_idx:
        if spans[i] is None:
            continue
        start, end = spans[i]
        truth_mask = np.zeros(len(labels[i]), dtype=np.float32)
        truth_mask[start:end] = 1.0
        inter = float((labels[i] * truth_mask).sum())
        union = float(((labels[i] + truth_mask) > 0).sum())
        overlaps.append(inter / max(union, 1.0))
    mean_iou = float(np.mean(overlaps)) if overlaps else 0.0
    print(f"fine: mean IoU with the injected burst over {len(overlaps)} true "
          f"anomalies: {mean_iou:.3f}")

    # shape contract: one label per segment, values in {0, 1}
    assert len(labels) == len(feats)
    for feat, label in zip(feats, labels):
        assert label.shape == (feat.shape[0],), (label.shape, feat.shape)
        assert set(np.unique(label)).issubset({0.0, 1.0})
    for i in normal_idx:
        assert labels[i].sum() == 0.0, "normal videos must be all-zero"

    failures = []
    if abnormal_precision < 0.8:
        failures.append(f"abnormal bag precision {abnormal_precision:.3f} < 0.8")
    if normal_precision < 0.8:
        failures.append(f"normal bag precision {normal_precision:.3f} < 0.8")
    if mean_iou < 0.5:
        failures.append(f"mean IoU {mean_iou:.3f} < 0.5")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
