"""K-fold cross validation of C2FPL on a per-video feature directory.

Per fold:
  1. train / cal / test folds are formed from splits_*.txt (cal = next fold)
  2. the C2FPL coarse-to-fine stage builds SEGMENT level pseudo labels from the
     training features alone -- no ground truth is involved
  3. Model_V2 is trained with BCE on those pseudo labels
  4. each test video gets one score (aggregated over its segments); ROC-AUC and
     PR-AUC come from the continuous scores, accuracy and F1 from a threshold
     calibrated on the cal fold

Video level ground truth is therefore used ONLY for the cal-fold threshold and
for the final metrics.

Example:
    python run_cv.py \
        --feat-dir /home/jovyan/work/toyota2026/tracking_src/output_i3d_feature \
        --label-dir /home/jovyan/work/toyota2026/data/label \
        --split-dir /home/jovyan/work/toyota2026/data/splits \
        --max-epoch 20 --batch-size 128 --lr 0.01
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

import data_adapter as da
from pseudo_label import generate_pseudo_labels


# --------------------------------------------------------------------------- #
# model -- architecturally identical to model.py:Model_V2, minus the unused
# fc_att1 / fc_att2 branches and without the global cuda tensor type that
# model.py sets at import time.
# --------------------------------------------------------------------------- #
def weight_init(m):
    name = m.__class__.__name__
    if name.find("Conv") != -1 or name.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model_V2(nn.Module):
    def __init__(self, n_features, dropout=0.6):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        x = self.dropout(self.relu(self.fc1(inputs)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x.mean(dim=1)  # average over the crop axis -> (B, 1)


# --------------------------------------------------------------------------- #
# segment level dataset over a list of (T, C, D) arrays
# --------------------------------------------------------------------------- #
class SegmentDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels
        self.index = [
            (v, s) for v, f in enumerate(feats) for s in range(f.shape[0])
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        v, s = self.index[i]
        return self.feats[v][s], np.float32(self.labels[v][s])


# --------------------------------------------------------------------------- #
# train / score
# --------------------------------------------------------------------------- #
def train_detector(feats, labels, feature_size, args, device):
    model = Model_V2(feature_size, dropout=args.dropout).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )
    loss_fn = nn.BCELoss()

    loader = DataLoader(
        SegmentDataset(feats, labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )

    model.train()
    for epoch in range(1, args.max_epoch + 1):
        epoch_losses = []
        for batch_feat, batch_label in loader:
            batch_feat = batch_feat.to(device, non_blocking=True).float()
            batch_label = batch_label.to(device, non_blocking=True).float()

            scores = model(batch_feat).flatten()
            loss = loss_fn(scores, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        if args.verbose_train:
            print(f"    epoch {epoch:3d}/{args.max_epoch}  "
                  f"lr {optimizer.param_groups[0]['lr']:.5f}  "
                  f"loss {np.mean(epoch_losses):.4f}")
    return model


@torch.no_grad()
def video_scores(model, feats, args, device):
    """One score per video, aggregated from its segment scores."""
    model.eval()
    out = []
    for feat in feats:
        tensor = torch.from_numpy(feat).to(device).float()
        segment = []
        for start in range(0, tensor.shape[0], args.eval_batch_size):
            chunk = tensor[start:start + args.eval_batch_size]
            segment.append(model(chunk).flatten().cpu().numpy())
        segment = np.concatenate(segment)

        if args.video_agg == "max":
            out.append(float(segment.max()))
        elif args.video_agg == "mean":
            out.append(float(segment.mean()))
        else:  # topk
            k = max(1, int(round(len(segment) * args.topk_ratio)))
            out.append(float(np.sort(segment)[-k:].mean()))
    return np.array(out, dtype=np.float64)


def calibrate_threshold(scores, labels):
    """Threshold maximising F1 on the calibration fold."""
    if len(np.unique(labels)) < 2:
        print("  [warn] calibration fold is single-class; falling back to 0.5")
        return 0.5
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = f1_score(labels, (scores >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    print(f"  [cal] threshold {best_threshold:.4f} (F1 {best_f1:.4f} on cal fold)")
    return best_threshold


def evaluate(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(labels, pred)),
        "F1": float(f1_score(labels, pred, zero_division=0)),
        "ROC_AUC": float("nan"),
        "PR_AUC": float("nan"),
    }
    if len(np.unique(labels)) >= 2:
        result["ROC_AUC"] = float(roc_auc_score(labels, scores))
        precision, recall, _ = precision_recall_curve(labels, scores)
        result["PR_AUC"] = float(auc(recall, precision))
    else:
        print("  [warn] test fold is single-class; AUC metrics are undefined")
    return result


def coarse_diagnostics(files, info, file_to_label):
    """Agreement of the unsupervised coarse split with the real video labels."""
    truth = np.array([file_to_label[f] for f in files])
    normal_idx, abnormal_idx = info["normal_idx"], info["abnormal_idx"]
    diag = {}
    if abnormal_idx.size:
        diag["abnormal_bag_precision"] = float(truth[abnormal_idx].mean())
    if normal_idx.size:
        diag["normal_bag_precision"] = float(1.0 - truth[normal_idx].mean())
    diag["bag_sizes"] = [int(normal_idx.size), int(abnormal_idx.size)]
    print(f"  [diag] coarse split vs truth: {diag}")
    return diag


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="C2FPL k-fold CV on a feature directory")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--crop-axis", default="auto", choices=["auto", "0", "1"],
                   help="which axis of a 3-D feature file is the crop axis")

    p.add_argument("--folds", type=int, default=-1,
                   help="number of folds to run; -1 = all splits_*.txt files")
    p.add_argument("--max-normal-train", type=int, default=-1)
    p.add_argument("--max-anomaly-train", type=int, default=-1)
    p.add_argument("--max-normal-test", type=int, default=-1)
    p.add_argument("--max-anomaly-test", type=int, default=-1)
    p.add_argument("--max-cal-videos", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)

    # C2FPL pseudo label stage
    p.add_argument("--nu", type=float, default=1.0, help="target |abnormal|/|normal| ratio")
    p.add_argument("--pct", type=float, default=3.0, help="percentile moved per coarse step")
    p.add_argument("--window-ratio", type=float, default=0.2,
                   help="fine-stage anomalous window length, as a fraction of a video")
    p.add_argument("--abnormal-by", default="var", choices=["var", "mean", "gmm"],
                   help="how the abnormal GMM cluster is identified")
    p.add_argument("--fine-score", default="pdf", choices=["pdf", "logpdf"],
                   help="fine-stage window score; logpdf is more robust to "
                        "strongly out-of-distribution anomalies (see pseudo_label.py)")

    # detector
    p.add_argument("--max-epoch", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--milestones", type=int, nargs="*", default=[15, 25])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--video-agg", default="max", choices=["max", "mean", "topk"])
    p.add_argument("--topk-ratio", type=float, default=0.1)

    p.add_argument("--device", default="auto")
    p.add_argument("--verbose-train", action="store_true")
    p.add_argument("--out-json", default="")
    return p


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}")

    file_to_label, splits, feat_dir = da.discover_dataset(
        args.feat_dir, args.label_dir, args.split_dir
    )
    n_anomaly = sum(file_to_label.values())
    print(f"{len(file_to_label)} feature files: "
          f"{n_anomaly} anomaly / {len(file_to_label) - n_anomaly} normal, "
          f"{len(splits)} splits")

    k = len(splits) if args.folds < 0 else min(args.folds, len(splits))
    fold_scores = {"accuracy": [], "ROC_AUC": [], "PR_AUC": [], "F1": []}
    per_fold = []

    for fold_idx in range(k):
        print(f"\n===== fold {fold_idx + 1}/{k} =====")
        test_all = splits[fold_idx]
        cal_idx = (fold_idx + 1) % len(splits)
        cal_all = splits[cal_idx]
        train_all = [
            p for j, fold in enumerate(splits) if j not in (fold_idx, cal_idx) for p in fold
        ]

        train_files = da.limit_data(train_all, file_to_label,
                                    args.max_normal_train, args.max_anomaly_train,
                                    args.seed + fold_idx)
        test_files = da.limit_data(test_all, file_to_label,
                                   args.max_normal_test, args.max_anomaly_test,
                                   args.seed + 100 + fold_idx)
        rng = random.Random(args.seed + fold_idx)
        cal_files = (cal_all if args.max_cal_videos < 0
                     else rng.sample(cal_all, min(args.max_cal_videos, len(cal_all))))
        print(f"  train {len(train_files)} / cal {len(cal_files)} / test {len(test_files)} videos")

        train_feats, train_files = da.load_fold(train_files, feat_dir, args.crop_axis)
        cal_feats, cal_files = da.load_fold(cal_files, feat_dir, args.crop_axis)
        test_feats, test_files = da.load_fold(test_files, feat_dir, args.crop_axis)
        feature_size = da.feature_dim(train_feats)
        print(f"  feature dim {feature_size}, crops {train_feats[0].shape[1]}")

        pseudo, info = generate_pseudo_labels(
            train_feats, nu=args.nu, pct=args.pct, length=args.window_ratio,
            seed=args.seed + fold_idx, abnormal_by=args.abnormal_by,
            score=args.fine_score,
        )
        diag = coarse_diagnostics(train_files, info, file_to_label)

        model = train_detector(train_feats, pseudo, feature_size, args, device)

        y_cal = np.array([file_to_label[f] for f in cal_files])
        y_test = np.array([file_to_label[f] for f in test_files])
        threshold = calibrate_threshold(video_scores(model, cal_feats, args, device), y_cal)
        scores = video_scores(model, test_feats, args, device)
        result = evaluate(scores, y_test, threshold)

        print("  " + "  ".join(f"{k2}={v:.4f}" for k2, v in result.items()))
        for key in fold_scores:
            fold_scores[key].append(result[key])
        per_fold.append({"fold": fold_idx, "threshold": threshold,
                         "metrics": result, "coarse_diag": diag})

    print("\n===== summary =====")
    summary = {}
    for key, values in fold_scores.items():
        values = np.array(values, dtype=np.float64)
        summary[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
        print(f"{key:9s} {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "folds": per_fold, "summary": summary},
                      fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
