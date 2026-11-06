"""C2FPL coarse-to-fine pseudo label generation, as a reusable function.

This is a port of ``notebooks/Unsupervised_PL_C2FPL.ipynb`` that works on a
list of per-video feature arrays instead of one big concatenated .npy plus
``nalist.npy``.  The algorithm is unchanged:

  Coarse (video level)
    1. segment energy  = mean over crops of the squared L2 norm  -> (T,)
    2. each video       -> 2-D point (mean, variance) of that series
    3. 2-component GMM  -> normal bag / abnormal bag
    4. iteratively refit the GMM on the normal bag and move the lowest
       ``pct``-percentile of log-likelihood into the abnormal bag, until
       |abnormal| / |normal| >= nu

  Fine (segment level)
    5. fit one Gaussian on the segment energies of all normal videos
    6. for every abnormal video, take the sliding window of length
       ``length * T`` that maximises the total absolute first difference of
       the Gaussian likelihood, and label those segments 1
    7. normal videos are labelled 0 everywhere
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


# --------------------------------------------------------------------------- #
# step 1: per-video segment energy
# --------------------------------------------------------------------------- #
def segment_energy(feat):
    """(T, C, D) -> (T,): squared L2 norm, averaged over crops."""
    return np.mean(np.sum(np.square(feat), axis=2), axis=1)


def video_stats(energy):
    """(T,) -> (mean, variance). Matches estimate_gauss() in the notebook."""
    energy = np.asarray(energy, dtype=np.float64)
    mu = float(np.mean(energy))
    var = float(np.var(energy, ddof=1)) if energy.size > 1 else 0.0
    return mu, var


def _new_gmm(seed):
    return GaussianMixture(
        n_components=2, max_iter=150, random_state=seed, covariance_type="spherical"
    )


# --------------------------------------------------------------------------- #
# steps 2-4: coarse stage
# --------------------------------------------------------------------------- #
def coarse_split(params, nu=1.0, pct=3.0, seed=0, abnormal_by="var", max_steps=200,
                 verbose=True):
    """Split videos into normal / abnormal bags.

    params      : (N, 2) array of (mean, variance) per video
    abnormal_by : which centroid coordinate marks the abnormal cluster --
                  "var" (larger variance), "mean" (larger mean) or "gmm"
                  (trust the raw GMM label 1, as the original notebook does)

    Returns (normal_idx, abnormal_idx) as int arrays of indices into ``params``.
    """
    params = np.asarray(params, dtype=np.float64)
    n = len(params)
    if n < 2:
        return np.arange(n), np.empty(0, dtype=int)

    gmm = _new_gmm(seed)
    labels = gmm.fit_predict(params)

    if abnormal_by == "gmm":
        abnormal_label = 1
    else:
        col = 1 if abnormal_by == "var" else 0
        abnormal_label = int(np.argmax(gmm.means_[:, col]))

    abnormal = list(np.where(labels == abnormal_label)[0])
    normal = list(np.where(labels != abnormal_label)[0])
    if verbose:
        print(f"  [coarse] initial GMM split: {len(normal)} normal / {len(abnormal)} abnormal")

    # iterative refinement: drain the normal bag until the bags are balanced
    step = 1
    while normal and len(abnormal) / max(len(normal), 1) < nu and step <= max_steps:
        if len(normal) < 2:
            break
        sub = params[normal]
        gmm = _new_gmm(seed)
        gmm.fit(sub)
        score = gmm.score_samples(sub)
        threshold = np.percentile(score, pct)
        moved = np.where(score < threshold)[0]
        if moved.size == 0:  # percentile too coarse for a bag this small
            moved = np.array([int(np.argmin(score))])

        moved_set = set(moved.tolist())
        abnormal += [normal[i] for i in moved]
        normal = [v for i, v in enumerate(normal) if i not in moved_set]
        if verbose:
            print(f"  [coarse] step {step}: threshold {threshold:.2f}, "
                  f"moved {moved.size} -> {len(normal)} normal / {len(abnormal)} abnormal")
        step += 1

    return np.array(sorted(normal), dtype=int), np.array(sorted(abnormal), dtype=int)


# --------------------------------------------------------------------------- #
# steps 5-6: fine stage
# --------------------------------------------------------------------------- #
def _best_window(values, window_size):
    """Index of the window maximising the total absolute first difference."""
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return 0
    kernel_len = max(window_size - 1, 1)
    if kernel_len > diffs.size:
        return 0
    sums = np.convolve(diffs, np.ones(kernel_len), mode="valid")
    return int(np.argmax(sums))


def fine_localize(energies, normal_idx, abnormal_idx, length=0.2, score="pdf",
                  verbose=True):
    """Produce one 0/1 label array per video (same length as its segment count)."""
    normal_pool = np.concatenate([energies[i] for i in normal_idx]) if len(normal_idx) else None
    if normal_pool is None or normal_pool.size < 2:
        raise RuntimeError("the normal bag is empty or too small to fit the fine-stage Gaussian")

    mu, var = video_stats(normal_pool)
    if var <= 0:
        var = 1e-12
    gauss = multivariate_normal(mu, var)
    if verbose:
        print(f"  [fine] normal Gaussian: mu={mu:.4g}, var={var:.4g} "
              f"({normal_pool.size} segments from {len(normal_idx)} videos)")

    labels = [None] * len(energies)
    for i in normal_idx:
        labels[i] = np.zeros(len(energies[i]), dtype=np.float32)

    degenerate = 0
    for i in abnormal_idx:
        energy = energies[i]
        t = len(energy)
        out = np.zeros(t, dtype=np.float32)

        if score == "pdf":
            probs = np.atleast_1d(np.asarray(gauss.pdf(energy), dtype=np.float64))
            if not np.any(np.abs(np.diff(probs)) > 0):
                # pdf underflowed to a constant -> the window search has nothing
                # to maximise, so fall back to the log density
                probs = np.atleast_1d(np.asarray(gauss.logpdf(energy), dtype=np.float64))
                degenerate += 1
        else:
            probs = np.atleast_1d(np.asarray(gauss.logpdf(energy), dtype=np.float64))
        probs = np.nan_to_num(probs, neginf=np.finfo(np.float64).min / 4)

        window_size = max(int(t * length), 1)
        window_size = min(window_size, t)
        start = _best_window(probs, window_size)
        out[start:start + window_size] = 1.0
        labels[i] = out

    if degenerate and verbose:
        print(f"  [fine] {degenerate} abnormal video(s) fell back to logpdf (pdf underflow)")
    return labels


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def generate_pseudo_labels(feats, nu=1.0, pct=3.0, length=0.2, seed=0,
                           abnormal_by="var", score="pdf", verbose=True):
    """Full coarse-to-fine pass over a list of (T, C, D) feature arrays.

    Returns ``(labels, info)`` where ``labels`` is a list of (T,) float32 arrays
    and ``info`` carries the bag indices for diagnostics.
    """
    energies = [segment_energy(f) for f in feats]
    params = np.array([video_stats(e) for e in energies], dtype=np.float64)

    normal_idx, abnormal_idx = coarse_split(
        params, nu=nu, pct=pct, seed=seed, abnormal_by=abnormal_by, verbose=verbose
    )
    labels = fine_localize(energies, normal_idx, abnormal_idx, length=length,
                           score=score, verbose=verbose)

    total = sum(len(l) for l in labels)
    positive = sum(float(l.sum()) for l in labels)
    if verbose:
        print(f"  [pseudo] {len(labels)} videos, {total} segments, "
              f"{positive:.0f} positive ({positive / max(total, 1):.1%})")

    info = {
        "normal_idx": normal_idx,
        "abnormal_idx": abnormal_idx,
        "params": params,
        "segment_total": total,
        "segment_positive": positive,
    }
    return labels, info
