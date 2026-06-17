#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v13.2.1 Bayesian-coupled full-diagnostics block discovery.

This version keeps the v13.1 residual-driven iterative K expansion, then adds
a posterior coupling layer:

    W/H posterior diagnostics -> ARD-style inclusion probabilities
    -> optional low-posterior pruning -> block co-association posterior.

The aim is not full MCMC.  It is a structured empirical-Bayes / variational
first pass that makes uncertainty visible and lets W/H/K/block evidence talk to
each other through diagnostics and lightweight posterior scores.

Main stages
-----------
1. Clustered-Gaussian block-identifiable toy.
   - Each component has a Gaussian neural loading.
   - Components within one functional group form a local Gaussian cluster.
   - H-side uses whitened polynomial subspaces by default, so q, q^2, q^3, q^4
     are all observable.
2. Identifiability audit.
3. True iterative K expansion:
   - Fit/refit H/W by ALS on training trials.
   - Compute residual.
   - Search residual split-repeat reliable directions.
   - Try top residual candidates.
   - Globally refit all accepted components.
   - Accept only if held-out gain and residual reliability pass thresholds.
4. Final all-data refit.
5. Component posterior diagnostics and optional ARD-style pruning.
6. Gaussian localization of the active K-dimensional subspace.
7. Block posterior from W locality, H dependency, split/seed perturbations, and
   co-association stability.

The file is self-contained and writes all diagnostic tables/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from numpy.linalg import eigh, norm, svd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

EPS = 1e-12

# =============================================================================
# Basic utilities
# =============================================================================

def deep_update(base: Dict, override: Dict) -> Dict:
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def zscore(x: np.ndarray, axis=None, eps: float = EPS) -> np.ndarray:
    return (x - np.mean(x, axis=axis, keepdims=True)) / (np.std(x, axis=axis, keepdims=True) + eps)


def center_cols(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def qr_basis(A: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    A = np.asarray(A, float)
    if A.size == 0:
        return A.copy()
    Q, _ = np.linalg.qr(A)
    if k is not None:
        Q = Q[:, :k]
    for j in range(Q.shape[1]):
        ix = np.argmax(np.abs(Q[:, j]))
        if Q[ix, j] < 0:
            Q[:, j] *= -1
    return Q


def participation_ratio(vals: np.ndarray, eps: float = EPS) -> float:
    vals = np.maximum(np.asarray(vals, float), 0.0)
    return float(vals.sum() ** 2 / (np.sum(vals ** 2) + eps))


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    a = np.asarray(a).ravel() - np.asarray(a).ravel().mean()
    b = np.asarray(b).ravel() - np.asarray(b).ravel().mean()
    return float(np.dot(a, b) / (norm(a) * norm(b) + eps))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, float) / max(float(temperature), EPS)
    s = s - np.nanmax(s)
    w = np.exp(s)
    return w / (np.nansum(w) + EPS)


def r2_score(X: np.ndarray, Xhat: np.ndarray, eps: float = EPS) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    return float(1.0 - np.sum((X - Xhat) ** 2) / (np.sum(Xc ** 2) + eps))


def project_r2(X: np.ndarray, Q: np.ndarray, eps: float = EPS) -> float:
    Xc = center_cols(X)
    if Q.shape[1] == 0:
        Xhat = np.zeros_like(Xc)
    else:
        Q = qr_basis(Q)
        Xhat = Xc @ Q @ Q.T
    return float(1.0 - np.sum((Xc - Xhat) ** 2) / (np.sum(Xc ** 2) + eps))


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n <= 1:
        return 1.0
    tv, ti = np.unique(labels_true, return_inverse=True)
    pv, pi = np.unique(labels_pred, return_inverse=True)
    tab = np.zeros((len(tv), len(pv)), dtype=np.int64)
    for i in range(n):
        tab[ti[i], pi[i]] += 1

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb = np.sum(comb2(tab))
    sum_t = np.sum(comb2(tab.sum(axis=1)))
    sum_p = np.sum(comb2(tab.sum(axis=0)))
    total = comb2(n)
    expected = sum_t * sum_p / total if total > 0 else 0.0
    max_index = 0.5 * (sum_t + sum_p)
    den = max_index - expected
    if abs(den) < 1e-12:
        return 0.0
    return float((sum_comb - expected) / den)


def one_dim_kmeans(x: np.ndarray, F: int, n_starts: int = 64, n_iter: int = 100, seed: int = 0):
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(seed)
    n = len(x)
    if F >= n:
        labels = np.arange(n)
        centers = x.copy()
        return labels, centers, 0.0
    xs = np.sort(x)
    starts = []
    starts.append(np.quantile(xs, np.linspace(0, 1, F + 2)[1:-1]))
    starts.append(np.linspace(xs.min(), xs.max(), F))
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(np.sort(rng.choice(xs, size=F, replace=False)))
    best = None
    for c0 in starts:
        centers = np.asarray(c0, float).copy()
        for _ in range(n_iter):
            labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            new_centers = centers.copy()
            for f in range(F):
                if np.any(labels == f):
                    new_centers[f] = x[labels == f].mean()
                else:
                    new_centers[f] = rng.choice(xs)
            if np.max(np.abs(new_centers - centers)) < 1e-9:
                centers = new_centers
                break
            centers = new_centers
        order = np.argsort(centers)
        remap = {old: new for new, old in enumerate(order)}
        labels2 = np.array([remap[l] for l in labels])
        centers2 = centers[order]
        sse = float(np.sum((x - centers2[labels2]) ** 2))
        if best is None or sse < best[2]:
            best = (labels2, centers2, sse)
    return best


def gaussian_atom(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    g = np.exp(-0.5 * ((z - mu) / max(sig, 1e-4)) ** 2)
    g = g - g.mean()
    g = g / (norm(g) + EPS)
    return g


def ridge_solve_left(A: np.ndarray, B: np.ndarray, ridge: float) -> np.ndarray:
    """Solve X = argmin ||A X - B||^2 + ridge ||X||^2."""
    K = A.shape[1]
    return np.linalg.solve(A.T @ A + ridge * np.eye(K), A.T @ B)


def smooth_H_trials(H_flat: np.ndarray, R: int, T: int, K: int, lam: float) -> np.ndarray:
    if K == 0 or lam <= 0:
        return H_flat
    # Use Gaussian filter as a fast smoother; earlier versions used second-derivative smoothing.
    sigma = max(0.0, float(lam))
    H = H_flat.reshape(R, T, K)
    out = np.empty_like(H)
    for r in range(R):
        for k in range(K):
            out[r, :, k] = gaussian_filter1d(H[r, :, k], sigma=sigma, mode="nearest")
    return out.reshape(R * T, K)


def polynomial_dependency_matrix(H: np.ndarray, max_samples: int = 6000, seed: int = 0) -> np.ndarray:
    """Nonlinear dependency proxy for component expressions.

    Uses max absolute correlation among simple polynomial transforms.  This is
    intentionally lightweight but detects q/q^2/q^3/q^4 dependence better than
    linear correlation alone.
    """
    rng = np.random.default_rng(seed)
    M, K = H.shape
    if M > max_samples:
        idx = rng.choice(M, size=max_samples, replace=False)
        X = H[idx]
    else:
        X = H.copy()
    X = zscore(X, axis=0)
    transforms = [X, zscore(X ** 2, axis=0), zscore(X ** 3, axis=0)]
    dep = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            best = 0.0
            for A in transforms:
                ai = A[:, i]
                for B in transforms:
                    bj = B[:, j]
                    best = max(best, abs(safe_corr(ai, bj)))
            dep[i, j] = dep[j, i] = best
    return dep

# =============================================================================
# Configurations
# =============================================================================

DEFAULT_CONFIG: Dict = {
    "version": "v13.2.1_full_diagnostics_block_discovery",
    "toy": {
        "seed": 20270714,
        "n_trials": 120,
        "n_time": 90,
        "n_neurons": 120,
        "n_groups": 3,
        "comps_per_group": 4,
        "group_centers": [0.22, 0.52, 0.80],
        "group_widths": [0.040, 0.040, 0.040],
        "within_group_center_span": 0.20,
        "center_jitter": 0.004,
        "width_jitter": 0.05,
        "amp_jitter": 0.10,
        "random_signs": True,
        "noise_sd": 0.04,
        "shift_sd": 0.035,
        "phase_sd": 0.018,
        "trial_amp_sd": 0.18,
        "smooth_noise_sd": 0.20,
        "h_basis": "whitened_powers",
        "standardize_X": True,
        "neuron_shuffle": True
    },
    "audit": {
        "rank_energy_warning": 0.90,
        "min_effective_rank_per_group": 2.5
    },
    "iterative": {
        "train_fraction": 0.75,
        "max_K": 18,
        "candidate_top": 4,
        "als_iters": 8,
        "ridge_H": 1e-3,
        "ridge_W": 1e-3,
        "h_smooth_sigma": 0.0,
        "n_residual_splits": 80,
        "n_null": 60,
        "null_quantile": 0.95,
        "min_gain_val": 0.0010,
        "min_gain_train": 0.0005,
        "min_eig_over_null": 1.02,
        "max_duplicate_corr": 0.95,
        "patience": 2,
        "seed": 123
    },
    "localization": {
        "center_grid_size": 520,
        "width_grid": [0.025, 0.032, 0.040, 0.050, 0.062, 0.078],
        "omp_min_center_sep": 0.025,
        "omp_redundancy_penalty": 0.25,
        "max_atoms": None
    },
    "block": {
        "F_grid": [1, 2, 3, 4, 5, 6, 7, 8],
        "kmeans_starts": 96,
        "seed": 456,
        "w_spatial_weight": 0.45,
        "h_dep_weight": 0.35,
        "stability_weight": 0.10,
        "complexity_penalty": 0.050,
        "min_block_size": 2
    },
    "posterior": {
        "inclusion_threshold": 0.55,
        "prune_threshold": 0.25,
        "enable_prune": True,
        "gain_scale": 0.001,
        "gain_weight": 1.20,
        "reliability_weight": 0.65,
        "energy_weight": 0.45,
        "posterior_bias": -2.40,
        "ridge_prior_w": 1e-3,
        "ridge_prior_h": 1e-3,
        "noise_floor": 1e-8
    },
    "block_posterior": {
        "n_runs": 80,
        "temperature": 0.035,
        "center_jitter_sd": 0.006,
        "dep_jitter_sd": 0.015,
        "extra_complexity_penalty": 0.012,
        "balance_weight": 0.030,
        "singleton_penalty": 0.120,
        "seed": 789
    },
    "plot": {"dpi": 160}
}

QUICK_OVERRIDE = {
    "toy": {"n_trials": 50, "n_time": 45, "n_neurons": 65, "noise_sd": 0.04},
    "iterative": {"max_K": 16, "candidate_top": 3, "als_iters": 5, "n_residual_splits": 35, "n_null": 25, "min_gain_val": 0.0010, "patience": 2},
    "localization": {"center_grid_size": 240},
    "block": {"F_grid": [1, 2, 3, 4, 5, 6], "kmeans_starts": 40},
    "block_posterior": {"n_runs": 36}
}

FULL_OVERRIDE = {
    "toy": {"n_trials": 120, "n_time": 90, "n_neurons": 120, "noise_sd": 0.04},
    "iterative": {"max_K": 20, "candidate_top": 5, "als_iters": 8, "n_residual_splits": 120, "n_null": 80, "patience": 3},
    "localization": {"center_grid_size": 560},
    "block": {"F_grid": [1, 2, 3, 4, 5, 6, 7, 8], "kmeans_starts": 128},
    "block_posterior": {"n_runs": 96}
}

WIDE_OVERRIDE = {
    "toy": {"n_trials": 180, "n_time": 110, "n_neurons": 160, "noise_sd": 0.035},
    "iterative": {"max_K": 24, "candidate_top": 6, "als_iters": 10, "n_residual_splits": 180, "n_null": 120, "patience": 4},
    "localization": {"center_grid_size": 700},
    "block": {"F_grid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "kmeans_starts": 180},
    "block_posterior": {"n_runs": 140}
}

# =============================================================================
# Toy generation
# =============================================================================

def make_clustered_gaussian_toy(cfg: Dict):
    toy = cfg["toy"]
    rng = np.random.default_rng(int(toy["seed"]))
    R = int(toy["n_trials"])
    T = int(toy["n_time"])
    N = int(toy["n_neurons"])
    G = int(toy["n_groups"])
    M = int(toy["comps_per_group"])
    Ktrue = G * M
    z = np.linspace(0, 1, N)
    group_centers = np.array(toy["group_centers"], float)
    widths = np.array(toy["group_widths"], float)
    span = float(toy["within_group_center_span"])
    base_offsets = np.linspace(-span / 2, span / 2, M)

    W_cols, labels, comp_centers, comp_widths, comp_amp = [], [], [], [], []
    for g in range(G):
        for m in range(M):
            cen = group_centers[g] + base_offsets[m] + float(toy["center_jitter"]) * rng.normal()
            sig = max(0.012, widths[g] * (1.0 + float(toy["width_jitter"]) * rng.normal()))
            amp = 1.0 + float(toy["amp_jitter"]) * rng.normal()
            if bool(toy["random_signs"]) and rng.random() < 0.5:
                amp *= -1.0
            w = amp * np.exp(-0.5 * ((z - cen) / sig) ** 2)
            w = zscore(w)
            w = w / (norm(w) + EPS)
            W_cols.append(w)
            labels.append(g)
            comp_centers.append(cen)
            comp_widths.append(sig)
            comp_amp.append(amp)
    W_true_ordered = np.column_stack(W_cols)
    labels = np.array(labels, dtype=int)

    H = np.zeros((R, T, Ktrue))
    t = np.linspace(0, 1, T)
    for g in range(G):
        block = np.zeros((R * T, M))
        for r in range(R):
            shift = float(toy["shift_sd"]) * rng.normal()
            phase = float(toy["phase_sd"]) * rng.normal()
            amp = 1.0 + float(toy["trial_amp_sd"]) * rng.normal()
            local = np.exp(-0.5 * ((t - (group_centers[g] + shift)) / 0.16) ** 2)
            osc = np.sin(2 * np.pi * ((g + 1) * 0.75 * t + phase + rng.uniform()))
            smooth_noise = gaussian_filter1d(rng.normal(size=T), sigma=2)
            q = 0.68 * local + 0.31 * osc + float(toy["smooth_noise_sd"]) * smooth_noise
            q = np.tanh(amp * zscore(q))
            q = zscore(q)
            for m in range(M):
                block[r * T:(r + 1) * T, m] = zscore(q ** (m + 1))
        if toy.get("h_basis", "whitened_powers") == "whitened_powers":
            block = zscore(block, axis=0)
            C = block.T @ block / block.shape[0]
            vals, vecs = eigh(C)
            ix = np.argsort(vals)[::-1]
            vals, vecs = vals[ix], vecs[:, ix]
            block = block @ vecs @ np.diag(1.0 / np.sqrt(vals + 1e-7))
            block = zscore(block, axis=0)
        else:
            block = zscore(block, axis=0)
        for m in range(M):
            H[:, :, g * M + m] = block[:, m].reshape(R, T)

    signal = np.einsum("rtk,nk->rtn", H, W_true_ordered)
    X = signal + float(toy["noise_sd"]) * rng.normal(size=signal.shape)

    if bool(toy["standardize_X"]):
        mean = X.mean(axis=(0, 1), keepdims=True)
        sd = X.std(axis=(0, 1), keepdims=True) + EPS
        X = (X - mean) / sd
        signal = (signal - mean) / sd

    if bool(toy["neuron_shuffle"]):
        perm = rng.permutation(N)
        X = X[:, :, perm]
        signal = signal[:, :, perm]
        W_true = W_true_ordered[perm, :]
        z_shuffled = z[perm]
    else:
        perm = np.arange(N)
        W_true = W_true_ordered
        z_shuffled = z

    truth = {
        "Ktrue": int(Ktrue),
        "Ftrue": int(G),
        "comps_per_group": int(M),
        "labels": labels,
        "group_centers": group_centers,
        "component_centers": np.array(comp_centers),
        "component_widths": np.array(comp_widths),
        "component_amp": np.array(comp_amp),
        "neuron_permutation": perm
    }
    return X, signal, H, W_true, z_shuffled, truth

# =============================================================================
# Identifiability audit
# =============================================================================

def identifiability_audit(H: np.ndarray, W: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    Hflat = H.reshape(-1, H.shape[-1])
    rows = []
    for g in np.unique(labels):
        cols = labels == g
        Sw = svd(W[:, cols], compute_uv=False) ** 2
        Sh = svd(Hflat[:, cols], compute_uv=False) ** 2
        rows.append({
            "group": int(g),
            "W_eff_rank": participation_ratio(Sw),
            "W_s1_energy": float(Sw[0] / (Sw.sum() + EPS)),
            "H_eff_rank": participation_ratio(Sh),
            "H_s1_energy": float(Sh[0] / (Sh.sum() + EPS)),
            "W_spectrum_norm": ";".join(f"{x:.6g}" for x in Sw / (Sw.sum() + EPS)),
            "H_spectrum_norm": ";".join(f"{x:.6g}" for x in Sh / (Sh.sum() + EPS)),
        })
    return pd.DataFrame(rows)

# =============================================================================
# ALS refit and validation
# =============================================================================

def flatten_trials(X: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
    if idx is None:
        Y = X
    else:
        Y = X[idx]
    return Y.reshape(-1, X.shape[-1])


def refit_HW_als(X_trials: np.ndarray, W_init: np.ndarray, cfg: Dict, trial_idx: Optional[np.ndarray] = None):
    """Global H/W refit for a fixed K using ALS.

    This is the refit step that was missing in the diagnostic v13 version.
    """
    X = center_cols(flatten_trials(X_trials, trial_idx))
    R = X_trials.shape[0] if trial_idx is None else len(trial_idx)
    T = X_trials.shape[1]
    if W_init.shape[1] == 0:
        return np.zeros((X.shape[0], 0)), np.zeros((X.shape[1], 0)), np.zeros_like(X), 0.0
    K = W_init.shape[1]
    W = qr_basis(W_init, K)
    for _ in range(int(cfg["iterative"]["als_iters"])):
        # H update
        H = X @ W @ np.linalg.inv(W.T @ W + float(cfg["iterative"]["ridge_H"]) * np.eye(K))
        H = center_cols(H)
        H = smooth_H_trials(H, R, T, K, float(cfg["iterative"].get("h_smooth_sigma", 0.0)))
        # W update
        W = X.T @ H @ np.linalg.inv(H.T @ H + float(cfg["iterative"]["ridge_W"]) * np.eye(K))
        W = center_cols(W)
        W = qr_basis(W, K)
    H = X @ W
    H = center_cols(H)
    H = smooth_H_trials(H, R, T, K, float(cfg["iterative"].get("h_smooth_sigma", 0.0)))
    W_ls = X.T @ H @ np.linalg.inv(H.T @ H + float(cfg["iterative"]["ridge_W"]) * np.eye(K))
    W = qr_basis(center_cols(W_ls), K)
    H = X @ W
    Xhat = H @ W.T
    r2 = r2_score(X, Xhat)
    return H, W, Xhat, r2


def project_reconstruct_r2(X_trials: np.ndarray, W: np.ndarray, trial_idx: Optional[np.ndarray] = None) -> float:
    X = center_cols(flatten_trials(X_trials, trial_idx))
    if W.shape[1] == 0:
        Xhat = np.zeros_like(X)
    else:
        Q = qr_basis(W)
        Xhat = X @ Q @ Q.T
    return r2_score(X, Xhat)


# =============================================================================
# Posterior diagnostics and ARD-style pruning
# =============================================================================

def component_posterior_diagnostics(
    X_trials: np.ndarray,
    iter_res: Dict,
    cfg: Dict,
    local: Optional[Dict] = None,
    truth: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Lightweight empirical-Bayes posterior summary for accepted components.

    The ALS fit remains the point-estimate backbone.  This function adds the
    missing uncertainty layer: component inclusion probability, ARD precision
    proxy, and Gaussian posterior variance proxies for W and H.
    """
    pcfg = cfg["posterior"]
    W = np.asarray(iter_res["W"], float)
    H = np.asarray(iter_res["H"], float)
    K = W.shape[1]
    Xflat = center_cols(flatten_trials(X_trials, None))
    if K == 0:
        summary = {
            "K_hard": 0,
            "K_eff_posterior": 0,
            "K_eff_soft": 0.0,
            "min_inclusion_prob": np.nan,
            "residual_noise_var": float(np.var(Xflat)),
            "posterior_pruned": 0,
        }
        return pd.DataFrame(), summary

    Xhat = H @ W.T
    resid = Xflat - Xhat
    noise_var = max(float(np.var(resid)), float(pcfg["noise_floor"]))
    beta = 1.0 / noise_var
    total_energy = float(np.sum(center_cols(Xflat) ** 2) + EPS)
    component_energy = np.array([(norm(H[:, k]) ** 2) * (norm(W[:, k]) ** 2) for k in range(K)], float)
    median_energy = float(np.median(component_energy) + EPS)

    trace = iter_res["trace"]
    accepted_trace = trace[trace["accepted"].astype(bool)].reset_index(drop=True) if len(trace) else pd.DataFrame()
    centers = None if local is None else np.asarray(local.get("centers", []), float)
    widths = None if local is None else np.asarray(local.get("widths", []), float)
    true_group_by_center = None
    if centers is not None and truth is not None and len(centers) == K:
        gc = np.asarray(truth["group_centers"], float)
        true_group_by_center = np.argmin(np.abs(centers[:, None] - gc[None, :]), axis=1)

    rows = []
    for k in range(K):
        if k < len(accepted_trace):
            tr = accepted_trace.iloc[k]
            gain_val = float(tr.get("best_gain_val", np.nan))
            gain_train = float(tr.get("best_gain_train", np.nan))
            eig_over_null = float(tr.get("best_eig_over_null", np.nan))
            accepted_iteration = int(tr.get("iteration", k + 1))
        else:
            gain_val = np.nan
            gain_train = np.nan
            eig_over_null = np.nan
            accepted_iteration = k + 1

        gain_term = math.log1p(max(gain_val if np.isfinite(gain_val) else 0.0, 0.0) / max(float(pcfg["gain_scale"]), EPS))
        reliability_term = math.log(max(eig_over_null if np.isfinite(eig_over_null) else 1.0, EPS))
        energy_term = math.log1p(component_energy[k] / median_energy)
        logit = (
            float(pcfg["posterior_bias"])
            + float(pcfg["gain_weight"]) * gain_term
            + float(pcfg["reliability_weight"]) * reliability_term
            + float(pcfg["energy_weight"]) * energy_term
        )
        incl = float(sigmoid(logit))
        w_norm2 = float(np.dot(W[:, k], W[:, k]) + EPS)
        h_norm2 = float(np.dot(H[:, k], H[:, k]) + EPS)
        var_w = float(1.0 / (beta * h_norm2 + float(pcfg["ridge_prior_w"])))
        var_h = float(1.0 / (beta * w_norm2 + float(pcfg["ridge_prior_h"])))
        r2_contrib = float(component_energy[k] / total_energy)
        rows.append({
            "component": k,
            "accepted_iteration": accepted_iteration,
            "inclusion_prob": incl,
            "active_posterior": bool(incl >= float(pcfg["inclusion_threshold"])),
            "ard_alpha_proxy": float(1.0 / (r2_contrib + 1e-9)),
            "posterior_var_w_mean": var_w,
            "posterior_var_h_mean": var_h,
            "component_energy": float(component_energy[k]),
            "component_R2_proxy": r2_contrib,
            "gain_val_at_accept": gain_val,
            "gain_train_at_accept": gain_train,
            "eig_over_null_at_accept": eig_over_null,
            "gaussian_center": np.nan if centers is None or k >= len(centers) else float(centers[k]),
            "gaussian_width": np.nan if widths is None or k >= len(widths) else float(widths[k]),
            "nearest_true_group_by_center": -1 if true_group_by_center is None else int(true_group_by_center[k]),
        })
    df = pd.DataFrame(rows)
    summary = {
        "K_hard": int(K),
        "K_eff_posterior": int(df["active_posterior"].sum()),
        "K_eff_soft": float(df["inclusion_prob"].sum()),
        "min_inclusion_prob": float(df["inclusion_prob"].min()),
        "mean_inclusion_prob": float(df["inclusion_prob"].mean()),
        "residual_noise_var": float(noise_var),
        "posterior_pruned": 0,
    }
    return df, summary


def apply_posterior_pruning(X_trials: np.ndarray, iter_res: Dict, comp_df: pd.DataFrame, cfg: Dict) -> Tuple[Dict, Dict]:
    pcfg = cfg["posterior"]
    prune_info = {
        "enabled": bool(pcfg.get("enable_prune", True)),
        "threshold": float(pcfg["prune_threshold"]),
        "n_pruned": 0,
        "kept_components": [],
        "pruned_components": [],
        "r2_before": float(iter_res["all_r2"]),
        "r2_after": float(iter_res["all_r2"]),
    }
    if not prune_info["enabled"] or len(comp_df) == 0:
        return iter_res, prune_info
    keep = comp_df["inclusion_prob"].to_numpy(float) >= prune_info["threshold"]
    if keep.all() or keep.sum() == 0:
        prune_info["kept_components"] = [int(x) for x in np.where(keep)[0]]
        prune_info["pruned_components"] = [int(x) for x in np.where(~keep)[0]]
        return iter_res, prune_info

    W_keep = np.asarray(iter_res["W"])[:, keep]
    H_new, W_new, Xhat_new, r2_new = refit_HW_als(X_trials, W_keep, cfg, None)
    out = dict(iter_res)
    out["H"] = H_new
    out["W"] = W_new
    out["Xhat"] = Xhat_new
    out["all_r2"] = float(r2_new)
    prune_info["n_pruned"] = int((~keep).sum())
    prune_info["kept_components"] = [int(x) for x in np.where(keep)[0]]
    prune_info["pruned_components"] = [int(x) for x in np.where(~keep)[0]]
    prune_info["r2_after"] = float(r2_new)
    return out, prune_info

# =============================================================================
# Residual reliability candidate search
# =============================================================================

def make_split_templates(Xres_train: np.ndarray, n_splits: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create split-trial mean views from residual training trials.

    Xres_train: R_train x T x N.
    Returns two matrices [n_splits*T, N].
    """
    rng = np.random.default_rng(seed)
    R, T, N = Xres_train.shape
    A, B = [], []
    for _ in range(n_splits):
        perm = rng.permutation(R)
        half = R // 2
        aidx = perm[:half]
        bidx = perm[half:]
        XA = Xres_train[aidx].mean(axis=0)
        XB = Xres_train[bidx].mean(axis=0)
        A.append(zscore(XA.reshape(T, N), axis=0))
        B.append(zscore(XB.reshape(T, N), axis=0))
    return np.vstack(A), np.vstack(B)


def residual_reliability_candidates(Xres_train: np.ndarray, W_existing: np.ndarray, cfg: Dict, iter_seed: int):
    itcfg = cfg["iterative"]
    XA, XB = make_split_templates(Xres_train, int(itcfg["n_residual_splits"]), seed=iter_seed)
    XA = center_cols(XA)
    XB = center_cols(XB)
    S = XA.shape[0]
    Crel = (XA.T @ XB + XB.T @ XA) / (2 * max(1, S - 1))
    Cavg = ((XA + XB) / 2).T @ ((XA + XB) / 2) / max(1, S - 1)
    # Remove already accepted subspace to force true residual search.
    if W_existing.shape[1] > 0:
        Q = qr_basis(W_existing)
        P = np.eye(Crel.shape[0]) - Q @ Q.T
        Crel = P @ Crel @ P
        Cavg = P @ Cavg @ P
    vals, vecs = eigh((Crel + Crel.T) / 2)
    ix = np.argsort(vals)[::-1]
    vals = vals[ix]
    vecs = vecs[:, ix]
    cand_top = min(int(itcfg["candidate_top"]), vecs.shape[1])
    cands = []
    for j in range(cand_top):
        v = vecs[:, j].copy()
        if W_existing.shape[1] > 0:
            Q = qr_basis(W_existing)
            v = v - Q @ (Q.T @ v)
        v = v / (norm(v) + EPS)
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
        duplicate = 0.0 if W_existing.shape[1] == 0 else float(np.max(np.abs(qr_basis(W_existing).T @ v)))
        cands.append({"rank": j + 1, "eig": float(vals[j]), "w": v, "duplicate_corr": duplicate})
    # Null: shuffle XB time rows to destroy repeat relationship.
    rng = np.random.default_rng(iter_seed + 999)
    null_top = []
    n_null = int(itcfg["n_null"])
    for _ in range(n_null):
        perm = rng.permutation(XB.shape[0])
        XBp = XB[perm]
        Cn = (XA.T @ XBp + XBp.T @ XA) / (2 * max(1, S - 1))
        if W_existing.shape[1] > 0:
            Q = qr_basis(W_existing)
            P = np.eye(Cn.shape[0]) - Q @ Q.T
            Cn = P @ Cn @ P
        vn = eigh((Cn + Cn.T) / 2, eigvals_only=True) if False else np.linalg.eigvalsh((Cn + Cn.T) / 2)
        null_top.append(float(np.max(vn)))
    null_threshold = float(np.quantile(null_top, float(itcfg["null_quantile"]))) if n_null > 0 else 0.0
    return cands, null_threshold, np.array(null_top)

# =============================================================================
# True iterative K expansion
# =============================================================================

def true_iterative_K_expansion(X: np.ndarray, signal: Optional[np.ndarray], cfg: Dict, outdir: Path):
    rng = np.random.default_rng(int(cfg["iterative"]["seed"]))
    R = X.shape[0]
    perm = rng.permutation(R)
    n_train = int(round(float(cfg["iterative"]["train_fraction"]) * R))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:])
    if len(val_idx) < 2:
        val_idx = train_idx[-2:]
        train_idx = train_idx[:-2]

    W_current = np.zeros((X.shape[-1], 0))
    H_current = np.zeros((len(train_idx) * X.shape[1], 0))
    train_r2 = project_reconstruct_r2(X, W_current, train_idx)
    val_r2 = project_reconstruct_r2(X, W_current, val_idx)
    accepted_no = 0
    rejected_in_row = 0
    trace_rows, candidate_rows = [], []
    final_fit = None

    for it in range(1, int(cfg["iterative"]["max_K"]) + 1):
        # Fit current model on train, compute residual on train trials.
        if W_current.shape[1] > 0:
            Hcur, Wcur_fit, Xhat_train_flat, train_r2 = refit_HW_als(X, W_current, cfg, train_idx)
            W_current = Wcur_fit
            final_fit = (Hcur, Wcur_fit, Xhat_train_flat)
            # reconstruct residual for individual train trials by projection onto W.
            Xtrain_flat = center_cols(flatten_trials(X, train_idx))
            Q = qr_basis(W_current)
            Rtrain_flat = Xtrain_flat - Xtrain_flat @ Q @ Q.T
        else:
            Xtrain_flat = center_cols(flatten_trials(X, train_idx))
            Rtrain_flat = Xtrain_flat.copy()
        Xres_train = Rtrain_flat.reshape(len(train_idx), X.shape[1], X.shape[-1])

        cands, null_thr, null_top = residual_reliability_candidates(
            Xres_train, W_current, cfg, iter_seed=int(cfg["iterative"]["seed"]) + 31 * it
        )
        best = None
        for cand in cands:
            if cand["duplicate_corr"] > float(cfg["iterative"]["max_duplicate_corr"]):
                status = "skip_duplicate"
                gain_val = np.nan
                gain_train = np.nan
                W_try_fit = None
                tr2 = np.nan
                vr2 = np.nan
            else:
                W_try0 = np.column_stack([W_current, cand["w"]]) if W_current.shape[1] else cand["w"][:, None]
                H_try, W_try_fit, _, tr2 = refit_HW_als(X, W_try0, cfg, train_idx)
                vr2 = project_reconstruct_r2(X, W_try_fit, val_idx)
                gain_val = float(vr2 - val_r2)
                gain_train = float(tr2 - train_r2)
                status = "evaluated"
            eig_over_null = float(cand["eig"] / (null_thr + EPS))
            row = {
                "iteration": it,
                "candidate_rank": cand["rank"],
                "K_try": int(W_current.shape[1] + 1),
                "eig": cand["eig"],
                "null_threshold": null_thr,
                "eig_over_null": eig_over_null,
                "duplicate_corr": cand["duplicate_corr"],
                "train_r2_try": tr2,
                "val_r2_try": vr2,
                "gain_train": gain_train,
                "gain_val": gain_val,
                "status": status,
            }
            candidate_rows.append(row)
            if status == "evaluated":
                score = gain_val + 0.25 * gain_train + 0.0001 * eig_over_null - 0.001 * cand["duplicate_corr"]
                if best is None or score > best["score"]:
                    best = {**row, "score": score, "W_fit": W_try_fit}
        if best is None:
            accepted = False
            reason = "no_candidate"
        else:
            accepted = (
                best["gain_val"] >= float(cfg["iterative"]["min_gain_val"])
                and best["gain_train"] >= float(cfg["iterative"]["min_gain_train"])
                and best["eig_over_null"] >= float(cfg["iterative"]["min_eig_over_null"])
            )
            if accepted:
                reason = "accepted"
            elif best["gain_val"] < float(cfg["iterative"]["min_gain_val"]):
                reason = "low_val_gain"
            elif best["gain_train"] < float(cfg["iterative"]["min_gain_train"]):
                reason = "low_train_gain"
            else:
                reason = "below_residual_null"
        if accepted:
            W_current = best["W_fit"]
            H_current, W_current, _, train_r2 = refit_HW_als(X, W_current, cfg, train_idx)
            val_r2 = project_reconstruct_r2(X, W_current, val_idx)
            accepted_no = W_current.shape[1]
            rejected_in_row = 0
        else:
            rejected_in_row += 1
        signal_r2 = np.nan
        if signal is not None and W_current.shape[1] > 0:
            signal_r2 = project_reconstruct_r2(signal, W_current, None)
        trace_rows.append({
            "iteration": it,
            "K_current": int(W_current.shape[1]),
            "accepted": bool(accepted),
            "reason": reason,
            "best_candidate_rank": None if best is None else int(best["candidate_rank"]),
            "best_gain_val": np.nan if best is None else float(best["gain_val"]),
            "best_gain_train": np.nan if best is None else float(best["gain_train"]),
            "best_eig_over_null": np.nan if best is None else float(best["eig_over_null"]),
            "residual_null_threshold": null_thr,
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "signal_R2_oracle": float(signal_r2) if np.isfinite(signal_r2) else np.nan,
            "rejected_in_row": int(rejected_in_row),
        })
        print(f"iter={it:02d} K={W_current.shape[1]:02d} accepted={accepted} reason={reason} "
              f"val_r2={val_r2:.4f} gain={np.nan if best is None else best['gain_val']:.5f}", flush=True)
        if rejected_in_row >= int(cfg["iterative"]["patience"]):
            break

    # Final all-data refit with accepted K.
    if W_current.shape[1] > 0:
        H_all, W_all, Xhat_all, all_r2 = refit_HW_als(X, W_current, cfg, None)
    else:
        H_all, W_all, Xhat_all, all_r2 = np.zeros((X.shape[0] * X.shape[1], 0)), W_current, center_cols(flatten_trials(X)), 0.0
    return {
        "W": W_all,
        "H": H_all,
        "Xhat": Xhat_all,
        "all_r2": float(all_r2),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "trace": pd.DataFrame(trace_rows),
        "candidates": pd.DataFrame(candidate_rows),
    }

# =============================================================================
# Gaussian localization and block discovery
# =============================================================================

def gaussian_dictionary(z: np.ndarray, cfg: Dict):
    loc = cfg["localization"]
    grid = np.linspace(0.02, 0.98, int(loc["center_grid_size"]))
    widths = [float(x) for x in loc["width_grid"]]
    atoms, centers, sigmas = [], [], []
    for mu in grid:
        for sig in widths:
            atoms.append(gaussian_atom(z, float(mu), float(sig)))
            centers.append(float(mu))
            sigmas.append(float(sig))
    D = np.column_stack(atoms)
    return D, np.array(centers), np.array(sigmas)


def localize_subspace_gaussian_omp(W: np.ndarray, H: np.ndarray, X: np.ndarray, z: np.ndarray, cfg: Dict, truth: Optional[Dict] = None):
    K = W.shape[1]
    Q = qr_basis(W, K)
    D, centers_all, sigmas_all = gaussian_dictionary(z, cfg)
    # Energy of each Gaussian atom inside the accepted subspace.
    base_scores = np.sum((Q.T @ D) ** 2, axis=0)
    selected = []
    residual_Q = Q.copy()
    # Greedy atom selection with redundancy penalty; not a PCA step, only an interpretation rotation.
    for k in range(K):
        if residual_Q.shape[1] > 0:
            scores = np.sum((residual_Q.T @ D) ** 2, axis=0)
        else:
            scores = base_scores.copy()
        if selected:
            dist = np.min(np.abs(centers_all[:, None] - centers_all[np.array(selected)][None, :]), axis=1)
            sep = float(cfg["localization"]["omp_min_center_sep"])
            penalty = float(cfg["localization"]["omp_redundancy_penalty"]) * np.exp(-(dist / max(sep, 1e-6)) ** 2)
            scores = scores - penalty
        j = int(np.argmax(scores))
        selected.append(j)
        # update residual subspace against selected atom span
        A = D[:, selected]
        U = qr_basis(A, min(len(selected), A.shape[1]))
        Rproj = Q - U @ (U.T @ Q)
        if Rproj.size and norm(Rproj) > 1e-9:
            residual_Q = qr_basis(Rproj, max(1, K - len(selected))) if len(selected) < K else np.zeros((Q.shape[0], 0))
        else:
            residual_Q = np.zeros((Q.shape[0], 0))
    selected = np.array(selected, dtype=int)
    Dsel = D[:, selected]
    # Project selected Gaussian atoms back into accepted subspace and orthonormalize.
    Wloc_raw = Q @ (Q.T @ Dsel)
    Wloc = qr_basis(Wloc_raw, K)
    Xflat = center_cols(flatten_trials(X, None))
    Hloc = Xflat @ Wloc
    r2_loc = project_reconstruct_r2(X, Wloc, None)

    rows = []
    true_group_by_center = None
    if truth is not None:
        gc = np.array(truth["group_centers"])
        true_group_by_center = np.argmin(np.abs(centers_all[selected][:, None] - gc[None, :]), axis=1)
    for k, j in enumerate(selected):
        rows.append({
            "component": k,
            "gaussian_center": float(centers_all[j]),
            "gaussian_width": float(sigmas_all[j]),
            "subspace_energy": float(base_scores[j]),
            "nearest_true_group_by_center": int(true_group_by_center[k]) if true_group_by_center is not None else -1,
        })
    return {
        "W_localized": Wloc,
        "H_localized": Hloc,
        "centers": centers_all[selected],
        "widths": sigmas_all[selected],
        "r2_X_localized": float(r2_loc),
        "components_df": pd.DataFrame(rows),
    }


def discover_blocks(local: Dict, cfg: Dict, truth: Optional[Dict] = None):
    centers = np.asarray(local["centers"], float)
    H = np.asarray(local["H_localized"], float)
    K = len(centers)
    dep = polynomial_dependency_matrix(H, seed=int(cfg["block"]["seed"]))
    true_by_center = None
    if truth is not None:
        gc = np.array(truth["group_centers"])
        true_by_center = np.argmin(np.abs(centers[:, None] - gc[None, :]), axis=1)
    rows, assign = [], []
    for F in cfg["block"]["F_grid"]:
        F = int(F)
        if F > K:
            continue
        labels, ccent, sse = one_dim_kmeans(centers, F, n_starts=int(cfg["block"]["kmeans_starts"]), seed=int(cfg["block"]["seed"]) + F)
        sizes = np.array([np.sum(labels == f) for f in range(F)])
        wd = bd = wh = bh = 0.0
        nw = nb = 0
        for i in range(K):
            for j in range(i + 1, K):
                if labels[i] == labels[j]:
                    wd += abs(centers[i] - centers[j])
                    wh += dep[i, j]
                    nw += 1
                else:
                    bd += abs(centers[i] - centers[j])
                    bh += dep[i, j]
                    nb += 1
        within_dist = wd / max(nw, 1)
        between_dist = bd / max(nb, 1)
        within_dep = wh / max(nw, 1)
        between_dep = bh / max(nb, 1)
        sep_ratio = between_dist / (within_dist + 1e-6)
        dep_contrast = within_dep - between_dep
        spatial_score = math.tanh(sep_ratio / 3.0)
        dep_score = math.tanh(3.0 * dep_contrast)
        balance = float(1.0 - np.std(sizes) / (np.mean(sizes) + EPS)) if len(sizes) else 0.0
        min_size = int(sizes.min()) if len(sizes) else 0
        size_pen = 0.2 if min_size < int(cfg["block"]["min_block_size"]) else 0.0
        score = (
            float(cfg["block"]["w_spatial_weight"]) * spatial_score
            + float(cfg["block"]["h_dep_weight"]) * dep_score
            + float(cfg["block"]["stability_weight"]) * balance
            - float(cfg["block"]["complexity_penalty"]) * F
            - size_pen
        )
        ari = adjusted_rand_index(true_by_center, labels) if true_by_center is not None else np.nan
        rows.append({
            "F": F,
            "block_score": float(score),
            "spatial_score": float(spatial_score),
            "dep_score": float(dep_score),
            "spatial_separation_ratio": float(sep_ratio),
            "within_center_distance": float(within_dist),
            "between_center_distance": float(between_dist),
            "within_H_dependency": float(within_dep),
            "between_H_dependency": float(between_dep),
            "H_dependency_contrast": float(dep_contrast),
            "center_sse": float(sse),
            "min_block_size": min_size,
            "block_size_balance": float(balance),
            "ARI_nearest_true_group_center": float(ari),
            "cluster_centers": ";".join(f"{x:.4f}" for x in ccent),
            "block_sizes": ";".join(str(int(x)) for x in sizes),
        })
        for k in range(K):
            assign.append({
                "F": F,
                "component": k,
                "component_center": float(centers[k]),
                "block": int(labels[k]),
                "block_center": float(ccent[labels[k]]),
                "nearest_true_group_by_center": int(true_by_center[k]) if true_by_center is not None else -1,
            })
    df = pd.DataFrame(rows)
    if len(df):
        best_ix = df["block_score"].idxmax()
        df["selected"] = False
        df.loc[best_ix, "selected"] = True
    return df, pd.DataFrame(assign)


def score_block_partition(
    centers: np.ndarray,
    dep: np.ndarray,
    labels: np.ndarray,
    F: int,
    cfg: Dict,
    true_by_center: Optional[np.ndarray] = None,
) -> Dict:
    K = len(centers)
    sizes = np.array([np.sum(labels == f) for f in range(F)])
    wd = bd = wh = bh = 0.0
    nw = nb = 0
    for i in range(K):
        for j in range(i + 1, K):
            if labels[i] == labels[j]:
                wd += abs(centers[i] - centers[j])
                wh += dep[i, j]
                nw += 1
            else:
                bd += abs(centers[i] - centers[j])
                bh += dep[i, j]
                nb += 1
    within_dist = wd / max(nw, 1)
    between_dist = bd / max(nb, 1)
    within_dep = wh / max(nw, 1)
    between_dep = bh / max(nb, 1)
    sep_ratio = between_dist / (within_dist + 1e-6)
    dep_contrast = within_dep - between_dep
    spatial_score = math.tanh(sep_ratio / 3.0)
    dep_score = math.tanh(3.0 * dep_contrast)
    balance = float(1.0 - np.std(sizes) / (np.mean(sizes) + EPS)) if len(sizes) else 0.0
    min_size = int(sizes.min()) if len(sizes) else 0
    size_pen = 0.2 if min_size < int(cfg["block"]["min_block_size"]) else 0.0
    score = (
        float(cfg["block"]["w_spatial_weight"]) * spatial_score
        + float(cfg["block"]["h_dep_weight"]) * dep_score
        + float(cfg["block"]["stability_weight"]) * balance
        - float(cfg["block"]["complexity_penalty"]) * F
        - size_pen
    )
    ari = adjusted_rand_index(true_by_center, labels) if true_by_center is not None else np.nan
    return {
        "block_score": float(score),
        "spatial_score": float(spatial_score),
        "dep_score": float(dep_score),
        "spatial_separation_ratio": float(sep_ratio),
        "within_center_distance": float(within_dist),
        "between_center_distance": float(between_dist),
        "within_H_dependency": float(within_dep),
        "between_H_dependency": float(between_dep),
        "H_dependency_contrast": float(dep_contrast),
        "min_block_size": min_size,
        "block_size_balance": float(balance),
        "ARI_nearest_true_group_center": float(ari),
        "block_sizes": ";".join(str(int(x)) for x in sizes),
    }


def block_posterior_diagnostics(
    local: Dict,
    block_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    cfg: Dict,
    truth: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Approximate posterior over F and component co-association."""
    if len(block_df) == 0:
        return block_df, assign_df, pd.DataFrame(), pd.DataFrame()

    centers = np.asarray(local["centers"], float)
    H = np.asarray(local["H_localized"], float)
    K = len(centers)
    bpcfg = cfg["block_posterior"]
    rng = np.random.default_rng(int(bpcfg["seed"]))
    dep_base = polynomial_dependency_matrix(H, seed=int(cfg["block"]["seed"]))
    true_by_center = None
    if truth is not None:
        gc = np.array(truth["group_centers"])
        true_by_center = np.argmin(np.abs(centers[:, None] - gc[None, :]), axis=1)

    run_rows = []
    label_runs: List[np.ndarray] = []
    for run in range(int(bpcfg["n_runs"])):
        centers_run = centers + rng.normal(0.0, float(bpcfg["center_jitter_sd"]), size=K)
        dep_noise = rng.normal(0.0, float(bpcfg["dep_jitter_sd"]), size=(K, K))
        dep_noise = (dep_noise + dep_noise.T) / 2.0
        dep_run = np.clip(dep_base + dep_noise, 0.0, 1.0)
        for F in cfg["block"]["F_grid"]:
            F = int(F)
            if F > K:
                continue
            labels, ccent, sse = one_dim_kmeans(
                centers_run,
                F,
                n_starts=max(8, int(cfg["block"]["kmeans_starts"]) // 3),
                seed=int(cfg["block"]["seed"]) + 1009 * run + F,
            )
            stats = score_block_partition(centers_run, dep_run, labels, F, cfg, true_by_center)
            singleton_pen = float(bpcfg["singleton_penalty"]) if stats["min_block_size"] < int(cfg["block"]["min_block_size"]) else 0.0
            posterior_log_score = (
                stats["block_score"]
                - float(bpcfg["extra_complexity_penalty"]) * F
                + float(bpcfg["balance_weight"]) * stats["block_size_balance"]
                - singleton_pen
            )
            run_rows.append({
                "run": run,
                "F": F,
                "posterior_log_score": float(posterior_log_score),
                "center_sse": float(sse),
                "cluster_centers": ";".join(f"{x:.4f}" for x in ccent),
                **stats,
            })
            label_runs.append(labels.astype(int))

    runs_df = pd.DataFrame(run_rows)
    weights = softmax(runs_df["posterior_log_score"].to_numpy(float), float(bpcfg["temperature"]))
    runs_df["partition_weight"] = weights

    C = np.zeros((K, K), float)
    for weight, labels in zip(weights, label_runs):
        same = (labels[:, None] == labels[None, :]).astype(float)
        C += float(weight) * same
    C = np.clip(C, 0.0, 1.0)

    fpost = (
        runs_df.groupby("F", as_index=False)
        .agg(
            posterior_weight=("partition_weight", "sum"),
            posterior_log_score_mean=("posterior_log_score", "mean"),
            posterior_log_score_max=("posterior_log_score", "max"),
            run_count=("run", "count"),
            mean_ARI_nearest_true_group_center=("ARI_nearest_true_group_center", "mean"),
        )
        .sort_values("F")
        .reset_index(drop=True)
    )
    fsel = int(fpost.loc[fpost["posterior_weight"].idxmax(), "F"])
    fpost["selected_posterior"] = fpost["F"] == fsel

    updated_block = block_df.copy()
    updated_block["selected_by_score"] = updated_block["selected"].astype(bool)
    updated_block["selected"] = False
    updated_block.loc[updated_block["F"] == fsel, "selected"] = True
    updated_block = updated_block.merge(fpost[["F", "posterior_weight"]], on="F", how="left")
    updated_block["posterior_weight"] = updated_block["posterior_weight"].fillna(0.0)

    updated_assign = assign_df.copy()
    updated_assign["selected_posterior"] = updated_assign["F"] == fsel
    cdf = pd.DataFrame(C, columns=[f"component_{j}" for j in range(K)])
    cdf.insert(0, "component", np.arange(K))
    cdf.insert(1, "gaussian_center", centers)
    return updated_block, updated_assign, cdf, fpost


# =============================================================================
# Full space recovery diagnostics
# =============================================================================

def subspace_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.array([], float)
    QA = qr_basis(np.asarray(A, float))
    QB = qr_basis(np.asarray(B, float))
    s = svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def upper_tri_values(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, float)
    if M.shape[0] < 2:
        return np.array([], float)
    ix = np.triu_indices(M.shape[0], k=1)
    return M[ix]


def binary_auc_like(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    pos = np.asarray(scores_pos, float)
    neg = np.asarray(scores_neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def pairwise_recovery_stats(M: np.ndarray, labels: np.ndarray, prefix: str, pac_low: float = 0.1, pac_high: float = 0.9) -> Dict:
    M = np.asarray(M, float)
    labels = np.asarray(labels)
    within, between = [], []
    ambiguous = []
    entropy = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p = float(M[i, j])
            if labels[i] == labels[j]:
                within.append(p)
            else:
                between.append(p)
            ambiguous.append(pac_low < p < pac_high)
            pc = min(max(p, EPS), 1.0 - EPS)
            entropy.append(-(pc * math.log(pc) + (1.0 - pc) * math.log(1.0 - pc)))
    within = np.asarray(within, float)
    between = np.asarray(between, float)
    return {
        f"{prefix}_within_mean": float(np.mean(within)) if len(within) else np.nan,
        f"{prefix}_within_min": float(np.min(within)) if len(within) else np.nan,
        f"{prefix}_within_max": float(np.max(within)) if len(within) else np.nan,
        f"{prefix}_between_mean": float(np.mean(between)) if len(between) else np.nan,
        f"{prefix}_between_min": float(np.min(between)) if len(between) else np.nan,
        f"{prefix}_between_max": float(np.max(between)) if len(between) else np.nan,
        f"{prefix}_mean_diff": float(np.mean(within) - np.mean(between)) if len(within) and len(between) else np.nan,
        f"{prefix}_auc_like": binary_auc_like(within, between),
        f"{prefix}_PAC_0p1_0p9": float(np.mean(ambiguous)) if len(ambiguous) else np.nan,
        f"{prefix}_pair_entropy_mean": float(np.mean(entropy)) if len(entropy) else np.nan,
    }


def center_matching_diagnostics(est_centers: np.ndarray, truth: Dict) -> Tuple[pd.DataFrame, Dict]:
    true_centers = np.asarray(truth["component_centers"], float)
    est_centers = np.asarray(est_centers, float)
    cost = np.abs(est_centers[:, None] - true_centers[None, :])
    row, col = linear_sum_assignment(cost)
    rows = []
    for r, c in zip(row, col):
        rows.append({
            "estimated_component": int(r),
            "matched_true_component": int(c),
            "estimated_center": float(est_centers[r]),
            "true_center": float(true_centers[c]),
            "abs_center_error": float(abs(est_centers[r] - true_centers[c])),
            "true_group": int(truth["labels"][c]),
        })
    df = pd.DataFrame(rows).sort_values("estimated_component").reset_index(drop=True)
    summary = {
        "center_match_mae": float(df["abs_center_error"].mean()) if len(df) else np.nan,
        "center_match_max_error": float(df["abs_center_error"].max()) if len(df) else np.nan,
        "center_match_median_error": float(df["abs_center_error"].median()) if len(df) else np.nan,
    }
    return df, summary


def component_alignment_diagnostics(
    W_est: np.ndarray,
    H_est: np.ndarray,
    W_true: np.ndarray,
    H_true_flat: np.ndarray,
    truth: Dict,
    est_centers: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict]:
    K_est = W_est.shape[1]
    K_true = W_true.shape[1]
    if K_est == 0 or K_true == 0:
        return pd.DataFrame(), {}
    Wn = W_est / (norm(W_est, axis=0, keepdims=True) + EPS)
    WTn = W_true / (norm(W_true, axis=0, keepdims=True) + EPS)
    Hn = H_est / (norm(H_est, axis=0, keepdims=True) + EPS)
    HTn = H_true_flat / (norm(H_true_flat, axis=0, keepdims=True) + EPS)
    Wcorr = np.abs(Wn.T @ WTn)
    Hcorr = np.abs(Hn.T @ HTn)
    score = 0.55 * Wcorr + 0.45 * Hcorr
    row, col = linear_sum_assignment(-score)
    rows = []
    for r, c in zip(row, col):
        rows.append({
            "estimated_component": int(r),
            "matched_true_component": int(c),
            "W_abs_corr": float(Wcorr[r, c]),
            "H_abs_corr": float(Hcorr[r, c]),
            "combined_alignment_score": float(score[r, c]),
            "estimated_center": np.nan if est_centers is None or r >= len(est_centers) else float(est_centers[r]),
            "true_center": float(truth["component_centers"][c]),
            "center_error": np.nan if est_centers is None or r >= len(est_centers) else float(abs(est_centers[r] - truth["component_centers"][c])),
            "true_group": int(truth["labels"][c]),
        })
    df = pd.DataFrame(rows).sort_values("estimated_component").reset_index(drop=True)
    summary = {
        "component_alignment_W_abs_corr_mean": float(df["W_abs_corr"].mean()) if len(df) else np.nan,
        "component_alignment_W_abs_corr_min": float(df["W_abs_corr"].min()) if len(df) else np.nan,
        "component_alignment_H_abs_corr_mean": float(df["H_abs_corr"].mean()) if len(df) else np.nan,
        "component_alignment_H_abs_corr_min": float(df["H_abs_corr"].min()) if len(df) else np.nan,
        "component_alignment_score_mean": float(df["combined_alignment_score"].mean()) if len(df) else np.nan,
        "component_alignment_score_min": float(df["combined_alignment_score"].min()) if len(df) else np.nan,
    }
    return df, summary


def full_space_diagnostics(
    outdir: Path,
    X: np.ndarray,
    signal: np.ndarray,
    H_true: np.ndarray,
    W_true: np.ndarray,
    z_true: np.ndarray,
    truth: Dict,
    iter_res: Dict,
    local: Dict,
    coassoc_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    block_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    H_true_flat = H_true.reshape(-1, H_true.shape[-1])
    W_est = np.asarray(iter_res["W"], float)
    H_est = np.asarray(iter_res["H"], float)
    W_loc = np.asarray(local["W_localized"], float)
    H_loc = np.asarray(local["H_localized"], float)
    centers = np.asarray(local["centers"], float)
    labels_true = np.asarray(truth["labels"], int)
    C = coassoc_df.drop(columns=["component", "gaussian_center"]).to_numpy(float) if len(coassoc_df) else np.zeros((W_loc.shape[1], W_loc.shape[1]))
    D_raw = polynomial_dependency_matrix(H_est, seed=int(cfg["block"]["seed"]) + 17) if H_est.shape[1] else np.zeros((0, 0))
    D_loc = polynomial_dependency_matrix(H_loc, seed=int(cfg["block"]["seed"]) + 29) if H_loc.shape[1] else np.zeros((0, 0))
    D_true = polynomial_dependency_matrix(H_true_flat, seed=int(cfg["block"]["seed"]) + 41)
    same_true = (labels_true[:, None] == labels_true[None, :]).astype(float)
    group_centers = np.asarray(truth["group_centers"], float)
    est_labels_by_center = np.argmin(np.abs(centers[:, None] - group_centers[None, :]), axis=1) if len(centers) else np.array([], int)

    w_angles = subspace_angles_deg(W_est, W_true)
    wloc_angles = subspace_angles_deg(W_loc, W_true)
    h_angles = subspace_angles_deg(H_est, H_true_flat)
    hloc_angles = subspace_angles_deg(H_loc, H_true_flat)
    center_df, center_summary = center_matching_diagnostics(centers, truth)
    align_df, align_summary = component_alignment_diagnostics(W_loc, H_loc, W_true, H_true_flat, truth, centers)

    selected_F = int(block_df.loc[block_df["selected"], "F"].iloc[0]) if len(block_df) else -1
    selected_assign = assign_df[(assign_df["F"] == selected_F) & (assign_df.get("selected_posterior", False) == True)].copy() if len(assign_df) else pd.DataFrame()
    block_ari = np.nan
    if len(selected_assign):
        block_ari = adjusted_rand_index(
            selected_assign.sort_values("component")["nearest_true_group_by_center"].to_numpy(int),
            selected_assign.sort_values("component")["block"].to_numpy(int),
        )

    summary = {
        "W_raw_subspace_angle_mean_deg": float(np.mean(w_angles)) if len(w_angles) else np.nan,
        "W_raw_subspace_angle_max_deg": float(np.max(w_angles)) if len(w_angles) else np.nan,
        "W_localized_subspace_angle_mean_deg": float(np.mean(wloc_angles)) if len(wloc_angles) else np.nan,
        "W_localized_subspace_angle_max_deg": float(np.max(wloc_angles)) if len(wloc_angles) else np.nan,
        "H_raw_subspace_angle_mean_deg": float(np.mean(h_angles)) if len(h_angles) else np.nan,
        "H_raw_subspace_angle_max_deg": float(np.max(h_angles)) if len(h_angles) else np.nan,
        "H_localized_subspace_angle_mean_deg": float(np.mean(hloc_angles)) if len(hloc_angles) else np.nan,
        "H_localized_subspace_angle_max_deg": float(np.max(hloc_angles)) if len(hloc_angles) else np.nan,
        "W_raw_canonical_corr_min": float(np.cos(np.radians(np.max(w_angles)))) if len(w_angles) else np.nan,
        "W_localized_canonical_corr_min": float(np.cos(np.radians(np.max(wloc_angles)))) if len(wloc_angles) else np.nan,
        "H_raw_canonical_corr_min": float(np.cos(np.radians(np.max(h_angles)))) if len(h_angles) else np.nan,
        "H_localized_canonical_corr_min": float(np.cos(np.radians(np.max(hloc_angles)))) if len(hloc_angles) else np.nan,
        "selected_block_ARI_vs_nearest_true_group": float(block_ari),
        **center_summary,
        **align_summary,
        **pairwise_recovery_stats(C, est_labels_by_center, "coassociation"),
        **pairwise_recovery_stats(D_loc, est_labels_by_center, "D_localized"),
        **pairwise_recovery_stats(D_true, labels_true, "D_true"),
    }

    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    summary_df = pd.DataFrame(rows)
    angles_df = pd.DataFrame({
        "index": np.arange(max(len(w_angles), len(wloc_angles), len(h_angles), len(hloc_angles))),
        "W_raw_angle_deg": pd.Series(w_angles),
        "W_localized_angle_deg": pd.Series(wloc_angles),
        "H_raw_angle_deg": pd.Series(h_angles),
        "H_localized_angle_deg": pd.Series(hloc_angles),
    })

    np.savez_compressed(
        outdir / "v13_2_1_latent_matrices.npz",
        W_raw=W_est,
        H_raw=H_est,
        W_localized=W_loc,
        H_localized=H_loc,
        W_true=W_true,
        H_true_flat=H_true_flat,
        z_true=z_true,
        component_centers=centers,
        true_component_centers=np.asarray(truth["component_centers"], float),
        true_labels=labels_true,
        estimated_labels_by_center=est_labels_by_center,
        D_raw=D_raw,
        D_localized=D_loc,
        D_true=D_true,
        same_group_true=same_true,
        block_coassociation=C,
        X_shape=np.asarray(X.shape),
        signal_shape=np.asarray(signal.shape),
    )
    return summary_df, angles_df, center_df, align_df

# =============================================================================
# Plots and reports
# =============================================================================

def save_plots(
    outdir: Path,
    audit_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    local: Dict,
    block_df: pd.DataFrame,
    truth: Dict,
    cfg: Dict,
    comp_df: Optional[pd.DataFrame] = None,
    block_post_df: Optional[pd.DataFrame] = None,
    coassoc_df: Optional[pd.DataFrame] = None,
):
    dpi = int(cfg["plot"]["dpi"])
    plt.figure(figsize=(7.2, 4.2))
    x = np.arange(len(audit_df))
    plt.plot(x, audit_df["W_eff_rank"], marker="o", label="W effective rank")
    plt.plot(x, audit_df["H_eff_rank"], marker="o", label="H effective rank")
    plt.xticks(x, [f"group {g}" for g in audit_df["group"]])
    plt.ylabel("effective rank")
    plt.title("Identifiability audit")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "v13_2_1_identifiability_rank.png", dpi=dpi); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(trace_df["iteration"], trace_df["K_current"], marker="o")
    plt.axhline(truth["Ktrue"], linestyle="--", label="true K")
    plt.xlabel("outer iteration"); plt.ylabel("accepted K")
    plt.title("True residual-driven iterative K expansion")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "v13_2_1_iterative_K_trace.png", dpi=dpi); plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(trace_df["iteration"], trace_df["best_gain_val"], marker="o", label="best validation gain")
    plt.plot(trace_df["iteration"], trace_df["best_gain_train"], marker="o", label="best train gain")
    plt.axhline(float(cfg["iterative"]["min_gain_val"]), linestyle="--", label="val gain threshold")
    plt.xlabel("outer iteration"); plt.ylabel("incremental R2 gain")
    plt.title("Held-out acceptance gain")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "v13_2_1_iterative_gain_trace.png", dpi=dpi); plt.close()

    if len(cand_df):
        plt.figure(figsize=(7.2, 4.2))
        best = cand_df.groupby("iteration")["eig_over_null"].max()
        plt.plot(best.index, best.values, marker="o")
        plt.axhline(float(cfg["iterative"]["min_eig_over_null"]), linestyle="--", label="threshold")
        plt.xlabel("outer iteration"); plt.ylabel("residual eig / null")
        plt.title("Residual reliability evidence")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "v13_2_1_residual_reliability_trace.png", dpi=dpi); plt.close()

    plt.figure(figsize=(8.0, 3.6))
    centers = local["centers"]
    plt.scatter(centers, np.zeros_like(centers), label="localized accepted components")
    for c in truth["group_centers"]:
        plt.axvline(c, linestyle="--", alpha=0.6)
    plt.yticks([]); plt.xlabel("z / Gaussian center")
    plt.title("Component-level Gaussian localization")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "v13_2_1_localized_component_centers.png", dpi=dpi); plt.close()

    if len(block_df):
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(block_df["F"], block_df["block_score"], marker="o", label="block score")
        plt.plot(block_df["F"], block_df["spatial_score"], marker="o", label="spatial")
        plt.plot(block_df["F"], block_df["dep_score"], marker="o", label="H dependency")
        if "posterior_weight" in block_df.columns:
            plt.plot(block_df["F"], block_df["posterior_weight"], marker="o", label="posterior weight")
        plt.axvline(truth["Ftrue"], linestyle="--", label="true F")
        plt.xlabel("F"); plt.ylabel("score")
        plt.title("Block posterior after iterative K")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "v13_2_1_F_block_posterior.png", dpi=dpi); plt.close()

    if comp_df is not None and len(comp_df):
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(comp_df["component"], comp_df["inclusion_prob"])
        plt.axhline(float(cfg["posterior"]["inclusion_threshold"]), linestyle="--", label="active threshold")
        plt.xlabel("component"); plt.ylabel("posterior inclusion probability")
        plt.ylim(0, 1.05)
        plt.title("ARD-style component posterior")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "v13_2_1_component_inclusion_posterior.png", dpi=dpi); plt.close()

    if block_post_df is not None and len(block_post_df):
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(block_post_df["F"], block_post_df["posterior_weight"])
        plt.axvline(truth["Ftrue"], linestyle="--", label="true F")
        plt.xlabel("F"); plt.ylabel("posterior mass")
        plt.title("Posterior over block count F")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "v13_2_1_F_posterior_mass.png", dpi=dpi); plt.close()

    if coassoc_df is not None and len(coassoc_df):
        C = coassoc_df.drop(columns=["component", "gaussian_center"]).to_numpy(float)
        plt.figure(figsize=(5.8, 5.2))
        plt.imshow(C, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(label="P(same block)")
        plt.xlabel("component"); plt.ylabel("component")
        plt.title("Block co-association posterior")
        plt.tight_layout()
        plt.savefig(outdir / "v13_2_1_block_coassociation_heatmap.png", dpi=dpi); plt.close()


def write_report(
    outdir: Path,
    cfg: Dict,
    truth: Dict,
    audit_df: pd.DataFrame,
    iter_res: Dict,
    local: Dict,
    block_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    comp_summary: Dict,
    block_post_df: pd.DataFrame,
    space_diag_df: pd.DataFrame,
    subspace_angles_df: pd.DataFrame,
    center_match_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    prune_info: Dict,
    elapsed: float,
):
    Ksel = int(iter_res["W"].shape[1])
    Fsel = int(block_df.loc[block_df["selected"], "F"].iloc[0]) if len(block_df) else None
    Fscore = int(block_df.loc[block_df["selected_by_score"], "F"].iloc[0]) if len(block_df) and "selected_by_score" in block_df.columns else Fsel
    report = {
        "version": cfg["version"],
        "elapsed_sec": float(elapsed),
        "design_notes": [
            "v13.2 keeps the v13.1 residual-driven iterative K core.",
            "Accepted components get empirical-Bayes posterior diagnostics.",
            "ARD-style inclusion probabilities estimate an active K posterior.",
            "Optional low-posterior pruning is applied before localization.",
            "Block discovery is upgraded to a perturbation-based co-association posterior.",
            "F_selected is selected by posterior mass, while F_selected_by_score records the old hard score choice."
        ],
        "truth": {
            "Ktrue": int(truth["Ktrue"]),
            "Ftrue": int(truth["Ftrue"]),
            "comps_per_group": int(truth["comps_per_group"]),
            "group_centers": [float(x) for x in truth["group_centers"]],
            "component_centers": [float(x) for x in truth["component_centers"]],
        },
        "selected": {
            "K_selected": Ksel,
            "K_eff_posterior": int(comp_summary.get("K_eff_posterior", Ksel)),
            "K_eff_soft": float(comp_summary.get("K_eff_soft", Ksel)),
            "F_selected": Fsel,
            "F_selected_by_score": Fscore,
            "all_data_R2": float(iter_res["all_r2"]),
            "localized_R2_X": float(local["r2_X_localized"]),
        },
        "posterior_summary": {
            **{k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
               for k, v in comp_summary.items()},
            "pruning": prune_info,
            "F_posterior": block_post_df.to_dict(orient="records") if len(block_post_df) else [],
        },
        "space_diagnostics": {
            row["metric"]: (float(row["value"]) if pd.notna(row["value"]) else None)
            for _, row in space_diag_df.iterrows()
        } if len(space_diag_df) else {},
        "audit_summary": {
            "W_eff_rank_mean": float(audit_df["W_eff_rank"].mean()),
            "H_eff_rank_mean": float(audit_df["H_eff_rank"].mean()),
            "W_s1_energy_mean": float(audit_df["W_s1_energy"].mean()),
            "H_s1_energy_mean": float(audit_df["H_s1_energy"].mean()),
        },
        "config": cfg,
    }
    with open(outdir / "v13_2_1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# v13.2.1 full diagnostics block discovery report\n")
    lines.append("## Core result\n")
    lines.append(f"- True K: {truth['Ktrue']}\n")
    lines.append(f"- Selected K: {Ksel}\n")
    lines.append(f"- Posterior active K: {comp_summary.get('K_eff_posterior', Ksel)}\n")
    lines.append(f"- Soft active K: {comp_summary.get('K_eff_soft', Ksel):.3f}\n")
    lines.append(f"- True F: {truth['Ftrue']}\n")
    lines.append(f"- Selected F by posterior: {Fsel}\n")
    lines.append(f"- Selected F by hard score: {Fscore}\n")
    lines.append(f"- All-data R2: {iter_res['all_r2']:.4f}\n")
    lines.append(f"- Localized R2: {local['r2_X_localized']:.4f}\n")
    lines.append(f"- Posterior pruned components: {prune_info.get('pruned_components', [])}\n")
    lines.append("\n## Identifiability audit\n")
    lines.append(audit_df.to_string(index=False))
    lines.append("\n\n## Component posterior\n")
    lines.append(comp_df.to_string(index=False) if len(comp_df) else "No components")
    lines.append("\n\n## Space diagnostics\n")
    lines.append(space_diag_df.to_string(index=False) if len(space_diag_df) else "No space diagnostics")
    lines.append("\n\n## Subspace angles\n")
    lines.append(subspace_angles_df.to_string(index=False) if len(subspace_angles_df) else "No subspace angles")
    lines.append("\n\n## Center matching\n")
    lines.append(center_match_df.to_string(index=False) if len(center_match_df) else "No center matching")
    lines.append("\n\n## Component alignment\n")
    lines.append(alignment_df.to_string(index=False) if len(alignment_df) else "No component alignment")
    lines.append("\n\n## Iterative K trace\n")
    lines.append(iter_res["trace"].to_string(index=False))
    lines.append("\n\n## F posterior\n")
    lines.append(block_post_df.to_string(index=False) if len(block_post_df) else "No F posterior")
    lines.append("\n\n## Block candidate surface\n")
    lines.append(block_df.to_string(index=False) if len(block_df) else "No blocks")
    with open(outdir / "v13_2_1_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report

# =============================================================================
# Main runner
# =============================================================================

def run_pipeline(cfg: Dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(outdir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    X, signal, H_true, W_true, z_true, truth = make_clustered_gaussian_toy(cfg)
    audit_df = identifiability_audit(H_true, W_true, truth["labels"])
    audit_df.to_csv(outdir / "v13_2_1_identifiability_audit.csv", index=False)

    iter_res = true_iterative_K_expansion(X, signal, cfg, outdir)
    iter_res["trace"].to_csv(outdir / "v13_2_1_iterative_trace.csv", index=False)
    iter_res["candidates"].to_csv(outdir / "v13_2_1_candidate_trace.csv", index=False)

    pre_comp_df, _ = component_posterior_diagnostics(X, iter_res, cfg, local=None, truth=truth)
    iter_res, prune_info = apply_posterior_pruning(X, iter_res, pre_comp_df, cfg)

    local = localize_subspace_gaussian_omp(iter_res["W"], iter_res["H"], X, z_true, cfg, truth)
    local["components_df"].to_csv(outdir / "v13_2_1_gaussian_components.csv", index=False)

    comp_df, comp_summary = component_posterior_diagnostics(X, iter_res, cfg, local=local, truth=truth)
    comp_summary["posterior_pruned"] = int(prune_info.get("n_pruned", 0))
    comp_df.to_csv(outdir / "v13_2_1_component_posterior.csv", index=False)

    block_df, assign_df = discover_blocks(local, cfg, truth)
    block_df, assign_df, coassoc_df, block_post_df = block_posterior_diagnostics(local, block_df, assign_df, cfg, truth)
    block_df.to_csv(outdir / "v13_2_1_block_candidates.csv", index=False)
    assign_df.to_csv(outdir / "v13_2_1_block_assignment.csv", index=False)
    coassoc_df.to_csv(outdir / "v13_2_1_block_coassociation.csv", index=False)
    block_post_df.to_csv(outdir / "v13_2_1_block_posterior_summary.csv", index=False)

    space_diag_df, subspace_angles_df, center_match_df, alignment_df = full_space_diagnostics(
        outdir,
        X,
        signal,
        H_true,
        W_true,
        z_true,
        truth,
        iter_res,
        local,
        coassoc_df,
        assign_df,
        block_df,
        cfg,
    )
    space_diag_df.to_csv(outdir / "v13_2_1_space_diagnostics.csv", index=False)
    subspace_angles_df.to_csv(outdir / "v13_2_1_subspace_angles.csv", index=False)
    center_match_df.to_csv(outdir / "v13_2_1_center_matching.csv", index=False)
    alignment_df.to_csv(outdir / "v13_2_1_component_alignment.csv", index=False)

    elapsed = time.time() - t0
    save_plots(
        outdir,
        audit_df,
        iter_res["trace"],
        iter_res["candidates"],
        local,
        block_df,
        truth,
        cfg,
        comp_df=comp_df,
        block_post_df=block_post_df,
        coassoc_df=coassoc_df,
    )
    report = write_report(
        outdir,
        cfg,
        truth,
        audit_df,
        iter_res,
        local,
        block_df,
        comp_df,
        comp_summary,
        block_post_df,
        space_diag_df,
        subspace_angles_df,
        center_match_df,
        alignment_df,
        prune_info,
        elapsed,
    )
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(f"Outputs saved to: {outdir}", flush=True)
    return report


def load_config(args) -> Dict:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = deep_update(DEFAULT_CONFIG, json.load(f))
    else:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.quick:
        cfg = deep_update(cfg, QUICK_OVERRIDE)
    if args.full:
        cfg = deep_update(cfg, FULL_OVERRIDE)
    if args.wide:
        cfg = deep_update(cfg, WIDE_OVERRIDE)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="v13.2.1 full-diagnostics Bayesian-coupled iterative block discovery")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="v13_2_1_outputs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args)
    run_pipeline(cfg, Path(args.outdir))


if __name__ == "__main__":
    main()
