"""Input-only augmentations for stage-1 VQ training.

All augmentations operate on a single sequence numpy array of shape [T, 58]
with channel layout:
    [0:50]  expression
    [50:53] global pose (gpose)
    [53:56] jaw
    [56:58] eyelids (placeholder)

Augmentations are applied to the INPUT only. The target stays clean.
"""
from __future__ import annotations

import numpy as np


EXPR_SLICE = slice(0, 50)
GPOSE_SLICE = slice(50, 53)
JAW_SLICE = slice(53, 56)
EYELID_SLICE = slice(56, 58)


def _moving_average(x, window):
    # x: [T, D], window: int
    if window <= 1 or x.shape[0] < window:
        return x
    kernel = np.ones((window,), dtype=x.dtype) / float(window)
    out = np.empty_like(x)
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")[: x.shape[0]]
    return out


def _interpolate_missing(x, mask_drop):
    """Replace timesteps where mask_drop=True with linear interpolation from neighbors."""
    T = x.shape[0]
    if not mask_drop.any():
        return x
    idx = np.arange(T)
    valid = ~mask_drop
    if valid.sum() < 2:
        return x
    out = x.copy()
    valid_idx = idx[valid]
    for d in range(x.shape[1]):
        out[:, d] = np.interp(idx, valid_idx, x[valid_idx, d])
    return out


class Augmentor:
    def __init__(self, cfg=None):
        cfg = cfg or {}

        # Gaussian jitter
        self.jitter_prob = float(cfg.get("jitter_prob", 0.7))
        self.jitter_std_expr = float(cfg.get("jitter_std_expr", 0.010))
        self.jitter_std_jaw = float(cfg.get("jitter_std_jaw", 0.006))
        self.jitter_std_gpose = float(cfg.get("jitter_std_gpose", 0.003))

        # Frame dropout + interpolation
        self.dropout_prob = float(cfg.get("dropout_prob", 0.5))
        self.dropout_ratio = float(cfg.get("dropout_ratio", 0.10))

        # Short-window temporal smoothing corruption
        self.smooth_prob = float(cfg.get("smooth_prob", 0.4))
        self.smooth_window_min = int(cfg.get("smooth_window_min", 8))
        self.smooth_window_max = int(cfg.get("smooth_window_max", 24))
        self.smooth_segment_min = int(cfg.get("smooth_segment_min", 8))
        self.smooth_segment_max = int(cfg.get("smooth_segment_max", 32))

        # Time resample perturbation
        self.resample_prob = float(cfg.get("resample_prob", 0.25))
        self.resample_min = float(cfg.get("resample_min", 0.95))
        self.resample_max = float(cfg.get("resample_max", 1.05))

        # Contiguous segment masking + interpolation
        self.segmask_prob = float(cfg.get("segmask_prob", 0.4))
        self.segmask_n_min = int(cfg.get("segmask_n_min", 1))
        self.segmask_n_max = int(cfg.get("segmask_n_max", 3))
        self.segmask_len_min = int(cfg.get("segmask_len_min", 2))
        self.segmask_len_max = int(cfg.get("segmask_len_max", 6))

    def _gaussian_jitter(self, x, rng):
        noise = np.zeros_like(x)
        noise[:, EXPR_SLICE] = rng.normal(0.0, self.jitter_std_expr, size=x[:, EXPR_SLICE].shape)
        noise[:, JAW_SLICE] = rng.normal(0.0, self.jitter_std_jaw, size=x[:, JAW_SLICE].shape)
        noise[:, GPOSE_SLICE] = rng.normal(0.0, self.jitter_std_gpose, size=x[:, GPOSE_SLICE].shape)
        # Eyelids are placeholder values -> leave untouched.
        return x + noise.astype(x.dtype)

    def _frame_dropout(self, x, rng):
        T = x.shape[0]
        n_drop = max(1, int(round(self.dropout_ratio * T)))
        drop_idx = rng.choice(T, size=n_drop, replace=False)
        mask_drop = np.zeros(T, dtype=bool)
        mask_drop[drop_idx] = True
        # Keep the first and last frames valid for stable interpolation.
        mask_drop[0] = False
        mask_drop[-1] = False
        return _interpolate_missing(x, mask_drop)

    def _smooth_segment(self, x, rng):
        T = x.shape[0]
        if T < self.smooth_segment_min + 2:
            return x
        seg_len = int(rng.integers(self.smooth_segment_min, max(self.smooth_segment_min + 1, min(self.smooth_segment_max, T - 1))))
        start = int(rng.integers(0, max(1, T - seg_len)))
        end = start + seg_len
        window = int(rng.integers(self.smooth_window_min, self.smooth_window_max + 1))
        out = x.copy()
        out[start:end] = _moving_average(out[start:end], window=window)
        return out

    def _time_resample(self, x, rng):
        T = x.shape[0]
        scale = float(rng.uniform(self.resample_min, self.resample_max))
        new_T = max(2, int(round(T * scale)))
        idx_src = np.linspace(0.0, T - 1, num=new_T)
        # Resample to new_T then back to T to keep tensor shape consistent.
        warped = np.empty((new_T, x.shape[1]), dtype=x.dtype)
        base = np.arange(T)
        for d in range(x.shape[1]):
            warped[:, d] = np.interp(idx_src, base, x[:, d])
        idx_back = np.linspace(0.0, new_T - 1, num=T)
        out = np.empty_like(x)
        base2 = np.arange(new_T)
        for d in range(x.shape[1]):
            out[:, d] = np.interp(idx_back, base2, warped[:, d])
        return out

    def _segment_mask(self, x, rng):
        T = x.shape[0]
        n_seg = int(rng.integers(self.segmask_n_min, self.segmask_n_max + 1))
        mask_drop = np.zeros(T, dtype=bool)
        for _ in range(n_seg):
            seg_len = int(rng.integers(self.segmask_len_min, self.segmask_len_max + 1))
            if T - seg_len - 2 <= 1:
                continue
            start = int(rng.integers(1, T - seg_len - 1))
            mask_drop[start : start + seg_len] = True
        mask_drop[0] = False
        mask_drop[-1] = False
        return _interpolate_missing(x, mask_drop)

    def __call__(self, x, rng=None):
        """Apply augmentations to input. Returns a new array of same shape as x."""
        if rng is None:
            rng = np.random.default_rng()

        out = x.copy()
        if self.jitter_prob > 0 and rng.random() < self.jitter_prob:
            out = self._gaussian_jitter(out, rng)
        if self.dropout_prob > 0 and rng.random() < self.dropout_prob:
            out = self._frame_dropout(out, rng)
        if self.smooth_prob > 0 and rng.random() < self.smooth_prob:
            out = self._smooth_segment(out, rng)
        if self.resample_prob > 0 and rng.random() < self.resample_prob:
            out = self._time_resample(out, rng)
        if self.segmask_prob > 0 and rng.random() < self.segmask_prob:
            out = self._segment_mask(out, rng)
        return out


def build_augmentor_from_cfg(cfg):
    """Build an Augmentor from a flat cfg namespace if enabled, else return None."""
    if not bool(getattr(cfg, "augment", False)):
        return None
    keys = [
        "jitter_prob", "jitter_std_expr", "jitter_std_jaw", "jitter_std_gpose",
        "dropout_prob", "dropout_ratio",
        "smooth_prob", "smooth_window_min", "smooth_window_max",
        "smooth_segment_min", "smooth_segment_max",
        "resample_prob", "resample_min", "resample_max",
        "segmask_prob", "segmask_n_min", "segmask_n_max",
        "segmask_len_min", "segmask_len_max",
    ]
    params = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}
    return Augmentor(params)
