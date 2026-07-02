#!/usr/bin/env python3
"""
Differentiable, in-training-loop re-implementation of the offline style metric
defined in ``precompute_style_disp.py``.

Given a batch of (decoded) blendshapes ``[B, T, 58]`` and a validity mask
``[B, T]``, :class:`StyleMeasurer` runs a FLAME forward pass and returns the
z-scored per-region (disp, speed) p95 style vector ``[B, N_REGIONS * 2]`` with
gradients flowing back to the input blendshapes through FLAME.

This is used for the style-consistency loss: decode the same content codes with
a *swapped* style vector, measure the style actually produced, and require it to
match the requested style.  Because the metric is just vertex-displacement norms
plus a (differentiable) percentile, the decoder receives a clean gradient — to
raise e.g. ``lips_disp`` it must physically make the lips move more.

The computation must mirror the offline script exactly:
  * global head pose (gpose) is zeroed — only expression + jaw drive the verts,
  * disp(t)  = mean_{v in region} || verts(t) - verts_neutral ||,
  * speed(t) = mean_{v in region} || verts(t) - verts(t-1) ||,  speed(0) = 0,
  * per-chunk p95 over valid frames, then z-scored with the saved train stats.
"""
import numpy as np
import torch
import torch.nn as nn

from flame_model.FLAME import FLAMEModel
from dataset.precompute_style_disp import (
    MASKS_PATH,
    N_REGIONS,
    REGION_NAMES,
    load_region_masks,
)

# Channel layout (per timestep, 58 dims total) — matches losses.py / precompute.
EXPR_SLICE = slice(0, 50)
GPOSE_SLICE = slice(50, 53)
JAW_SLICE = slice(53, 56)


class StyleMeasurer(nn.Module):
    def __init__(self, stats_path, masks_path=MASKS_PATH, n_shape=300, n_exp=50,
                 quantile=0.95):
        super().__init__()
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.quantile = quantile

        self.flame = FLAMEModel(n_shape=n_shape, n_exp=n_exp, no_lmks=True)
        self.flame.eval()
        for p in self.flame.parameters():
            p.requires_grad_(False)

        # Region vertex-index masks (LongTensors) as non-persistent buffers so
        # they move with .to(device) but are not saved in checkpoints.
        masks = load_region_masks(masks_path)
        self._region_names = list(REGION_NAMES)
        for name in self._region_names:
            self.register_buffer(f"mask_{name}", masks[name], persistent=False)

        # Neutral-pose vertices (everything zero).
        with torch.no_grad():
            neutral = self.flame.forward(
                shape_params=torch.zeros(1, n_shape),
                expression_params=torch.zeros(1, n_exp),
                pose_params=torch.zeros(1, 6),
                eye_pose_params=torch.zeros(1, 6),
            )
            if isinstance(neutral, tuple):
                neutral = neutral[0]
        self.register_buffer("neutral_verts", neutral.squeeze(0), persistent=False)  # (V, 3)

        # z-score stats written by precompute_style_disp.py, shape (R, 2) where
        # R = N_REGIONS vertex regions, optionally +1 for a 'pose' region.
        stats = np.load(stats_path)
        self.register_buffer(
            "style_mean", torch.from_numpy(stats["mean"].astype(np.float32)), persistent=False
        )
        self.register_buffer(
            "style_std", torch.from_numpy(stats["std"].astype(np.float32)), persistent=False
        )

        # Pose is measured in parameter space from the decoded gpose channels
        # (||gpose(t)|| and its frame-to-frame change), NOT as vertex
        # displacement — head rotation rigidly moves the whole mesh.  It is
        # present iff the stats file carries an extra region row beyond the
        # vertex regions, matching precompute_style_disp.py --include_pose.
        self.n_regions = int(self.style_mean.shape[0])
        self.include_pose = self.n_regions > N_REGIONS

    def _region_idx(self, name):
        return getattr(self, f"mask_{name}")

    def forward(self, blendshapes, mask=None):
        """blendshapes: [B, T, 58], mask: [B, T] (bool/0-1) -> style [B, n_regions*2]."""
        B = blendshapes.shape[0]
        expr = blendshapes[..., EXPR_SLICE]   # [B, T, 50]
        gpose = blendshapes[..., GPOSE_SLICE] # [B, T, 3]
        jaw = blendshapes[..., JAW_SLICE]     # [B, T, 3]

        out = []
        for b in range(B):
            if mask is not None:
                valid = mask[b].bool()
                expr_b = expr[b][valid]
                gpose_b = gpose[b][valid]
                jaw_b = jaw[b][valid]
            else:
                expr_b = expr[b]
                gpose_b = gpose[b]
                jaw_b = jaw[b]

            L = expr_b.shape[0]
            if L < 2:
                out.append(blendshapes.new_zeros(self.n_regions * 2))
                continue

            # gpose zeroed: only expression + jaw drive the vertices.
            pose_b = torch.cat([jaw_b.new_zeros(L, 3), jaw_b], dim=-1)  # [L, 6]
            verts = self.flame.forward(
                shape_params=expr_b.new_zeros(L, self.n_shape),
                expression_params=expr_b,
                pose_params=pose_b,
                eye_pose_params=expr_b.new_zeros(L, 6),
            )
            if isinstance(verts, tuple):
                verts = verts[0]                                  # [L, V, 3]

            delta = verts - self.neutral_verts.unsqueeze(0)       # [L, V, 3]
            shifted = torch.cat([verts[:1], verts[:-1]], dim=0)   # speed(0)=0
            vel = verts - shifted                                 # [L, V, 3]

            feats = []
            for name in self._region_names:
                idx = self._region_idx(name)
                disp_r = delta[:, idx, :].norm(dim=-1).mean(dim=-1)   # [L]
                speed_r = vel[:, idx, :].norm(dim=-1).mean(dim=-1)    # [L]
                feats.append(torch.stack([disp_r, speed_r], dim=-1))  # [L, 2]

            if self.include_pose:
                # Parameter-space head-rotation metric from the decoded gpose,
                # mirroring precompute_style_disp.py exactly.  disp is the
                # excursion from the clip's MEAN orientation (range of motion),
                # so a static held turn scores ~0 and a high-disp request can
                # only be met by sustained head movement, not a single turn.
                g_ref = gpose_b.mean(dim=0, keepdim=True)            # [1, 3] mean orientation
                pose_disp = (gpose_b - g_ref).norm(dim=-1)          # [L]
                gpose_shift = torch.cat([gpose_b[:1], gpose_b[:-1]], dim=0)
                pose_speed = (gpose_b - gpose_shift).norm(dim=-1)    # speed(0)=0
                feats.append(torch.stack([pose_disp, pose_speed], dim=-1))  # [L, 2]

            feats = torch.stack(feats, dim=1)                         # [L, n_regions, 2]

            p95 = torch.quantile(feats, self.quantile, dim=0)         # [n_regions, 2]
            if self.include_pose:
                # Pose disp = head range of motion, aggregated as the temporal
                # std (RMS excursion from the clip mean) over ALL frames, NOT the
                # p95 tail.  A p95 only constrains ~5% of frames, so it is met by
                # a single brief swing; the RMS forces the head to spend a large
                # fraction of the clip away from centre -> sustained motion.
                pose_exc = feats[:, N_REGIONS, 0]                     # [L]
                p95[N_REGIONS, 0] = pose_exc.pow(2).mean().clamp_min(1e-12).sqrt()
            z = (p95 - self.style_mean) / self.style_std              # [n_regions, 2]
            out.append(z.flatten())                                   # [n_regions*2]

        return torch.stack(out, dim=0)                                # [B, n_regions*2]
