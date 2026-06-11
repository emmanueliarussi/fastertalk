#!/usr/bin/env python3
"""
Offline precompute script for per-frame region displacement features.

For each .npz file in the dataset, runs a FLAME forward pass and writes a
sidecar  <stem>.style_disp.npy  (shape [T, N_REGIONS, 2], float32) containing
for every frame and every semantic face region two scalars:

  [..., 0]  disp(t)  = mean_{v in M_r} || verts(t,v) - verts_neutral(v) ||_2
                       "how deformed is this region right now" (position)

  [..., 1]  speed(t) = mean_{v in M_r} || verts(t,v) - verts(t-1,v) ||_2
                       "how fast is this region moving"        (velocity)
                       speed(0) is set to 0.

After processing all files, writes  style_disp_stats.npz  to the npz root
with per-region per-feature (mean, std) of the per-chunk p95 values computed
over the train split.  These are used for z-score normalization at training
time.  Stats arrays have shape (N_REGIONS, 2).

Regions
-------
  0  lips         – mouth / smile expressiveness
  1  forehead     – brow raise
  2  eyes         – eye widening / squint  (left_eye_region ∪ right_eye_region)

Usage
-----
  python dataset/precompute_style_disp.py \\
      --data_root /mnt/Datasets/ARTalk_data \\
      --npz_subdir npz \\
      --batch_size 256 \\
      --device cuda
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.FLAME import FLAMEModel

# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------
REGION_NAMES = ["lips", "forehead", "eyes"]
_MASK_KEYS = {
    "lips":     ["lips"],
    "forehead": ["forehead"],
    "eyes":     ["left_eye_region", "right_eye_region"],
}
N_REGIONS = len(REGION_NAMES)

MASKS_PATH = PROJECT_ROOT / "flame_model" / "assets" / "FLAME_masks.pkl"

# Chunk params must match the dataset loader defaults.
MAX_SEQ_LEN = 600
MIN_SEQ_LEN = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_region_masks(masks_path: Path) -> dict:
    with open(masks_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    out = {}
    for name, keys in _MASK_KEYS.items():
        idx = np.concatenate([raw[k] for k in keys])
        out[name] = torch.from_numpy(np.unique(idx).astype(np.int64))
    return out


def extract_params(npz_path: Path):
    """Extract (expr, gpose, jaw) from a .npz file.  Returns None on failure."""
    param = np.load(npz_path, allow_pickle=True)
    files = set(param.files)

    if "exp" in files:
        expr = param["exp"]
    elif "expression_params" in files:
        expr = param["expression_params"]
    else:
        return None
    expr = expr.reshape(expr.shape[0], -1).astype(np.float32)

    pose = None
    for key in ("pose", "pose_params", "gpose"):
        if key in files:
            pose = param[key].reshape(param[key].shape[0], -1).astype(np.float32)
            break
    if pose is None:
        return None

    if pose.shape[1] >= 6:
        gpose = pose[:, :3]
        jaw = pose[:, 3:6]
    elif pose.shape[1] == 3:
        jaw_key = next((k for k in ("jaw", "jaw_params") if k in files), None)
        if jaw_key is None:
            return None
        gpose = pose
        jaw = param[jaw_key].reshape(param[jaw_key].shape[0], -1).astype(np.float32)
    else:
        return None

    T = expr.shape[0]
    if gpose.shape[0] != T or jaw.shape[0] != T:
        return None

    return expr, jaw


@torch.no_grad()
def compute_neutral_verts(flame: FLAMEModel, device: torch.device) -> torch.Tensor:
    """Returns neutral-pose vertices, shape (V, 3)."""
    verts = flame.forward(
        shape_params=torch.zeros(1, 300, device=device),
        expression_params=torch.zeros(1, 50, device=device),
        pose_params=torch.zeros(1, 6, device=device),
        eye_pose_params=torch.zeros(1, 6, device=device),
    )
    if isinstance(verts, tuple):
        verts = verts[0]
    return verts.squeeze(0)  # (V, 3)


@torch.no_grad()
def compute_disp_sequence(
    flame: FLAMEModel,
    expr_np: np.ndarray,
    jaw_np: np.ndarray,
    verts_neutral: torch.Tensor,
    region_masks: dict,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Returns per-frame per-region (disp, speed) features: (T, N_REGIONS, 2).

      disp(t)  = mean_{v in M_r} || verts(t,v) - verts_neutral(v) ||_2
      speed(t) = mean_{v in M_r} || verts(t,v) - verts(t-1,v)     ||_2
      speed(0) = 0
    """
    T = expr_np.shape[0]
    out = np.zeros((T, N_REGIONS, 2), dtype=np.float32)

    shape_zeros = torch.zeros(1, 300, device=device)
    eye_zeros = torch.zeros(1, 6, device=device)
    verts_neutral_b = verts_neutral.unsqueeze(0)  # (1, V, 3)

    # Pre-move masks to the device once.
    masks_on_device = {name: region_masks[name].to(device) for name in REGION_NAMES}

    # Track the last frame's vertices across batches so the velocity is
    # computed correctly across batch boundaries.
    prev_verts = None

    for start in range(0, T, batch_size):
        end = min(T, start + batch_size)
        B = end - start

        expr_t = torch.from_numpy(expr_np[start:end]).to(device)
        jaw_t  = torch.from_numpy(jaw_np[start:end]).to(device)
        # Zero out global head pose: we want displacement due to expression
        # and jaw only.  Including gpose would rigidly rotate/translate all
        # vertices, producing large spurious displacements unrelated to facial
        # expression.
        gpose_zeros = torch.zeros(B, 3, device=device)
        pose_t = torch.cat([gpose_zeros, jaw_t], dim=-1)  # (B, 6)

        verts = flame.forward(
            shape_params=shape_zeros.expand(B, -1),
            expression_params=expr_t,
            pose_params=pose_t,
            eye_pose_params=eye_zeros.expand(B, -1),
        )
        if isinstance(verts, tuple):
            verts = verts[0]
        # verts: (B, V, 3)

        # --- Position: deformation from neutral ---
        delta = verts - verts_neutral_b                       # (B, V, 3)

        # --- Velocity: per-frame change in vertex position ---
        # shifted[i] = verts[i-1], with shifted[0] = prev_verts or verts[0]
        if prev_verts is None:
            first_prev = verts[:1]                           # use frame 0 itself -> speed(0)=0
        else:
            first_prev = prev_verts.unsqueeze(0)             # (1, V, 3)
        shifted = torch.cat([first_prev, verts[:-1]], dim=0)  # (B, V, 3)
        vel = verts - shifted                                 # (B, V, 3)

        for ri, name in enumerate(REGION_NAMES):
            idx = masks_on_device[name]
            # (B, |M_r|, 3) -> per-vertex L2 norm -> mean over region
            mean_disp  = delta[:, idx, :].norm(dim=-1).mean(dim=-1)  # (B,)
            mean_speed = vel[:,  idx, :].norm(dim=-1).mean(dim=-1)   # (B,)
            out[start:end, ri, 0] = mean_disp.cpu().numpy()
            out[start:end, ri, 1] = mean_speed.cpu().numpy()

        prev_verts = verts[-1].detach()  # (V, 3)

    return out


def read_split_stems(txt_path: Path) -> set:
    """Return the set of file stems listed in a split .txt file."""
    if not txt_path.exists():
        return set()
    stems = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.add(Path(os.path.basename(line)).stem)
    return stems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-frame FLAME region displacement for style conditioning."
    )
    parser.add_argument("--data_root",  type=str, required=True,
                        help="Dataset root (contains the npz subdir and train.txt / test.txt).")
    parser.add_argument("--npz_subdir", type=str, default="npz",
                        help="Subdirectory under data_root that holds the .npz files.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of frames per FLAME forward pass.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even when a sidecar already exists.")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    npz_root   = data_root / args.npz_subdir
    annot_root = data_root / "annotations"
    annot_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Data root        : {data_root}")
    print(f"NPZ root         : {npz_root}")
    print(f"Annotations root : {annot_root}")
    print(f"Device           : {device}")
    print(f"Regions          : {REGION_NAMES}")

    # --- Models & masks ---
    print("\nLoading FLAME model (no_lmks=True for speed)...")
    flame = FLAMEModel(n_shape=300, n_exp=50, no_lmks=True).to(device)
    flame.eval()

    print("Loading region masks...")
    region_masks = load_region_masks(MASKS_PATH)
    for name, idx in region_masks.items():
        print(f"  {name}: {len(idx)} vertices")

    verts_neutral = compute_neutral_verts(flame, device)
    print(f"Neutral verts: {tuple(verts_neutral.shape)}")

    # --- Train split (for stats) ---
    train_stems = read_split_stems(data_root / "train.txt")
    print(f"\nTrain split: {len(train_stems)} entries "
          f"{'(not found - stats from all files)' if not train_stems else ''}")

    # --- Collect npz files ---
    npz_files = sorted(npz_root.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files\n")

    # -----------------------------------------------------------------------
    # Pass 1: compute and save per-file sidecars
    # -----------------------------------------------------------------------
    n_done, n_skipped, n_failed = 0, 0, 0

    for npz_path in tqdm(npz_files, desc="Computing displacements"):
        # Mirror the directory structure of npz_root inside annot_root so that
        # files with the same name in different subdirectories don't collide.
        rel       = npz_path.relative_to(npz_root)
        sidecar   = (annot_root / rel).with_suffix(".style_disp.npy")
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        if sidecar.exists() and not args.overwrite:
            n_skipped += 1
            continue

        params = extract_params(npz_path)
        if params is None:
            n_failed += 1
            continue

        expr_np, jaw_np = params
        try:
            disp = compute_disp_sequence(
                flame, expr_np, jaw_np,
                verts_neutral, region_masks, device,
                batch_size=args.batch_size,
            )
            np.save(sidecar, disp)
            n_done += 1
        except Exception as exc:
            tqdm.write(f"  FAILED {npz_path.name}: {exc}")
            n_failed += 1

    print(f"\nPass 1 complete — done: {n_done}  skipped: {n_skipped}  failed: {n_failed}")

    # -----------------------------------------------------------------------
    # Pass 2: normalization stats from train set
    # -----------------------------------------------------------------------
    print("\nComputing normalization stats (p95 per chunk, train split only)...")

    p95_list = []
    for npz_path in tqdm(npz_files, desc="Stats"):
        if train_stems and npz_path.stem not in train_stems:
            continue
        rel     = npz_path.relative_to(npz_root)
        sidecar = (annot_root / rel).with_suffix(".style_disp.npy")
        if not sidecar.exists():
            continue
        feats = np.load(sidecar)       # (T, N_REGIONS, 2)
        T = feats.shape[0]
        for start in range(0, T, MAX_SEQ_LEN):
            end = min(T, start + MAX_SEQ_LEN)
            if (end - start) < MIN_SEQ_LEN:
                continue
            chunk = feats[start:end]                              # (L, N_REGIONS, 2)
            p95_list.append(np.percentile(chunk, 95, axis=0))     # (N_REGIONS, 2)

    if not p95_list:
        print("Warning: no training chunks found — stats file not written.")
        return

    p95_array = np.stack(p95_list, axis=0)        # (N_chunks, N_REGIONS, 2)
    mean = p95_array.mean(axis=0)                 # (N_REGIONS, 2)
    std  = p95_array.std(axis=0) + 1e-8           # (N_REGIONS, 2)

    stats_path = annot_root / "style_disp_stats.npz"
    np.savez(
        stats_path,
        mean=mean,
        std=std,
        region_names=np.array(REGION_NAMES),
        feature_names=np.array(["disp", "speed"]),
    )

    print(f"\nStats saved → {stats_path}")
    print(f"{'Region':<12}  {'feat':<6}  {'mean':>12}  {'std':>12}")
    print("-" * 46)
    for i, name in enumerate(REGION_NAMES):
        for j, feat in enumerate(["disp", "speed"]):
            print(f"{name:<12}  {feat:<6}  {mean[i, j]:>12.6f}  {std[i, j]:>12.6f}")


if __name__ == "__main__":
    main()
