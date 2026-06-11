#!/usr/bin/env python3
"""
Visualize top examples ranked by per-region style displacement metrics.

For each region (lips, forehead, eyes) and each metric (disp, speed) finds
the N clips with the highest p95 value and renders them as MP4 videos.

Output structure:
  <out_dir>/
    lips_disp/      top-N clips ranked by lips  displacement-from-neutral
    lips_speed/     top-N clips ranked by lips  motion speed
    forehead_disp/  ...
    forehead_speed/
    eyes_disp/
    eyes_speed/

Usage
-----
  python dataset/visualize_style_disp.py \\
      --data_root /mnt/Datasets/ARTalk_data \\
      --npz_subdir npz \\
      --out_dir /mnt/fastertalk/demo/style_disp_viz \\
      --top_n 5 \\
      --fps 25 \\
      --device cuda
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from flame_model.FLAME import FLAMEModel
from renderer.renderer import Renderer

# Must match precompute_style_disp.py
REGION_NAMES  = ["lips", "forehead", "eyes"]
FEATURE_NAMES = ["disp", "speed"]
N_REGIONS     = len(REGION_NAMES)   # 3
N_FEATURES    = len(FEATURE_NAMES)  # 2


# ---------------------------------------------------------------------------
# Param extraction (same logic as data_loader)
# ---------------------------------------------------------------------------

def extract_params(npz_path: Path):
    """Returns (expr, gpose, jaw) as float32 numpy arrays, or None on failure."""
    param = np.load(npz_path, allow_pickle=True)
    files = set(param.files)

    expr = None
    for key in ("exp", "expression_params"):
        if key in files:
            expr = param[key].reshape(param[key].shape[0], -1).astype(np.float32)
            break
    if expr is None:
        return None

    pose = None
    for key in ("pose", "pose_params", "gpose"):
        if key in files:
            pose = param[key].reshape(param[key].shape[0], -1).astype(np.float32)
            break
    if pose is None:
        return None

    if pose.shape[1] >= 6:
        gpose = pose[:, :3]
        jaw   = pose[:, 3:6]
    elif pose.shape[1] == 3:
        jaw_key = next((k for k in ("jaw", "jaw_params") if k in files), None)
        if jaw_key is None:
            return None
        gpose = pose
        jaw   = param[jaw_key].reshape(param[jaw_key].shape[0], -1).astype(np.float32)
    else:
        return None

    return expr, gpose, jaw


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_sequence(flame, renderer, expr_np, gpose_np, jaw_np, device,
                    batch_size=64):
    """Return a list of (H, W, 3) uint8 numpy frames."""
    T = expr_np.shape[0]
    frames = []
    cam = torch.tensor([5.0, 0.0, 0.0], device=device).unsqueeze(0)

    for start in range(0, T, batch_size):
        end = min(T, start + batch_size)
        B   = end - start

        expr_t  = torch.from_numpy(expr_np[start:end]).to(device)
        gpose_t = torch.from_numpy(gpose_np[start:end]).to(device)
        jaw_t   = torch.from_numpy(jaw_np[start:end]).to(device)
        pose_t  = torch.cat([gpose_t, jaw_t], dim=-1)

        verts = flame.forward(
            shape_params   = torch.zeros(B, 300, device=device),
            expression_params = expr_t,
            pose_params    = pose_t,
            eye_pose_params= torch.zeros(B, 6, device=device),
        )
        if isinstance(verts, tuple):
            verts = verts[0]

        cam_b = cam.expand(B, -1)
        out   = renderer.forward(verts, cam_b)["rendered_img"]  # (B, 3, H, W) [0,1]

        for i in range(B):
            img = out[i].permute(1, 2, 0).cpu().numpy()
            frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

    return frames


def save_video(frames, path: Path, fps: int = 25):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True
    )
    ani.save(str(path), writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Score sidecars
# ---------------------------------------------------------------------------

def collect_scores(npz_files, npz_root, annot_root):
    """
    Returns scores[region_idx][feat_idx] = list of (score, npz_path).
    score = p95 of the feature over the full sidecar (all frames).
    """
    scores = [[[] for _ in FEATURE_NAMES] for _ in REGION_NAMES]

    for npz_path in npz_files:
        rel     = npz_path.relative_to(npz_root)
        sidecar = (annot_root / rel).with_suffix(".style_disp.npy")
        if not sidecar.exists():
            continue

        feats = np.load(sidecar)   # (T, N_REGIONS, 2)
        if feats.ndim != 3 or feats.shape[1] != N_REGIONS or feats.shape[2] != N_FEATURES:
            continue

        for ri in range(N_REGIONS):
            for fi in range(N_FEATURES):
                p95 = float(np.percentile(feats[:, ri, fi], 95))
                scores[ri][fi].append((p95, npz_path))

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render top examples per region/feature from style displacement sidecars."
    )
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--npz_subdir", default="npz")
    parser.add_argument("--out_dir",    default="demo/style_disp_viz")
    parser.add_argument("--top_n",      type=int, default=5,
                        help="Number of top clips to render per region/feature.")
    parser.add_argument("--fps",        type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Truncate clips to this many frames to keep videos short.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    npz_root   = data_root / args.npz_subdir
    annot_root = data_root / "annotations"
    out_dir    = Path(args.out_dir)
    device     = torch.device(args.device)

    print(f"Data root  : {data_root}")
    print(f"Annot root : {annot_root}")
    print(f"Out dir    : {out_dir}")
    print(f"Device     : {device}")

    # --- Models ---
    print("\nLoading FLAME + Renderer...")
    flame    = FLAMEModel(n_shape=300, n_exp=50).to(device)
    flame.eval()
    renderer = Renderer(render_full_head=True).to(device)
    renderer.eval()

    # --- Collect all npz files ---
    npz_files = sorted(npz_root.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files")

    # --- Score ---
    print("Scoring sidecars...")
    scores = collect_scores(npz_files, npz_root, annot_root)

    total_scored = sum(len(scores[ri][fi]) for ri in range(N_REGIONS) for fi in range(N_FEATURES))
    print(f"Scored {total_scored // (N_REGIONS * N_FEATURES)} clips with valid sidecars")

    if total_scored == 0:
        print("ERROR: no sidecar files found. Run precompute_style_disp.py first.")
        return

    # --- Render top-N per region/feature ---
    for ri, region in enumerate(REGION_NAMES):
        for fi, feat in enumerate(FEATURE_NAMES):
            bucket = sorted(scores[ri][fi], key=lambda x: x[0], reverse=True)
            top    = bucket[: args.top_n]

            print(f"\n=== {region} / {feat} (top {len(top)}) ===")
            sub_dir = out_dir / f"{region}_{feat}"

            for rank, (score, npz_path) in enumerate(top, start=1):
                print(f"  [{rank}] score={score:.5f}  {npz_path.name}")

                params = extract_params(npz_path)
                if params is None:
                    print("      SKIP: could not parse params")
                    continue
                expr_np, gpose_np, jaw_np = params

                # Truncate to keep videos manageable
                T = min(expr_np.shape[0], args.max_frames)
                expr_np  = expr_np[:T]
                gpose_np = gpose_np[:T]
                jaw_np   = jaw_np[:T]

                try:
                    frames = render_sequence(
                        flame, renderer,
                        expr_np, gpose_np, jaw_np,
                        device,
                    )
                    out_path = sub_dir / f"rank{rank:02d}_{npz_path.stem}_s{score:.4f}.mp4"
                    save_video(frames, out_path, fps=args.fps)
                except Exception as exc:
                    print(f"      FAILED: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
