import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from tqdm import tqdm

from dataset.augment import build_augmentor_from_cfg


class BlendshapeDataset(data.Dataset):
    def __init__(self, items, augmentor=None, style_mean=None, style_std=None):
        """
        items       : list of (blendshapes_chunk np.ndarray [T,58],
                               style_disp_chunk  np.ndarray [T, N_REGIONS, 2] or None)
        style_mean  : np.ndarray [N_REGIONS, 2] or None  (z-score stats)
        style_std   : np.ndarray [N_REGIONS, 2] or None
        """
        self.items       = items
        self.augmentor   = augmentor
        self.style_mean  = style_mean
        self.style_std   = style_std

    def __getitem__(self, index):
        blendshapes, disp = self.items[index]
        if self.augmentor is not None:
            seed = (index * 2654435761) & 0xFFFFFFFF
            rng  = np.random.default_rng(seed ^ np.random.randint(0, 2**31))
            inp  = self.augmentor(blendshapes, rng=rng)
        else:
            inp  = blendshapes

        # Style conditioning scalar: p95 over the chunk for each (region, feature),
        # then z-scored.  Shape: (N_REGIONS * 2,)  i.e. [lips_disp, lips_speed,
        #                                                  forehead_disp, forehead_speed,
        #                                                  eyes_disp, eyes_speed]
        if disp is not None:
            # disp: (T, N_REGIONS, 2)
            p95 = np.percentile(disp, 95, axis=0)          # (N_REGIONS, 2)
            if self.style_mean is not None and self.style_std is not None:
                p95 = (p95 - self.style_mean) / self.style_std
            style = torch.from_numpy(p95.flatten().astype(np.float32))  # (N_REGIONS*2,)
        else:
            # No sidecar available: emit zeros (model treats this as "average style"
            # after z-scoring, since mean≈0).  Shape matches the normal case.
            n_style = 6  # N_REGIONS * N_FEATURES = 3 * 2
            style = torch.zeros(n_style, dtype=torch.float32)

        return (
            torch.from_numpy(inp).float(),        # augmented input
            torch.from_numpy(blendshapes).float(), # clean target
            style,                                 # (6,) style conditioning
        )

    def __len__(self):
        return len(self.items)


def _read_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    # Accept both absolute/relative paths and plain filenames.
    return {os.path.basename(line) for line in lines}


def _extract_blendshapes(npz_path):
    param = np.load(npz_path, allow_pickle=True)
    files = set(param.files)

    if "exp" in files:
        expr = param["exp"]
    elif "expression_params" in files:
        expr = param["expression_params"]
    else:
        return None

    if "pose" in files:
        pose = param["pose"]
    elif "pose_params" in files:
        pose = param["pose_params"]
    elif "gpose" in files:
        pose = param["gpose"]
    else:
        return None

    if "jaw" in files:
        jaw = param["jaw"]
    elif "jaw_params" in files:
        jaw = param["jaw_params"]
    else:
        jaw = None

    expr = expr.reshape(expr.shape[0], -1)
    pose = pose.reshape(pose.shape[0], -1)

    if pose.shape[1] >= 6:
        gpose = pose[:, 0:3]
        jaw_from_pose = pose[:, 3:6]
        jaw = jaw_from_pose if jaw is None else jaw.reshape(jaw.shape[0], -1)
    elif pose.shape[1] == 3:
        if jaw is None:
            return None
        gpose = pose
        jaw = jaw.reshape(jaw.shape[0], -1)
    else:
        return None

    # Reduce global motion drift for easier learning.
    gpose = gpose - gpose.mean(axis=0, keepdims=True)

    if "eyelids" in files:
        eyelids = param["eyelids"].reshape(param["eyelids"].shape[0], -1)
        if eyelids.shape[0] != expr.shape[0] or eyelids.shape[1] != 2:
            return None
        eyelids = eyelids.astype(expr.dtype)
    else:
        eyelids = np.ones((expr.shape[0], 2), dtype=expr.dtype)
    blendshapes = np.concatenate([expr, gpose, jaw, eyelids], axis=1)

    if blendshapes.shape[1] != 58:
        return None
    return blendshapes.astype(np.float32)


def _load_style_stats(annot_root: Path):
    """Load z-score stats written by precompute_style_disp.py.  Returns (mean, std) or (None, None)."""
    stats_path = annot_root / "style_disp_stats.npz"
    if not stats_path.exists():
        return None, None
    stats = np.load(stats_path)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def read_data(args):
    print("Loading data...")
    data_root = Path(args.data_root)
    npz_root  = data_root / args.vertices_path
    annot_root = data_root / "annotations"
    audio_path = data_root / getattr(args, "wav_path", "wav")

    print("Data root:", str(data_root))
    print("Audio path:", str(audio_path))
    print("Vertices path:", str(npz_root))
    print("Annotations root:", str(annot_root))

    style_mean, style_std = _load_style_stats(annot_root)
    if style_mean is not None:
        print("Style disp stats loaded (z-score normalization enabled).")
    else:
        print("Style disp stats NOT found — style conditioning will use zeros.")

    train_wavs = _read_split_file(data_root / "train.txt")
    test_wavs  = _read_split_file(data_root / "test.txt")
    print("Train lines read:", len(train_wavs))
    print("Test lines read:", len(test_wavs))

    max_seq_len = int(getattr(args, "max_seq_len", 600))
    min_seq_len = int(getattr(args, "min_seq_len", 8))

    train_items = []
    test_items  = []
    frames_count = 0
    counter = 0
    non_existent_files = 0
    n_style_missing = 0

    npz_files = []
    for root, _, files in os.walk(npz_root):
        for file_name in files:
            if file_name.endswith(".npz"):
                npz_files.append(Path(root) / file_name)

    for npz_path in tqdm(npz_files):
        file_name = npz_path.name
        wav_name  = f"{npz_path.stem}.wav"

        if wav_name not in train_wavs and wav_name not in test_wavs:
            continue

        if not npz_path.exists():
            non_existent_files += 1
            print("File does not exist:", str(npz_path))
            continue

        blendshapes = _extract_blendshapes(npz_path)
        if blendshapes is None or blendshapes.shape[0] < min_seq_len:
            continue

        # Load matching style displacement sidecar (optional).
        rel          = npz_path.relative_to(npz_root)
        sidecar_path = (annot_root / rel).with_suffix(".style_disp.npy")
        if sidecar_path.exists():
            disp_full = np.load(sidecar_path)  # (T_full, N_REGIONS, 2)
            if disp_full.shape[0] != blendshapes.shape[0]:
                disp_full = None  # length mismatch — ignore
                n_style_missing += 1
        else:
            disp_full = None
            n_style_missing += 1

        total_frames = blendshapes.shape[0]
        for start in range(0, total_frames, max_seq_len):
            end = min(total_frames, start + max_seq_len)
            if (end - start) < min_seq_len:
                continue
            chunk_bs   = blendshapes[start:end]
            chunk_disp = disp_full[start:end] if disp_full is not None else None
            item = (chunk_bs, chunk_disp)
            if wav_name in train_wavs:
                train_items.append(item)
            else:
                test_items.append(item)
            frames_count += int(chunk_bs.shape[0])
            counter += 1

    print(f"Style sidecar missing/mismatched for {n_style_missing} files.")

    print("Total sequences:", counter, " Non-existent files:", non_existent_files)

    all_seen_wavs = {f"{p.stem}.wav" for p in npz_files}
    train_matches = len([w for w in all_seen_wavs if w in train_wavs])
    test_matches = len([w for w in all_seen_wavs if w in test_wavs])
    print("Train matches (basename):", train_matches, "Test matches (basename):", test_matches)

    # Validation size roughly tracks test size for simple, stable feedback.
    train_items.sort(key=lambda x: x[0].shape[0])
    val_target = len(test_items) if len(test_items) > 0 else max(1, int(0.1 * len(train_items)))
    val_target = min(val_target, len(train_items))
    split_idx = len(train_items) - val_target

    valid_items = train_items[split_idx:]
    train_items = train_items[:split_idx]

    print("Loaded data: Train-{}, Val-{}, Test-{}".format(len(train_items), len(valid_items), len(test_items)))
    print("Total hours of data: {:.2f}".format(frames_count / 25 / 60 / 60))
    return train_items, valid_items, test_items, style_mean, style_std


def collate_fn(batch):
    # batch: list of (input_tensor, target_tensor, style_tensor)
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    inputs  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    styles  = [item[2] for item in batch]   # each (6,)
    lengths = [t.shape[0] for t in inputs]

    padded_in  = pad_sequence(inputs,  batch_first=True, padding_value=0.0)
    padded_tgt = pad_sequence(targets, batch_first=True, padding_value=0.0)
    style_batch = torch.stack(styles, dim=0)   # (B, 6)
    mask = torch.zeros(padded_in.shape[:2], dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return padded_in, padded_tgt, mask, style_batch


def get_dataloaders(args):
    train_items, valid_items, test_items, style_mean, style_std = read_data(args)

    train_augmentor = build_augmentor_from_cfg(args)
    if train_augmentor is not None:
        print("Data augmentation enabled for training (input-only corruption).")
    else:
        print("Data augmentation disabled.")

    datasets = {
        "train": BlendshapeDataset(train_items, augmentor=train_augmentor,
                                   style_mean=style_mean, style_std=style_std),
        "valid": BlendshapeDataset(valid_items, augmentor=None,
                                   style_mean=style_mean, style_std=style_std),
        "test":  BlendshapeDataset(test_items,  augmentor=None,
                                   style_mean=style_mean, style_std=style_std),
    }

    workers = int(getattr(args, "workers", 4))
    batch_size = int(getattr(args, "batch_size", 8))

    return {
        "train": data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            collate_fn=collate_fn,
        ),
        "valid": data.DataLoader(
            datasets["valid"],
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            collate_fn=collate_fn,
        ),
        "test": data.DataLoader(
            datasets["test"],
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            collate_fn=collate_fn,
        ),
    }


# ---------------------------------------------------------------------------
# Paired (content, style) dataloader for the stage1_style training script.
#
# Each batch yields two independent clips per sample: (A, B). Each pad-group is
# padded independently to its own max length so masks are tight. The simplest
# approach is to take a batch of size N from the underlying dataset and pair it
# with a permuted copy of itself; this halves dataset-IO vs sampling 2N items.
# ---------------------------------------------------------------------------


def _pack(items):
    """Pack a list of (input, target) into padded tensors + bool mask."""
    inputs = [it[0] for it in items]
    targets = [it[1] for it in items]
    lengths = [t.shape[0] for t in inputs]
    padded_in = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_tgt = pad_sequence(targets, batch_first=True, padding_value=0.0)
    mask = torch.zeros(padded_in.shape[:2], dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    return padded_in, padded_tgt, mask


def paired_collate_fn(batch):
    """Collate that yields (A_in, A_tgt, A_mask, B_in, B_tgt, B_mask).

    A is the original batch order (content side). B is a permutation of the
    same batch (style side). We try to avoid self-pairing so each sample sees
    a different clip as its style reference.
    """
    n = len(batch)
    perm = list(range(n))
    if n > 1:
        for _ in range(8):
            random.shuffle(perm)
            if all(p != i for i, p in enumerate(perm)):
                break
        else:  # fall back to a single-step rotation
            perm = list(range(1, n)) + [0]

    A = [batch[i] for i in range(n)]
    B = [batch[perm[i]] for i in range(n)]

    A_in, A_tgt, A_mask = _pack(A)
    B_in, B_tgt, B_mask = _pack(B)
    return A_in, A_tgt, A_mask, B_in, B_tgt, B_mask


def get_paired_dataloaders(args):
    """Same dataset as :func:`get_dataloaders` but with paired-collate."""
    train_items, valid_items, test_items = read_data(args)

    train_augmentor = build_augmentor_from_cfg(args)
    if train_augmentor is not None:
        print("Data augmentation enabled for training (input-only corruption).")
    else:
        print("Data augmentation disabled.")

    datasets = {
        "train": BlendshapeDataset(train_items, augmentor=train_augmentor),
        "valid": BlendshapeDataset(valid_items, augmentor=None),
        "test": BlendshapeDataset(test_items, augmentor=None),
    }

    workers = int(getattr(args, "workers", 4))
    batch_size = int(getattr(args, "batch_size", 8))

    return {
        "train": data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            collate_fn=paired_collate_fn,
        ),
        "valid": data.DataLoader(
            datasets["valid"],
            batch_size=max(2, min(batch_size, 4)),
            shuffle=True,  # need >=2 distinct clips per batch to form pairs
            num_workers=workers,
            drop_last=True,
            collate_fn=paired_collate_fn,
        ),
        "test": data.DataLoader(
            datasets["test"],
            batch_size=max(2, min(batch_size, 4)),
            shuffle=False,
            num_workers=workers,
            drop_last=True,
            collate_fn=paired_collate_fn,
        ),
    }
