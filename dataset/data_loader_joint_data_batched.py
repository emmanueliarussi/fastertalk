import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from tqdm import tqdm

from dataset.augment import build_augmentor_from_cfg


class BlendshapeDataset(data.Dataset):
    def __init__(self, items, augmentor=None):
        self.items = items
        self.augmentor = augmentor

    def __getitem__(self, index):
        clean = self.items[index]
        if self.augmentor is not None:
            # Use a per-sample RNG seeded by index + worker info for reproducibility per epoch shuffle.
            seed = (index * 2654435761) & 0xFFFFFFFF
            rng = np.random.default_rng(seed ^ np.random.randint(0, 2**31))
            inp = self.augmentor(clean, rng=rng)
        else:
            inp = clean
        return (
            torch.from_numpy(inp).float(),
            torch.from_numpy(clean).float(),
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


def read_data(args):
    print("Loading data...")
    data_root = Path(args.data_root)
    npz_root = data_root / args.vertices_path
    audio_path = data_root / getattr(args, "wav_path", "wav")

    print("Data root:", str(data_root))
    print("Audio path:", str(audio_path))
    print("Vertices path:", str(npz_root))

    train_wavs = _read_split_file(data_root / "train.txt")
    test_wavs = _read_split_file(data_root / "test.txt")
    print("Train lines read:", len(train_wavs))
    print("Test lines read:", len(test_wavs))

    max_seq_len = int(getattr(args, "max_seq_len", 600))
    min_seq_len = int(getattr(args, "min_seq_len", 8))

    train_items = []
    test_items = []
    frames_count = 0
    counter = 0
    non_existent_files = 0

    npz_files = []
    for root, _, files in os.walk(npz_root):
        for file_name in files:
            if file_name.endswith(".npz"):
                npz_files.append(Path(root) / file_name)

    for npz_path in tqdm(npz_files):
        file_name = npz_path.name
        wav_name = f"{npz_path.stem}.wav"

        if wav_name not in train_wavs and wav_name not in test_wavs:
            continue

        if not npz_path.exists():
            non_existent_files += 1
            print("File does not exist:", str(npz_path))
            continue

        blendshapes = _extract_blendshapes(npz_path)
        if blendshapes is None or blendshapes.shape[0] < min_seq_len:
            continue

        total_frames = blendshapes.shape[0]
        chunk_idx = 0
        for start in range(0, total_frames, max_seq_len):
            end = min(total_frames, start + max_seq_len)
            if (end - start) < min_seq_len:
                continue
            chunk = blendshapes[start:end]
            if wav_name in train_wavs:
                train_items.append(chunk)
            else:
                test_items.append(chunk)
            frames_count += int(chunk.shape[0])
            counter += 1
            chunk_idx += 1

    print("Total sequences:", counter, " Non-existent files:", non_existent_files)

    all_seen_wavs = {f"{p.stem}.wav" for p in npz_files}
    train_matches = len([w for w in all_seen_wavs if w in train_wavs])
    test_matches = len([w for w in all_seen_wavs if w in test_wavs])
    print("Train matches (basename):", train_matches, "Test matches (basename):", test_matches)

    # Validation size roughly tracks test size for simple, stable feedback.
    train_items.sort(key=lambda x: x.shape[0])
    val_target = len(test_items) if len(test_items) > 0 else max(1, int(0.1 * len(train_items)))
    val_target = min(val_target, len(train_items))
    split_idx = len(train_items) - val_target

    valid_items = train_items[split_idx:]
    train_items = train_items[:split_idx]

    print("Loaded data: Train-{}, Val-{}, Test-{}".format(len(train_items), len(valid_items), len(test_items)))
    print("Total hours of data: {:.2f}".format(frames_count / 25 / 60 / 60))
    return train_items, valid_items, test_items


def collate_fn(batch):
    # batch: list of (input_tensor, target_tensor)
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [t.shape[0] for t in inputs]

    padded_in = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_tgt = pad_sequence(targets, batch_first=True, padding_value=0.0)
    mask = torch.zeros(padded_in.shape[:2], dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return padded_in, padded_tgt, mask


def get_dataloaders(args):
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
