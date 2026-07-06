#!/usr/bin/env python
"""Stage-2 training: audio -> stage-1 latent codes -> blendshapes.

Single-GPU trainer in the same lightweight style as ``train_stage1_style.py``.
The stage-1 StyleVQAutoEncoder is loaded frozen inside the model; only the
audio encoder, the Transformer decoder and the projection heads are trained.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from dataset.data_loader_joint_data_batched import get_dataloaders
from models import get_model
from utils.config import load_flat_config

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")


def run_epoch(loader, model, optimizer, criterion, device, cfg, epoch, epochs, train_mode=True):
    model.train(train_mode)
    # The frozen stage-1 autoencoder always stays in eval mode (BN/dropout off).
    model.autoencoder.eval()
    # If the audio encoder is frozen, keep it in eval too (no dropout drift).
    if getattr(model, "freeze_audio_encoder", False):
        model.audio_encoder.eval()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    total_meter = AverageMeter()
    bs_meter = AverageMeter()
    reg_meter = AverageMeter()
    delta_meter = AverageMeter()
    print_freq = int(getattr(cfg, "print_freq", 10))

    max_iter = epochs * len(loader)
    end = time.time()

    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)

        padded_in, padded_tgt, mask, style, padded_audio, audio_mask = batch
        padded_tgt   = padded_tgt.to(device)
        mask         = mask.to(device)
        style        = style.to(device)
        padded_audio = padded_audio.to(device)
        audio_mask   = audio_mask.to(device)

        with torch.set_grad_enabled(train_mode):
            total_loss, parts = model(
                padded_tgt, mask, padded_audio, audio_mask, criterion, style=style
            )

        grad_norm = 0.0
        if train_mode:
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = float(model.feat_map.weight.grad.norm().item())
            optimizer.step()

        n = padded_tgt.shape[0]
        total_meter.update(total_loss.item(), n)
        bs_meter.update(parts[0].item(), n)
        reg_meter.update(parts[1].item(), n)
        if len(parts) > 2:
            delta_meter.update(parts[2].item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if train_mode and (i % print_freq == 0):
            current_iter = (epoch - 1) * len(loader) + i + 1
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time_str = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

            print(
                f"Epoch: [{epoch}/{epochs}][{i}/{len(loader)}] "
                f"Data: {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Remain: {remain_time_str} "
                f"Grad norm: {grad_norm:.4f} "
                f"Loss total: {total_meter.avg:.4f} "
                f"blendshapes: {bs_meter.avg:.4f} latent: {reg_meter.avg:.4f} delta: {delta_meter.avg:.4f}"
            )

    return {
        "total": total_meter.avg,
        "recon_blendshapes": bs_meter.avg,
        "recon_latent": reg_meter.avg,
        "recon_delta": delta_meter.avg,
    }


@torch.no_grad()
def collapse_probe(model, loader, device, n_clips=4):
    """Free-running (autoregressive) collapse diagnostic.

    Runs ``model.predict`` on a few clips and measures MOTION rather than error,
    because teacher-forced loss stays healthy even after the model has collapsed.
    Returns ratios of predicted-to-GT statistics (1.0 == matches real motion):

      vel_ratio  : mean |frame-to-frame delta|, pred / GT   (temporal liveliness)
      tstd_ratio : mean temporal std,           pred / GT   (amount of movement)
      diversity  : cross-clip std of the per-clip mean pose, pred / GT
                   (does the output actually depend on the audio?)

    Collapse  => all ratios ~ 0 (frozen, input-agnostic).
    Untrained but healthy => ratios are O(1) or > 1 while MSE stays high.
    """
    model.eval()
    model.autoencoder.eval()
    pred_vel = gt_vel = pred_tstd = gt_tstd = 0.0
    pred_means, gt_means = [], []
    count = 0
    for batch in loader:
        if count >= n_clips:
            break
        _, padded_tgt, mask, style, padded_audio, _ = batch
        padded_tgt = padded_tgt.to(device)
        style = style.to(device)
        padded_audio = padded_audio.to(device)

        pred = model.predict(padded_audio, style=style).squeeze(0)          # [Tp, 58]
        gt = padded_tgt[0][mask[0].bool()] if mask.dtype == torch.bool else padded_tgt[0]
        T = min(pred.shape[0], gt.shape[0])
        if T < 3:
            continue
        pred, gt = pred[:T], gt[:T].to(device)

        pred_vel  += (pred[1:] - pred[:-1]).abs().mean().item()
        gt_vel    += (gt[1:] - gt[:-1]).abs().mean().item()
        pred_tstd += pred.std(dim=0).mean().item()
        gt_tstd   += gt.std(dim=0).mean().item()
        pred_means.append(pred.mean(dim=0))
        gt_means.append(gt.mean(dim=0))
        count += 1

    if count == 0:
        return {}
    eps = 1e-8
    diversity = 1.0
    if len(pred_means) > 1:
        pdiv = torch.stack(pred_means).std(dim=0).mean().item()
        gdiv = torch.stack(gt_means).std(dim=0).mean().item()
        diversity = pdiv / (gdiv + eps)
    return {
        "vel_ratio": pred_vel / (gt_vel + eps),
        "tstd_ratio": pred_tstd / (gt_tstd + eps),
        "diversity": diversity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="config/talkinghead-1kh/stage2.yaml")
    args = parser.parse_args()

    cfg = load_flat_config(args.config)
    set_seed(int(getattr(cfg, "manual_seed", 131)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device.type

    # ── Data (audio enabled) ────────────────────────────────────────────────
    cfg.read_audio = True
    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    valid_loader = loaders["valid"]

    # ── Model ───────────────────────────────────────────────────────────────
    model = get_model(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable) / 1e6
    print(f"Trainable parameters: {n_trainable:.2f} M")

    criterion = nn.MSELoss()
    base_lr = float(getattr(cfg, "base_lr", 1e-5))
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=int(getattr(cfg, "step_size", 100)),
                       gamma=float(getattr(cfg, "gamma", 0.5)))

    epochs = int(getattr(cfg, "epochs", 300))
    save_dir = Path(getattr(cfg, "save_path", "logs/stage2/checkpoints"))
    log_dir = Path(getattr(cfg, "log_dir", "logs/stage2"))
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    save_freq = int(getattr(cfg, "save_freq", 5))
    eval_freq = int(getattr(cfg, "eval_freq", 5))

    use_wandb = bool(getattr(cfg, "wandb", False)) and wandb is not None
    if use_wandb:
        wandb.init(
            project=getattr(cfg, "wandb_project", "fastertalk_stage2"),
            name=getattr(cfg, "wandb_run_name", "stage2"),
            dir=str(log_dir),
            mode=getattr(cfg, "wandb_mode", "online"),
            config=vars(cfg),
        )

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, model, optimizer, criterion,
                                  device, cfg, epoch, epochs, train_mode=True)
        scheduler.step()

        log_row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        log_row.update({f"train/{k}": v for k, v in train_metrics.items()})

        print(
            f"[Epoch {epoch:03d}] "
            f"total_loss_train={train_metrics['total']:.6f} "
            f"blendshapes_loss_train={train_metrics['recon_blendshapes']:.6f} "
            f"latent_loss_train={train_metrics['recon_latent']:.6f}"
        )

        if bool(getattr(cfg, "evaluate", True)) and (epoch % eval_freq == 0):
            val_metrics = run_epoch(valid_loader, model, optimizer, criterion,
                                    device, cfg, epoch, epochs, train_mode=False)
            log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            print(
                f"[Epoch {epoch:03d}] "
                f"total_loss_val={val_metrics['total']:.6f} "
                f"blendshapes_loss_val={val_metrics['recon_blendshapes']:.6f} "
                f"latent_loss_val={val_metrics['recon_latent']:.6f}"
            )

            # Free-running collapse probe (same eval cadence, a few rollouts).
            probe_clips = int(getattr(cfg, "collapse_probe_clips", 4))
            if probe_clips > 0:
                probe = collapse_probe(model, valid_loader, device, n_clips=probe_clips)
                if probe:
                    log_row.update({f"probe/{k}": v for k, v in probe.items()})
                    print(
                        f"[Epoch {epoch:03d}] PROBE "
                        f"vel_ratio={probe['vel_ratio']:.3f} "
                        f"tstd_ratio={probe['tstd_ratio']:.3f} "
                        f"diversity={probe['diversity']:.3f}  "
                        f"(all ->0 = collapse)"
                    )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")
        if use_wandb:
            wandb.log(log_row, step=epoch)

        if epoch % save_freq == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, save_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
