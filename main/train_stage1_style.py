#!/usr/bin/env python
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
from torch.optim.lr_scheduler import StepLR

from dataset.data_loader_joint_data_batched import get_dataloaders
from losses import calc_vq_loss
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


def run_epoch(loader, model, optimizer, device, cfg, epoch, epochs, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    recon_total = 0.0
    quant_total = 0.0
    parts_total = {"recon_expr": 0.0, "recon_gpose": 0.0, "recon_jaw": 0.0, "recon_eyelids": 0.0}
    n_steps = 0
    end = time.time()
    max_iter = int(epochs) * max(1, len(loader))

    w_expr = float(getattr(cfg, "recon_w_expr", 1.0))
    w_gpose = float(getattr(cfg, "recon_w_gpose", 5.0))
    w_jaw = float(getattr(cfg, "recon_w_jaw", 2.0))
    w_eyelids = float(getattr(cfg, "recon_w_eyelids", 1.0))

    if train_mode:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"----> Total trainable parameters: {trainable_params}")

    for step, batch in enumerate(loader, start=1):
        current_iter = (int(epoch) - 1) * len(loader) + step
        data_time.update(time.time() - end)

        # New dataset returns (input, target, mask). Input may be augmented; target is clean.
        blendshapes_in, blendshapes_tgt, mask = batch
        blendshapes_in = blendshapes_in.to(device)
        blendshapes_tgt = blendshapes_tgt.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(train_mode):
            pred, q_loss = model(blendshapes_in, mask)
            loss, details = calc_vq_loss(
                pred,
                blendshapes_tgt,
                q_loss,
                mask,
                quant_loss_weight=float(cfg.quant_loss_weight),
                w_expr=w_expr,
                w_gpose=w_gpose,
                w_jaw=w_jaw,
                w_eyelids=w_eyelids,
            )

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                grad_norm_first_layer = float(model.encoder_proj.weight.grad.norm().item())
                optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        recon_total += float(details["recon"].item())
        quant_total += float(details["quant"].item())
        for k in parts_total:
            if k in details:
                parts_total[k] += float(details[k].item())
        n_steps += 1

        if train_mode and step % int(cfg.print_freq) == 0:
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time_str = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

            print(
                f"Epoch: [{epoch}/{epochs}][{step}/{len(loader)}] "
                f"Data: {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Remain: {remain_time_str} "
                f"Grad norm: {grad_norm_first_layer:.4f} "
                f"Loss blendshapes: {details['recon'].item():.4f}"
            )

    if n_steps == 0:
        return {"recon": 0.0, "quant": 0.0, "recon_expr": 0.0, "recon_gpose": 0.0, "recon_jaw": 0.0, "recon_eyelids": 0.0}

    return {
        "recon": recon_total / n_steps,
        "quant": quant_total / n_steps,
        "recon_expr": parts_total["recon_expr"] / n_steps,
        "recon_gpose": parts_total["recon_gpose"] / n_steps,
        "recon_jaw": parts_total["recon_jaw"] / n_steps,
        "recon_eyelids": parts_total["recon_eyelids"] / n_steps,
    }


def maybe_resume(model, optimizer, resume_path, device):
    if not resume_path:
        return 1
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt["epoch"]) + 1
    print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    return start_epoch


def init_wandb(cfg, run_name, log_dir):
    if not bool(getattr(cfg, "wandb", True)):
        print("W&B disabled by config (wandb: false).")
        return None

    if wandb is None:
        print("wandb package not available, continuing without W&B logging.")
        return None

    wandb_dir = Path(log_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = str(getattr(cfg, "wandb_mode", "online"))
    project = str(getattr(cfg, "wandb_project", "multitalk_custom_vq"))

    run = wandb.init(
        project=project,
        name=run_name,
        dir=str(wandb_dir),
        mode=mode,
        config=vars(cfg),
        reinit=True,
    )
    return run


def main():
    parser = argparse.ArgumentParser(description="Minimal stage1 VQ training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/talkinghead-1kh/stage1_style.yaml",
        help="Path to yaml config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = load_flat_config(args.config)
    set_seed(int(getattr(cfg, "manual_seed", 131)))

    device = torch.device(args.device)
    dataloaders = get_dataloaders(cfg)
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]

    model = get_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.base_lr))
    scheduler = StepLR(optimizer, step_size=int(cfg.step_size), gamma=float(cfg.gamma))

    save_root = Path(cfg.save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(getattr(cfg, "log_dir", save_root.parent))
    log_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(getattr(cfg, "wandb_run_name", save_root.parent.name))
    wandb_run = init_wandb(cfg, run_name=run_name, log_dir=log_dir)
    local_log_path = log_dir / "metrics.jsonl"

    start_epoch = 1
    resume_path = getattr(cfg, "resume", False)
    if isinstance(resume_path, str) and resume_path:
        start_epoch = maybe_resume(model, optimizer, resume_path, device)

    epochs = int(cfg.epochs)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(train_loader, model, optimizer, device, cfg, epoch, epochs, train_mode=True)
        scheduler.step()

        metrics = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "blendshapes_loss_train": train_metrics["recon"],
            "quan_loss_train": train_metrics["quant"],
            "recon_expr_train": train_metrics["recon_expr"],
            "recon_gpose_train": train_metrics["recon_gpose"],
            "recon_jaw_train": train_metrics["recon_jaw"],
            "recon_eyelids_train": train_metrics["recon_eyelids"],
        }

        print(
            f"[Epoch {epoch:03d}] "
            f"blendshapes_loss_train={metrics['blendshapes_loss_train']:.6f} "
            f"quan_loss_train={metrics['quan_loss_train']:.6f} "
            f"expr={metrics['recon_expr_train']:.6f} "
            f"gpose={metrics['recon_gpose_train']:.6f} "
            f"jaw={metrics['recon_jaw_train']:.6f} "
            f"eyelids={metrics['recon_eyelids_train']:.6f}"
        )

        if bool(getattr(cfg, "evaluate", True)) and epoch % int(cfg.eval_freq) == 0:
            val_metrics = run_epoch(valid_loader, model, optimizer, device, cfg, epoch, epochs, train_mode=False)
            metrics["blendshapes_loss_val"] = val_metrics["recon"]
            metrics["quan_loss_val"] = val_metrics["quant"]
            metrics["recon_expr_val"] = val_metrics["recon_expr"]
            metrics["recon_gpose_val"] = val_metrics["recon_gpose"]
            metrics["recon_jaw_val"] = val_metrics["recon_jaw"]
            metrics["recon_eyelids_val"] = val_metrics["recon_eyelids"]
            print(
                f"[Epoch {epoch:03d}] "
                f"blendshapes_loss_val={metrics['blendshapes_loss_val']:.6f} "
                f"quan_loss_val={metrics['quan_loss_val']:.6f} "
                f"expr={metrics['recon_expr_val']:.6f} "
                f"gpose={metrics['recon_gpose_val']:.6f} "
                f"jaw={metrics['recon_jaw_val']:.6f} "
                f"eyelids={metrics['recon_eyelids_val']:.6f}"
            )

        with local_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch)

        if epoch % int(cfg.eval_freq) == 0:
            save_checkpoint(model, optimizer, epoch, save_root)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
