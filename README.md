# FasterTalk

A transformer-based VQ-VAE for compressing and reconstructing FLAME facial motion sequences (expression, jaw, global pose, eyelids) from talking-head video data.

## Overview

Stage 1 trains a `GroupedResidualVQ` autoencoder over per-frame FLAME parameters (58-dim: 50 expression + 3 jaw + 3 global pose + 2 eyelids), producing a discrete motion codebook intended as a backbone for downstream audio-driven talking-head synthesis.

## Structure

- `models/stage1.py` — VQ autoencoder (transformer encoder/decoder + grouped residual VQ).
- `dataset/` — joint FLAME parameter dataloader with augmentations (jitter, dropout, smoothing, resampling, segment masking).
- `flame_model/` — FLAME mesh model and assets.
- `renderer/` — mesh rendering utilities.
- `losses.py` — per-group reconstruction + quantization losses.
- `config/talkinghead-1kh/stage1.yaml` — training config.
- `main/train_joint_data_vq_bs.py` — stage 1 training entry point.

## Train

```bash
python main/train_joint_data_vq_bs.py --config config/talkinghead-1kh/stage1.yaml
```

Checkpoints and metrics are written under `logs/stage1/`.

## Requirements

See [requirements.txt](requirements.txt).
