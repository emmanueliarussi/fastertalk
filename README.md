# FasterTalk Minimal (Stage 1 Only)

A clean, minimal extraction of the original project focused on one goal:
train and test the stage-1 VQ model on blendshapes.

## What is included

- `dataset/data_loader_joint_data_batched.py`: minimal chunked dataloader for `.npz` FLAME params
- `models/stage1.py`: stage-1 VQ autoencoder
- `config/talkinghead-1kh/stage1.yaml`: training config
- `main/train_joint_data_vq_bs.py`: minimal training script
- `main/test_joint_data_vq_bs.ipynb`: minimal test/eval notebook

## Install

```bash
cd /mnt/fastertalk
pip install -r requirements.txt
```

## Train

```bash
cd /mnt/fastertalk
python main/train_joint_data_vq_bs.py --config config/talkinghead-1kh/stage1.yaml
```

Checkpoints are saved under `save_path` from the config.

## Test (Notebook)

Open:

- `main/test_joint_data_vq_bs.ipynb`

Run all cells. The notebook:

- loads config
- loads the latest checkpoint from `save_path` (if available)
- runs one test batch
- prints reconstruction metrics

## Notes

- This version intentionally removes distributed training, W&B, rendering, and stage-2 logic.
- It does not modify the original `/mnt/fasttalk` project.
