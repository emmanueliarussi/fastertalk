#!/usr/bin/env python3
"""
Temporal realism critic for the style-swap branch.

The style-consistency loss only constrains a clip-level aggregate (p95 per
region/feature), which a flexible decoder can satisfy with unnatural motion
(e.g. a transient spike then a revert to a standard pose). This discriminator
judges the realism of *local temporal patches* of motion so that the swap decode
must look like real motion everywhere, not just hit the aggregate style number.

It is a PatchGAN over time: stride-1 dilated 1D convolutions keep the output
length equal to the input length, so per-frame logits can be masked directly
with the frame validity mask. A growing dilation gives a wide temporal receptive
field (it sees dynamics / speed), which is exactly what penalizes the
spike-then-freeze pattern.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TemporalPatchDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 58, hidden: int = 128, n_layers: int = 4,
                 kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        layers = []
        dilation = 1
        c_in = in_dim
        for _ in range(n_layers):
            pad = ((kernel - 1) // 2) * dilation
            layers.append(spectral_norm(
                nn.Conv1d(c_in, hidden, kernel, stride=1, padding=pad, dilation=dilation)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c_in = hidden
            dilation *= 2
        self.body = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Conv1d(hidden, 1, 1))

    def forward(self, x):              # x: [B, T, in_dim]
        h = x.transpose(1, 2)          # [B, in_dim, T]
        h = self.body(h)
        logit = self.head(h)           # [B, 1, T]
        return logit.squeeze(1)        # [B, T]


def _masked_mean(x, mask):
    m = mask.float()
    return (x * m).sum() / m.sum().clamp(min=1.0)


def disc_hinge_loss(d_real, d_fake, mask):
    """Hinge discriminator loss (masked over valid frames)."""
    loss_real = _masked_mean(torch.relu(1.0 - d_real), mask)
    loss_fake = _masked_mean(torch.relu(1.0 + d_fake), mask)
    return loss_real + loss_fake


def gen_hinge_loss(d_fake, mask):
    """Hinge generator loss: push fake logits up (masked over valid frames)."""
    return -_masked_mean(d_fake, mask)
