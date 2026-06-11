import torch


# Channel layout (per timestep, 58 dims total):
#   [0:50]  expression
#   [50:53] global pose (gpose)
#   [53:56] jaw
#   [56:58] eyelids  (placeholder values in this dataset)
EXPR_SLICE = slice(0, 50)
GPOSE_SLICE = slice(50, 53)
JAW_SLICE = slice(53, 56)
EYELID_SLICE = slice(56, 58)


def _masked_group_mse(pred, target, mask, group_slice):
    """MSE within a channel group, averaged over valid time and over the group's dims."""
    p = pred[..., group_slice]
    t = target[..., group_slice]
    m = mask.unsqueeze(-1).float()  # [B, T, 1]
    sq = (p - t) ** 2
    denom = torch.clamp(m.sum() * p.shape[-1], min=1.0)
    return (sq * m).sum() / denom


def masked_latent_mse(a, b, mask):
    """MSE between two latent sequences [B, T, D], averaged over valid timesteps.

    Used for the content / code-preservation anchor: re-encode the style-swapped
    decode and require its continuous encoder latent to match the original
    content latent (so swapping style does not change *what* is being said).
    """
    m = mask.unsqueeze(-1).float()           # [B, T, 1]
    sq = (a - b) ** 2
    denom = torch.clamp(m.sum() * a.shape[-1], min=1.0)
    return (sq * m).sum() / denom


def masked_grouped_recon(pred, target, mask, w_expr=1.0, w_gpose=5.0, w_jaw=2.0, w_eyelids=1.0):
    """Group-weighted reconstruction loss.

    Each group's MSE is averaged over its own dims, so a 50-dim group (expression)
    does not drown a 3-dim group (gpose/jaw). Groups are then combined with explicit
    weights. Pass weight=0 to disable a group (eyelids are placeholder data here).

    Returns (total_loss, parts_dict).
    """
    parts = {}
    total = pred.new_zeros(())

    if w_expr > 0:
        l = _masked_group_mse(pred, target, mask, EXPR_SLICE)
        total = total + w_expr * l
        parts["recon_expr"] = l.detach()
    if w_gpose > 0:
        l = _masked_group_mse(pred, target, mask, GPOSE_SLICE)
        total = total + w_gpose * l
        parts["recon_gpose"] = l.detach()
    if w_jaw > 0:
        l = _masked_group_mse(pred, target, mask, JAW_SLICE)
        total = total + w_jaw * l
        parts["recon_jaw"] = l.detach()
    if w_eyelids > 0:
        l = _masked_group_mse(pred, target, mask, EYELID_SLICE)
        total = total + w_eyelids * l
        parts["recon_eyelids"] = l.detach()

    return total, parts


def calc_vq_loss(
    pred,
    target,
    quant_loss,
    mask,
    quant_loss_weight=1.0,
    w_expr=1.0,
    w_gpose=5.0,
    w_jaw=2.0,
    w_eyelids=1.0,
):
    recon, parts = masked_grouped_recon(
        pred, target, mask,
        w_expr=w_expr, w_gpose=w_gpose, w_jaw=w_jaw, w_eyelids=w_eyelids,
    )
    q = quant_loss.mean()
    total = recon + quant_loss_weight * q

    info = {"recon": recon.detach(), "quant": q.detach()}
    info.update(parts)
    return total, info
