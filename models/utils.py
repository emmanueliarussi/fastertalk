# Attention-bias helpers for the autoregressive stage2 decoder.
# Borrowed from FaceFormer (https://github.com/EvelynFan/FaceFormer).
import math

import torch


def init_biased_mask(n_head, max_seq_len, period):
    """ALiBi-style temporal bias combined with a causal mask (shape [n_head, T, T])."""
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(
        torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1),
        period,
        rounding_mode="floor",
    )
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, : i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def enc_dec_mask(device, dataset, T, S):
    """Alignment mask between T motion frames (25 fps) and S audio frames.

    For every dataset except BIWI/vocaset each motion frame attends to its two
    corresponding audio frames (audio runs at ~50 fps -> 2x the motion rate).
    """
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    else:
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    return (mask == 1).to(device=device)
