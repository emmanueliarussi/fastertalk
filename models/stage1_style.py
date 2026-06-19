import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import GroupedResidualVQ


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

    def code(self, T):
        """Return the raw positional code for the first T positions: (1, T, dim).

        Used by the time-varying FiLM as a content-free per-frame clock so the
        style modulation can vary across time (the same sinusoid for every clip).
        """
        return self.pe[:, :T]


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class StyleConditionedTransformerBlock(nn.Module):
    """Decoder transformer where each layer is followed by a FiLM modulation.

    Applying one FiLM per layer (rather than once before the whole decoder)
    keeps the style signal alive through every stage of decoding and gives the
    model more capacity to translate the style embedding into output variation.
    All per-layer FiLM modules are independent (not weight-shared).
    """

    def __init__(self, dim, heads=8, depth=6, dropout=0.1, style_dim=32,
                 time_varying=False, pos_dim=None, time_dim=16, smooth_win=3):
        super().__init__()
        self.time_varying = time_varying
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        # One FiLM per layer, all zero-initialized → identity at start of training.
        if time_varying:
            self.films = nn.ModuleList(
                [TimeVaryingFiLM(style_dim=style_dim, feature_dim=dim,
                                 pos_dim=pos_dim if pos_dim is not None else dim,
                                 time_dim=time_dim, smooth_win=smooth_win)
                 for _ in range(depth)]
            )
        else:
            self.films = nn.ModuleList(
                [FiLM(style_dim=style_dim, feature_dim=dim) for _ in range(depth)]
            )

    def forward(self, x, style_emb, src_key_padding_mask=None, pos_code=None):
        for layer, film in zip(self.layers, self.films):
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            if self.time_varying:
                x = film(x, style_emb, pos_code)
            else:
                x = film(x, style_emb)
        return x


class StyleMLP(nn.Module):
    """Maps a precomputed style scalar vector (B, n_scalars) to a style embedding (B, style_dim).

    Two-layer MLP with GELU so the model can learn non-linear combinations of
    the input scalars (e.g. lips_disp + lips_speed together mean something
    different from either alone).
    """

    def __init__(self, n_scalars: int, style_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_scalars, hidden),
            nn.GELU(),
            nn.Linear(hidden, style_dim),
        )

    def forward(self, scalars):          # (B, n_scalars) → (B, style_dim)
        return self.net(scalars)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: produces per-feature (gamma, beta) from a style embedding.

    Applied as:  output = x * (1 + gamma) + beta
    Initialized to zero weights so the branch starts as identity and the
    model learns to use it gradually — avoids disrupting early training.
    """

    def __init__(self, style_dim: int, feature_dim: int):
        super().__init__()
        self.to_gb = nn.Linear(style_dim, 2 * feature_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, x, style):         # x: (B, T, D),  style: (B, style_dim)
        gb = self.to_gb(style)           # (B, 2*D)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class TimeVaryingFiLM(nn.Module):
    """FiLM whose (gamma, beta) vary across time, driven by style + a positional clock.

    The per-clip style embedding is fused with a content-free positional code so
    the modulation can differ frame-to-frame. This adds the temporal degree of
    freedom a time-invariant FiLM lacks: it can change the speed/displacement
    ratio (the frequency of motion), not just its amplitude.

    A mild temporal low-pass (moving average of width ``smooth_win``) caps how
    spiky the modulation can be, preventing the swap-consistency loss from being
    gamed with single-frame high-frequency wiggles while still allowing natural
    fast transitions.  ``smooth_win=1`` disables smoothing.

    Zero-initialized output layer → starts as identity, like the plain FiLM.
    """

    def __init__(self, style_dim: int, feature_dim: int, pos_dim: int,
                 time_dim: int = 16, smooth_win: int = 3):
        super().__init__()
        self.smooth_win = int(smooth_win)
        self.pos_proj = nn.Linear(pos_dim, time_dim)
        self.to_gb = nn.Linear(style_dim + time_dim, 2 * feature_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def _smooth(self, m):                 # m: (B, T, D)
        k = self.smooth_win
        if k <= 1:
            return m
        k = k if k % 2 == 1 else k + 1    # force odd so length is preserved
        m = m.transpose(1, 2)             # (B, D, T)
        m = F.avg_pool1d(m, kernel_size=k, stride=1, padding=k // 2,
                         count_include_pad=False)
        return m.transpose(1, 2)          # (B, T, D)

    def forward(self, x, style, pos_code):
        # x: (B, T, D)  style: (B, S)  pos_code: (1, T, pos_dim) or (B, T, pos_dim)
        B, T, _ = x.shape
        t_emb = self.pos_proj(pos_code)               # (1 or B, T, time_dim)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(B, -1, -1)           # (B, T, time_dim)
        style_t = style.unsqueeze(1).expand(-1, T, -1)  # (B, T, S)
        gb = self.to_gb(torch.cat([style_t, t_emb], dim=-1))  # (B, T, 2D)
        gamma, beta = gb.chunk(2, dim=-1)             # (B, T, D) each
        gamma = self._smooth(gamma)
        beta = self._smooth(beta)
        return x * (1.0 + gamma) + beta


class StyleVQAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim  = getattr(args, "in_dim", 58)
        hidden_dim = getattr(args, "hidden_size", 512)
        codebook_size = getattr(args, "n_embed", 256)
        num_layers = getattr(args, "num_hidden_layers", 6)
        num_heads  = getattr(args, "num_attention_heads", 8)
        style_dim  = getattr(args, "style_dim", 32)
        # n_style_scalars must match N_REGIONS * N_FEATURES in the dataloader (3*2=6).
        n_scalars  = getattr(args, "n_style_scalars", 6)

        # Time-varying FiLM: let style modulation vary across frames so it can
        # control the speed/displacement ratio, not just amplitude.
        self.style_time_varying = bool(getattr(args, "style_time_varying", False))
        style_time_dim = int(getattr(args, "style_time_dim", 16))
        style_smooth_win = int(getattr(args, "style_film_smooth_win", 3))

        self.encoder_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(dim=hidden_dim)
        self.encoder_transformer = TransformerBlock(dim=hidden_dim, heads=num_heads, depth=num_layers)

        self.vq = GroupedResidualVQ(
            dim=hidden_dim,
            codebook_size=codebook_size,
            groups=32,
            num_quantizers=4,
            commitment_weight=0.1,
            decay=0.97,
            use_cosine_sim=False,
            rotation_trick=False,
        )

        self.style_mlp = StyleMLP(n_scalars=n_scalars, style_dim=style_dim)

        self.decoder_transformer = StyleConditionedTransformerBlock(
            dim=hidden_dim, heads=num_heads, depth=num_layers, style_dim=style_dim,
            time_varying=self.style_time_varying, pos_dim=hidden_dim,
            time_dim=style_time_dim, smooth_win=style_smooth_win,
        )
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)

    def _pos_code(self, T):
        """Positional clock for the decoder FiLM (None when not time-varying)."""
        return self.pos_enc.code(T) if self.style_time_varying else None

    def forward(self, blendshapes, mask=None, style=None):
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, _, vq_loss = self.vq(x)

        if style is None:
            style = torch.zeros(blendshapes.shape[0], self.style_mlp.net[0].in_features,
                                device=blendshapes.device)
        style_emb = self.style_mlp(style)          # (B, style_dim)

        x = self.decoder_transformer(quantized, style_emb,
                                     src_key_padding_mask=~mask if mask is not None else None,
                                     pos_code=self._pos_code(quantized.shape[1]))
        decoded = self.decoder_proj(x)
        return decoded, vq_loss

    def encode(self, blendshapes, mask=None):
        """Encode to continuous latent z and quantized codes.

        Returns (z, quantized, vq_loss) where z is the pre-quantization encoder
        output (continuous, fully differentiable) and quantized are the VQ codes.
        """
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        z = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        quantized, _, vq_loss = self.vq(z)
        return z, quantized, vq_loss

    def encode_continuous(self, blendshapes, mask=None):
        """Encoder output BEFORE quantization (no VQ call, no codebook EMA update).

        Used for the content / code-preservation anchor where we only need the
        continuous latent and must not pollute the VQ codebook statistics with a
        synthetic (decoded) sample.
        """
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        return self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

    def get_quant(self, blendshapes, mask=None):
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        quantized, indices, _ = self.vq(x)
        return quantized, indices

    def decode(self, quantized, mask=None, style=None):
        """Decode from quantized features with an explicit style vector (B, n_scalars)."""
        if style is None:
            style = torch.zeros(quantized.shape[0], self.style_mlp.net[0].in_features,
                                device=quantized.device)
        style_emb = self.style_mlp(style)
        x = self.decoder_transformer(quantized, style_emb,
                                     src_key_padding_mask=~mask if mask is not None else None,
                                     pos_code=self._pos_code(quantized.shape[1]))
        return self.decoder_proj(x)
