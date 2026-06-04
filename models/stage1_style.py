import math

import torch
import torch.nn as nn
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


class StyleVQAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = getattr(args, "in_dim", 58)
        hidden_dim = getattr(args, "hidden_size", 512)
        codebook_size = getattr(args, "n_embed", 256)
        num_layers = getattr(args, "num_hidden_layers", 6)
        num_heads = getattr(args, "num_attention_heads", 8)

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

        self.decoder_transformer = TransformerBlock(dim=hidden_dim, heads=num_heads, depth=num_layers)
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, blendshapes, mask=None):
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, _, vq_loss = self.vq(x)

        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)
        decoded = self.decoder_proj(x)
        return decoded, vq_loss

    def get_quant(self, blendshapes, mask=None):
        x = self.encoder_proj(blendshapes)
        x = self.pos_enc(x)
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        quantized, indices, _ = self.vq(x)
        return quantized, indices

    def decode(self, quantized, mask=None):
        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)
        return self.decoder_proj(x)
