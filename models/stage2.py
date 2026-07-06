"""Stage-2 audio-to-motion predictor for the fastertalk pipeline.

Maps an input waveform to the stage-1 (StyleVQAutoEncoder) latent codebook
space via a HuBERT/Wav2Vec2 audio encoder and an autoregressive Transformer
decoder, then decodes to 58-dim blendshapes through the *frozen* stage-1 model.

This is a stripped-down version of the original ``stage2v2`` FasterTalk: all
AdaIN / style-encoder machinery has been removed. Expressiveness ("style") is
no longer injected here — it is controlled directly by the stage-1 decoder,
which takes an explicit per-clip style-scalar vector. Stage-2 therefore only
learns the audio -> content-code mapping; the style vector is passed straight
through to ``autoencoder.decode`` at both train and inference time.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from transformers import Wav2Vec2Model

from models.utils import init_biased_mask, enc_dec_mask


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, T, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FasterTalk(nn.Module):
    """Audio -> stage-1 latent code -> blendshapes (frozen stage-1 decode)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.device = args.device

        feature_dim = int(getattr(args, "feature_dim", 1024))
        blendshapes_dim = int(getattr(args, "blendshapes_dim", 58))
        # Latent width the stage-1 codebook lives in (its transformer hidden dim).
        self.embed_dim = int(getattr(args, "hidden_size", 512))

        # ── Audio encoder (HuBERT / Wav2Vec2) ───────────────────────────────
        self.audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec2model_path)
        print("Loading pretrained audio encoder: {}".format(args.wav2vec2model_path))
        self.audio_encoder.feature_extractor._freeze_parameters()
        # Optionally freeze the *entire* audio encoder (not just the conv feature
        # extractor). Recommended when training from scratch: a trainable HuBERT
        # can overpower the tiny decoder early and collapse to a mean pose.
        self.freeze_audio_encoder = bool(getattr(args, "freeze_audio_encoder", False))
        if self.freeze_audio_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            self.audio_encoder.eval()
            print("Audio encoder fully frozen (freeze_audio_encoder=True).")
        audio_hidden = int(self.audio_encoder.config.hidden_size)
        self.audio_feature_map = nn.Linear(audio_hidden, feature_dim)

        # ── Motion (target) embedding + positional encoding ─────────────────
        self.blendshapes_map = nn.Linear(blendshapes_dim, feature_dim)
        self.pos_enc = PositionalEncoding(dim=feature_dim)

        # Temporal ALiBi bias for the autoregressive self-attention.
        self.biased_mask = init_biased_mask(
            n_head=1, max_seq_len=600, period=int(getattr(args, "period", 25))
        )

        # ── Plain Transformer decoder (no AdaIN / style conditioning) ───────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=int(getattr(args, "n_head", 4)),
            dim_feedforward=feature_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(getattr(args, "num_layers", 4)),
            norm=nn.LayerNorm(feature_dim),
        )

        # Map decoder output to the stage-1 latent width.
        self.feat_map = nn.Linear(feature_dim, self.embed_dim)

        # ── Frozen stage-1 autoencoder (StyleVQAutoEncoder) ─────────────────
        from models.stage1_style import StyleVQAutoEncoder

        self.autoencoder = StyleVQAutoEncoder(args)
        ckpt = torch.load(args.vqvae_pretrained_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.autoencoder.load_state_dict(state_dict)
        print("Loading pretrained stage-1 vq: {}".format(args.vqvae_pretrained_path))
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

        # Number of style scalars the stage-1 decoder expects (e.g. 8).
        self.n_style_scalars = int(getattr(args, "n_style_scalars", 8))

    # -----------------------------------------------------------------------
    def _zero_style(self, batch, device):
        return torch.zeros(batch, self.n_style_scalars, device=device)

    def _encode_audio(self, audio, attention_mask=None):
        """Waveform [B, 1, N] or [B, N] -> audio features [B, T_audio, feature_dim]."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        hidden = self.audio_encoder(audio, attention_mask=attention_mask).last_hidden_state
        return self.audio_feature_map(hidden)

    # -----------------------------------------------------------------------
    def forward(self, padded_blendshapes, blendshapes_mask, padded_audios, audio_mask,
                criterion, style=None):
        """Teacher-forced training step.

        padded_blendshapes : [B, T, 58]  (clean target motion)
        blendshapes_mask   : [B, T]      (bool, True = valid frame)
        padded_audios      : [B, 1, N]   (raw waveform samples)
        audio_mask         : [B, N]      (bool/long, wav2vec2 attention mask)
        style              : [B, n_style_scalars] or None (stage-1 decode style)
        """
        B, T, D = padded_blendshapes.shape

        # ── Audio features ──────────────────────────────────────────────────
        hidden_states = self._encode_audio(padded_audios, audio_mask)
        _, T_audio, _ = hidden_states.shape

        frame_num = blendshapes_mask.sum(dim=1)                              # [B]
        valid_audio_lens = torch.clamp(frame_num * 2, max=T_audio)          # [B]
        time_range_audio = torch.arange(T_audio, device=hidden_states.device).unsqueeze(0)
        memory_key_padding_mask = time_range_audio >= valid_audio_lens.unsqueeze(1)  # [B, T_audio]

        # ── Stage-1 target latent (frozen encoder) ──────────────────────────
        with torch.no_grad():
            encoded = self.autoencoder.encode_continuous(padded_blendshapes, blendshapes_mask)

        if style is None:
            style = self._zero_style(B, padded_blendshapes.device)

        # ── Autoregressive teacher forcing (shift target right by one) ──────
        zero_frame = torch.zeros(B, 1, D, device=padded_blendshapes.device)
        shifted = torch.cat((zero_frame, padded_blendshapes[:, :-1, :]), dim=1)
        tgt = self.blendshapes_map(shifted)
        tgt = self.pos_enc(tgt)

        tgt_mask = self.biased_mask[0, :T, :T].clone().detach().to(self.device)
        memory_mask = enc_dec_mask(self.device, self.dataset, T, T_audio)

        feat_out = self.transformer_decoder(
            tgt=tgt,
            memory=hidden_states,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=~blendshapes_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        feat_out = self.feat_map(feat_out)                                   # [B, T, embed_dim]

        valid = blendshapes_mask.unsqueeze(-1).float()                       # [B, T, 1]

        # Regression: predicted latent must match the stage-1 encoder latent
        # (masked so padded frames do not distort the target).
        diff_reg = (feat_out - encoded.detach()) ** 2 * valid
        loss_reg = diff_reg.sum() / valid.sum().clamp_min(1.0) / feat_out.shape[-1]

        # Reconstruction: quantize the prediction and decode through stage-1.
        feat_out_q, _, _ = self.autoencoder.vq(feat_out)
        blendshapes_out = self.autoencoder.decode(feat_out_q, blendshapes_mask, style=style)
        diff_bs = (blendshapes_out - padded_blendshapes) ** 2 * valid
        loss_blendshapes = diff_bs.sum() / valid.sum().clamp_min(1.0) / D

        # Velocity / delta loss: penalise static ("move once and freeze") motion
        # by matching frame-to-frame differences. This is the key anti-collapse
        # term the working stage-2 relied on.
        delta_weight = float(getattr(self.args, "blendshape_delta_weight", 0.0))
        loss_delta = torch.zeros((), device=blendshapes_out.device)
        if delta_weight > 0:
            pred_delta = blendshapes_out[:, 1:] - blendshapes_out[:, :-1]
            gt_delta = padded_blendshapes[:, 1:] - padded_blendshapes[:, :-1]
            delta_mask = (blendshapes_mask[:, 1:] & blendshapes_mask[:, :-1]).unsqueeze(-1).float()
            # Optionally exclude the pose block (gpose+jaw) from the velocity term,
            # matching the working recipe (pose dynamics are shaped elsewhere).
            channel_mask = torch.ones(D, device=blendshapes_out.device)
            pose_start = int(getattr(self.args, "pose_start_idx", 50))
            pose_dim = int(getattr(self.args, "pose_dim", 6))
            pose_start = max(0, min(pose_start, D))
            pose_end = max(pose_start, min(pose_start + max(0, pose_dim), D))
            if pose_end > pose_start:
                channel_mask[pose_start:pose_end] = 0.0
            diff_d = (pred_delta - gt_delta) ** 2 * delta_mask * channel_mask.view(1, 1, -1)
            denom = delta_mask.sum().clamp_min(1.0) * channel_mask.sum().clamp_min(1.0)
            loss_delta = diff_d.sum() / denom

        total_loss = loss_blendshapes + loss_reg + delta_weight * loss_delta
        return total_loss, [loss_blendshapes, loss_reg, loss_delta]

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, audio, style=None):
        """Autoregressive inference with VQ quantization. Returns [1, T, 58]."""
        hidden_states = self._encode_audio(audio)
        frame_num = hidden_states.shape[1] // 2

        if style is None:
            style = self._zero_style(hidden_states.shape[0], self.device)

        blendshapes_emb = None
        feat_out = None
        for i in range(frame_num):
            if i == 0:
                blendshapes_emb = torch.zeros(
                    (hidden_states.shape[0], 1, self.args.feature_dim), device=self.device
                )
            blendshapes_input = self.pos_enc(blendshapes_emb)

            tgt_mask = self.biased_mask[:, :blendshapes_input.shape[1], :blendshapes_input.shape[1]] \
                .clone().detach().to(self.device).squeeze(0)
            memory_mask = enc_dec_mask(self.device, self.dataset,
                                       blendshapes_input.shape[1], hidden_states.shape[1])

            feat_out = self.transformer_decoder(
                tgt=blendshapes_input,
                memory=hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            feat_out = self.feat_map(feat_out)
            feat_out_q, _, _ = self.autoencoder.vq(feat_out)

            # Decode current prefix; on the first step duplicate a frame so the
            # stage-1 decoder (which needs T>=1) yields a usable last frame.
            if i == 0:
                dec = self.autoencoder.decode(torch.cat([feat_out_q, feat_out_q], dim=1), style=style)
                blendshapes_out_q = dec[:, 0].unsqueeze(1)
            else:
                blendshapes_out_q = self.autoencoder.decode(feat_out_q, style=style)

            if i != frame_num - 1:
                last_frame = blendshapes_out_q[:, -1, :]
                new_output = self.blendshapes_map(last_frame).unsqueeze(1)
                blendshapes_emb = torch.cat((blendshapes_emb, new_output), dim=1)

        feat_out_q, _, _ = self.autoencoder.vq(feat_out)
        blendshapes_out = self.autoencoder.decode(feat_out_q, style=style)
        return blendshapes_out

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def predict_no_quantizer(self, audio, style=None):
        """Autoregressive inference WITHOUT VQ quantization (uses continuous latent)."""
        hidden_states = self._encode_audio(audio)
        frame_num = hidden_states.shape[1] // 2

        if style is None:
            style = self._zero_style(hidden_states.shape[0], self.device)

        blendshapes_emb = None
        feat_out = None
        for i in range(frame_num):
            if i == 0:
                blendshapes_emb = torch.zeros(
                    (hidden_states.shape[0], 1, self.args.feature_dim), device=self.device
                )
            blendshapes_input = self.pos_enc(blendshapes_emb)

            tgt_mask = self.biased_mask[:, :blendshapes_input.shape[1], :blendshapes_input.shape[1]] \
                .clone().detach().to(self.device).squeeze(0)
            memory_mask = enc_dec_mask(self.device, self.dataset,
                                       blendshapes_input.shape[1], hidden_states.shape[1])

            feat_out = self.transformer_decoder(
                tgt=blendshapes_input,
                memory=hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            feat_out = self.feat_map(feat_out)

            if i == 0:
                dec = self.autoencoder.decode(torch.cat([feat_out, feat_out], dim=1), style=style)
                blendshapes_out_q = dec[:, 0].unsqueeze(1)
            else:
                blendshapes_out_q = self.autoencoder.decode(feat_out, style=style)

            if i != frame_num - 1:
                last_frame = blendshapes_out_q[:, -1, :]
                new_output = self.blendshapes_map(last_frame).unsqueeze(1)
                blendshapes_emb = torch.cat((blendshapes_emb, new_output), dim=1)

        blendshapes_out = self.autoencoder.decode(feat_out, style=style)
        return blendshapes_out
