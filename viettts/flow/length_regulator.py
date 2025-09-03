from typing import Tuple
import torch.nn as nn
import torch
from torch.nn import functional as F
from viettts.utils.mask import make_pad_mask


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='linear')
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens

    def inference(self, x1, x2, mel_len1, mel_len2, input_frame_rate=50):
        # in inference mode, interpolate prompt token and token(head/mid/tail) separately
        # x in (B, T, D)

        if mel_len2 <= 0 or x2.shape[1] == 0:
            raise ValueError(
                f"Invalid mel_len2={mel_len2}, x2 shape={x2.shape}. "
                f"Check upstream tokenization or model output."
            )

        if x2.shape[1] > 40:
            x2_head = F.interpolate(
                x2[:, :20].transpose(1, 2).contiguous(),
                size=int(20 / input_frame_rate * 22050 / 256),
                mode='linear'
            )
            x2_mid = F.interpolate(
                x2[:, 20:-20].transpose(1, 2).contiguous(),
                size=max(mel_len2 - int(20 / input_frame_rate * 22050 / 256) * 2, 1),
                mode='linear'
            )
            x2_tail = F.interpolate(
                x2[:, -20:].transpose(1, 2).contiguous(),
                size=int(20 / input_frame_rate * 22050 / 256),
                mode='linear'
            )
            x2 = torch.concat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(
                x2.transpose(1, 2).contiguous(),
                size=max(mel_len2, 1),   # tránh size=0
                mode='linear'
            )

        if x1.shape[1] != 0 and mel_len1 > 0:
            x1 = F.interpolate(
                x1.transpose(1, 2).contiguous(),
                size=max(mel_len1, 1),
                mode='linear'
            )
            x = torch.concat([x1, x2], dim=2)
        else:
            x = x2

        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2

