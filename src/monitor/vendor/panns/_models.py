"""
Vendored ``Cnn14_DecisionLevelMax`` architecture (derived from
``panns_inference``, MIT).

Inference-only.  Training-specific machinery (SpecAugmentation, mixup and
weight initialisation) has been removed: it cannot execute under
``model.eval()`` / ``torch.no_grad()``, and none of it contributes entries to
the checkpoint's ``state_dict``.

The module/attribute names below are load-bearing — the pre-trained
checkpoint's 84 ``state_dict`` keys are derived from them and the model is
loaded with ``strict=True``.  Do not rename anything in this module.

See ``LICENSE-third-party.txt`` for attribution and the list of changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._stft import LogmelFilterBank, Spectrogram

# AudioSet / PANNs front-end parameters. These must match the values the
# checkpoint was trained with, otherwise the STFT and mel weights loaded from
# the checkpoint will not correspond to the audio being fed in.
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
CLASSES_NUM = 527


def pad_framewise_output(
    framewise_output: torch.Tensor, frames_num: int
) -> torch.Tensor:
    """Repeat the final frame until *framewise_output* has *frames_num* frames.

    Args:
        framewise_output: ``(batch, frames, classes)``.
        frames_num: Target number of frames.

    Returns:
        ``(batch, frames_num, classes)``.
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    return torch.cat((framewise_output, pad), dim=1)


class NearestInterpolator(nn.Module):
    """Upsample along the time axis by an integer *ratio* (nearest neighbour)."""

    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``(batch, time, classes)`` → ``(batch, time*ratio, classes)``."""
        batch_size, time_steps, classes_num = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, self.ratio, 1)
        return upsampled.reshape(batch_size, time_steps * self.ratio, classes_num)


class Interpolator(nn.Module):
    """Thin dispatcher kept for structural parity with the upstream model."""

    def __init__(self, ratio: int, interpolate_mode: str = "nearest") -> None:
        super().__init__()
        if interpolate_mode != "nearest":
            raise ValueError(
                f"Only 'nearest' interpolation is vendored, got {interpolate_mode!r}"
            )
        self.interpolator = NearestInterpolator(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.interpolator(x)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch-norm, followed by pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(
        self,
        input: torch.Tensor,
        pool_size: tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg+max":
            return (
                F.avg_pool2d(x, kernel_size=pool_size)
                + F.max_pool2d(x, kernel_size=pool_size)
            )
        raise ValueError(f"Unknown pool_type: {pool_type!r}")


class Cnn14_DecisionLevelMax(nn.Module):  # noqa: N801 - checkpoint-defined name
    """PANNs CNN14 with frame-level (decision-level max) outputs."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_size: int = WINDOW_SIZE,
        hop_size: int = HOP_SIZE,
        mel_bins: int = MEL_BINS,
        fmin: int = FMIN,
        fmax: int = FMAX,
        classes_num: int = CLASSES_NUM,
        interpolate_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate_ratio = 32  # Downsampling factor of the conv stack.

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
            fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.interpolator = Interpolator(
            ratio=self.interpolate_ratio, interpolate_mode=interpolate_mode,
        )

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args: input ``(batch, samples)`` at 32 kHz.

        Returns a dict with ``framewise_output`` ``(batch, frames, classes)``
        and ``clipwise_output`` ``(batch, classes)``.
        """
        x = self.spectrogram_extractor(input)  # (batch, 1, time, freq)
        x = self.logmel_extractor(x)           # (batch, 1, time, mel)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # NOTE: dropout is retained with training=False so it is a no-op. It is
        # kept for structural fidelity with the reference implementation.
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output, _ = torch.max(segmentwise_output, dim=1)

        framewise_output = self.interpolator(segmentwise_output)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        return {
            "framewise_output": framewise_output,
            "clipwise_output": clipwise_output,
        }
