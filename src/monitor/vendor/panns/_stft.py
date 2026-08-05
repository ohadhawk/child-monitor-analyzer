"""
Vendored STFT / log-mel front-end (derived from ``torchlibrosa``, MIT).

Only the two modules required by ``Cnn14_DecisionLevelMax`` are kept:
``Spectrogram`` and ``LogmelFilterBank``.  Attribute names are preserved
exactly, because the pre-trained checkpoint's ``state_dict`` keys depend on
them (``spectrogram_extractor.stft.conv_real.weight``,
``logmel_extractor.melW``).  Do not rename anything in this module.

See ``LICENSE-third-party.txt`` for attribution and the list of changes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    """Short-time Fourier transform implemented with two ``Conv1d`` layers.

    Numerically equivalent to ``librosa.stft`` for the parameters used by
    PANNs.  The DFT basis is baked into the convolution weights, which is why
    those weights appear in the checkpoint.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()

        if pad_mode not in ("constant", "reflect"):
            raise ValueError(f"Unsupported pad_mode: {pad_mode!r}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        if self.win_length is None:
            self.win_length = n_fft
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # librosa is already a first-class dependency of this project; it is
        # imported lazily so that importing this module stays cheap.
        import librosa

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)

        # DFT matrix: W[x, y] = exp(-2j*pi/n)**(x*y)
        (x, y) = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
        omega = np.exp(-2 * np.pi * 1j / n_fft)
        dft_matrix = np.power(omega, x * y)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=n_fft,
            stride=self.hop_length, padding=0, dilation=1, groups=1, bias=False,
        )
        self.conv_imag = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=n_fft,
            stride=self.hop_length, padding=0, dilation=1, groups=1, bias=False,
        )

        windowed = dft_matrix[:, 0:out_channels] * fft_window[:, None]
        self.conv_real.weight.data = torch.Tensor(np.real(windowed).T)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(np.imag(windowed).T)[:, None, :]

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: input ``(batch, samples)``.

        Returns ``(real, imag)``, each ``(batch, 1, time_steps, n_fft//2+1)``.
        """
        x = input[:, None, :]

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        return real, imag


class Spectrogram(nn.Module):
    """Power spectrogram built on :class:`STFT`."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.power = power
        self.stft = STFT(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Args: input ``(batch, samples)`` → ``(batch, 1, time, n_fft//2+1)``."""
        real, imag = self.stft.forward(input)
        spectrogram = real ** 2 + imag ** 2
        if self.power != 2.0:
            spectrogram = spectrogram ** (self.power / 2.0)
        return spectrogram


class LogmelFilterBank(nn.Module):
    """Mel filter bank + power-to-dB, matching ``librosa.filters.mel``."""

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        n_mels: int = 64,
        fmin: float = 0.0,
        fmax: float | None = None,
        is_log: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float | None = 80.0,
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax is None:
            fmax = sr // 2

        import librosa

        mel_w = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        ).T  # (n_fft // 2 + 1, mel_bins)
        self.melW = nn.Parameter(torch.Tensor(mel_w))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Args: input ``(..., n_fft//2+1)`` → ``(..., mel_bins)``."""
        mel_spectrogram = torch.matmul(input, self.melW)
        if self.is_log:
            return self.power_to_db(mel_spectrogram)
        return mel_spectrogram

    def power_to_db(self, input: torch.Tensor) -> torch.Tensor:
        """PyTorch equivalent of ``librosa.power_to_db``."""
        log_spec = 10.0 * torch.log10(
            torch.clamp(input, min=self.amin, max=np.inf)
        )
        log_spec = log_spec - 10.0 * np.log10(np.maximum(self.amin, self.ref))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.clamp(
                log_spec, min=log_spec.max().item() - self.top_db, max=np.inf,
            )
        return log_spec
