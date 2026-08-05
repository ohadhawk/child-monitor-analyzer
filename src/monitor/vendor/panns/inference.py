"""
Sound-event-detection inference wrapper (derived from
``panns_inference.inference``, MIT).

Security-relevant differences from upstream:

* The checkpoint is loaded with ``weights_only=True``.  Upstream used
  unrestricted ``torch.load``, i.e. full pickle deserialisation, which
  executes arbitrary code contained in the checkpoint file.
* There is no auto-download.  Upstream ran
  ``os.system('wget -O "{path}" ...')`` with an interpolated path, a shell
  command-injection vector.  Callers must supply an already-verified path
  (see ``monitor.model_cache.ensure_panns_checkpoint``, which verifies a
  pinned SHA-256 digest).
* ``matplotlib.pyplot`` is no longer imported.

See ``LICENSE-third-party.txt`` for attribution and the list of changes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from ._models import CLASSES_NUM, Cnn14_DecisionLevelMax

log = logging.getLogger(__name__)

#: Devices this wrapper knows how to place the model on.
_SUPPORTED_DEVICES = ("cpu", "cuda", "xpu")


class SoundEventDetection:
    """Frame-level AudioSet event detection using PANNs CNN14.

    Args:
        checkpoint_path: Path to a verified ``Cnn14_DecisionLevelMax`` ``.pth``.
        device: ``"cpu"``, ``"cuda"`` or ``"xpu"``.  Falls back to CPU when the
            requested accelerator is unavailable.
        classes_num: Number of output classes (527 for AudioSet).

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        ValueError: If *device* is not supported.
        RuntimeError: If the checkpoint does not match the model architecture.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        classes_num: int = CLASSES_NUM,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PANNs checkpoint not found: {checkpoint_path}"
            )
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device {device!r}; expected one of "
                f"{_SUPPORTED_DEVICES}"
            )

        self.device = self._resolve_device(device)
        self.classes_num = classes_num

        t0 = time.perf_counter()
        self.model = Cnn14_DecisionLevelMax(classes_num=classes_num)

        # weights_only=True restricts the unpickler to plain tensors and
        # primitives. A tampered checkpoint can therefore not execute code.
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True,
        )
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            raise RuntimeError(
                f"PANNs checkpoint has no 'model' entry: {checkpoint_path}"
            )

        # strict=True: every one of the 84 tensors must be present and shaped
        # correctly. A silent partial load would produce plausible-looking but
        # wrong detections, which is worse than a crash.
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(torch.device(self.device))

        log.info(
            "PANNs SED model loaded from %s on %s in %.2fs "
            "(checkpoint iteration=%s).",
            checkpoint_path.name,
            self.device,
            time.perf_counter() - t0,
            checkpoint.get("iteration", "?"),
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Return *requested* if the accelerator is usable, else ``"cpu"``."""
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            log.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        if requested == "xpu":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            log.warning("XPU requested but unavailable; falling back to CPU.")
            return "cpu"
        return "cpu"

    def to(self, device: str) -> "SoundEventDetection":
        """Move the model to *device* in place and return ``self``."""
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device {device!r}")
        resolved = self._resolve_device(device)
        self.model.to(torch.device(resolved))
        self.device = resolved
        log.info("PANNs SED model moved to %s.", resolved)
        return self

    def inference(self, audio: np.ndarray) -> np.ndarray:
        """Run frame-level event detection on a batch of waveforms.

        Args:
            audio: ``(batch, samples)`` float32 waveform at 32 kHz, or a 1-D
                ``(samples,)`` array which is promoted to a batch of one.

        Returns:
            ``(batch, frames, classes)`` float32 probabilities in ``[0, 1]``.

        Raises:
            ValueError: If *audio* has an unsupported shape or dtype.
        """
        array = np.asarray(audio)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2:
            raise ValueError(
                f"Expected audio of shape (batch, samples), got {array.shape}"
            )
        if array.size == 0:
            raise ValueError("Cannot run inference on empty audio.")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(
                f"Expected floating-point audio, got dtype {array.dtype}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        tensor = tensor.to(torch.device(self.device))

        with torch.no_grad():
            self.model.eval()
            output = self.model(tensor)

        return output["framewise_output"].detach().cpu().numpy()
