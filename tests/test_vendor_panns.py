"""
Regression tests for the vendored PANNs inference code.

The critical property is **numerical parity with upstream**: vendoring must not
change a single detection.  Where the upstream ``panns_inference`` /
``torchlibrosa`` packages are still installed, the tests compare against them
directly.  Once those packages are uninstalled the comparison tests skip, and
the remaining tests still lock in the structural contract (state_dict key set,
output shapes, dtype/range invariants) so future edits cannot silently break
the model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from monitor.vendor.panns import SoundEventDetection  # noqa: E402
from monitor.vendor.panns._models import (  # noqa: E402
    CLASSES_NUM,
    HOP_SIZE,
    SAMPLE_RATE,
    Cnn14_DecisionLevelMax,
    pad_framewise_output,
)
from monitor.vendor.panns._stft import (  # noqa: E402
    LogmelFilterBank,
    Spectrogram,
    STFT,
)

# The checkpoint's state_dict has exactly this many tensors. If a refactor
# adds or removes a parameter, strict loading breaks -- catch it here first.
EXPECTED_STATE_DICT_KEYS = 84

# Two seconds is enough to exercise every pooling stage and the interpolator.
_TEST_SECONDS = 2


def _deterministic_audio(seconds: int = _TEST_SECONDS) -> np.ndarray:
    """Reproducible pseudo-audio batch of shape (1, seconds * 32000)."""
    rng = np.random.default_rng(20260803)
    return rng.standard_normal(
        (1, seconds * SAMPLE_RATE), dtype=np.float32
    ) * 0.1


def _upstream_available() -> bool:
    try:
        import panns_inference  # noqa: F401
        import torchlibrosa  # noqa: F401
    except Exception:
        return False
    return True


upstream_only = pytest.mark.skipif(
    not _upstream_available(),
    reason="upstream panns_inference/torchlibrosa not installed (expected "
           "after vendoring); parity was verified before removal",
)


# ---------------------------------------------------------------------------
# Structural contract -- runs with or without upstream, without the checkpoint
# ---------------------------------------------------------------------------


def test_state_dict_key_count_is_stable() -> None:
    """The vendored model must expose exactly the checkpoint's tensor set."""
    model = Cnn14_DecisionLevelMax()
    assert len(model.state_dict()) == EXPECTED_STATE_DICT_KEYS


def test_state_dict_contains_checkpoint_critical_keys() -> None:
    """These names are baked into the checkpoint and must never be renamed."""
    keys = set(Cnn14_DecisionLevelMax().state_dict())
    for required in (
        "spectrogram_extractor.stft.conv_real.weight",
        "spectrogram_extractor.stft.conv_imag.weight",
        "logmel_extractor.melW",
        "bn0.weight",
        "conv_block1.conv1.weight",
        "conv_block6.bn2.running_var",
        "fc1.weight",
        "fc_audioset.bias",
    ):
        assert required in keys, f"missing checkpoint key: {required}"


def test_no_training_only_modules_in_state_dict() -> None:
    """SpecAugmentation was dropped; assert nothing training-only crept back."""
    keys = " ".join(Cnn14_DecisionLevelMax().state_dict())
    assert "spec_augmenter" not in keys
    assert "dropper" not in keys


def test_vendored_package_does_not_import_matplotlib() -> None:
    """Dropping matplotlib/Pillow from the bundle is the point of vendoring."""
    import importlib
    import sys

    for name in list(sys.modules):
        if name.startswith(("matplotlib", "PIL")):
            del sys.modules[name]

    importlib.import_module("monitor.vendor.panns")

    assert not any(
        name.startswith(("matplotlib", "PIL")) for name in sys.modules
    ), "importing monitor.vendor.panns must not pull in matplotlib or Pillow"


def test_forward_output_shapes_and_ranges() -> None:
    """Untrained forward pass still has to obey the documented contract."""
    model = Cnn14_DecisionLevelMax()
    model.eval()
    audio = torch.from_numpy(_deterministic_audio())

    with torch.no_grad():
        out = model(audio)

    frames = out["framewise_output"]
    clips = out["clipwise_output"]

    expected_frames = _TEST_SECONDS * SAMPLE_RATE // HOP_SIZE + 1
    assert frames.shape == (1, expected_frames, CLASSES_NUM)
    assert clips.shape == (1, CLASSES_NUM)
    # Both heads are sigmoid outputs.
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    assert float(clips.min()) >= 0.0 and float(clips.max()) <= 1.0


def test_pad_framewise_output_repeats_last_frame() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    padded = pad_framewise_output(x, frames_num=4)
    assert padded.shape == (1, 4, 2)
    assert torch.equal(padded[0, 2], padded[0, 3])
    assert torch.equal(padded[0, 3], x[0, 1])


def test_spectrogram_matches_librosa_stft() -> None:
    """The Conv1d STFT must agree with librosa's reference implementation."""
    librosa = pytest.importorskip("librosa")

    n_fft, hop = 1024, 320
    audio = _deterministic_audio(seconds=1)

    spec = Spectrogram(
        n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann",
        center=True, pad_mode="reflect", freeze_parameters=True,
    )
    spec.eval()
    with torch.no_grad():
        ours = spec(torch.from_numpy(audio))[0, 0].numpy()

    reference = np.abs(
        librosa.stft(
            audio[0], n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window="hann", center=True, pad_mode="reflect",
        )
    ).T ** 2

    # Conv1d accumulates in float32; librosa uses an FFT. Relative agreement
    # to ~1e-4 is the expected level for a 1024-point transform.
    np.testing.assert_allclose(ours, reference, rtol=1e-3, atol=1e-3)


def test_logmel_filterbank_matches_librosa_mel() -> None:
    librosa = pytest.importorskip("librosa")

    fb = LogmelFilterBank(
        sr=SAMPLE_RATE, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
        ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
    )
    reference = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
    ).T
    np.testing.assert_allclose(
        fb.melW.detach().numpy(), reference, rtol=0, atol=0,
    )


def test_stft_rejects_bad_pad_mode() -> None:
    with pytest.raises(ValueError, match="pad_mode"):
        STFT(n_fft=64, pad_mode="nonsense")


# ---------------------------------------------------------------------------
# Numerical parity with upstream -- the core vendoring guarantee
# ---------------------------------------------------------------------------


@upstream_only
def test_vendored_state_dict_keys_match_upstream() -> None:
    """Key-for-key identity is what makes the checkpoint loadable."""
    from panns_inference.models import (
        Cnn14_DecisionLevelMax as UpstreamModel,
    )

    upstream = UpstreamModel(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
        fmin=50, fmax=14000, classes_num=CLASSES_NUM,
    )
    assert set(Cnn14_DecisionLevelMax().state_dict()) == set(
        upstream.state_dict()
    )


@upstream_only
def test_vendored_forward_matches_upstream_bitwise() -> None:
    """Same weights in → same probabilities out, to float32 precision."""
    from panns_inference.models import (
        Cnn14_DecisionLevelMax as UpstreamModel,
    )

    torch.manual_seed(0)
    upstream = UpstreamModel(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
        fmin=50, fmax=14000, classes_num=CLASSES_NUM,
    )
    ours = Cnn14_DecisionLevelMax()
    ours.load_state_dict(upstream.state_dict(), strict=True)

    upstream.eval()
    ours.eval()
    audio = torch.from_numpy(_deterministic_audio())

    with torch.no_grad():
        expected = upstream(audio, None)["framewise_output"].numpy()
        actual = ours(audio)["framewise_output"].numpy()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.slow
@upstream_only
def test_real_checkpoint_parity_with_upstream(panns_checkpoint) -> None:
    """End-to-end parity using the actual 312 MB pre-trained checkpoint."""
    from panns_inference import SoundEventDetection as UpstreamSED

    audio = _deterministic_audio()

    upstream = UpstreamSED(checkpoint_path=str(panns_checkpoint), device="cpu")
    expected = upstream.inference(audio)

    ours = SoundEventDetection(checkpoint_path=panns_checkpoint, device="cpu")
    actual = ours.inference(audio)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# SoundEventDetection wrapper behaviour
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sed_loads_real_checkpoint_and_infers(panns_checkpoint) -> None:
    sed = SoundEventDetection(checkpoint_path=panns_checkpoint, device="cpu")
    out = sed.inference(_deterministic_audio())

    expected_frames = _TEST_SECONDS * SAMPLE_RATE // HOP_SIZE + 1
    assert out.shape == (1, expected_frames, CLASSES_NUM)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.slow
def test_sed_accepts_1d_audio(panns_checkpoint) -> None:
    sed = SoundEventDetection(checkpoint_path=panns_checkpoint, device="cpu")
    out = sed.inference(_deterministic_audio()[0])
    assert out.ndim == 3 and out.shape[0] == 1


def test_sed_rejects_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        SoundEventDetection(checkpoint_path=tmp_path / "nope.pth")


def test_sed_rejects_unknown_device(panns_checkpoint) -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        SoundEventDetection(checkpoint_path=panns_checkpoint, device="tpu")


def test_sed_rejects_checkpoint_without_model_key(tmp_path) -> None:
    """A structurally wrong checkpoint must fail loudly, not load partially."""
    bad = tmp_path / "bad.pth"
    torch.save({"not_model": {}}, bad)
    with pytest.raises(RuntimeError, match="no 'model' entry"):
        SoundEventDetection(checkpoint_path=bad)


def test_sed_rejects_mismatched_checkpoint(tmp_path) -> None:
    """strict=True must reject a checkpoint missing tensors."""
    bad = tmp_path / "partial.pth"
    torch.save({"model": {"fc1.weight": torch.zeros(2048, 2048)}}, bad)
    with pytest.raises(RuntimeError):
        SoundEventDetection(checkpoint_path=bad)


@pytest.mark.slow
@pytest.mark.parametrize(
    "bad_audio, message",
    [
        (np.zeros((1, 1, 100), dtype=np.float32), "shape"),
        (np.zeros((1, 0), dtype=np.float32), "empty"),
        (np.zeros((1, 100), dtype=np.int16), "floating"),
    ],
)
def test_sed_inference_input_validation(
    panns_checkpoint, bad_audio, message
) -> None:
    sed = SoundEventDetection(checkpoint_path=panns_checkpoint, device="cpu")
    with pytest.raises(ValueError, match=message):
        sed.inference(bad_audio)
