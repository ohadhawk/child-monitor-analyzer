"""Regression tests for :mod:`monitor.model_cache` integrity checking.

These cover the supply-chain hardening added alongside the vendoring of the
PANNs inference code:

* every model download must go over HTTPS,
* the PANNs checkpoint must match a pinned SHA-256 before it is ever loaded,
* a file that fails verification must be quarantined, not silently reused.

The tests never touch the network: :func:`monitor.model_cache._download_file`
is monkeypatched with a local fake.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from monitor import model_cache


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(path: Path, payload: bytes) -> str:
    """Write *payload* to *path* and return its hex SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class _FakeDownloader:
    """Stand-in for ``_download_file`` that writes fixed bytes."""

    def __init__(self, *payloads: bytes) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, url, dest, label, on_progress=None):
        self.calls.append((url, Path(dest)))
        payload = self._payloads.pop(0) if self._payloads else b""
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(payload)


# ---------------------------------------------------------------------------
# URL scheme enforcement
# ---------------------------------------------------------------------------

def test_pinned_urls_are_https():
    """Both hard-coded model URLs must be HTTPS.

    Upstream PANNs fetches the AudioSet labels over plain HTTP, which lets an
    on-path attacker remap every detection class.
    """
    assert model_cache._PANNS_CHECKPOINT_URL.startswith("https://")
    assert model_cache._PANNS_LABELS_URL.startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/model.pth",
        "ftp://example.invalid/model.pth",
        "file:///etc/passwd",
        "HTTP://example.invalid/model.pth",
    ],
)
def test_require_https_rejects_non_https(url):
    with pytest.raises(ValueError, match="Refusing to download"):
        model_cache._require_https(url)


def test_require_https_accepts_https():
    model_cache._require_https("https://example.invalid/model.pth")


def test_download_file_refuses_plaintext(tmp_path):
    """The scheme check must fire before any socket is opened."""
    with pytest.raises(ValueError):
        model_cache._download_file(
            "http://example.invalid/x.bin", tmp_path / "x.bin", "x",
        )
    assert not (tmp_path / "x.bin").exists()
    assert not (tmp_path / "x.tmp").exists()


# ---------------------------------------------------------------------------
# digest helpers
# ---------------------------------------------------------------------------

def test_file_sha256_matches_hashlib(tmp_path):
    payload = b"the quick brown fox" * 1000
    target = tmp_path / "blob.bin"
    expected = _write(target, payload)
    assert model_cache.file_sha256(target) == expected


def test_file_sha256_streams_in_chunks(tmp_path):
    """A tiny chunk size must produce the same digest as a single read."""
    payload = bytes(range(256)) * 500
    target = tmp_path / "blob.bin"
    expected = _write(target, payload)
    assert model_cache.file_sha256(target, chunk_size=7) == expected


def test_verify_file_digest_accepts_match(tmp_path):
    target = tmp_path / "blob.bin"
    digest = _write(target, b"payload")
    assert model_cache.verify_file_digest(target, digest, "blob") is True


def test_verify_file_digest_is_case_insensitive(tmp_path):
    target = tmp_path / "blob.bin"
    digest = _write(target, b"payload")
    assert model_cache.verify_file_digest(target, digest.upper(), "blob") is True


def test_verify_file_digest_rejects_mismatch(tmp_path):
    target = tmp_path / "blob.bin"
    _write(target, b"payload")
    assert model_cache.verify_file_digest(target, "00" * 32, "blob") is False


def test_verify_file_digest_rejects_missing_file(tmp_path):
    """A missing file must return False, not raise."""
    assert model_cache.verify_file_digest(
        tmp_path / "nope.bin", "00" * 32, "blob",
    ) is False


# ---------------------------------------------------------------------------
# checkpoint acquisition
# ---------------------------------------------------------------------------

@pytest.fixture()
def pinned_small(monkeypatch):
    """Shrink the pinned checkpoint to a small in-test payload.

    Lets the full download/verify/quarantine flow run in milliseconds instead
    of hashing 312 MB.
    """
    payload = b"GOOD-CHECKPOINT-BYTES" * 64
    monkeypatch.setattr(model_cache, "_PANNS_CHECKPOINT_SIZE", len(payload))
    monkeypatch.setattr(
        model_cache, "_PANNS_CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest(),
    )
    return payload


def test_ensure_checkpoint_downloads_and_verifies(tmp_path, monkeypatch, pinned_small):
    fake = _FakeDownloader(pinned_small)
    monkeypatch.setattr(model_cache, "_download_file", fake)

    result = model_cache.ensure_panns_checkpoint(panns_dir=tmp_path)

    assert result == tmp_path / model_cache._PANNS_CHECKPOINT_NAME
    assert result.read_bytes() == pinned_small
    assert len(fake.calls) == 1


def test_ensure_checkpoint_reuses_verified_file(tmp_path, monkeypatch, pinned_small):
    """An already-correct file must not be re-downloaded."""
    (tmp_path / model_cache._PANNS_CHECKPOINT_NAME).write_bytes(pinned_small)
    fake = _FakeDownloader()
    monkeypatch.setattr(model_cache, "_download_file", fake)

    model_cache.ensure_panns_checkpoint(panns_dir=tmp_path)

    assert fake.calls == []


def test_ensure_checkpoint_replaces_tampered_file(tmp_path, monkeypatch, pinned_small):
    """A cached file with the right size but wrong bytes must be re-fetched.

    This is the core tamper scenario: something on disk swapped the model for
    a same-length payload.
    """
    checkpoint = tmp_path / model_cache._PANNS_CHECKPOINT_NAME
    checkpoint.write_bytes(b"X" * len(pinned_small))
    fake = _FakeDownloader(pinned_small)
    monkeypatch.setattr(model_cache, "_download_file", fake)

    result = model_cache.ensure_panns_checkpoint(panns_dir=tmp_path)

    assert result.read_bytes() == pinned_small
    assert len(fake.calls) == 1
    # The bad copy is preserved for inspection, out of the load path.
    assert (tmp_path / "Cnn14_DecisionLevelMax.untrusted").exists()


def test_ensure_checkpoint_raises_when_download_never_verifies(
    tmp_path, monkeypatch, pinned_small,
):
    """Two bad downloads in a row must be fatal, never silently accepted."""
    bad = b"Y" * len(pinned_small)
    fake = _FakeDownloader(bad, bad)
    monkeypatch.setattr(model_cache, "_download_file", fake)

    with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
        model_cache.ensure_panns_checkpoint(panns_dir=tmp_path)

    assert len(fake.calls) == 2
    # Nothing loadable is left behind.
    assert not (tmp_path / model_cache._PANNS_CHECKPOINT_NAME).exists()


def test_ensure_checkpoint_rejects_truncated_download(
    tmp_path, monkeypatch, pinned_small,
):
    """A short file must be rejected on the size check alone."""
    truncated = pinned_small[:-10]
    fake = _FakeDownloader(truncated, truncated)
    monkeypatch.setattr(model_cache, "_download_file", fake)

    with pytest.raises(RuntimeError):
        model_cache.ensure_panns_checkpoint(panns_dir=tmp_path)


def test_ensure_panns_ready_fetches_labels_and_checkpoint(
    tmp_path, monkeypatch, pinned_small,
):
    calls: list[Path] = []

    def fake_labels(panns_dir=None):
        calls.append(Path(panns_dir))
        return Path(panns_dir) / model_cache._PANNS_LABELS_NAME

    monkeypatch.setattr(model_cache, "ensure_panns_labels", fake_labels)
    monkeypatch.setattr(model_cache, "_download_file", _FakeDownloader(pinned_small))

    result = model_cache.ensure_panns_ready(panns_dir=tmp_path)

    assert calls == [tmp_path]
    assert result.name == model_cache._PANNS_CHECKPOINT_NAME


def test_no_legacy_home_directory_writes():
    """The ``~/panns_data`` shim must be gone.

    It only existed because upstream ``panns_inference.config`` read that path
    at import time. Vendoring removed the need, and writing outside the
    portable models directory broke the "no traces on the host" guarantee.
    """
    assert not hasattr(model_cache, "_ensure_legacy_panns_labels")
    source = Path(model_cache.__file__).read_text(encoding="utf-8")
    assert "panns_data" not in source


# ---------------------------------------------------------------------------
# labels CSV
# ---------------------------------------------------------------------------

_LABELS_HEADER = "index,mid,display_name\n"


def _labels_payload(count=527):
    rows = "".join(f'{i},/m/{i:04x},"Class {i}"\n' for i in range(count))
    return (_LABELS_HEADER + rows).encode("utf-8")


def test_ensure_labels_accepts_valid_cached_file(tmp_path, monkeypatch):
    (tmp_path / model_cache._PANNS_LABELS_NAME).write_bytes(_labels_payload())
    fake = _FakeDownloader()
    monkeypatch.setattr(model_cache, "_download_file", fake)

    result = model_cache.ensure_panns_labels(panns_dir=tmp_path)

    assert result.name == model_cache._PANNS_LABELS_NAME
    assert fake.calls == []


def test_ensure_labels_replaces_truncated_cached_file(tmp_path, monkeypatch):
    """A short label file must not be reused forever.

    Without this, one bad download wedges detection permanently: the file
    exists, so it is never re-fetched, and every run fails to map class
    indices to names.
    """
    (tmp_path / model_cache._PANNS_LABELS_NAME).write_bytes(_labels_payload(10))
    fake = _FakeDownloader(_labels_payload())
    monkeypatch.setattr(model_cache, "_download_file", fake)

    result = model_cache.ensure_panns_labels(panns_dir=tmp_path)

    assert len(fake.calls) == 1
    assert result.read_bytes() == _labels_payload()


def test_ensure_labels_raises_when_download_stays_invalid(tmp_path, monkeypatch):
    bad = _labels_payload(3)
    monkeypatch.setattr(model_cache, "_download_file", _FakeDownloader(bad, bad))

    with pytest.raises(RuntimeError, match="valid 527-class label file"):
        model_cache.ensure_panns_labels(panns_dir=tmp_path)

    assert not (tmp_path / model_cache._PANNS_LABELS_NAME).exists()


@pytest.mark.slow
def test_real_labels_file_is_accepted(panns_labels_csv):
    assert model_cache.ensure_panns_labels(
        panns_dir=panns_labels_csv.parent,
    ) == panns_labels_csv


# ---------------------------------------------------------------------------
# pinned digest sanity
# ---------------------------------------------------------------------------

def test_pinned_digest_is_wellformed():
    digest = model_cache._PANNS_CHECKPOINT_SHA256
    assert len(digest) == 64
    assert digest == digest.lower()
    int(digest, 16)  # raises ValueError if not hex


@pytest.mark.slow
def test_real_checkpoint_matches_pinned_digest(panns_checkpoint):
    """The checkpoint actually on this machine must match the pin.

    Skipped when the model has not been downloaded yet.
    """
    assert panns_checkpoint.stat().st_size == model_cache._PANNS_CHECKPOINT_SIZE
    assert model_cache.file_sha256(panns_checkpoint) == (
        model_cache._PANNS_CHECKPOINT_SHA256
    )
