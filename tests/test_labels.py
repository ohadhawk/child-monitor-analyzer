"""Regression tests for the vendored AudioSet label loader.

The label file maps model output indices to human-readable class names. If it
is truncated, reordered, or substituted, every detection in the report is
silently mislabelled -- a wrong crying/screaming classification with no error
anywhere. These tests lock in the fail-closed behaviour.
"""

from __future__ import annotations

import pytest

from monitor.vendor.panns import CLASSES_NUM, load_labels, load_labels_cached
from monitor.vendor.panns.labels import _EXPECTED_COLUMNS

HEADER = "index,mid,display_name\n"


def _write_csv(path, rows):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _valid_rows(count=CLASSES_NUM):
    return [f'{i},/m/{i:04x},"Class {i}"\n' for i in range(count)]


# ---------------------------------------------------------------------------
# happy path, against the real file
# ---------------------------------------------------------------------------

def test_real_labels_file_loads(panns_labels_csv):
    labels = load_labels(panns_labels_csv)
    assert len(labels) == CLASSES_NUM
    assert labels[0] == "Speech"


@pytest.mark.parametrize(
    "name,index",
    [
        ("Speech", 0),
        ("Shout", 8),
        ("Screaming", 14),
        ("Crying, sobbing", 22),
    ],
)
def test_detection_classes_keep_their_indices(panns_labels_csv, name, index):
    """These four names are looked up by ``audio_events`` at runtime.

    A change here means the shipped label file no longer matches the
    checkpoint the model was trained with.
    """
    labels = load_labels(panns_labels_csv)
    assert labels.index(name) == index


# ---------------------------------------------------------------------------
# fail-closed behaviour
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "nope.csv")


def test_empty_file_raises(tmp_path):
    target = tmp_path / "labels.csv"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_labels(target)


def test_bad_header_raises(tmp_path):
    target = tmp_path / "labels.csv"
    target.write_text("index\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected header"):
        load_labels(target)


def test_truncated_file_raises(tmp_path):
    """A short label file must be an error, never a silent partial load."""
    target = _write_csv(tmp_path / "labels.csv", _valid_rows(10))
    with pytest.raises(ValueError, match=f"expected {CLASSES_NUM}"):
        load_labels(target)


def test_too_many_rows_raises(tmp_path):
    target = _write_csv(tmp_path / "labels.csv", _valid_rows(CLASSES_NUM + 1))
    with pytest.raises(ValueError, match=f"expected {CLASSES_NUM}"):
        load_labels(target)


def test_malformed_row_raises(tmp_path):
    rows = _valid_rows()
    rows[100] = "100,/m/0064\n"  # missing display_name column
    target = _write_csv(tmp_path / "labels.csv", rows)
    with pytest.raises(ValueError, match="Malformed"):
        load_labels(target)


def test_blank_rows_are_skipped(tmp_path):
    rows = _valid_rows()
    rows.insert(50, "\n")
    target = _write_csv(tmp_path / "labels.csv", rows)
    assert len(load_labels(target)) == CLASSES_NUM


def test_expected_column_count_is_three():
    assert _EXPECTED_COLUMNS == 3


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def test_cached_loader_returns_immutable_tuple(tmp_path):
    target = _write_csv(tmp_path / "labels.csv", _valid_rows())
    labels = load_labels_cached(str(target))
    assert isinstance(labels, tuple)
    with pytest.raises(TypeError):
        labels[0] = "hacked"  # type: ignore[index]


def test_cached_loader_reuses_the_same_object(tmp_path):
    target = _write_csv(tmp_path / "labels.csv", _valid_rows())
    first = load_labels_cached(str(target))
    second = load_labels_cached(str(target))
    assert first is second


def test_cached_loader_propagates_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labels_cached(str(tmp_path / "nope.csv"))
