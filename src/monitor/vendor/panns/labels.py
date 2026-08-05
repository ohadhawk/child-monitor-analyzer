"""
AudioSet class labels (derived from ``panns_inference.config``, MIT).

Upstream read ``~/panns_data/class_labels_indices.csv`` **at import time** and,
if missing, shelled out to ``wget``.  That made a plain ``import`` perform
network I/O and write to the user's home directory, and it failed outright on
Windows.  Here the path is always explicit and no network access occurs.

See ``LICENSE-third-party.txt`` for attribution and the list of changes.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

#: Number of AudioSet classes the pre-trained checkpoint predicts.
CLASSES_NUM = 527

#: Columns of ``class_labels_indices.csv``: index, mid, display_name.
_EXPECTED_COLUMNS = 3


def load_labels(csv_path: str | Path) -> list[str]:
    """Load AudioSet display names from ``class_labels_indices.csv``.

    Args:
        csv_path: Path to the AudioSet label CSV (index, mid, display_name).

    Returns:
        Display names ordered by class index, length :data:`CLASSES_NUM`.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the file is malformed or does not contain exactly
            :data:`CLASSES_NUM` rows.  A truncated or substituted label file
            would silently mislabel every detection, so this is treated as a
            hard error rather than a warning.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"AudioSet label CSV not found: {path}")

    labels: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=",")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"AudioSet label CSV is empty: {path}") from None
        if len(header) < _EXPECTED_COLUMNS:
            raise ValueError(
                f"AudioSet label CSV has an unexpected header {header!r}: {path}"
            )
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < _EXPECTED_COLUMNS:
                raise ValueError(
                    f"Malformed AudioSet label CSV at line {row_num}: {row!r}"
                )
            labels.append(row[2])

    if len(labels) != CLASSES_NUM:
        raise ValueError(
            f"AudioSet label CSV has {len(labels)} labels, expected "
            f"{CLASSES_NUM}: {path}"
        )

    log.debug("Loaded %d AudioSet labels from %s", len(labels), path)
    return labels


@lru_cache(maxsize=4)
def load_labels_cached(csv_path: str) -> tuple[str, ...]:
    """Memoised :func:`load_labels`, returning an immutable tuple.

    The tuple prevents a caller from mutating the shared cached list.
    """
    return tuple(load_labels(csv_path))
