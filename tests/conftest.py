"""
Shared pytest fixtures and helpers.

Adds ``src/`` to ``sys.path`` so tests run against the working tree without
requiring an editable install, and provides markers/helpers for the tests that
need the large downloaded model files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


#: Real PANNs checkpoint (~312 MB), downloaded at first run. Tests that need it
#: skip when it is absent so that a fresh clone can still run the suite.
PANNS_CHECKPOINT = PROJECT_ROOT / "models" / "panns" / "Cnn14_DecisionLevelMax.pth"
PANNS_LABELS_CSV = PROJECT_ROOT / "models" / "panns" / "class_labels_indices.csv"


@pytest.fixture(scope="session")
def panns_checkpoint() -> Path:
    """Path to the real PANNs checkpoint, or skip the test."""
    if not PANNS_CHECKPOINT.is_file():
        pytest.skip(f"PANNs checkpoint not downloaded: {PANNS_CHECKPOINT}")
    return PANNS_CHECKPOINT


@pytest.fixture(scope="session")
def panns_labels_csv() -> Path:
    """Path to the real AudioSet label CSV, or skip the test."""
    if not PANNS_LABELS_CSV.is_file():
        pytest.skip(f"AudioSet label CSV not downloaded: {PANNS_LABELS_CSV}")
    return PANNS_LABELS_CSV


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: test needs the large downloaded model files"
    )
    config.addinivalue_line(
        "markers", "network: test performs real network access"
    )
