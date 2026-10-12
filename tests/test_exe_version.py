"""The built exe must carry the app version, read from ``monitor.__version__``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import monitor

ROOT = Path(__file__).resolve().parents[1]


def _load_version_resource():
    src = (ROOT / "monitor-gui.spec").read_text(encoding="utf-8")
    fn = next(
        n for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_version_resource"
    )
    ns = {"PROJECT_ROOT": ROOT}
    exec(compile(ast.Module([fn], []), "monitor-gui.spec", "exec"), ns)
    return ns["_version_resource"]()


def test_spec_passes_a_version_resource_to_the_exe():
    src = (ROOT / "monitor-gui.spec").read_text(encoding="utf-8")
    assert "version=_version_resource()" in src


def test_version_resource_matches_package_version():
    pytest.importorskip("PyInstaller")
    text = str(_load_version_resource())
    assert f"StringStruct('ProductVersion', '{monitor.__version__}')" in text
    assert f"StringStruct('FileVersion', '{monitor.__version__}')" in text
    numeric = tuple(int(p) for p in monitor.__version__.split(".")) + (0,)
    assert f"prodvers={numeric[:4]}" in text
