# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Child Monitor Analyzer GUI.

Build with:
    pyinstaller monitor-gui.spec

Output:
    dist/monitor-gui/  (one-directory bundle)

Models (Whisper STT, PANNs) are NOT bundled -- they are downloaded at
first run into ``dist/monitor-gui/models/``.
"""

import sys
from pathlib import Path

block_cipher = None

# Project root (where this .spec file lives).
PROJECT_ROOT = Path(SPECPATH)
SITE_PACKAGES = Path(sys.executable).parent / ".." / "Lib" / "site-packages"
# Fallback: resolve via importlib if the relative path doesn't exist.
if not SITE_PACKAGES.exists():
    import importlib
    SITE_PACKAGES = Path(importlib.import_module("_soundfile_data").__path__[0]).parent

a = Analysis(
    [str(PROJECT_ROOT / "src" / "run_gui.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=[
        # Bundle the profanity word lists.
        (str(PROJECT_ROOT / "data"), "data"),
        # _soundfile_data contains libsndfile DLL (required by soundfile).
        (str(SITE_PACKAGES / "_soundfile_data"), "_soundfile_data"),
    ],
    hiddenimports=[
        # --- monitor subpackages ---
        "monitor",
        "monitor.models",
        "monitor.model_cache",
        "monitor.stt",
        "monitor.audio_events",
        "monitor.profanity",
        "monitor.pipeline",
        "monitor.cli",
        "monitor.gui",
        "monitor.gui.main_window",
        "monitor.gui.report_table",
        "monitor.gui.audio_player",
        "monitor.gui.sensitivity_panel",
        "monitor.gui.transcript_widget",
        "monitor.gui.player_icons",
        "monitor.gui.strings",
        # --- PySide6 ---
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtMultimedia",
        # --- ML / audio ---
        "faster_whisper",
        "ctranslate2",
        "panns_inference",
        "panns_inference.models",
        "librosa",
        "torchlibrosa",
        "soundfile",
        "audioread",
        "scipy",
        "scipy.signal",
        "numba",
        # --- torch (XPU build) ---
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.xpu",
        # --- transformers (for profanity AI model) ---
        "transformers",
        "transformers.pipelines",
        "transformers.models.auto",
        "safetensors",
        "tokenizers",
        # --- networking / download ---
        "huggingface_hub",
        "requests",
        "urllib3",
        "certifi",
        "tqdm",
        "filelock",
        "fsspec",
        # --- other ---
        "numpy",
        "cffi",
        "_cffi_backend",
        "yaml",
        "regex",
        "packaging",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "pip",
        "setuptools",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="monitor-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Debug: show console for error messages.
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="monitor-gui",
)
