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

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Bundle python-docx: its data files (default.docx template), hidden
# imports, and distribution metadata (read via importlib.metadata).
_docx_datas, _docx_binaries, _docx_hiddenimports = collect_all("docx")

# NOTE: matplotlib is deliberately NOT bundled any more.
# It was only ever present because upstream ``panns_inference.models`` had a
# module-level ``import matplotlib.pyplot`` that was never exercised at
# runtime. That transitively pulled in Pillow, which as of 2026-08-03 carries
# 27 open advisories -- shipped attack surface for zero functionality.
# The PANNs inference code is now vendored in ``monitor.vendor.panns`` without
# that import, so both packages are excluded below.

# Bundle faster-whisper data assets: the Silero VAD model
# (assets/silero_vad_v6.onnx) is loaded from disk at transcription time and is
# not picked up by module analysis.
_fw_datas = collect_data_files("faster_whisper")

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
    binaries=[] + _docx_binaries,
    datas=[
        # Bundle the profanity word lists.
        (str(PROJECT_ROOT / "data"), "data"),
        # _soundfile_data contains libsndfile DLL (required by soundfile).
        (str(SITE_PACKAGES / "_soundfile_data"), "_soundfile_data"),
        # Attribution for the vendored PANNs/torchlibrosa code (MIT requires
        # the licence text to ship with redistributed binaries).
        (
            str(PROJECT_ROOT / "src" / "monitor" / "vendor" / "panns"
                / "LICENSE-third-party.txt"),
            "monitor/vendor/panns",
        ),
    ] + _docx_datas + _fw_datas,
    hiddenimports=[
        # --- monitor subpackages ---
        "monitor",
        "monitor.models",
        "monitor.model_cache",
        "monitor.cancellation",
        "monitor.stt",
        "monitor.audio_events",
        "monitor.profanity",
        "monitor.pipeline",
        "monitor.cli",
        "monitor.priority",
        "monitor.analysis_worker",
        "monitor.model_updates",
        "monitor.gui",
        "monitor.gui.main_window",
        "monitor.gui.report_table",
        "monitor.gui.audio_player",
        "monitor.gui.sensitivity_panel",
        "monitor.gui.transcript_widget",
        "monitor.gui.player_icons",
        "monitor.gui.strings",
        "monitor.gui.google_account",
        "monitor.log_redaction",
        # --- Google Drive upload ---
        "monitor.gdrive",
        "monitor.gdrive.auth",
        "monitor.gdrive.client",
        "monitor.gdrive.naming",
        "monitor.gdrive.store",
        # keyring discovers backends through entry points, which PyInstaller
        # cannot see.  Without these the frozen build silently resolves to the
        # null backend and the feature disables itself.
        "keyring",
        "keyring.backends",
        "keyring.backends.Windows",
        "keyring.backends.fail",
        "win32ctypes",
        "win32ctypes.pywin32",
        # --- vendored PANNs inference (replaces panns-inference/torchlibrosa) ---
        "monitor.vendor",
        "monitor.vendor.panns",
        "monitor.vendor.panns.inference",
        "monitor.vendor.panns.labels",
        "monitor.vendor.panns._models",
        "monitor.vendor.panns._stft",
        # --- PySide6 ---
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtMultimedia",
        # --- ML / audio ---
        "faster_whisper",
        "ctranslate2",
        "librosa",
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
        # --- docx export ---
        "docx",
    ] + _docx_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "pip",
        "setuptools",
        # Removed with the panns_inference dependency -- see the note at the
        # top of this file. Excluding them explicitly means an accidental
        # re-introduction fails the build instead of silently adding ~40 MB
        # and 27 Pillow advisories back into the bundle.
        "matplotlib",
        "PIL",
        "panns_inference",
        "torchlibrosa",
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
