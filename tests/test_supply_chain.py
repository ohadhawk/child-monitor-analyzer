"""Supply-chain regression tests.

These lock in the outcome of the 2026-08-03 dependency review. They are cheap,
import-only checks whose job is to fail loudly if a future change quietly
re-introduces a dependency that was deliberately removed.

What is guarded here:

* ``panns_inference`` / ``torchlibrosa`` -- abandoned single-maintainer packages
  (last released early 2023), replaced by ``monitor.vendor.panns``.
* ``matplotlib`` / ``PIL`` -- pulled in only by upstream ``panns_inference``'s
  unused module-level ``import matplotlib.pyplot``. Pillow alone carried 27
  open advisories at review time.
* ``os.system`` / ``subprocess`` shell use in the vendored code -- upstream
  PANNs shelled out to ``wget`` with an interpolated path.
* ``torch.load`` without ``weights_only=True`` -- unrestricted pickle load is
  arbitrary code execution.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "monitor"
VENDOR = SRC / "vendor" / "panns"

BANNED_IMPORTS = {
    "panns_inference",
    "torchlibrosa",
    "matplotlib",
    "PIL",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level module names imported by *path*.

    Uses the AST rather than a text search so that the banned names appearing
    in comments or docstrings (as they do, explaining why they were removed)
    do not trip the check.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "source_file",
    _python_files(SRC),
    ids=lambda p: str(p.relative_to(SRC)).replace("\\", "/"),
)
def test_no_banned_imports_in_source(source_file):
    """No module under ``src/monitor`` may import a removed package."""
    offending = _imported_roots(source_file) & BANNED_IMPORTS
    assert not offending, (
        f"{source_file.relative_to(PROJECT_ROOT)} imports {sorted(offending)}, "
        "which were deliberately removed. See "
        "src/monitor/vendor/panns/LICENSE-third-party.txt."
    )


@pytest.mark.parametrize(
    "source_file",
    _python_files(VENDOR),
    ids=lambda p: p.name,
)
def test_vendored_code_never_shells_out(source_file):
    """The vendored code must not spawn processes.

    Upstream ``panns_inference`` ran ``os.system`` on a format-string built
    ``wget`` command with an interpolated path -- a shell command injection,
    and a silent network fetch triggered by merely constructing the inference
    object.

    The check is AST-based on purpose: the banned names legitimately appear in
    this repository's docstrings and comments, which document why they were
    removed.
    """
    assert "subprocess" not in _imported_roots(source_file)

    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute):
            owner = func.value
            owner_name = owner.id if isinstance(owner, ast.Name) else ""
            assert not (owner_name == "os" and func.attr in {"system", "popen"}), (
                f"{source_file.name}: os.{func.attr} executes a shell command"
            )
            assert owner_name != "subprocess", (
                f"{source_file.name}: subprocess call in vendored code"
            )

        for keyword in node.keywords:
            if keyword.arg == "shell":
                value = keyword.value
                assert not (
                    isinstance(value, ast.Constant) and value.value is True
                ), f"{source_file.name}: shell=True"


def test_vendored_torch_load_is_weights_only():
    """Every ``torch.load`` in the vendored code must pass weights_only=True."""
    found = 0
    for source_file in _python_files(VENDOR):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "load"):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "torch"):
                continue
            found += 1
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            assert "weights_only" in kwargs, (
                f"{source_file.name}: torch.load without weights_only "
                "allows arbitrary code execution from a crafted checkpoint."
            )
            value = kwargs["weights_only"]
            assert isinstance(value, ast.Constant) and value.value is True
    assert found >= 1, "expected at least one torch.load in the vendored code"


def test_requirements_do_not_list_vendored_packages():
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    # Strip comments -- the removal is explained in a comment block.
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "panns-inference" not in body
    assert "torchlibrosa" not in body


def test_pyproject_does_not_list_vendored_packages():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "panns-inference" not in body
    assert "torchlibrosa" not in body


# ---------------------------------------------------------------------------
# lockfile
# ---------------------------------------------------------------------------

LOCKFILE = PROJECT_ROOT / "requirements.lock"

_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+)")


def _lock_pins() -> dict[str, str]:
    """Return ``{package: version}`` for every pin in the lockfile."""
    pins: dict[str, str] = {}
    for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "#")) or not line.strip():
            continue
        match = _PIN_RE.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
    return pins


def test_lockfile_exists():
    """Without the lock, install.ps1 silently falls back to an unverified install."""
    assert LOCKFILE.is_file(), (
        "requirements.lock is missing. Regenerate it with the uv command in "
        "the header of requirements.txt."
    )


def test_every_locked_package_is_hash_pinned():
    """``--require-hashes`` only protects packages that actually carry hashes."""
    lines = LOCKFILE.read_text(encoding="utf-8").splitlines()
    missing = []
    for index, line in enumerate(lines):
        match = _PIN_RE.match(line)
        if not match:
            continue
        # uv writes "name==version \" followed by indented --hash lines.
        following = lines[index + 1: index + 3]
        if not any("--hash=sha256:" in nxt for nxt in following):
            missing.append(match.group("name"))
    assert not missing, f"lockfile entries without a hash: {missing}"


def test_lockfile_has_no_removed_packages():
    pins = _lock_pins()
    for name in ("panns-inference", "torchlibrosa", "matplotlib", "pillow"):
        assert name not in pins, (
            f"{name} is back in requirements.lock; it was removed on purpose."
        )


def test_lockfile_respects_security_floors():
    """The floors in requirements.txt must actually be reflected in the lock."""
    pins = _lock_pins()
    floors = {"urllib3": (2, 7, 0), "idna": (3, 15), "msgpack": (1, 2, 1)}
    for name, minimum in floors.items():
        assert name in pins, f"{name} missing from the lockfile"
        actual = tuple(int(part) for part in pins[name].split(".")[: len(minimum)])
        assert actual >= minimum, (
            f"{name}=={pins[name]} is below the security floor {minimum}"
        )


def test_pyinstaller_spec_excludes_removed_packages():
    """The shipped binary must not regain matplotlib/Pillow."""
    spec = (PROJECT_ROOT / "monitor-gui.spec").read_text(encoding="utf-8")
    assert "collect_all(\"matplotlib\")" not in spec
    assert "_mpl_" not in spec
    for name in ("matplotlib", "PIL", "panns_inference", "torchlibrosa"):
        assert f'"{name}",' in spec, f"{name} should be listed in excludes"


def test_pyinstaller_spec_bundles_vendored_package():
    spec = (PROJECT_ROOT / "monitor-gui.spec").read_text(encoding="utf-8")
    for module in (
        "monitor.vendor",
        "monitor.vendor.panns",
        "monitor.vendor.panns.inference",
        "monitor.vendor.panns.labels",
        "monitor.vendor.panns._models",
        "monitor.vendor.panns._stft",
    ):
        assert f'"{module}"' in spec, f"{module} missing from hiddenimports"


def test_pyinstaller_spec_lists_every_monitor_module():
    """A module absent from hiddenimports can go missing from the binary.

    PyInstaller's static analysis misses modules that are only imported lazily
    inside functions, which this codebase does throughout to keep startup
    fast. The failure mode is an ImportError that only appears in the packaged
    build, so it is cheap to catch here instead.
    """
    spec = (PROJECT_ROOT / "monitor-gui.spec").read_text(encoding="utf-8")
    missing = []
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC.parent)
        module = ".".join(relative.with_suffix("").parts)
        module = module.removesuffix(".__init__")
        if module.endswith("__main__"):
            continue
        if f'"{module}"' not in spec:
            missing.append(module)
    assert not missing, f"modules missing from monitor-gui.spec hiddenimports: {missing}"


def test_vendored_licence_text_is_present():
    """MIT requires the licence to ship with redistributed code."""
    licence = VENDOR / "LICENSE-third-party.txt"
    text = licence.read_text(encoding="utf-8")
    assert "MIT" in text
    assert "Qiuqiang Kong" in text


# ---------------------------------------------------------------------------
# model revision pinning
# ---------------------------------------------------------------------------

_SHA1_LEN = 40


def _assert_commit_sha(value: str, what: str) -> None:
    assert len(value) == _SHA1_LEN, f"{what} is not a full commit SHA: {value!r}"
    assert value == value.lower(), f"{what} must be lowercase"
    int(value, 16)


def test_stt_models_are_revision_pinned():
    """Every STT repo we download must map to an immutable commit."""
    from monitor import stt

    assert set(stt._STT_REVISIONS) == {stt.DEFAULT_MODEL, stt.TURBO_MODEL}
    for model, revision in stt._STT_REVISIONS.items():
        _assert_commit_sha(revision, f"revision for {model}")


def test_toxicity_model_is_revision_pinned():
    from monitor import profanity

    _assert_commit_sha(profanity._TOXICITY_REVISION, "toxicity model revision")


@pytest.mark.parametrize(
    "source_name,call_name",
    [
        ("stt.py", "snapshot_download"),
        ("stt.py", "WhisperModel"),
        ("profanity.py", "hf_pipeline"),
    ],
)
def test_hub_downloads_pass_a_revision(source_name, call_name):
    """No Hugging Face download may be left tracking a moving branch.

    A model registry is the most realistic supply-chain entry point here: the
    artefacts are large, binary, fetched automatically, and not covered by the
    Python package lockfile.

    ``**kwargs`` unpacking counts as satisfying the rule, because the
    conditional-pin pattern in ``stt.py`` builds the revision that way.
    """
    path = SRC / source_name
    tree = ast.parse(path.read_text(encoding="utf-8"))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    ]
    assert calls, f"no {call_name} call found in {source_name}"

    for call in calls:
        # kw.arg is None for a ``**mapping`` unpacking.
        arg_names = {kw.arg for kw in call.keywords}
        assert "revision" in arg_names or None in arg_names, (
            f"{source_name}: {call_name}() at line {call.lineno} does not pass "
            "a revision, so it would follow the moving 'main' branch."
        )


def test_importing_audio_events_does_not_pull_in_matplotlib():
    """Guard the actual runtime cost, not just the source text.

    Runs in a clean interpreter: within the pytest process another test module
    may already have imported the upstream package (for parity comparisons),
    which would make an in-process ``sys.modules`` check pass vacuously.
    """
    probe = (
        "import sys;"
        "import monitor.audio_events;"
        "bad=sorted(n for n in sys.modules"
        " if n.split('.')[0] in {'matplotlib','PIL','panns_inference',"
        "'torchlibrosa'});"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert not leaked, f"importing monitor.audio_events pulled in: {leaked}"


# ===========================
# SOURCE / TOOLING COUPLINGS
# ===========================

SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup-google-drive.ps1"
AUTH_SOURCE = SRC / "gdrive" / "auth.py"


def test_setup_script_can_still_find_the_client_id_constant():
    """The setup script rewrites this line by regex from outside Python.

    Renaming or reformatting ``DEFAULT_CLIENT_ID`` would leave the test suite
    green while silently breaking ``setup-google-drive.ps1 -PatchSource``,
    which is the only supported way to configure the packaged build.
    """
    assert SETUP_SCRIPT.exists(), "the Drive setup script has moved or been deleted"

    auth_text = AUTH_SOURCE.read_text(encoding="utf-8")
    script_text = SETUP_SCRIPT.read_text(encoding="utf-8")

    # The exact pattern the PowerShell script greps for.
    pattern = r"(?m)^DEFAULT_CLIENT_ID = "
    assert pattern.replace("(?m)", "") in script_text, (
        "the setup script no longer looks for DEFAULT_CLIENT_ID; update this test"
    )
    assert re.search(pattern, auth_text), (
        f"{AUTH_SOURCE} has no line starting 'DEFAULT_CLIENT_ID = ', so "
        f"{SETUP_SCRIPT.name} -PatchSource can no longer configure the build."
    )


def test_no_oauth_client_id_is_committed():
    """A real client id must reach the repo only via a deliberate release step."""
    auth_text = AUTH_SOURCE.read_text(encoding="utf-8")
    match = re.search(r'(?m)^DEFAULT_CLIENT_ID = "(.*)"', auth_text)
    assert match, "DEFAULT_CLIENT_ID is no longer a plain string literal"
    assert match.group(1) == "", (
        "an OAuth client id has been committed to source; it belongs in the "
        "release build only"
    )


def test_every_ui_string_is_used():
    """Unused S.* keys are dead translations that drift out of sync.

    ``KNOWN_UNUSED`` records debt that predates this guard. Do not add to it --
    it exists so that *new* dead strings fail the build, not to bless old ones.
    """
    # Player chrome that ships as icons, and status text the pipeline emits
    # directly. Pre-dates this test; tracked, not blessed.
    known_unused = {
        "PLAYER_BACK", "PLAYER_PLAY", "PLAYER_PAUSE", "PLAYER_FORWARD",
        "PLAYER_VOLUME", "TRANSCRIPT_NO_MATCHES", "TRANSCRIPT_DOWNLOAD_TITLE",
        "DT_PROFANITY", "PIPE_LOADED_CACHE", "PIPE_STARTING", "PIPE_STT_LOADED",
    }
    strings_file = SRC / "gui" / "strings.py"
    tree = ast.parse(strings_file.read_text(encoding="utf-8"))

    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "S":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            keys.append(target.id)
    assert keys, "no S.* keys found; the strings module changed shape"

    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_files(SRC)
        if path != strings_file
    )
    unused = [
        key for key in keys
        if f"S.{key}" not in corpus and key not in known_unused
    ]
    assert not unused, f"unused UI strings (delete them or wire them up): {unused}"

    stale = sorted(name for name in known_unused if name not in set(keys))
    assert not stale, f"remove these from known_unused, they no longer exist: {stale}"


def test_every_ui_string_is_translated():
    """A key with no entry for a language renders as a raw identifier."""
    from monitor.gui.strings import _STRINGS, Lang, S

    keys = [name for name in vars(S) if not name.startswith("_")]
    missing = [
        (key, lang.name)
        for key in keys
        for lang in Lang
        if (getattr(S, key), lang) not in _STRINGS
    ]
    assert not missing, f"untranslated UI strings: {missing}"
