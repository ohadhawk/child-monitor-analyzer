"""Every shipped PowerShell script must parse.

Two release scripts have been broken by characters that read fine in an editor
and then fail in Windows PowerShell 5.1: an em-dash inside a string literal,
and a pipeline whose left side was a try/catch. Both were only caught by
running the script, which is exactly when it is most expensive to find out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(REPO.glob("scripts/*.ps1")) + sorted(REPO.glob("*.ps1"))

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

# Windows PowerShell 5.1 is what end users run patches with, and it is stricter
# about non-ASCII in scripts than pwsh 7 is. The path arrives by environment
# rather than as an argument, because -Command does not populate $args.
PARSE = (
    "$errs = $null; "
    "[void][System.Management.Automation.Language.Parser]::ParseFile("
    "$env:CMA_SCRIPT_PATH, [ref]$null, [ref]$errs); "
    "if ($errs) { $errs | ForEach-Object { Write-Output $_.Message }; exit 1 }"
)


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no PowerShell scripts found; the glob is wrong"


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not available")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_script_parses(script: Path):
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", PARSE],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "CMA_SCRIPT_PATH": str(script)},
    )
    assert result.returncode == 0, f"{script.name} does not parse:\n{result.stdout}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_the_script_is_ascii(script: Path):
    """Non-ASCII is what broke the parser both times, so keep it out entirely."""
    text = script.read_text(encoding="utf-8")
    offenders = {ch for ch in text if ord(ch) > 127}
    assert not offenders, (
        f"{script.name} contains non-ASCII characters {sorted(offenders)!r}; "
        f"they have broken Windows PowerShell parsing before"
    )
