"""Refuse to commit a real Google OAuth credential.

The packaged build needs a client id and secret baked into
``monitor/gdrive/auth.py``, and ``setup-google-drive.ps1 -PatchSource`` writes
them into the working tree just before PyInstaller runs. That leaves a window
where a routine ``git commit -a`` would publish them to a public repository,
where GitHub's secret scanning may have Google revoke the credential outright
and break Drive for everyone who already installed the app.

So this inspects *staged* content rather than the working tree: mid-build the
two disagree, and it is the staged bytes that are about to become public.

Run with no arguments as a pre-commit hook, or ``--all`` in CI to sweep the
whole checkout.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

#: Google's client secrets are "GOCSPX-" plus about 28 characters. The length
#: floor is what separates a real one from test fixtures like "GOCSPX-secret".
CLIENT_SECRET_RE = re.compile(r"GOCSPX-[A-Za-z0-9_-]{20,}")

#: Real client ids are a long numeric project part and a long random part,
#: unlike the "1-abc.apps.googleusercontent.com" used in tests.
CLIENT_ID_RE = re.compile(
    r"[0-9]{10,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com"
)

AUTH_PATH = "src/monitor/gdrive/auth.py"

#: Constants that must stay empty in committed source.
BAKED_CONSTANTS = ("DEFAULT_CLIENT_ID", "DEFAULT_CLIENT_SECRET")

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
                 ".pyd", ".dll", ".exe", ".pth", ".onnx", ".bin", ".svg"}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")


def _staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACM")
    return [line for line in out.splitlines() if line.strip()]


def _staged_text(path: str) -> str | None:
    blob = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True,
    )
    if blob.returncode != 0 or b"\0" in blob.stdout:
        return None
    return blob.stdout.decode("utf-8", "replace")


def _tracked_paths() -> list[str]:
    return [line for line in _git("ls-files").splitlines() if line.strip()]


def _worktree_text(path: str) -> str | None:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    return data.decode("utf-8", "replace")


def _scan(path: str, text: str) -> list[str]:
    """Return a description of every credential found in *text*."""
    problems = []
    if CLIENT_SECRET_RE.search(text):
        problems.append(f"{path}: contains a Google client secret (GOCSPX-...)")
    if CLIENT_ID_RE.search(text):
        problems.append(f"{path}: contains a real Google OAuth client id")
    if path.replace("\\", "/").endswith(AUTH_PATH):
        problems.extend(_scan_auth_constants(path, text))
    return problems


def _scan_auth_constants(path: str, text: str) -> list[str]:
    """Catch a baked-in credential even in a shape the regexes miss."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [f"{path}: could not be parsed, so it could not be checked"]
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Name)
                    and target.id in BAKED_CONSTANTS
                    and isinstance(node.value, ast.Constant)
                    and node.value.value):
                problems.append(
                    f"{path}: {target.id} is not empty; credentials belong in "
                    f"the build, not in a commit"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true",
        help="scan every tracked file in the checkout instead of the index",
    )
    args = parser.parse_args()

    if args.all:
        paths, read = _tracked_paths(), _worktree_text
    else:
        paths, read = _staged_paths(), _staged_text

    problems: list[str] = []
    for path in paths:
        if Path(path).suffix.lower() in SKIP_SUFFIXES:
            continue
        text = read(path)
        if text is not None:
            problems.extend(_scan(path, text))

    if problems:
        print("Refusing to commit: a Google OAuth credential is present.\n",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nThe repository is public. Bake credentials in at build time"
            "\n(scripts\\setup-google-drive.ps1 -PatchSource) and leave the"
            "\nconstants empty in git. To stage the rest of your work:"
            "\n    git restore --staged " + AUTH_PATH,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
