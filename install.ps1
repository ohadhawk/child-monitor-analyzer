<#
.SYNOPSIS
    Lightweight installer / updater for Child Monitor Analyzer (Windows).

.DESCRIPTION
    A tiny bootstrap script — no multi-GB installer required. It clones (or
    updates) the source from GitHub, creates a Python virtual environment,
    installs the dependencies, and can launch the GUI.

    Run it again any time to pull the latest version: it does a `git pull`
    and re-installs so new code and any changed dependencies are applied.

    Typical first-time use on a fresh machine:
        # download just this one file, then:
        powershell -ExecutionPolicy Bypass -File install.ps1 -Launch

    Update an existing install to the latest version:
        powershell -ExecutionPolicy Bypass -File install.ps1 -Launch

.PARAMETER InstallDir
    Where to install. Defaults to the repo folder if this script is run from
    inside an existing clone, otherwise %USERPROFILE%\child-monitor-analyzer.

.PARAMETER RepoUrl
    Git remote to clone from.

.PARAMETER Branch
    Branch to track (default: main).

.PARAMETER Proxy
    Optional HTTP/HTTPS proxy (e.g. http://proxy:911) used for git and pip.

.PARAMETER Launch
    Launch the GUI after a successful install/update.

.PARAMETER SkipInstall
    Only pull the latest source; skip the pip dependency install step.
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$RepoUrl = "https://github.com/ohadhawk/child-monitor-analyzer.git",
    [string]$Branch  = "main",
    [string]$Proxy   = "",
    [switch]$Launch,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# --- Resolve install directory ---------------------------------------------
# If run from inside an existing clone, update that clone in place.
if (-not $InstallDir) {
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot ".git"))) {
        $InstallDir = $PSScriptRoot
    } else {
        $InstallDir = Join-Path $env:USERPROFILE "child-monitor-analyzer"
    }
}

# --- Prerequisite checks ----------------------------------------------------
Write-Step "Checking prerequisites"

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    throw "Git is not installed or not on PATH. Install Git for Windows: https://git-scm.com/download/win"
}

# Prefer the 'py' launcher, fall back to 'python'.
$pyExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyExe = "py"; $pyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = "python"; $pyArgs = @()
} else {
    throw "Python 3 is not installed or not on PATH. Install Python 3.10+: https://www.python.org/downloads/windows/"
}

$pyVer = (& $pyExe @pyArgs -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
Write-Ok "git: $($git.Source)"
Write-Ok "python: $pyExe $pyArgs (version $pyVer)"

# --- Optional proxy ---------------------------------------------------------
$gitProxyArgs = @()
if ($Proxy) {
    Write-Ok "Using proxy: $Proxy"
    $gitProxyArgs = @("-c", "http.proxy=$Proxy", "-c", "https.proxy=$Proxy")
    $env:HTTP_PROXY  = $Proxy
    $env:HTTPS_PROXY = $Proxy
}

# --- Clone or update --------------------------------------------------------
if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Step "Updating existing install: $InstallDir"
    Push-Location $InstallDir
    try {
        & git @gitProxyArgs fetch origin $Branch
        & git checkout $Branch
        & git @gitProxyArgs pull --ff-only origin $Branch
    } finally {
        Pop-Location
    }
} else {
    Write-Step "Cloning into: $InstallDir"
    $parent = Split-Path -Parent $InstallDir
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    & git @gitProxyArgs clone --branch $Branch $RepoUrl $InstallDir
}

$version = (Get-Content (Join-Path $InstallDir "src\monitor\__init__.py") |
    Select-String '__version__\s*=\s*"([^"]+)"').Matches.Groups[1].Value
Write-Ok "Source is now at version $version"

# --- Virtual environment + dependencies ------------------------------------
$venvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"

if (-not $SkipInstall) {
    if (-not (Test-Path $venvPy)) {
        Write-Step "Creating virtual environment (.venv)"
        & $pyExe @pyArgs -m venv (Join-Path $InstallDir ".venv")
    }

    Write-Step "Installing / updating dependencies (this can take a while the first time)"
    $pipProxyArgs = @()
    if ($Proxy) { $pipProxyArgs = @("--proxy", $Proxy) }

    # Supply-chain hardening for every pip invocation below:
    #   --index-url            pin to the official PyPI index explicitly, so a
    #                          stray PIP_EXTRA_INDEX_URL / pip.ini entry cannot
    #                          shadow a real package with a look-alike from
    #                          another index (dependency confusion).
    #   --disable-pip-version-check / --no-input
    #                          keep the run non-interactive and quiet.
    $pipSafeArgs = @(
        "--index-url", "https://pypi.org/simple",
        "--disable-pip-version-check",
        "--no-input"
    )

    # pip 26.1.2+ fixes several advisories in 25.x; upgrade it first so the
    # rest of the install runs on the patched resolver.
    & $venvPy -m pip install @pipProxyArgs @pipSafeArgs --upgrade "pip>=26.1.2"
    if ($LASTEXITCODE -ne 0) { throw "pip self-upgrade failed (exit $LASTEXITCODE)" }

    # Prefer the hash-pinned lockfile when it is present: --require-hashes
    # makes pip reject any artifact whose SHA-256 is not listed, which is the
    # single most effective defence against a compromised or substituted
    # package release. Fall back to the loose install if the lock is absent.
    $lockFile = Join-Path $InstallDir "requirements.lock"
    if (Test-Path $lockFile) {
        Write-Step "Installing pinned dependencies from requirements.lock (hash-verified)"
        & $venvPy -m pip install @pipProxyArgs @pipSafeArgs `
            --require-hashes --only-binary=:all: -r $lockFile
        if ($LASTEXITCODE -ne 0) {
            throw "Hash-verified dependency install failed (exit $LASTEXITCODE). " +
                  "Do NOT bypass this: a hash mismatch means the downloaded " +
                  "package does not match the reviewed release."
        }
        # Install the project itself without touching the pinned dependency set.
        & $venvPy -m pip install @pipProxyArgs @pipSafeArgs --no-deps -e $InstallDir
    } else {
        Write-Warn "requirements.lock not found - installing without hash verification"
        & $venvPy -m pip install @pipProxyArgs @pipSafeArgs -e $InstallDir
    }
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed (exit $LASTEXITCODE)" }
    Write-Ok "Dependencies are up to date"
}

# --- Done -------------------------------------------------------------------
Write-Step "Ready"
Write-Host ""
Write-Host "Child Monitor Analyzer v$version installed at:" -ForegroundColor Green
Write-Host "    $InstallDir"
Write-Host ""
Write-Host "To launch later, run:" -ForegroundColor Green
Write-Host "    $venvPy `"$([IO.Path]::Combine($InstallDir,'src','run_gui.py'))`""
Write-Host ""

if ($Launch) {
    Write-Step "Launching GUI"
    & $venvPy (Join-Path $InstallDir "src\run_gui.py")
}
