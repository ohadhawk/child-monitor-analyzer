<#
.SYNOPSIS
    Apply a Child Monitor Analyzer patch to an existing portable release.

.DESCRIPTION
    Patches are cumulative: each one is built by comparing a fresh build
    against the original v1.0.0 release, so applying it leaves the install
    identical to a fresh build of that version no matter which earlier patches
    were applied.

    The script backs up the current monitor-gui.exe and base_library.zip,
    replaces the exe, and overlays the new _internal files. Re-copying files
    that are already present is harmless (identical content). The downloaded
    models\ folder is left untouched.

    It also retires files the new version no longer ships, listed in
    retired.txt. They are moved aside rather than deleted, so the step is
    reversible. Without it a patched install keeps metadata for two versions
    of the same package, and Python can then report whichever it finds first.

.PARAMETER ReleaseDir
    Path to the installed portable release -- the folder that contains
    monitor-gui.exe and the _internal sub-folder. If omitted, the script
    tries the current directory and its parent.

.PARAMETER KeepRetiredFiles
    Skip the retirement step and leave superseded files in place.
#>
[CmdletBinding()]
param(
    [string]$ReleaseDir,
    [switch]$KeepRetiredFiles
)

$ErrorActionPreference = "Stop"
$patchExe      = Join-Path $PSScriptRoot "monitor-gui.exe"
$patchInternal = Join-Path $PSScriptRoot "_internal"
$retiredList   = Join-Path $PSScriptRoot "retired.txt"
$versionFile   = Join-Path $PSScriptRoot "version.txt"

if (-not (Test-Path -LiteralPath $patchExe))      { throw "Patched monitor-gui.exe not found next to this script." }
if (-not (Test-Path -LiteralPath $patchInternal)) { throw "Patch _internal folder not found next to this script." }

$version = "this patch"
if (Test-Path -LiteralPath $versionFile) {
    $read = (Get-Content -LiteralPath $versionFile -Raw).Trim()
    if ($read) { $version = "v$read" }
}

function Test-Release($dir) {
    return ($dir -and (Test-Path -LiteralPath (Join-Path $dir "monitor-gui.exe")) -and (Test-Path -LiteralPath (Join-Path $dir "_internal")))
}

if (-not $ReleaseDir) {
    foreach ($cand in @((Get-Location).Path, (Split-Path -Parent $PSScriptRoot))) {
        if (Test-Release $cand) { $ReleaseDir = $cand; break }
    }
}
if (-not (Test-Release $ReleaseDir)) {
    throw "Could not find the release. Pass -ReleaseDir pointing to the folder that contains monitor-gui.exe and _internal."
}

$ReleaseDir = (Resolve-Path -LiteralPath $ReleaseDir).Path
Write-Host "Release folder : $ReleaseDir" -ForegroundColor Cyan
Write-Host "Applying       : $version" -ForegroundColor Cyan

$running = Get-Process -Name "monitor-gui" -ErrorAction SilentlyContinue
if ($running) { throw "monitor-gui.exe is currently running. Close the application and try again." }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 1) Back up + replace the exe.
$targetExe = Join-Path $ReleaseDir "monitor-gui.exe"
Copy-Item -LiteralPath $targetExe -Destination (Join-Path $ReleaseDir "monitor-gui.exe.bak_$stamp") -Force
Copy-Item -LiteralPath $patchExe  -Destination $targetExe -Force
Write-Host "Executable updated (backup: monitor-gui.exe.bak_$stamp)" -ForegroundColor Green

# 2) Overlay the _internal delta files.
$targetInternal = Join-Path $ReleaseDir "_internal"
$srcRoot = (Resolve-Path -LiteralPath $patchInternal).Path
$copied = 0
Get-ChildItem -LiteralPath $srcRoot -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($srcRoot.Length + 1)
    $dst = Join-Path $targetInternal $rel
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    if ($rel -ieq "base_library.zip" -and (Test-Path -LiteralPath $dst)) {
        Copy-Item -LiteralPath $dst -Destination "$dst.bak_$stamp" -Force
    }
    # -LiteralPath, because python-docx ships "[Content_Types].xml" and square
    # brackets are wildcards to Copy-Item -Path: it would skip it in silence.
    Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
    $copied++
}
Write-Host "Added/updated $copied files in _internal" -ForegroundColor Green

# 3) Move aside what this version no longer ships.
if ($KeepRetiredFiles) {
    Write-Host "Skipping retirement of superseded files (-KeepRetiredFiles)." -ForegroundColor Yellow
} elseif (Test-Path -LiteralPath $retiredList) {
    $attic = Join-Path $ReleaseDir "removed_by_patch_$stamp"
    $moved = 0
    foreach ($rel in (Get-Content -LiteralPath $retiredList | Where-Object { $_.Trim() })) {
        $src = Join-Path $ReleaseDir $rel
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $dst = Join-Path $attic $rel
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
        Move-Item -LiteralPath $src -Destination $dst -Force
        $moved++
    }
    if ($moved -gt 0) {
        Write-Host "Retired $moved superseded files to removed_by_patch_$stamp" -ForegroundColor Green
    } else {
        Write-Host "No superseded files to retire." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Patch applied. Launch monitor-gui.exe ($version)." -ForegroundColor Green
Write-Host "If needed, restore with: Copy-Item `"$targetExe.bak_$stamp`" `"$targetExe`" -Force" -ForegroundColor Yellow
