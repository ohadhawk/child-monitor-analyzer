<#
.SYNOPSIS
    Prove a patch zip turns the original release into a fresh build.

.DESCRIPTION
    Applies the patch to a throwaway copy of the original v1.0.0 release, then
    compares the result against the fresh build file by file. Anything missing,
    extra or differing in content is a defect in the patch.

    This is the check worth running before publishing: "it built" says nothing
    about whether the people applying the patch end up with the same bytes.

.PARAMETER Zip
    The patch zip produced by build-patch.ps1.

.PARAMETER BaseDir
    An extracted copy of the original v1.0.0 portable release.

.PARAMETER FreshDir
    The fresh build the patch was made from, i.e. dist\monitor-gui.

.PARAMETER WorkDir
    Scratch folder. Defaults to a temp folder. Deleted and recreated.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Zip,
    [Parameter(Mandatory)][string]$BaseDir,
    [Parameter(Mandatory)][string]$FreshDir,
    [string]$WorkDir
)

$ErrorActionPreference = "Stop"

$Zip      = (Resolve-Path -LiteralPath $Zip).Path
$BaseDir  = (Resolve-Path -LiteralPath $BaseDir).Path
$FreshDir = (Resolve-Path -LiteralPath $FreshDir).Path
if (-not $WorkDir) { $WorkDir = Join-Path $env:TEMP "cma-verify" }

if (Test-Path -LiteralPath $WorkDir) { Remove-Item -LiteralPath $WorkDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

$install = Join-Path $WorkDir "install"
$patch   = Join-Path $WorkDir "patch"

Write-Host "Copying the baseline..." -ForegroundColor Cyan
robocopy $BaseDir $install /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying the baseline failed (robocopy $LASTEXITCODE)." }

Expand-Archive -LiteralPath $Zip -DestinationPath $patch -Force
& (Join-Path $patch "apply-patch.ps1") -ReleaseDir $install

# Rollback artefacts the patch leaves behind on purpose, plus its own files.
$ignore = '\.bak_\d{8}_\d{6}$|^removed_by_patch_|^apply-patch\.ps1$|^READ_ME_PATCH\.txt$|^retired\.txt$|^version\.txt$'

function Get-Index($root) {
    $index = @{}
    Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length + 1)
        if ($rel -notmatch $ignore) { $index[$rel] = $_ }
    }
    return $index
}

$p = Get-Index $install
$f = Get-Index $FreshDir

$missing = @($f.Keys | Where-Object { -not $p.ContainsKey($_) } | Sort-Object)
$extra   = @($p.Keys | Where-Object { -not $f.ContainsKey($_) } | Sort-Object)

$mismatch = New-Object System.Collections.ArrayList
foreach ($rel in $f.Keys) {
    if (-not $p.ContainsKey($rel)) { continue }
    $a = $f[$rel]; $b = $p[$rel]
    if ($a.Length -ne $b.Length) { [void]$mismatch.Add($rel); continue }
    if ((Get-FileHash -LiteralPath $a.FullName -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $b.FullName -Algorithm SHA256).Hash) {
        [void]$mismatch.Add($rel)
    }
}

Write-Host ""
Write-Host "compared      : $($f.Count) files"
Write-Host "missing       : $($missing.Count)"
Write-Host "extra         : $($extra.Count)"
Write-Host "hash mismatch : $($mismatch.Count)"

if ($missing.Count)  { ""; "=== missing from the patched install ==="; $missing | Select-Object -First 30 }
if ($extra.Count)    { ""; "=== present only in the patched install ==="; $extra | Select-Object -First 30 }
if ($mismatch.Count) { ""; "=== content differs ==="; ($mismatch | Sort-Object | Select-Object -First 30) }

if ($missing.Count -or $extra.Count -or $mismatch.Count) {
    throw "The patched install does NOT match a fresh build."
}

Write-Host ""
Write-Host "RESULT: patched install is identical to a fresh build." -ForegroundColor Green
Write-Host "Patched install left at: $install" -ForegroundColor Yellow
