<#
.SYNOPSIS
    Build a cumulative patch zip from a fresh build and the original release.

.DESCRIPTION
    Compares a freshly built dist\monitor-gui against the ORIGINAL v1.0.0
    portable release and packages every file that is new or different, plus a
    manifest of files the new build no longer ships.

    The baseline is deliberately the original release rather than the previous
    patch: a patch diffed against "v1.0.0 plus some earlier patch" is only
    correct for people who applied that earlier patch, even though the notes
    promise it applies to any v1.x.

.PARAMETER NewDir
    The fresh build, i.e. dist\monitor-gui.

.PARAMETER BaseDir
    An extracted copy of the original v1.0.0 portable release.

.PARAMETER Version
    Version being shipped, without the leading v (e.g. 1.2.2).

.PARAMETER OutZip
    Where to write the patch zip.

.PARAMETER ReadMe
    Optional READ_ME_PATCH.txt to include.

.PARAMETER WorkDir
    Scratch folder for the staged payload. Defaults to a temp folder.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$NewDir,
    [Parameter(Mandatory)][string]$BaseDir,
    [Parameter(Mandatory)][string]$Version,
    [Parameter(Mandatory)][string]$OutZip,
    [string]$ReadMe,
    [string]$WorkDir
)

$ErrorActionPreference = "Stop"

$NewDir  = (Resolve-Path -LiteralPath $NewDir).Path
$BaseDir = (Resolve-Path -LiteralPath $BaseDir).Path
if (-not $WorkDir) { $WorkDir = Join-Path $env:TEMP "cma-patch-$Version" }
$payload = Join-Path $WorkDir "payload"

if (Test-Path -LiteralPath $WorkDir) { Remove-Item -LiteralPath $WorkDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $payload | Out-Null

Write-Host "New build : $NewDir" -ForegroundColor Cyan
Write-Host "Baseline  : $BaseDir" -ForegroundColor Cyan

# Index the baseline by relative path. Length is compared first, so an
# identical-size file is the only case that costs a hash.
$baseIndex = @{}
Get-ChildItem -LiteralPath $BaseDir -Recurse -File | ForEach-Object {
    $baseIndex[$_.FullName.Substring($BaseDir.Length + 1)] = $_
}

$added = New-Object System.Collections.ArrayList
$changed = New-Object System.Collections.ArrayList
$same = 0

Get-ChildItem -LiteralPath $NewDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($NewDir.Length + 1)
    $old = $baseIndex[$rel]
    if ($null -eq $old) {
        [void]$added.Add($rel)
    } elseif ($old.Length -ne $_.Length) {
        [void]$changed.Add($rel)
    } elseif ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -ne
              (Get-FileHash -LiteralPath $old.FullName -Algorithm SHA256).Hash) {
        [void]$changed.Add($rel)
    } else {
        $same++
    }
}

foreach ($rel in ($added + $changed)) {
    $dst = Join-Path $payload $rel
    $dir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Copy-Item -LiteralPath (Join-Path $NewDir $rel) -Destination $dst -Force
}

# Files the baseline has and the new build does not. A copy-only patch cannot
# remove these, so they are listed for apply-patch.ps1 to move aside.
$retired = @($baseIndex.Keys |
    Where-Object { -not (Test-Path -LiteralPath (Join-Path $NewDir $_)) } |
    Sort-Object)
Set-Content -LiteralPath (Join-Path $payload "retired.txt") -Value $retired -Encoding utf8
Set-Content -LiteralPath (Join-Path $payload "version.txt") -Value $Version -Encoding utf8

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "apply-patch.ps1") -Destination $payload -Force
if ($ReadMe) {
    Copy-Item -LiteralPath $ReadMe -Destination (Join-Path $payload "READ_ME_PATCH.txt") -Force
}

if (Test-Path -LiteralPath $OutZip) { Remove-Item -LiteralPath $OutZip -Force }
$zipDir = Split-Path -Parent $OutZip
if ($zipDir -and -not (Test-Path -LiteralPath $zipDir)) { New-Item -ItemType Directory -Path $zipDir -Force | Out-Null }
Compress-Archive -Path (Join-Path $payload '*') -DestinationPath $OutZip -CompressionLevel Optimal

$exe = Join-Path $NewDir "monitor-gui.exe"
Write-Host ""
Write-Host "identical : $same"
Write-Host "added     : $($added.Count)"
Write-Host "changed   : $($changed.Count)"
Write-Host "retired   : $($retired.Count)"
Write-Host ""
Write-Host "changed files:" -ForegroundColor Cyan
$changed | Sort-Object | ForEach-Object { "  $_" }
Write-Host ""
Write-Host ("Patch written to {0} ({1:N1} MB)" -f $OutZip, ((Get-Item -LiteralPath $OutZip).Length / 1MB)) -ForegroundColor Green
Write-Host "exe SHA-256: $((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)" -ForegroundColor Green
