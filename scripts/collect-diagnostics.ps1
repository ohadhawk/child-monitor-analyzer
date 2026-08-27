<#
.SYNOPSIS
    Collect diagnostic information for Child Monitor Analyzer support.

.DESCRIPTION
    Gathers logs and configuration state — no passwords, no credentials —
    and packages them into a zip file you can share.

.PARAMETER ReleaseDir
    Path to the portable release folder (the one containing monitor-gui.exe).
    If omitted the script tries the current directory and its parent.

.PARAMETER Out
    Where to write the zip. Defaults to the Desktop.
#>
[CmdletBinding()]
param(
    [string]$ReleaseDir,
    [string]$Out = (Join-Path $env:USERPROFILE 'Desktop\cma-diagnostics.zip')
)

$ErrorActionPreference = 'Stop'

# ── Find the release folder ──────────────────────────────────────────────────

function Test-Release($dir) {
    $dir -and (Test-Path -LiteralPath (Join-Path $dir 'monitor-gui.exe')) -and
               (Test-Path -LiteralPath (Join-Path $dir '_internal'))
}

if (-not $ReleaseDir) {
    foreach ($cand in @((Get-Location).Path, (Split-Path -Parent $PSScriptRoot), $PSScriptRoot)) {
        if (Test-Release $cand) { $ReleaseDir = $cand; break }
    }
}
if (-not (Test-Release $ReleaseDir)) {
    throw "Could not locate the release folder. Pass -ReleaseDir pointing to the folder that contains monitor-gui.exe and _internal."
}
$ReleaseDir = (Resolve-Path -LiteralPath $ReleaseDir).Path
$internal   = Join-Path $ReleaseDir '_internal'
Write-Host "Release folder : $ReleaseDir" -ForegroundColor Cyan

# ── Build output folder ──────────────────────────────────────────────────────

$tmp = Join-Path $env:TEMP "cma_diag_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

# ── Helper: write a section to a report file ─────────────────────────────────

$report = Join-Path $tmp 'report.txt'
function Write-Section([string]$title, [scriptblock]$body) {
    "=== $title ===" | Add-Content -LiteralPath $report
    $lines = try { & $body } catch { "ERROR: $_" }
    $lines | Add-Content -LiteralPath $report
    "" | Add-Content -LiteralPath $report
}

# ── System information ───────────────────────────────────────────────────────

Write-Section "System" {
    "Date        : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "OS          : $([System.Environment]::OSVersion.VersionString)"
    "Machine     : $env:COMPUTERNAME"
    "Username    : $env:USERNAME"
    $exe = Join-Path $ReleaseDir 'monitor-gui.exe'
    if (Test-Path -LiteralPath $exe) {
        $fi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($exe)
        "monitor-gui : $($fi.ProductVersion) ($($fi.FileVersion))"
        "exe SHA-256 : $((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)"
    }
}

# ── Google Drive configuration ───────────────────────────────────────────────

Write-Section "Google Drive: client_id" {
    # Report presence and source only; never print the value itself.
    $authPy = Join-Path $internal 'monitor\gdrive\auth.py'
    if (Test-Path -LiteralPath $authPy) {
        $line = Select-String -LiteralPath $authPy -Pattern 'DEFAULT_CLIENT_ID\s*=' |
                Select-Object -First 1
        if ($line) {
            $blank = $line.Line -match '""'
            "DEFAULT_CLIENT_ID in auth.py is: $(if ($blank) {'empty (not baked in)'} else {'set'})"
        } else {
            "DEFAULT_CLIENT_ID line not found in auth.py"
        }
    } else {
        "monitor\gdrive\auth.py not found under _internal"
    }

    # Check environment variable (value is omitted — just presence)
    $envVal = [System.Environment]::GetEnvironmentVariable('MONITOR_GOOGLE_CLIENT_ID')
    "MONITOR_GOOGLE_CLIENT_ID env: $(if ($envVal) {'set'} else {'not set'})"
}

# ── Google Drive: keyring / credential store ─────────────────────────────────

Write-Section "Google Drive: keyring" {
    $epFile = Join-Path $internal 'keyring-25.7.0.dist-info\entry_points.txt'
    if (Test-Path -LiteralPath $epFile) {
        "entry_points.txt found:"
        Get-Content -LiteralPath $epFile
    } else {
        "keyring-25.7.0.dist-info\entry_points.txt NOT FOUND - keyring may not load"
    }

    "Windows Credential Manager service (VaultSvc):"
    $svc = Get-Service -Name VaultSvc -ErrorAction SilentlyContinue
    if ($svc) { "  State=$($svc.Status)  StartType=$($svc.StartType)" }
    else { "  service not found" }
}

# ── _internal\monitor directory listing ──────────────────────────────────────

Write-Section "_internal\monitor layout" {
    $monDir = Join-Path $internal 'monitor'
    if (Test-Path -LiteralPath $monDir) {
        Get-ChildItem -LiteralPath $monDir -Recurse |
            Select-Object @{n='Path';e={$_.FullName.Substring($monDir.Length+1)}}, Length |
            Format-Table -AutoSize | Out-String
    } else {
        "_internal\monitor not found"
    }
}

# ── _internal\monitor\gdrive listing ────────────────────────────────────────

Write-Section "_internal\monitor\gdrive listing" {
    $gd = Join-Path $internal 'monitor\gdrive'
    if (Test-Path -LiteralPath $gd) {
        Get-ChildItem -LiteralPath $gd -File |
            Select-Object Name, Length |
            Format-Table -AutoSize | Out-String
        # Report presence of client_secrets.json without exposing its content
        $secrets = Join-Path $gd 'client_secrets.json'
        "client_secrets.json present: $(Test-Path -LiteralPath $secrets)"
    } else {
        "_internal\monitor\gdrive not found"
    }
}

# ── keyring dist-info ────────────────────────────────────────────────────────

Write-Section "keyring dist-info files" {
    $ki = Join-Path $internal 'keyring-25.7.0.dist-info'
    if (Test-Path -LiteralPath $ki) {
        Get-ChildItem -LiteralPath $ki -File | Select-Object Name, Length | Format-Table -AutoSize | Out-String
    } else {
        "keyring-25.7.0.dist-info not found - keyring was not patched in"
    }
}

# ── PySide6 QtSvg (needed for the Drive icon) ────────────────────────────────

Write-Section "PySide6 QtSvg" {
    $svg = Join-Path $internal 'PySide6\QtSvg.pyd'
    "QtSvg.pyd present: $(Test-Path -LiteralPath $svg)"
    if (Test-Path -LiteralPath $svg) {
        "size: $((Get-Item -LiteralPath $svg).Length) bytes"
    }
}

# ── Application logs ─────────────────────────────────────────────────────────

$logDir = Join-Path $env:USERPROFILE '.child-monitor-analyzer\logs'
Write-Section "Log files available" {
    if (Test-Path -LiteralPath $logDir) {
        Get-ChildItem -LiteralPath $logDir -Filter '*.log' |
            Sort-Object LastWriteTime -Descending |
            Select-Object Name, LastWriteTime, Length |
            Format-Table -AutoSize | Out-String
    } else {
        "Log folder not found: $logDir"
    }
}

# Copy the three most recent logs (they contain the Drive self-test lines)
if (Test-Path -LiteralPath $logDir) {
    $logsOut = Join-Path $tmp 'logs'
    New-Item -ItemType Directory -Path $logsOut -Force | Out-Null
    Get-ChildItem -LiteralPath $logDir -Filter '*.log' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 3 |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $logsOut -Force }
}

# ── Pack into zip ─────────────────────────────────────────────────────────────

if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }
Compress-Archive -Path "$tmp\*" -DestinationPath $Out -CompressionLevel Optimal
Remove-Item $tmp -Recurse -Force

$size = "{0:N1} KB" -f ((Get-Item -LiteralPath $Out).Length / 1KB)
Write-Host ""
Write-Host "Diagnostics written to: $Out  ($size)" -ForegroundColor Green
Write-Host "Share that file for support." -ForegroundColor Green
