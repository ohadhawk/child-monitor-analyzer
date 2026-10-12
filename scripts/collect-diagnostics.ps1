<#
.SYNOPSIS
    Collect diagnostic information for Child Monitor Analyzer support.

.DESCRIPTION
    Gathers logs and configuration state - no passwords, no credentials -
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

# -- Find the release folder --------------------------------------------------

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

# -- Build output folder ------------------------------------------------------

$tmp = Join-Path $env:TEMP "cma_diag_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

# -- Helper: write a section to a report file ---------------------------------

$report = Join-Path $tmp 'report.txt'
function Write-Section([string]$title, [scriptblock]$body) {
    # UTF8: Windows PowerShell 5.1 defaults to ANSI, which turns Hebrew paths into '?'.
    "=== $title ===" | Add-Content -LiteralPath $report -Encoding UTF8
    $lines = try { & $body } catch { "ERROR: $_" }
    $lines | Add-Content -LiteralPath $report -Encoding UTF8
    "" | Add-Content -LiteralPath $report -Encoding UTF8
}

$logDir = Join-Path $env:USERPROFILE '.child-monitor-analyzer\logs'

# -- System information -------------------------------------------------------

Write-Section "System" {
    "Date        : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "OS          : $([System.Environment]::OSVersion.VersionString)"
    "Machine     : $env:COMPUTERNAME"
    "Username    : $env:USERNAME"
    "CPU threads : $([System.Environment]::ProcessorCount)"
    $exe = Join-Path $ReleaseDir 'monitor-gui.exe'
    if (Test-Path -LiteralPath $exe) {
        $fi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($exe)
        $ver = $fi.ProductVersion
        if (-not $ver -and (Test-Path -LiteralPath $logDir)) {
            # The exe carries no version resource; the app logs it at startup.
            $newest = Get-ChildItem -LiteralPath $logDir -Filter 'monitor_*.log' |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($newest) {
                $hit = Select-String -LiteralPath $newest.FullName -Encoding UTF8 `
                    -Pattern 'GUI v(\S+)', 'Update check: (\S+) is current' | Select-Object -First 1
                if ($hit) { $ver = "$($hit.Matches[0].Groups[1].Value) (from $($newest.Name))" }
            }
        }
        "monitor-gui : $(if ($ver) { $ver } else { 'unknown' })"
        "exe SHA-256 : $((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)"
    }
}

# -- Google Drive configuration -----------------------------------------------

Write-Section "Google Drive: OAuth client" {
    # Report presence and source only; never print the values themselves.
    $authPy = Join-Path $internal 'monitor\gdrive\auth.py'
    if (Test-Path -LiteralPath $authPy) {
        foreach ($name in @('DEFAULT_CLIENT_ID', 'DEFAULT_CLIENT_SECRET')) {
            $line = Select-String -LiteralPath $authPy -Pattern "$name\s*=" | Select-Object -First 1
            if ($line) {
                $blank = $line.Line -match '""'
                "$name in auth.py is: $(if ($blank) {'empty (not baked in)'} else {'set'})"
            } else {
                "$name line not found in auth.py"
            }
        }
    } else {
        "auth.py is not on disk (normal: PyInstaller packs it into the exe)"
    }

    # Both halves are required. Google rejects the token exchange with
    # "client_secret is missing" if only the id is set.
    foreach ($var in @('MONITOR_GOOGLE_CLIENT_ID', 'MONITOR_GOOGLE_CLIENT_SECRET')) {
        foreach ($scope in @('Process', 'User', 'Machine')) {
            $v = [System.Environment]::GetEnvironmentVariable($var, $scope)
            if ($null -eq $v) { "$var [$scope] : NOT SET"; continue }
            # The app trims before use, so a value of spaces counts as absent.
            $t = $v.Trim()
            $notes = @("len=$($v.Length)")
            if ($t.Length -ne $v.Length) { $notes += "HAS SURROUNDING WHITESPACE" }
            if ($t.Length -eq 0) { $notes += "EMPTY AFTER TRIM (app sees this as unset)" }
            if ($t -match '^["'']|["'']$') { $notes += "WRAPPED IN QUOTES (remove them)" }
            if ($t -like '*.apps.googleusercontent.com') { $notes += "looks like a client ID" }
            if ($t -like 'GOCSPX-*') { $notes += "looks like a client SECRET" }
            "$var [$scope] : set  ($($notes -join ', '))"
        }
    }

    # A near-miss name is indistinguishable from the real thing at a glance:
    # a name carrying a stray newline prints normally under `set` but is a
    # different key entirely, so the app never finds it. Match only names that
    # ARE one of ours once punctuation is stripped, so OneDrive and HOMEDRIVE
    # are not accused of being broken credentials.
    $known = @('MONITOR_GOOGLE_CLIENT_ID', 'MONITOR_GOOGLE_CLIENT_SECRET')
    foreach ($scope in @('Process', 'User', 'Machine')) {
        [System.Environment]::GetEnvironmentVariables($scope).Keys |
            Where-Object {
                $known -notcontains $_ -and (
                    $known -contains ($_ -replace '[^A-Za-z0-9_]', '') -or
                    $_ -like '*MONITOR_GOOGLE*'
                )
            } |
            ForEach-Object {
                $shown = $_ -replace "`r", '<CR>' -replace "`n", '<LF>' -replace "`t", '<TAB>'
                $shown = $shown -replace '^ +', '<SPACE>' -replace ' +$', '<SPACE>'
                "MALFORMED NAME [$scope]: '$shown' - the app cannot read this; " +
                "delete it and recreate the variable without the marked character"
            }
    }
}

# -- Google Drive: keyring / credential store ---------------------------------

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

# -- _internal\monitor directory listing --------------------------------------

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

# -- _internal\monitor\gdrive listing ----------------------------------------

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
        "_internal\monitor\gdrive is not on disk (normal: PyInstaller packs it into the exe)"
    }
}

# -- keyring dist-info --------------------------------------------------------

Write-Section "keyring dist-info files" {
    $ki = Join-Path $internal 'keyring-25.7.0.dist-info'
    if (Test-Path -LiteralPath $ki) {
        Get-ChildItem -LiteralPath $ki -File | Select-Object Name, Length | Format-Table -AutoSize | Out-String
    } else {
        "keyring-25.7.0.dist-info not found - keyring was not patched in"
    }
}

# -- PySide6 QtSvg (needed for the Drive icon) --------------------------------

Write-Section "PySide6 QtSvg" {
    $svg = Join-Path $internal 'PySide6\QtSvg.pyd'
    "QtSvg.pyd present: $(Test-Path -LiteralPath $svg)"
    if (Test-Path -LiteralPath $svg) {
        "size: $((Get-Item -LiteralPath $svg).Length) bytes"
    }
}

# -- Application logs ---------------------------------------------------------

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

# Copy the newest GUI and worker logs (the worker log holds the analysis trace).
if (Test-Path -LiteralPath $logDir) {
    $logsOut = Join-Path $tmp 'logs'
    New-Item -ItemType Directory -Path $logsOut -Force | Out-Null
    foreach ($pattern in @('monitor_*.log', 'worker_*.log')) {
        Get-ChildItem -LiteralPath $logDir -Filter $pattern |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 3 |
            ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $logsOut -Force }
    }
}

# -- Recent analysis jobs -----------------------------------------------------
# Cache state only. Transcript text is never copied: recordings are private.

function Get-JsonFlag([string]$text, [string]$name) {
    $m = [regex]::Match($text, '"' + $name + '"\s*:\s*(true|false)')
    if ($m.Success) { $m.Groups[1].Value } else { 'absent' }
}

Write-Section "Recent analysis jobs (no transcript text)" {
    if (-not (Test-Path -LiteralPath $logDir)) { "Log folder not found: $logDir"; return }
    $jobs = @()
    $workerLogs = Get-ChildItem -LiteralPath $logDir -Filter 'worker_*.log' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 3
    foreach ($wl in $workerLogs) {
        foreach ($ln in (Get-Content -LiteralPath $wl.FullName -Encoding UTF8)) {
            if ($ln -match 'Job (\d+) started: (.+) \(model=(\w+)\)\s*$') {
                $jobs += [pscustomobject]@{
                    Log = $wl.Name; Job = $Matches[1]; Audio = $Matches[2]
                    Model = $Matches[3]; Dir = $null
                }
            } elseif ($ln -match 'Job (\d+) artifacts dir: (.+?)\s*$' -and $jobs.Count) {
                $jobs[-1].Dir = $Matches[2]
            }
        }
    }
    if (-not $jobs) { "No analysis jobs found in the newest worker logs."; return }

    foreach ($job in ($jobs | Select-Object -Last 5)) {
        "--- Job $($job.Job) [$($job.Model)] in $($job.Log)"
        "Audio : $($job.Audio)"
        $audio = $job.Audio -replace '/', '\'
        if (Test-Path -LiteralPath $audio) {
            # Get-Item reads metadata only, so a cloud placeholder is not downloaded.
            $item = Get-Item -LiteralPath $audio -Force
            $attr = [int]$item.Attributes
            $flags = @()
            if ($attr -band 0x1000)   { $flags += 'OFFLINE' }
            if ($attr -band 0x40000)  { $flags += 'RECALL_ON_OPEN' }
            if ($attr -band 0x400000) { $flags += 'RECALL_ON_DATA_ACCESS (cloud placeholder)' }
            "        size=$($item.Length) modified=$($item.LastWriteTime.ToString('s')) attrs=0x$($attr.ToString('x')) $($flags -join ',')"
        } else {
            "        NOT FOUND"
        }

        $dir = $job.Dir
        if (-not $dir) {
            $dir = Join-Path (Split-Path -Parent $audio) ([System.IO.Path]::GetFileNameWithoutExtension($audio))
            "Cache : $dir (guessed; older build did not log it)"
        } else {
            "Cache : $dir"
        }
        if (-not (Test-Path -LiteralPath $dir)) { "        folder not found"; continue }
        foreach ($f in (Get-ChildItem -LiteralPath $dir -File | Sort-Object Name)) {
            $line = "        {0,-28} {1,10}  {2}" -f $f.Name, $f.Length, $f.LastWriteTime.ToString('s')
            if ($f.Name -like 'stt_cache_*.json') {
                $raw = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
                $segs = ([regex]::Matches($raw, '"text"\s*:')).Count
                $vad = if ($raw -match '"vad_timestamps"') { 'yes' } else { 'no' }
                $line += "  completed=$(Get-JsonFlag $raw 'completed') gap_fill_completed=$(Get-JsonFlag $raw 'gap_fill_completed') segments=$segs vad_cached=$vad"
            } elseif ($f.Name -eq 'events_cache.json') {
                $raw = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
                $line += "  detections=$(([regex]::Matches($raw, '"type"\s*:')).Count)"
            }
            $line
        }
    }
}

# -- Pack into zip -------------------------------------------------------------

if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }
Compress-Archive -Path "$tmp\*" -DestinationPath $Out -CompressionLevel Optimal
Remove-Item $tmp -Recurse -Force

$size = "{0:N1} KB" -f ((Get-Item -LiteralPath $Out).Length / 1KB)
Write-Host ""
Write-Host "Diagnostics written to: $Out  ($size)" -ForegroundColor Green
Write-Host "Share that file for support." -ForegroundColor Green
