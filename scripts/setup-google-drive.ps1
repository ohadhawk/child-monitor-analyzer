<#
.SYNOPSIS
    Configure the Google Drive upload feature for Child Monitor Analyzer.

.DESCRIPTION
    Google does not expose an API for creating OAuth clients, so the client
    itself must be created by hand in the Cloud Console. This script automates
    everything around that: it opens the exact console pages in the right
    order, then validates, stores and verifies the resulting client ID.

    Run it once. The resulting client ID is not a secret and is not per-user.

.PARAMETER ClientId
    Supply the client ID non-interactively (skips the browser steps).

.PARAMETER ClientSecret
    The client secret shown next to the client ID. Google's token endpoint
    rejects the exchange without it, even for Desktop clients using PKCE.
    It is not a true secret -- Google states installed apps cannot keep one.

.PARAMETER PatchSource
    Write the ID into src/monitor/gdrive/auth.py instead of an environment
    variable. Use this for the packaged .exe, where an environment variable
    would never be set on the end user's machine.

.PARAMETER VerifyOnly
    Skip configuration and just report the current state.
#>
[CmdletBinding()]
param(
    [string] $ClientId,
    [string] $ClientSecret,
    [switch] $PatchSource,
    [switch] $VerifyOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$AuthFile = Join-Path $RepoRoot 'src\monitor\gdrive\auth.py'
$EnvVar = 'MONITOR_GOOGLE_CLIENT_ID'
$SecretEnvVar = 'MONITOR_GOOGLE_CLIENT_SECRET'

# Google issues these as <digits>-<random>.apps.googleusercontent.com. Checking
# the shape here turns a silent 401 during sign-in into an immediate error.
$ClientIdPattern = '^[0-9]+-[a-z0-9_]+\.apps\.googleusercontent\.com$'
$ClientSecretPattern = '^GOCSPX-[A-Za-z0-9_\-]+$'

function Write-Step { param([int]$N, [string]$Text) Write-Host "`n[$N] $Text" -ForegroundColor Cyan }
function Write-Ok { param([string]$Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "    !   $Text" -ForegroundColor Yellow }

function Get-PythonExe {
    $venv = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    throw "No Python found. Run install.ps1 first."
}

function Test-Configuration {
    <# Ask the application itself, so this reports what the app will actually see. #>
    $python = Get-PythonExe
    $probe = @'
import sys, os
sys.path.insert(0, "src")
from monitor.gdrive import auth, store
print("CONFIGURED", auth.is_configured())
print("CLIENTID", auth.client_id() or "<none>")
print("SECRET", "present" if auth.client_secret() else "<none>")
try:
    print("BACKEND", store.backend_name())
except Exception as exc:
    print("BACKEND <unavailable: %s>" % exc)
'@
    Push-Location $RepoRoot
    try { $probe | & $python - } finally { Pop-Location }
}

if ($VerifyOnly) {
    Write-Step 1 'Current configuration'
    Test-Configuration
    return
}

# --- 1. Create the Cloud project and enable the API ------------------------

if (-not $ClientId) {
    Write-Step 1 'Create a Google Cloud project and enable the Drive API'
    Write-Host '    A browser window will open. Create (or pick) a project, then'
    Write-Host '    click Enable on the Google Drive API page.'
    Read-Host '    Press Enter to open the browser'
    Start-Process 'https://console.cloud.google.com/apis/library/drive.googleapis.com'
    Read-Host '    Press Enter once the Drive API shows as Enabled'

    # --- 2. Consent screen -------------------------------------------------

    Write-Step 2 'Configure the consent screen'
    Write-Host '    User type:  External'
    Write-Host '    Scopes:     .../auth/drive.file , openid , email'
    Write-Host ''
    Write-Host '    drive.file lets the app touch only files it created itself —'
    Write-Host '    it can never read the rest of the Drive. Do not add more.'
    Read-Host '    Press Enter to open the consent screen'
    Start-Process 'https://console.cloud.google.com/auth/overview'
    Read-Host '    Press Enter once the three scopes are saved'

    # --- 3. Publish --------------------------------------------------------

    Write-Step 3 'Publish the app to production'
    Write-Warn 'While the app is in "Testing", Google expires refresh tokens after'
    Write-Warn '7 days, forcing a fresh sign-in every week. Publish it.'
    Read-Host '    Press Enter to open the audience page'
    Start-Process 'https://console.cloud.google.com/auth/audience'
    Read-Host '    Press Enter once the status reads "In production"'

    # --- 4. Create the client ----------------------------------------------

    Write-Step 4 'Create the OAuth client'
    Write-Host '    Create Credentials -> OAuth client ID -> Desktop app'
    Write-Host ''
    Write-Host '    The console shows a Client ID and a Client secret. You need both:'
    Write-Host '    Google rejects the token exchange without the secret, even though'
    Write-Host '    this app uses PKCE. It is not a real secret (an installed app'
    Write-Host '    cannot keep one) but it is still never written to the log.'
    Read-Host '    Press Enter to open the credentials page'
    Start-Process 'https://console.cloud.google.com/auth/clients'
    $ClientId = Read-Host '    Paste the Client ID'
    $ClientSecret = Read-Host '    Paste the Client secret'
}

$ClientId = $ClientId.Trim()
if ($ClientId -notmatch $ClientIdPattern) {
    throw "That does not look like a client ID (expected <digits>-<id>.apps.googleusercontent.com), got: '$ClientId'"
}

$ClientSecret = "$ClientSecret".Trim()
if ($ClientSecret -and $ClientSecret -notmatch $ClientSecretPattern) {
    throw "That does not look like a client secret (expected GOCSPX-...)."
}
if (-not $ClientSecret) {
    Write-Warn 'No client secret supplied. Google will reject the sign-in with'
    Write-Warn '"client_secret is missing" unless this client predates that rule.'
}

# --- 5. Store it -----------------------------------------------------------

Write-Step 5 'Store the credentials'
if ($PatchSource) {
    $content = Get-Content $AuthFile -Raw
    if ($content -notmatch '(?m)^DEFAULT_CLIENT_ID = ') {
        throw "Could not find DEFAULT_CLIENT_ID in $AuthFile"
    }
    if ($content -notmatch '(?m)^DEFAULT_CLIENT_SECRET = ') {
        throw "Could not find DEFAULT_CLIENT_SECRET in $AuthFile"
    }
    $updated = $content -replace '(?m)^DEFAULT_CLIENT_ID = .*$', "DEFAULT_CLIENT_ID = `"$ClientId`""
    $updated = $updated -replace '(?m)^DEFAULT_CLIENT_SECRET = .*$', "DEFAULT_CLIENT_SECRET = `"$ClientSecret`""
    Set-Content -Path $AuthFile -Value $updated -Encoding UTF8 -NoNewline
    Write-Ok "Written to $AuthFile (this ships with the build)"
    Write-Warn 'Do not commit this change: revert auth.py after building.'
} else {
    [Environment]::SetEnvironmentVariable($EnvVar, $ClientId, 'User')
    [Environment]::SetEnvironmentVariable($SecretEnvVar, $ClientSecret, 'User')
    $env:MONITOR_GOOGLE_CLIENT_ID = $ClientId
    $env:MONITOR_GOOGLE_CLIENT_SECRET = $ClientSecret
    Write-Ok "$EnvVar and $SecretEnvVar set for the current user"
    Write-Warn 'Restart VS Code / any open terminal so they inherit it.'
}

# --- 6. Verify -------------------------------------------------------------

Write-Step 6 'Verify'
Test-Configuration
Write-Host ''
Write-Host 'CONFIGURED True and a real BACKEND mean the cloud chip is now live.' -ForegroundColor Green
Write-Host 'Start the app and use the cloud chip to sign in.'
