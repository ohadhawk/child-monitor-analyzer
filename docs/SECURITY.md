# Supply-chain security posture

Status as of 2026-08-03. This document records what is defended, how, and what
is knowingly accepted. It is not a general threat model for the application.

## Why this matters here

The program runs on a parent's machine, downloads ~1 GB of third-party model
weights on first run, and produces reports about a child's speech. The two
realistic compromise paths are:

1. a Python package release being replaced or backdoored, and
2. a model artifact (a pickle or a Hugging Face repo) being replaced.

Both would execute or influence code on the user's machine with no visible
symptom.

## Controls in place

| # | Control | Where | Guards against |
|---|---------|-------|----------------|
| 1 | PANNs inference code **vendored** instead of installed | `src/monitor/vendor/panns/` | Abandoned upstream packages (`panns-inference`, `torchlibrosa`, last released early 2023, single maintainer, no release attestations) |
| 2 | `torch.load(..., weights_only=True)` | `vendor/panns/inference.py` | Arbitrary code execution from a crafted `.pth` pickle |
| 3 | Pinned **SHA-256** of the PANNs checkpoint, verified on every load; failures quarantined | `model_cache.ensure_panns_checkpoint` | Substituted or corrupted weights |
| 4 | **HTTPS enforced**, including across redirects | `model_cache._require_https`, `_HttpsOnlyRedirectHandler` | On-path tampering; the upstream labels URL was plain HTTP |
| 5 | Hugging Face downloads pinned to **commit SHAs** | `stt._STT_REVISIONS`, `profanity._TOXICITY_REVISION` | Moving `main` branch silently changing model behaviour |
| 6 | AudioSet label file validated structurally (exactly 527 rows, fail-closed) | `vendor/panns/labels.py` | Truncated/substituted labels silently mislabelling detections |
| 7 | **Hash-pinned lockfile**, 92 packages, `--require-hashes --only-binary=:all:` | `requirements.lock`, `install.ps1` | Compromised package release, dependency confusion |
| 8 | Single explicit `--index-url` for installs | `install.ps1` | A stray `PIP_EXTRA_INDEX_URL` shadowing a package |
| 9 | Upper version bounds on every direct dependency | `requirements.txt`, `pyproject.toml` | An unreviewed major release landing automatically |
| 10 | No shell execution anywhere in vendored code | enforced by test | Upstream's `os.system('wget -O "{path}" ...')` command injection |
| 11 | matplotlib and Pillow removed from the dependency tree **and** added to PyInstaller `excludes` | `monitor-gui.spec` | 27 open Pillow advisories shipped for zero functionality |
| 12 | All of the above enforced by tests | `tests/test_supply_chain.py`, `tests/test_model_cache.py`, `tests/test_labels.py` | Silent regression |

## Vendoring: what changed vs upstream

`src/monitor/vendor/panns/` is derived from `panns_inference` and
`torchlibrosa` (both MIT, Qiuqiang Kong). Full attribution and the complete
change list are in `src/monitor/vendor/panns/LICENSE-third-party.txt`.

Correctness of the port is proven, not assumed: `tests/test_vendor_panns.py`
compares the vendored model against the upstream packages **bit-for-bit**
(`rtol=0, atol=0`), including a run against the real 312 MB checkpoint. Those
comparisons skip automatically once the upstream packages are uninstalled.

## Accepted risks

- **`setuptools` / `pip`** advisories are build- and install-time only. They
  are excluded from the shipped binary. The lockfile pins `setuptools==83.0.0`,
  which resolves `PYSEC-2026-3447`.
- **`torch` `GHSA-rrmf-rvhw-rf47`** (LOW) affects `torch.jit.script`, which this
  project never calls. The lockfile pins `torch==2.13.0+xpu`, where it is fixed.
- **The AudioSet labels CSV is not digest-pinned.** It is served from a Google
  Cloud Storage bucket that has published new revisions historically. It is
  instead fetched over HTTPS and validated structurally (exactly 527 rows,
  3 columns); a substituted file of a different length fails closed. Pinning a
  digest would break first-run installs whenever upstream republishes.
- **The Hugging Face model weights are not digest-pinned**, only revision-pinned.
  A commit SHA in a Git-backed repository is itself a content commitment, so
  this is equivalent in strength for the repository contents.
- **The Google `id_token` signature is not verified.** Its claims are used only
  to display which account is connected; every authorisation decision is made
  by Google when it honours the access token. Verifying the signature would
  mean shipping a JWKS fetcher and key cache for no security gain. The `iss`,
  `aud` and `exp` claims *are* checked so a token from another application
  cannot be presented.

## Google Drive upload

The optional Drive upload uses OAuth 2.0 with PKCE and a loopback redirect
(RFC 8252). Design points:

- **The `client_secret` is not treated as a secret.** PKCE (`S256`) carries the
  real security. Google nevertheless issues a secret for Desktop clients and
  its token endpoint rejects the exchange without one, so it is sent when
  configured. Google's own documentation states installed apps "cannot keep
  secrets", so this value is a client identifier in practice, not a
  credential. Tests assert that no secret value is committed to source and
  that it never reaches a log record.
- **Least privilege.** The scopes are `drive.file`, `openid` and `email`.
  `drive.file` grants access *only* to files this app itself created — it can
  never read the user's existing Drive contents.
- **The redirect listener** binds `127.0.0.1` on an OS-assigned ephemeral port,
  never `0.0.0.0` and never the name `localhost`. `state` is compared with
  `hmac.compare_digest` *before* the authorization code is read, the result
  slot is write-once, and `BaseHTTPRequestHandler.log_message` is overridden so
  the code cannot reach a log file.
- **The refresh token lives only in the Windows Credential Manager**
  (via `keyring`). `~/.child-monitor-analyzer/gdrive.json` holds the account
  id, e-mail and folder id — never a token. Access tokens exist in memory only
  and are sent in an `Authorization` header, never in a query string.
- **Revocation happens before the local purge**, so "sign out" cannot claim to
  have revoked access it did not.
- **All logging passes through `monitor.log_redaction`**, which strips
  `ya29.*`, `1//*`, JWTs, `Authorization` headers and the OAuth form/JSON
  fields from records before any handler sees them.
- **TLS verification is never disabled** and redirects are refused outright on
  every token and Drive request.

### Configuration

The feature stays off until an OAuth client id is supplied:

```powershell
$env:MONITOR_GOOGLE_CLIENT_ID = "<id>.apps.googleusercontent.com"
$env:MONITOR_GOOGLE_CLIENT_SECRET = "GOCSPX-<secret>"
```

Or run `scripts\setup-google-drive.ps1`, which sets both for the current user
and verifies the result.

Create it in Google Cloud Console: enable the Drive API, create a **Desktop
app** OAuth client, and configure the consent screen with the three scopes
above. Publish the app to **In production** — while it is in *Testing*, Google
expires refresh tokens after 7 days, which would force a re-login every week.

## Routine maintenance

```powershell
# Audit the environment
.\.venv\Scripts\python.exe -m pip_audit

# Refresh the lockfile after changing requirements.txt
.\.venv\Scripts\uv.exe pip compile requirements.txt --generate-hashes `
    --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/xpu `
    --index-strategy unsafe-best-match `
    --output-file requirements.lock

# Verify nothing regressed
.\.venv\Scripts\python.exe -m pytest tests -q
```

Rotating a pinned model revision is a deliberate edit to `_STT_REVISIONS` or
`_TOXICITY_REVISION`, and should be reviewed like any other code change.
