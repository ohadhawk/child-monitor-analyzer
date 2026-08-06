# Feature Plan — Google Drive / Google Docs transcript upload

Status: **Implemented** · Shipped in: 1.2.1 · Author: design pass, 2026-08-03

---

## 1. Goals

1. Give the user a **Google Drive menu** plus a persistent, glanceable
   **signed-in / signed-out indicator** on the main window.
2. Add an **upload (↑)** action next to the existing **download (↓)** action in the
   transcript toolbar, using the *same* options dialog (timestamps / events / format).
3. Uploaded transcripts land in a dedicated **"Transcriptions" folder** in the user's
   Drive, converted to a native **Google Doc**.
4. **Unify the transcript file name** across all three destinations (`.txt`, `.docx`,
   Google Drive) so it includes: recording folder name + recording date/time + model
   quality label.
5. **Security is the top priority.** No regression in the app's current
   zero-credential posture may be introduced carelessly.
6. **Harden the dependency supply chain** *before* introducing credential handling
   (§13). Adding an OAuth token to a process built from ~109 unpinned packages would
   be putting a lock on a door with no frame.

### Explicit non-goals (v1)

- No automatic/silent upload on analysis completion. (Deliberate — see §8.1.)
- No Google Docs API usage, no editing of existing user documents.
- No download-from-Drive, no Drive browsing, no sharing/permission management.
- No multi-account support. One linked account at a time.

---

## 2. Current-state summary (verified against the code)

| Concern | Today |
|---|---|
| GUI | PySide6, `MainWindow(QMainWindow)`, RTL (`app.setLayoutDirection(RightToLeft)`) |
| Settings | `QSettings("ChildMonitorAnalyzer", "monitor-gui")` → Windows Registry (per-user) |
| Icons | **No image assets at all** — icons are drawn with `QPainter` in [player_icons.py](../src/monitor/gui/player_icons.py); the download button is the literal Unicode `"↓"` |
| Strings | Central table in [strings.py](../src/monitor/gui/strings.py): `class S` keys + `_STRINGS[(key, Lang)]`, accessed via `tr(S.KEY)` |
| Export dialog | `TranscriptExportDialog` in [transcript_widget.py](../src/monitor/gui/transcript_widget.py) — format combo (`txt`/`docx`) + 2 checkboxes, persisted to QSettings |
| Export name | `_default_export_name()` → `f"{audio_stem} {model_label}"` |
| Export source plumbing | `set_export_source(audio_path, model_key)` already receives the **full audio path** — mtime is available for free |
| Background work | Analysis in a `multiprocessing.Process`; network (`_ModelCheckWorker`) in a `QThread` + `moveToThread` + `finished`/`failed` signals |
| Networking | `urllib.request` only. **No `requests` in our own code**, no OAuth, no keyring |
| Artifact layout | `<audio parent>/<sanitize_artifact_stem(stem)>/` containing `analysis_<key>.json`, `transcript_<key>.txt`, caches |
| Model keys | `"thorough"` / `"fast"` / `"none"`; Hebrew labels via `S.EXPORT_NAME_THOROUGH` / `S.EXPORT_NAME_FAST` |
| Logging | `logging.getLogger("monitor")` → `~/.child-monitor-analyzer/logs/monitor_<ts>.log` |
| Packaging | PyInstaller **onedir**, [monitor-gui.spec](../monitor-gui.spec) with explicit `hiddenimports` |

---

## 3. Work item A — Unified transcript file naming

### 3.1 Requirement

Every transcript artifact — local `.txt`, local `.docx`, and the Google Doc — uses one
canonical base name containing:

- the **recording folder name** (`sanitize_artifact_stem(audio.stem)`),
- the **recording's modification date & time**,
- the **model quality label** (thorough / fast).

### 3.2 Format

```
<YYYY-MM-DD HH-MM> <folder name> - <תמלול יסודי | תמלול מהיר>
```

Example: `2026-04-16 15-59 אריאל 1 - תמלול יסודי.docx`

**Design notes:**

- **Date first** so both Explorer and Drive sort chronologically by default — the
  primary way the user will look for a recording.
- **`HH-MM`, not `HH:MM`.** The colon is illegal in Windows filenames *and* is the NTFS
  Alternate Data Stream separator — `report:HH` would silently create a hidden ADS.
  This is a correctness *and* a security issue.
- ISO-8601-ordered date (`YYYY-MM-DD`) for lexicographic = chronological ordering.
- Timestamp is `datetime.fromtimestamp(path.stat().st_mtime)` in **local time**
  (matches what Explorer shows the user, which is the mental model).
- On `OSError` or a missing/zero mtime, **omit the timestamp prefix entirely** rather
  than substituting "now" — a wrong date is worse than no date.
- Model label and its `" - "` separator are omitted when `model_key == "none"`
  (events-only).

**Under RTL rendering:** the leading ASCII timestamp inside an otherwise-Hebrew name is
displayed by Windows/Drive according to the bidi algorithm and may *appear* at the
right. The stored bytes are what matter for sorting, and they are unambiguous. Do **not**
"fix" this by inserting bidi control characters into the filename — that is precisely
the spoofing vector §3.4 strips.

### 3.3 Implementation

New single source of truth, in `transcript_widget.py` (or a small shared helper):

```python
def build_transcript_base_name(audio_path: str | None, model_key: str | None) -> str:
    """Canonical base name shared by .txt, .docx and Google Docs export."""
```

- `set_export_source()` stores `self._audio_path` (full path) in addition to the
  existing `_audio_stem` / `_model_key`. No call-site changes needed — it already
  receives the full path at both call sites
  ([main_window.py:575](../src/monitor/gui/main_window.py#L575) and
  [:843](../src/monitor/gui/main_window.py#L843)).
- `_default_export_name()` becomes a thin wrapper over `build_transcript_base_name()`.
- Uses `sanitize_artifact_stem()` from [models.py](../src/monitor/models.py) so the
  name matches the artifact folder exactly.

### 3.4 Filename hardening (security — see also §8.4)

A new `sanitize_display_name()` runs on the final base name before it is used as a
local filename **or** sent as the Drive `name` field:

| Threat | Mitigation |
|---|---|
| **Unicode RTLO spoofing** — a stem containing U+202E makes `…txt.exe` render as `…exe.txt` | Strip bidi controls U+202A–U+202E, U+2066–U+2069, and U+200E/U+200F |
| Path traversal / separator injection into the Drive name | Strip `/ \ : * ? " < > \|` and NUL |
| Control characters, newlines | Strip C0/C1 |
| Windows reserved device names (`CON`, `PRN`, `NUL`, `COM1`…) | Prefix with `_` if matched |
| Overlong names (Drive limit, Windows MAX_PATH) | Truncate the *stem portion* to ~120 chars, preserving the date prefix + model suffix |

> Note: the audio stem is attacker-influenced in the realistic case where the user is
> handed a recording file by a third party. Treating it as untrusted input is correct,
> not paranoid.

---

## 4. Work item B — Google account indicator on the main window

### 4.1 Placement — indicator *and* menu in one control

The app currently has **no `QMenuBar`** (verified — `menuBar()` is never called). Adding
one solely for this feature would be a disproportionate change to the window chrome and
would sit awkwardly with the RTL toolbar-only design.

**Decision:** the Google Drive control is a `QToolButton` with
`ToolButtonPopupMode.InstantPopup` and an attached `QMenu` — i.e. it *is* the Google
Drive menu, and its icon *is* the status indicator. This is exactly the existing
`_btn_recent` pattern ([main_window.py:328](../src/monitor/gui/main_window.py#L328)),
so it needs no new UI vocabulary.

Placed in `_build_top_bar()` to the **left of `_btn_open`** (under RTL, the visual far
edge — the conventional account-chip location). `setFixedHeight(36)` to match the row.

### 4.1.1 Menu contents

| Item | Signed out | Signed in |
|---|---|---|
| `חשבון Google…` (opens `GoogleAccountDialog`) | ✔ | ✔ |
| `התחבר עם Google` | ✔ | — |
| `מחובר כ־<email>` (disabled, informational) | — | ✔ |
| `העלה את התמליל הנוכחי…` | disabled | ✔ (disabled when no transcript is loaded) |
| `פתח את תיקיית Transcriptions בדרייב` | — | ✔ (disabled until the folder exists) |
| `שאל לפני כל העלאה` (checkable, default **on**) | — | ✔ |
| `נתק חשבון` | — | ✔ |

The menu is rebuilt in an `aboutToShow` handler so its state always reflects reality
(same lifecycle as `_recent_menu`). Disabled-with-tooltip is preferred over hiding
items, so the feature is discoverable before sign-in (*progressive disclosure*).

If a conventional `QMenuBar` is later wanted, the same `QMenu` instance can be attached
to it unchanged — the menu is built by a standalone factory for that reason.

### 4.2 Visual states — "account chip" pattern

| State | Icon | Tooltip |
|---|---|---|
| Not linked | Grey cloud-with-up-arrow glyph, 40 % opacity | "התחבר לחשבון Google" |
| Linked | Same glyph, full colour | "מחובר כ־user@example.com" |
| Working (auth/upload in flight) | Glyph + small busy indicator, button disabled | "מתחבר…" / "מעלה…" |
| Error / token invalid | Glyph with a small warning dot | "החיבור פג — התחבר מחדש" |

Clicking pops the menu (§4.1.1); `חשבון Google…` opens the account dialog (§4.4).

### 4.3 ⚠️ Trademark decision — do NOT use a greyed-out Google "G"

The original idea was a Google "G" that is grey when logged out and Google-coloured when
logged in. **This must not be implemented as described.** Google's *Branding Guidelines*
for Identity explicitly prohibit altering the "G" mark — including recolouring or
desaturating it — and restrict its use to the official "Sign in with Google" button
assets. The same applies to the Drive triangle.

Additionally, the app currently ships **zero image assets** and draws every icon with
`QPainter`; hand-drawing a facsimile of the Google logo would be both a trademark
problem and a rendering-quality problem.

**Decision:** draw a **neutral cloud-upload glyph** in the existing
[player_icons.py](../src/monitor/gui/player_icons.py) style, with a
`grey → colour` state change. The word "Google" appears only as *text* in the tooltip
and dialog, which is nominative fair use and permitted.

If an official-looking sign-in affordance is wanted later, the correct approach is to
bundle Google's **official** "Sign in with Google" button PNG/SVG assets unmodified,
inside the account dialog only — not on the toolbar.

New functions in `player_icons.py`: `icon_cloud_upload(active: bool) -> QIcon`,
`icon_upload_arrow() -> QIcon`.

### 4.4 Google account dialog (`GoogleAccountDialog`)

Signed out:
- One-paragraph plain-language explanation of exactly what access is requested:
  *"האפליקציה תוכל ליצור ולערוך רק קבצים שהיא עצמה יצרה. אין לה גישה לשאר הקבצים בדרייב שלך."*
- **"התחבר עם Google"** button → starts the OAuth flow (§5).
- Text noting the browser will open.

Signed in:
- Account email, folder name, link to the Drive folder.
- **"נתק חשבון"** button → revoke + purge (§8.6).
- Checkbox: *"שאל לפני כל העלאה"* (default **on**).

---

## 5. Work item C — OAuth 2.0 authorization (security core)

### 5.1 Standards and patterns used

This is a deliberately conventional implementation. Named concepts applied:

- **RFC 8252 — OAuth 2.0 for Native Apps.** System browser + loopback redirect. No
  embedded webview (Google rejects those with `disallowed_useragent`, and they defeat
  the whole point of not handling the password).
- **RFC 7636 — PKCE**, `code_challenge_method=S256`.
- **Public client** (no `client_secret`). Google's own docs state that for installed
  apps the client secret "is obviously not treated as a secret"; PKCE makes it
  unnecessary, so we omit it entirely rather than embed a fake secret.
- **`state` parameter** for CSRF / response-injection protection, compared with
  `hmac.compare_digest`.
- **Principle of least privilege** for scope selection (§5.2).
- **Fail-safe defaults**: every failure path leaves the feature disabled, never
  half-authorized.
- **Defence in depth** on the loopback listener (§5.4).
- **Exponential backoff with full jitter** on 429/5xx (Google's documented guidance).

The out-of-band (OOB) copy-paste flow is **removed by Google** and is not an option.

### 5.2 Scope — `https://www.googleapis.com/auth/drive.file` and nothing else

> *"See, edit, create, and delete **only the specific Google Drive files you use with
> this app**."*

Why this is the single most important security decision in the plan:

- **Blast radius containment.** A stolen token can only touch files this app created.
  It cannot read the user's other documents, cannot enumerate their Drive, cannot
  reach Gmail or Photos.
- **`drive.file` is classified non-sensitive.** Per Google's OAuth App Verification
  docs, apps using only non-sensitive scopes are **not required** to complete app
  verification. `drive` / `drive.readonly` are *restricted* scopes requiring a
  third-party **CASA security assessment**, re-done annually, at real cost.
- We will still complete **brand verification** (the lightweight process) so the
  consent screen shows our name/logo and to clear the 100-user cap on unverified
  external apps.

We do **not** request `userinfo.email`. Instead the account email is read once from the
`files` API context… *(revised — see §5.3)*.

### 5.3 Obtaining the account email — revision

`drive.file` alone does not return the user's email. Options considered:

| Option | Verdict |
|---|---|
| `userinfo.email` scope | Adds a scope, but it is **non-sensitive** and excluded from the "Testing = 7-day refresh token" penalty. Shows as "See your primary Google Account email address". |
| `drive.about.get` with `fields=user` | Requires `drive.readonly`/`drive.metadata` — **restricted/sensitive**. Rejected. |
| Show no email, just "מחובר" | Weakens the security UX — the user cannot tell *which* account is linked, which is exactly how mis-targeted uploads happen. |

**Decision:** request `openid email` alongside `drive.file` and read the email from the
returned `id_token`. Both are non-sensitive; verification posture is unchanged. The
`id_token` signature is **not** security-critical here (it arrives over TLS directly
from Google's token endpoint in response to our own PKCE-bound request), but we will
still validate `iss`, `aud` and `exp` before trusting the `email` claim, and treat it
as display-only — never as an authorization decision.

### 5.4 Loopback listener hardening

```
redirect_uri = http://127.0.0.1:<ephemeral-port>/oauth2/callback
```

| Control | Rationale |
|---|---|
| Bind `127.0.0.1` explicitly — never `0.0.0.0`, never the hostname `localhost` | `localhost` can resolve to `::1` or be poisoned; Google's own docs warn it can also trip firewalls. Binding `0.0.0.0` would expose the callback to the LAN. |
| Port `0` (OS-assigned), read the **actual** port after `bind()`, then build `redirect_uri` | Prevents another local process from squatting a fixed port and stealing the code |
| `ThreadingHTTPServer` with `allow_reuse_address = False` | Refuses to share the port |
| Serve **exactly one** request, then `shutdown()` in a `finally:` | Minimal exposure window |
| Hard timeout (120 s), then abort and tear down | No indefinitely-listening socket |
| Reject any request whose path ≠ `/oauth2/callback` with 404 | Reduces surface for local probing |
| Compare `state` with `hmac.compare_digest` before touching `code` | CSRF / code-injection |
| `code_verifier = secrets.token_urlsafe(64)`; `state = secrets.token_urlsafe(32)` | CSPRNG, adequate entropy |
| Response headers: `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, `Content-Security-Policy: default-src 'none'` | The URL contains the auth code; prevent it leaking via Referer or cache |
| Success page is static inline HTML with **no external links or resources** | Same |
| **Never log the request line** (override `log_message` to a no-op) | `BaseHTTPRequestHandler` logs the full URL — *including the auth code* — to stderr by default. This is a real, easy-to-miss leak. |
| Only one auth flow at a time; the chip is disabled while in flight | Avoids racing listeners |

### 5.5 Token exchange & refresh

- `POST https://oauth2.googleapis.com/token`, `urllib.request` with an explicit
  `ssl.create_default_context()`. Certificate verification is never disabled.
- No redirect following on the token endpoint.
- Bounded response read (e.g. 64 KiB) before JSON parsing.
- **Access token: memory only.** Never written to disk, never to QSettings, never to a
  log. Held in the auth object with a computed expiry; refreshed at `expiry - 60 s`.
- **Refresh token: Windows Credential Manager** (§5.6).
- Access token is sent **only** as an `Authorization: Bearer` header — never as an
  `access_token` query parameter (Google explicitly discourages this because URLs land
  in logs).
- On `invalid_grant`: purge the stored credential, flip the chip to the error state,
  prompt to re-link. Never retry-loop.

### 5.6 Credential storage

**Windows Credential Manager**, via the `keyring` package (Windows backend →
`CredWrite`/`CredRead`, DPAPI-encrypted, bound to the Windows user *and* machine).

```
service = "child-monitor-analyzer:google-oauth"
username = "<google account sub claim>"
password = <refresh token>
```

- Refresh tokens are ≤512 bytes (Google's documented limit); Credential Manager's
  ~2560-byte blob limit is not a concern.
- Nothing secret ever goes into `QSettings` (which is the **Registry**, plainly
  readable) or into the `~/.child-monitor-analyzer/config.json`.
- If `keyring` is unavailable or its Windows backend fails to load, the feature is
  **disabled with an explanatory message**. It must never silently fall back to a
  plaintext file. (Fail-safe defaults.)

**Threat model — stated honestly:**

| Threat | Protected |
|---|---|
| Another Windows user on the machine reads the token | ✅ DPAPI keys are per-user |
| Credential blob copied to another machine (backup, sync, exfil) | ✅ Machine+user bound, undecryptable elsewhere |
| Offline disk inspection | ✅ Encrypted at rest |
| Full Google account takeover from a stolen token | ✅ Contained — `drive.file` only reaches our own files |
| **Malware already running as the logged-in user** | ❌ **Not preventable.** It can call `CredRead` exactly as we do. |

That last row is inherent to every desktop app, including the user's browser — whose
Google session cookies sit under the same DPAPI protection and grant vastly more
access. This feature does not create a new class of risk; it creates a *smaller* one
than the browser workflow it replaces.

### 5.7 Deferred hardening — DPoP (RFC 9449)

Google's token endpoint now supports **DPoP**, binding the refresh token to an EC P-256
private key proven per-request via a signed JWT. With the key held in the **TPM** (CNG,
non-exportable), an exfiltrated token becomes useless off-device.

Deferred to a later version — it closes the exfiltration gap but not the local-malware
gap, and costs JWT signing + nonce handling + CNG interop. **The credential layer will
be designed behind an interface so DPoP can be added without touching call sites.**

---

## 6. Work item D — Upload UI & Drive integration

### 6.1 The ↑ button

In `TranscriptWidget`'s `search_row`, immediately beside the existing
`_btn_download` (`"↓"`, [transcript_widget.py:237](../src/monitor/gui/transcript_widget.py#L237)):

```python
self._btn_upload = QPushButton("↑")     # or icon_upload_arrow()
self._btn_upload.setFixedWidth(28)
self._btn_upload.setToolTip(tr(S.TRANSCRIPT_UPLOAD_DRIVE))
self._btn_upload.clicked.connect(self._upload_transcript)
```

Enablement mirrors `_btn_download` (disabled when there is nothing to export).

### 6.2 Shared options dialog — one dialog, two destinations

`TranscriptExportDialog` gains a `mode` parameter (`"download"` | `"upload"`) rather
than being forked. This guarantees the two paths can never drift apart, which is
exactly the requirement ("same selection … whether we want the timestamps or not").

| | `download` | `upload` |
|---|---|---|
| Title | "הורדת תמליל" | "העלאה ל‑Google Drive" |
| Format combo | `txt` / `docx` | `txt` / `docx` → both convert to a Google Doc; `docx` preserves RTL formatting and is the default |
| Include timestamps | ✔ shared QSettings key | ✔ same key |
| Include events | ✔ shared QSettings key | ✔ same key |
| Confirm button | "שמור" | "העלה" |
| Extra row | — | Target: "Google Drive › Transcriptions › `<final file name>`" + linked account email |

The upload dialog **shows the exact destination and final file name** before
confirming. This is the *informed-consent* / *no-surprise* principle — the user must
never discover after the fact where a transcript of their child went.

`_build_transcript_lines(include_ts, include_ev)` is reused unchanged for all three
destinations, so content is byte-identical between a local save and an upload.

### 6.3 If not signed in

Clicking ↑ while signed out opens the **account dialog** first, then continues to the
export dialog on success. No "sign in and upload in one silent step".

### 6.4 The "Transcriptions" folder

- Created on first upload: `files.create` with
  `mimeType=application/vnd.google-apps.folder`, `name="Transcriptions"`.
- Folder ID persisted in `QSettings` **keyed by account** (`gdrive/folder_id/<sub-hash>`)
  so switching accounts can never write into a stale foreign folder ID.
- **Known limitation of `drive.file`:** we cannot search the user's Drive for a folder
  we didn't create. If the stored ID is lost or the folder was trashed, a `404`/`trashed`
  response triggers creating a fresh folder. Documented in the UI help text. This is an
  accepted, deliberate cost of the least-privilege scope.

### 6.5 Upload mechanics

```
POST https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,webViewLink
Content-Type: multipart/related; boundary=<random>

--<boundary>
Content-Type: application/json; charset=UTF-8

{"name": "...", "mimeType": "application/vnd.google-apps.document", "parents": ["<folderId>"]}
--<boundary>
Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document

<docx bytes>
--<boundary>--
```

- `mimeType: application/vnd.google-apps.document` makes **Drive perform the
  conversion server-side** → a native Google Doc. No Google Docs API, no extra scope.
- Transcripts are kilobytes, so `uploadType=multipart` is correct; resumable upload is
  unnecessary complexity here.
- **Multipart boundary must be `secrets.token_hex(16)`, generated per request.** With a
  fixed boundary, a transcript containing that literal string could break out of the
  body part — a genuine injection vector. Additionally, assert the boundary does not
  occur in the payload before sending.
- **Metadata JSON is built with `json.dumps`**, never string concatenation. The `name`
  is attacker-influenced (§3.4).
- Retries: 429 / 500 / 502 / 503 / 504 → exponential backoff with full jitter, max 4
  attempts, hard cap ~30 s total. 4xx (other than 429) → fail immediately, no retry.

### 6.6 Idempotency — avoid duplicate documents

A sidecar `gdrive_<model_key>.json` in the artifact folder records
`{account_sub, file_id, uploaded_at, name}`. On re-upload of the same transcript:

- If a `file_id` exists for the current account → offer **"עדכן את המסמך הקיים"** vs
  **"צור מסמך חדש"**.
- Update path: `PATCH .../upload/drive/v3/files/<id>?uploadType=multipart`.
- A `404` (user deleted it in Drive) transparently falls back to creating a new file.

The sidecar contains **no secrets** — only opaque Drive IDs.

### 6.7 Threading

Upload runs on a `QThread` worker following the existing `_ModelCheckWorker` idiom
verbatim: `moveToThread` → `started→run` → `finished(object)` / `failed(str)` signals →
`quit`/`deleteLater`, with strong references held on `self` to prevent premature GC.

- The GUI never blocks.
- Upload **must not** be placed in the analysis `multiprocessing.Process` — that
  process is `terminate()`d on cancel, which would kill an in-flight upload
  mid-request and could leave a partial file. Credentials must also never cross the
  process boundary.
- On success: status message + a "פתח ב‑Google Docs" link (`webViewLink`).
- On failure: non-modal warning + a **"נסה שוב"** action. An upload failure must never
  affect the analysis or the local artifacts.

---

## 7. Module layout

```
src/monitor/gdrive/
    __init__.py       # public façade: is_linked(), link(), unlink(), upload_transcript()
    auth.py           # PKCE, loopback listener, token exchange/refresh/revoke
    store.py          # keyring wrapper (get/set/delete refresh token) — swappable for DPoP
    client.py         # Drive REST calls: ensure_folder, create_file, update_file, backoff
    naming.py         # build_transcript_base_name(), sanitize_display_name()

src/monitor/gui/
    google_account.py # GoogleAccountDialog + the toolbar account chip widget
```

`monitor.gdrive` has **no PySide6 import** — it is headless and unit-testable, and the
GUI layer owns all threading. `naming.py` is imported by both the GUI and (optionally)
the pipeline so the naming rule has exactly one definition.

### Dependencies

Add **`keyring`** only (pulls `pywin32-ctypes` on Windows).

Deliberately **not** adding `google-api-python-client` / `google-auth-oauthlib`: they
are heavy, pull a large transitive tree, are notoriously painful under PyInstaller
(discovery-doc data files, hidden imports), and the entire flow is ~250 lines against
`urllib.request` — which matches the existing convention in
[model_cache.py](../src/monitor/model_cache.py) and
[model_updates.py](../src/monitor/model_updates.py) that already avoid `requests`.

Fewer dependencies is also a **supply-chain security** argument, which matters more
than usual for a package that will hold OAuth credentials.

### Packaging ([monitor-gui.spec](../monitor-gui.spec))

Add to `hiddenimports`:

```
"monitor.gdrive", "monitor.gdrive.auth", "monitor.gdrive.store",
"monitor.gdrive.client", "monitor.gdrive.naming",
"monitor.gui.google_account",
"keyring", "keyring.backends", "keyring.backends.Windows",
"win32ctypes", "win32ctypes.pywin32",
```

`keyring` uses entry-point discovery for backends, which PyInstaller cannot see — the
explicit `keyring.backends.Windows` import is **required** or the frozen build silently
falls back to a null/failing backend. Add a startup self-test that logs the resolved
backend name.

---

## 8. Security review

### 8.1 Privacy — the highest-severity concern

These transcripts contain speech captured from a child's environment. Uploading them
to a cloud service is the single most consequential thing this feature does, and it
outranks token handling in real-world impact.

Controls:

1. **No auto-upload.** Explicitly out of scope for v1. Every upload is a deliberate,
   per-file user action.
2. **Destination shown before confirming** — folder + exact file name + account email.
3. **Opt-in only**, off by default; the feature is invisible until the user links an
   account.
4. **Files are private by default.** We never call the `permissions` API, so the
   document inherits the user's default (private). No sharing links are generated.
5. **Least-privilege scope** so a compromise cannot enumerate the rest of their Drive.
6. First-link dialog states in plain Hebrew what is uploaded and where.

### 8.2 OAuth / token handling

Covered in §5.4–§5.6. Checklist for review:

- [ ] Loopback bound to `127.0.0.1`, ephemeral port, single request, timeout
- [ ] `BaseHTTPRequestHandler.log_message` overridden to a no-op **(auth-code leak)**
- [ ] `state` verified with `compare_digest` before reading `code`
- [ ] PKCE `S256`; `code_verifier` ≥ 43 chars from `secrets`
- [ ] No `client_secret` shipped
- [ ] TLS verification never disabled; explicit `ssl.create_default_context()`
- [ ] Access token in memory only, `Authorization` header only
- [ ] Refresh token only in Credential Manager; never QSettings/Registry/JSON/log
- [ ] Revoke endpoint called on disconnect, before local purge
- [ ] `invalid_grant` → purge + re-prompt, never retry-loop

### 8.3 Log redaction

The app logs to `~/.child-monitor-analyzer/logs/`, and users are likely to attach logs
to bug reports. Add a `logging.Filter` on the `monitor` logger that regex-redacts
`ya29.*`, `1//*`, `Bearer *`, `code=`, `refresh_token`, `access_token`, `id_token`, and
`client_secret` values. Explicitly unit-test it. Never `log.debug()` a raw request body
or URL from `gdrive.*`.

### 8.4 Input handling

- Filename sanitization per §3.4 — RTLO spoofing, separators, reserved names, length.
- Drive metadata built via `json.dumps`.
- Random multipart boundary + collision assertion (§6.5).
- Bounded reads on all HTTP responses before parsing.
- Treat every API response field as untrusted: `webViewLink` is only ever passed to
  `QDesktopServices.openUrl` **after** validating `scheme in {"https"}` and
  `host.endswith("google.com")` — otherwise a malicious/compromised response could
  drive the user's browser to an arbitrary URL.

### 8.5 Client ID exposure

The `client_id` is embedded in the binary. This is expected and unavoidable for a
public client; a third party can extract it, but the consent screen would then display
*our* verified app name. That is the residual risk, it is industry-standard, and brand
verification is what makes the displayed name meaningful. No further mitigation exists
and none is claimed.

### 8.6 Disconnect must be complete

"נתק חשבון" performs, in order:

1. `POST https://oauth2.googleapis.com/revoke?token=<refresh_token>` (server-side kill).
2. Delete the Credential Manager entry.
3. Clear cached account email, folder ID, and in-memory access token.
4. Reset the chip to signed-out.

Step 1 must come first. Deleting locally while leaving a live grant on Google's side is
a classic and dangerous half-measure — the user believes they revoked access when they
did not.

### 8.7 Offline / failure behaviour

No network → the ↑ button shows a clear "אין חיבור לאינטרנט" message. Nothing about the
existing offline analysis workflow may regress. The Drive module is imported lazily so
a missing `keyring` cannot prevent the app from starting.

---

## 9. Strings

New `S` keys + Hebrew/English entries in [strings.py](../src/monitor/gui/strings.py)
(existing convention: `S.KEY = "key"` plus `_STRINGS[(S.KEY, Lang.HE/EN)]`):

`GOOGLE_ACCOUNT_TITLE`, `GOOGLE_SIGN_IN`, `GOOGLE_SIGN_OUT`, `GOOGLE_CONNECTED_AS`,
`GOOGLE_NOT_CONNECTED`, `GOOGLE_SCOPE_EXPLANATION`, `GOOGLE_BROWSER_HINT`,
`GOOGLE_AUTH_FAILED`, `GOOGLE_AUTH_TIMEOUT`, `GOOGLE_SESSION_EXPIRED`,
`GOOGLE_KEYRING_UNAVAILABLE`, `TRANSCRIPT_UPLOAD_DRIVE`, `UPLOAD_DIALOG_TITLE`,
`UPLOAD_OK`, `UPLOAD_TARGET_LABEL`, `UPLOAD_IN_PROGRESS`, `UPLOAD_SUCCESS`,
`UPLOAD_FAILED`, `UPLOAD_RETRY`, `UPLOAD_OPEN_IN_DOCS`, `UPLOAD_REPLACE_EXISTING`,
`UPLOAD_CREATE_NEW`, `UPLOAD_NO_NETWORK`, `DRIVE_FOLDER_NAME`, `GOOGLE_MENU_TITLE`,
`GOOGLE_MENU_ACCOUNT`, `GOOGLE_MENU_UPLOAD_CURRENT`, `GOOGLE_MENU_OPEN_FOLDER`,
`GOOGLE_MENU_ASK_EVERY_TIME`.

RTL note: the account chip and both dialogs inherit the app-wide
`RightToLeft` direction; email addresses and URLs must be wrapped in
`\u2066…\u2069` (LRI/PDI isolates) so they don't render scrambled inside Hebrew
sentences.

---

## 10. Testing

**Unit (no network, no Qt):**
- `build_transcript_base_name` — normal, missing mtime, `model_key="none"`, unicode stem
- `sanitize_display_name` — RTLO, `:`/ADS, reserved device names, over-length, empty
- PKCE challenge derivation against the RFC 7636 test vector
- `state` mismatch → rejected
- Log-redaction filter — each secret pattern
- Backoff schedule bounds
- Multipart body assembly + boundary-collision assertion

**Integration (mocked HTTP):**
- Full auth flow with a stubbed token endpoint
- Refresh-on-expiry; `invalid_grant` → purge + re-prompt
- Folder create → cached ID → 404 → recreate
- Upload create vs. update path

**Manual:**
- Link / unlink / re-link; confirm revocation at `myaccount.google.com/permissions`
- Cancel the browser consent screen → clean abort, listener closed (`netstat` check)
- Close the app mid-auth → no lingering listener
- Airplane mode → graceful message
- Frozen PyInstaller build → keyring backend resolves to `WinVaultKeyring`
- Verify local `.txt`, local `.docx` and the Google Doc are content-identical for the
  same options

---

## 11. Delivery phases

| Phase | Content | Independently shippable |
|---|---|---|
| **0** | **§13 supply-chain hardening** — lockfile, model integrity, `pip-audit` in CI | ✅ Yes — **blocking prerequisite for phase 2** |
| **1** | Work item A — unified naming + `sanitize_display_name` + tests | ✅ Yes — pure improvement, zero new deps |
| **2** | `monitor.gdrive` headless module: auth, store, client, log redaction + tests | ✅ Yes — no UI, dead code until phase 3 |
| **3** | Drive menu + status chip, `GoogleAccountDialog`, link/unlink | ✅ Yes |
| **4** | ↑ button, dialog `mode`, upload worker, folder, idempotency | ✅ Yes — feature complete |
| **5** | PyInstaller spec, README, brand verification submission | — |
| **later** | DPoP + TPM-bound key | — |

**Prerequisite (manual, blocking phase 3):** Google Cloud project → enable Drive API →
create **Desktop app** OAuth client → configure consent screen (External,
`drive.file` + `openid` + `email`) → **publish to "In production"**.

> ⚠️ While the consent screen is in **"Testing"** status, refresh tokens **expire after
> 7 days**, forcing weekly re-login. Because only non-sensitive scopes are requested,
> moving to "In production" does **not** require the full verification review.

---

## 12. Review pass — gaps found and closed

| # | Gap in the first draft | Resolution |
|---|---|---|
| 1 | Greyed-out Google "G" logo as the status indicator | **Trademark violation** — Google forbids altering the mark. Replaced with a neutral cloud glyph; §4.3 |
| 2 | `HH:MM` in the filename | Illegal on Windows **and** an NTFS ADS separator. Changed to `HH-MM`; §3.2 |
| 3 | Account email source unspecified | `drive.file` doesn't provide it; `drive.about` needs a restricted scope. Added `openid email` (non-sensitive) + `id_token` claim validation; §5.3 |
| 4 | `BaseHTTPRequestHandler` default logging | Would write the **auth code** to stderr/log. `log_message` overridden; §5.4 |
| 5 | Fixed multipart boundary | Body-injection vector if the transcript contains it. Random per-request + collision assert; §6.5 |
| 6 | Audio stem treated as trusted | RTLO / separator / reserved-name spoofing. `sanitize_display_name()`; §3.4 |
| 7 | `webViewLink` opened directly | Untrusted response driving the user's browser. Scheme + host validation before `openUrl`; §8.4 |
| 8 | Disconnect only deleted local token | Left a live grant on Google's side. Revoke-first ordering; §8.6 |
| 9 | Duplicate Google Docs on re-upload | Sidecar `file_id` + update-vs-create prompt; §6.6 |
| 10 | `keyring` backend under PyInstaller | Entry-point discovery fails when frozen → silent null backend. Explicit hiddenimports + startup self-test; §7 |
| 11 | Upload placed in the analysis subprocess | That process is `terminate()`d on cancel. Moved to a GUI-side `QThread`; §6.7 |
| 12 | No fallback policy if `keyring` is missing | Risk of a plaintext-file fallback. Explicitly forbidden — feature disables itself; §5.6 |
| 13 | Log redaction not considered | Users attach logs to bug reports. Added a `logging.Filter` + tests; §8.3 |
| 14 | Folder ID not scoped to an account | Switching accounts could target a stale foreign ID. Keyed by account `sub`; §6.4 |
| 15 | "Testing" publishing status | Silent 7-day refresh-token expiry during development. Called out as a blocking prerequisite; §11 |
| 16 | Two separate export dialogs | Would drift apart, violating the "same options" requirement. Single dialog with a `mode`; §6.2 |
| 17 | RTL rendering of emails/URLs in Hebrew text | Scrambled display. LRI/PDI isolates; §9 |
| 18 | Fixed loopback port | Local port squatting. Ephemeral port, read after bind; §5.4 |
| 19 | "Google Drive menu" requirement unaddressed; app has **no menu bar** | Status chip promoted to a `QToolButton` + `InstantPopup` `QMenu`, reusing the existing `_btn_recent` pattern — indicator and menu in one control; §4.1 |
| 20 | Dependency supply chain not considered at all | ~109 unpinned packages share the process with the OAuth token. Added §13 as **blocking Phase 0**: hash-pinned lockfile, single index, `pip-audit` in CI, signed reproducible build |
| 21 | Model weights treated as data | `Cnn14_DecisionLevelMax.pth` is an **unverified pickle** → arbitrary code execution on load, and `snapshot_download` tracks a moving `main`. SHA-256 + `revision=` pinning; §13.2 |
| 22 | Timestamp placed mid-name | Moved to a leading `YYYY-MM-DD HH-MM` prefix so Explorer and Drive sort chronologically by default; §3.2 |

### Accepted residual risks

1. **Local malware running as the user can read the refresh token.** Unpreventable by
   any desktop application; mitigated in impact by `drive.file`. Partially addressable
   later via DPoP + TPM (§5.7).
2. **`client_id` is extractable from the binary.** Inherent to public clients.
3. **A lost folder ID creates a second "Transcriptions" folder.** The deliberate cost of
   least privilege over a `drive` full-access scope. Documented for the user.

---

## 13. Supply-chain security (Phase 0 — blocking prerequisite)

### 13.1 Why this is in scope

Python has **no in-process isolation**. Once this app holds a Google refresh token, any
one of the ~109 installed packages can read it, read `keyring`, and read the child
audio recordings — with no privilege boundary to stop it. Therefore the dependency
supply chain must be hardened *before* credentials are introduced, not after.

Defence is **provenance + integrity + minimisation**, not sandboxing. Nothing below is
claimed to be a complete solution.

### 13.2 Current exposure — three concrete gaps in this repo

1. **Nothing is pinned.** [requirements.txt](../requirements.txt) and
   [pyproject.toml](../pyproject.toml) use `>=` only, and
   [install.ps1:130](../install.ps1#L130) runs `pip install -e .`. Every fresh install
   resolves to whatever is newest at that moment. A compromised release of any of ~109
   transitive packages is adopted automatically, with no integrity check.
   **This is the single largest gap.**
2. **`Cnn14_DecisionLevelMax.pth` is an unverified Python pickle.** Downloaded from a
   hard-coded Zenodo URL at [model_cache.py:37](../src/monitor/model_cache.py#L37) via
   `urlopen`, with **no `hashlib` / SHA-256 check anywhere in the file**, then handed to
   `panns_inference` → `torch.load`. Pickle deserialisation is **arbitrary code
   execution**. Model weights are executable content and must be treated as such.
3. **`snapshot_download` is unpinned.** [stt.py:344](../src/monitor/stt.py#L344) has no
   `revision=` argument, so it tracks the HuggingFace repo's moving `main`. A hijacked
   account or a force-push silently changes what runs on the user's machine.

### 13.3 Actions

| # | Action | Addresses |
|---|---|---|
| 1 | **Hash-pinned lockfile.** `pip-compile --generate-hashes` → `requirements.lock`; install with `--require-hashes --only-binary=:all:`. Hashes stop tampered artifacts; `--only-binary` stops arbitrary `setup.py` execution at install time. Update [install.ps1](../install.ps1) to use it. | Gap 1 |
| 2 | **Single index.** `--index-url` only; **never** `--extra-index-url`. Blocks dependency-confusion, the most common real-world PyPI attack. | Gap 1 |
| 3 | **Pin model artifacts.** Embed the expected SHA-256 of the PANNs `.pth` and verify before first use; pass an explicit commit `revision=` to every `snapshot_download`; verify HF file hashes. Refuse to load on mismatch. | Gaps 2, 3 |
| 4 | **`weights_only=True`** wherever a `torch.load` call is reachable from our code; where it is inside `panns_inference`, the SHA-256 gate in #3 is the compensating control. | Gap 2 |
| 5 | **`pip-audit` in CI** (fails the build on new advisories) + Dependabot/GHAS on `ohadhawk/child-monitor-analyzer`. Catches known CVEs; catches nothing novel — necessary, not sufficient. | ongoing |
| 6 | **Reproducible release build** on a clean runner from the lockfile only — never from a developer's `site-packages`. **Authenticode-sign** `monitor-gui.exe` so users can detect tampering. | integrity |
| 7 | **CycloneDX SBOM** per release, so "are we affected by X?" is answerable in minutes. | response |
| 8 | **Minimise.** Keep OAuth on `urllib` + `keyring`. Rejecting `google-api-python-client` avoids ~15 further packages *in the process that holds the token*. | §7 |
| 9 | **Freeze the ML stack.** `torch`/`transformers`/`librosa` gain nothing from routine upgrades. Upgrade only in response to a reachable advisory. Fewer version changes = fewer chances to ingest a bad release. | ongoing |

### 13.4 Live vulnerability scan — 2026-08-03

109 installed packages queried against **OSV.dev**. 8 packages carry advisories.
Triaged by **reachability**, which is what determines actual risk:

| Package | Installed | Advisories | Reachable? | Action |
|---|---|---|---|---|
| **urllib3** | 2.6.3 | GHSA-qccp-gfcp-xxvc (HIGH — sensitive headers forwarded across origins on proxied redirects), GHSA-mf9v-mfxr-j63j (HIGH — decompression-bomb bypass) | **YES** — `huggingface_hub` uses it, and [install.ps1](../install.ps1) shows **proxy use is a supported scenario**. Header-leak-through-proxy is directly on our path. | **Upgrade to ≥ 2.7.0. Highest priority.** |
| **setuptools** | 70.2.0 | GHSA-5rjg-fvgr-3xxf (HIGH — path traversal → arbitrary file write in `PackageIndex.download`), GHSA-h35f-9h28-mq5c | Build/install-time only | **Upgrade to ≥ 83.0.0.** Cheap, no runtime risk. |
| **pip** | 25.3 | 4 advisories (path traversal in entry-point names; tar/ZIP confusion) | Install-time; mitigated further by action #1 | Upgrade to ≥ 26.1.2 |
| **msgpack** | 1.1.2 | GHSA-6v7p-g79w-8964 (HIGH — OOB read on `Unpacker` reuse) | Indirect; not fed untrusted data by us | Upgrade to ≥ 1.2.1 with the next lock |
| **idna** | 3.11 | GHSA-65pc-fj4g-8rjx (MODERATE) | Indirect via requests | Upgrade to ≥ 3.15 |
| **click** | 8.3.2 | PYSEC-2026-2132 | Not used by the GUI path | Upgrade to ≥ 8.3.3 |
| **torch** | 2.11.0+xpu | GHSA-rrmf-rvhw-rf47 (LOW — memory corruption via `torch.jit.script`) | **No** — we never call `torch.jit.script` | Accept. Do not churn the XPU build for a LOW. |
| **Pillow** | 12.2.0 | **27 advisories**, many HIGH (heap OOB write in `paste`/`crop`, `ImageCmsTransform`, `RankFilter`; command injection in `WindowsViewer`) | **No** — verified: **no `PIL` or `matplotlib` import anywhere in `src/monitor/`.** It is a transitive package that is bundled but never executed. | **Upgrade to ≥ 12.3.0 anyway** (free), and **drop `matplotlib` from the PyInstaller spec** if it is truly unused — 27 advisories of attack surface removed from the shipped binary. |

**Bottom line:** no critical *reachable* vulnerability today. `urllib3` is the one that
genuinely matters; `Pillow` looks alarming but is dead weight — and dead weight in a
shipped binary should be deleted, not patched.

### 13.5 Known compromise / breach history

Checked separately from CVEs — a compromise is a different failure mode.

- **No package in our tree has a recorded PyPI account takeover or malicious-release
  incident.**
- Relevant near-misses in our neighbourhood: the **`torchtriton` dependency-confusion
  attack (Dec 2022)** hit `pytorch-nightly` — we use stable releases, and action #2
  (single index) is the specific control against a repeat. The **`ultralytics` build
  compromise (Dec 2024)** hit a peer ML package via a poisoned CI workflow — the reason
  action #6 (clean-room reproducible build) exists.
- **The more realistic vector for this app is the model registry, not PyPI.** HuggingFace
  has repeatedly hosted malicious pickle-based models. We download from HF at runtime,
  unpinned (gap 3). This is our most plausible compromise path and actions #3/#4 target
  it directly.
- **PEP 740 provenance attestations: none of our key dependencies publish them yet** —
  verified against the PyPI API for `panns-inference`, `faster-whisper`, `librosa`,
  `python-docx`, `PySide6`, `torchlibrosa`, `huggingface-hub`, `ctranslate2`, `keyring`,
  `audioread`, `soundfile`, `numba`, `transformers`. Attestation checking is therefore a
  *future* control, not an available one. Hash pinning is what we have.

### 13.6 Packages to replace or reconsider

Ranked by **provenance risk**, not by CVE count:

| Package | Concern | Recommendation |
|---|---|---|
| **`panns-inference`** | **Weakest link.** Last release **0.1.1, 2023-03-26**; single academic maintainer; sub-1.0; low download volume; and it is the package that pulls an **unverified pickle** from Zenodo. An abandoned single-maintainer package is the classic account-takeover target. | **Do not replace now** — no comparable Hebrew-suitable SED model exists. Instead: SHA-256-pin the checkpoint (#3), vendor the ~200 lines of inference code we actually use to drop the dependency, or isolate audio-event detection into a separate process. **Re-evaluate before v1.3.** |
| **`torchlibrosa`** | Same maintainer, last release **2023-02-21**, version 0.1.0. Pulled in only by `panns-inference`. | Same treatment; disappears if `panns-inference` is vendored. |
| **`matplotlib` / `Pillow`** | Bundled in [monitor-gui.spec](../monitor-gui.spec) but **never imported by our code**. 27 Pillow advisories shipping to users for nothing. | **Remove from the spec** after confirming no transitive runtime import. Pure attack-surface reduction. |
| **`requests` + `urllib3`** | Present only because `huggingface_hub` needs them. Our own code correctly uses `urllib` already. | Keep (unavoidable), but **upgrade urllib3 now** and never route OAuth through them. |
| **`keyring`** (new) | Actively maintained (25.7.0, 2025-11-16), 203 releases, jaraco — a long-standing, high-reputation maintainer. On Windows it is a thin `CredWrite`/`CredRead` shim. | **Accept.** If even this is unwanted, the ~60-line `ctypes` alternative against `advapi32` is a viable fallback that removes the dependency entirely. |
| `faster-whisper`, `ctranslate2`, `PySide6`, `librosa`, `numba`, `soundfile`, `huggingface_hub`, `transformers`, `python-docx` | Active, recent releases, organisational or well-established maintainers. | **Keep.** No action beyond pinning. |

**Explicitly not recommended:** replacing the ML stack. `torch`/`transformers`/
`faster-whisper` have no meaningfully more trustworthy alternative, and swapping a
well-maintained package for an obscure one *increases* supply-chain risk.

### 13.7 Honest limits

Hash pinning defends against *later* compromise of a package you already trust. It does
**not** help if the version you pinned was already backdoored, and it does not stop
malicious code once it is running in the process. This is exactly why the `drive.file`
scope decision (§5.2) carries more weight than any scanner: it bounds the worst case to
files this app created, regardless of how the code got there.
