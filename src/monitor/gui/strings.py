"""
Centralised UI string table for internationalisation.

All user-facing text in the GUI is defined here.  Switching language
requires setting ``current_lang`` and restarting the application.

Usage:
    from monitor.gui.strings import tr, S

    label = tr(S.OPEN_FILE)
"""

from __future__ import annotations

import enum


class Lang(enum.Enum):
    HE = "he"
    EN = "en"


# Active language — changed via set_language(), read via tr().
current_lang: Lang = Lang.HE


def set_language(lang: Lang) -> None:
    global current_lang
    current_lang = lang


class S:
    """String keys — use ``tr(S.KEY)`` to get localised text."""

    # --- Main window ---
    WINDOW_TITLE = "window_title"
    OPEN_FILE = "open_file"
    RECENT = "recent"
    NO_FILE_SELECTED = "no_file_selected"
    ANALYSE = "analyse"
    DROP_HINT = "drop_hint"
    NO_RECENT_FILES = "no_recent_files"
    LOADED_CACHED = "loaded_cached"
    SELECT_AUDIO_FILE = "select_audio_file"
    FILE_NOT_FOUND = "file_not_found"
    FILE_NOT_FOUND_MSG = "file_not_found_msg"
    ERROR = "error"
    ANALYSIS_FAILED = "analysis_failed"
    TASK_STT = "task_stt"
    TASK_AUDIO_EVENTS = "task_audio_events"

    # --- Audio player ---
    PLAYER_BACK = "player_back"
    PLAYER_PLAY = "player_play"
    PLAYER_PAUSE = "player_pause"
    PLAYER_FORWARD = "player_forward"
    PLAYER_VOLUME = "player_volume"
    PLAYER_PREV_EVENT = "player_prev_event"
    PLAYER_NEXT_EVENT = "player_next_event"

    # --- Report table ---
    COL_TIME = "col_time"
    COL_TYPE = "col_type"
    COL_DETAILS = "col_details"
    COL_CONFIDENCE = "col_confidence"
    COL_PLAY = "col_play"
    FILTER_SEARCH = "filter_search"
    FILTER_SELECT_ALL = "filter_select_all"
    FILTER_CLEAR_ALL = "filter_clear_all"

    # --- Transcript ---
    TRANSCRIPT_TITLE = "transcript_title"
    TRANSCRIPT_EMPTY = "transcript_empty"
    TRANSCRIPT_SEARCH = "transcript_search"
    TRANSCRIPT_NO_MATCHES = "transcript_no_matches"
    TRANSCRIPT_DOWNLOAD = "transcript_download"
    TRANSCRIPT_DOWNLOAD_TITLE = "transcript_download_title"
    TRANSCRIPT_DOWNLOAD_EMPTY = "transcript_download_empty"
    EXPORT_DIALOG_TITLE = "export_dialog_title"
    EXPORT_FORMAT_LABEL = "export_format_label"
    EXPORT_FORMAT_TXT = "export_format_txt"
    EXPORT_FORMAT_DOCX = "export_format_docx"
    EXPORT_INCLUDE_TIMESTAMPS = "export_include_timestamps"
    EXPORT_INCLUDE_EVENTS = "export_include_events"
    EXPORT_OK = "export_ok"
    EXPORT_CANCEL = "export_cancel"
    EXPORT_SAVED = "export_saved"
    EXPORT_FAILED = "export_failed"
    EXPORT_NAME_THOROUGH = "export_name_thorough"
    EXPORT_NAME_FAST = "export_name_fast"

    # --- Google Drive ---
    GOOGLE_ACCOUNT_TITLE = "google_account_title"
    GOOGLE_SIGN_IN = "google_sign_in"
    GOOGLE_SIGN_OUT = "google_sign_out"
    GOOGLE_CONNECTED_AS = "google_connected_as"
    GOOGLE_NOT_CONNECTED = "google_not_connected"
    GOOGLE_SCOPE_EXPLANATION = "google_scope_explanation"
    GOOGLE_BROWSER_HINT = "google_browser_hint"
    GOOGLE_AUTH_FAILED = "google_auth_failed"
    GOOGLE_AUTH_TIMEOUT = "google_auth_timeout"
    GOOGLE_SESSION_EXPIRED = "google_session_expired"
    GOOGLE_KEYRING_UNAVAILABLE = "google_keyring_unavailable"
    GOOGLE_NOT_CONFIGURED = "google_not_configured"
    GOOGLE_SIGNING_IN = "google_signing_in"
    GOOGLE_AUTH_DENIED = "google_auth_denied"
    GOOGLE_MISSING_SCOPE = "google_missing_scope"
    GOOGLE_MENU_ACCOUNT = "google_menu_account"
    GOOGLE_MENU_UPLOAD_CURRENT = "google_menu_upload_current"
    GOOGLE_MENU_OPEN_FOLDER = "google_menu_open_folder"
    GOOGLE_MENU_ASK_EVERY_TIME = "google_menu_ask_every_time"
    TRANSCRIPT_UPLOAD_DRIVE = "transcript_upload_drive"
    UPLOAD_DIALOG_TITLE = "upload_dialog_title"
    UPLOAD_OK = "upload_ok"
    UPLOAD_TARGET_LABEL = "upload_target_label"
    UPLOAD_NAME_LABEL = "upload_name_label"
    UPLOAD_IN_PROGRESS = "upload_in_progress"
    UPLOAD_SUCCESS = "upload_success"
    UPLOAD_FAILED = "upload_failed"
    UPLOAD_RETRY = "upload_retry"
    UPLOAD_OPEN_IN_DOCS = "upload_open_in_docs"
    UPLOAD_EXISTING_TITLE = "upload_existing_title"
    UPLOAD_EXISTING_QUESTION = "upload_existing_question"
    UPLOAD_REPLACE_EXISTING = "upload_replace_existing"
    UPLOAD_CREATE_NEW = "upload_create_new"
    UPLOAD_NO_NETWORK = "upload_no_network"

    # --- Detection type labels ---
    DT_PROFANITY = "dt_profanity"
    DT_SHOUT = "dt_shout"
    DT_SCREAM = "dt_scream"
    DT_CRY = "dt_cry"
    DT_WAIL = "dt_wail"
    DT_BABY_CRY = "dt_baby_cry"
    DT_LAUGHTER = "dt_laughter"
    DT_VOLUME_SPIKE = "dt_volume_spike"

    # --- Pipeline progress ---
    PIPE_LOADED_CACHE = "pipe_loaded_cache"
    PIPE_STARTING = "pipe_starting"
    PIPE_LOADING_STT = "pipe_loading_stt"
    PIPE_STT_LOADED = "pipe_stt_loaded"
    PIPE_ALL_MODELS_LOADED = "pipe_all_models_loaded"
    PIPE_PARALLEL = "pipe_parallel"
    PIPE_PROFANITY_SEARCH = "pipe_profanity_search"
    PIPE_DONE = "pipe_done"
    PIPE_STT_START = "pipe_stt_start"
    PIPE_EVENTS_START = "pipe_events_start"

    # --- STT progress ---
    STT_TRANSCRIBING = "stt_transcribing"
    STT_GAP_FILL = "stt_gap_fill"
    STT_STARTING = "stt_starting"
    STT_DOWNLOADING = "stt_downloading"

    # --- Toxicity model progress ---
    TOXICITY_DOWNLOADING = "toxicity_downloading"

    # --- Audio events progress ---
    AE_LOADING_AUDIO = "ae_loading_audio"
    AE_CHECKING_VOLUME = "ae_checking_volume"
    AE_ANALYSING_PANNS = "ae_analysing_panns"
    AE_LOADING_MODEL = "ae_loading_model"

    # --- Partial analysis warning ---
    PARTIAL_WARNING = "partial_warning"
    AI_PROFANITY_UNAVAILABLE = "ai_profanity_unavailable"

    # --- Sensitivity panel ---
    SENSITIVITY_TITLE = "sensitivity_title"
    SENSITIVITY_LOW = "sensitivity_low"
    SENSITIVITY_HIGH = "sensitivity_high"

    # --- New-model check ---
    CHECK_MODELS = "check_models"
    CHECK_MODELS_TOOLTIP = "check_models_tooltip"
    CHECK_MODELS_TITLE = "check_models_title"
    CHECK_MODELS_CHECKING = "check_models_checking"
    CHECK_MODELS_NONE = "check_models_none"
    CHECK_MODELS_FOUND = "check_models_found"
    CHECK_MODELS_FAILED = "check_models_failed"


# fmt: off
_STRINGS = {
    # --- Main window ---
    (S.WINDOW_TITLE, Lang.HE):       "מנתח ניטור ילדים",
    (S.WINDOW_TITLE, Lang.EN):       "Child Monitor Analyzer",
    (S.OPEN_FILE, Lang.HE):          "פתח קובץ שמע",
    (S.OPEN_FILE, Lang.EN):          "Open Audio File",
    (S.RECENT, Lang.HE):             "אחרונים",
    (S.RECENT, Lang.EN):             "Recent",
    (S.NO_FILE_SELECTED, Lang.HE):   "לא נבחר קובץ",
    (S.NO_FILE_SELECTED, Lang.EN):   "No file selected",
    (S.ANALYSE, Lang.HE):            "נתח",
    (S.ANALYSE, Lang.EN):            "Analyse",
    (S.DROP_HINT, Lang.HE):          "גרור קובץ שמע לכאן, או לחץ על כפתור הפתיחה",
    (S.DROP_HINT, Lang.EN):          "Drag and drop an audio file here, or use the Open button",
    (S.NO_RECENT_FILES, Lang.HE):    "(אין קבצים אחרונים)",
    (S.NO_RECENT_FILES, Lang.EN):    "(no recent files)",
    (S.LOADED_CACHED, Lang.HE):      "נטען מניתוח קודם",
    (S.LOADED_CACHED, Lang.EN):      "Loaded cached analysis",
    (S.SELECT_AUDIO_FILE, Lang.HE):  "בחר קובץ שמע",
    (S.SELECT_AUDIO_FILE, Lang.EN):  "Select Audio File",
    (S.FILE_NOT_FOUND, Lang.HE):     "קובץ לא נמצא",
    (S.FILE_NOT_FOUND, Lang.EN):     "File not found",
    (S.FILE_NOT_FOUND_MSG, Lang.HE): "קובץ לא נמצא:\n{path}",
    (S.FILE_NOT_FOUND_MSG, Lang.EN): "File not found:\n{path}",
    (S.ERROR, Lang.HE):              "שגיאה",
    (S.ERROR, Lang.EN):              "Error",
    (S.ANALYSIS_FAILED, Lang.HE):    "הניתוח נכשל:\n{msg}",
    (S.ANALYSIS_FAILED, Lang.EN):    "Analysis failed:\n{msg}",
    (S.TASK_STT, Lang.HE):           "תמלול",
    (S.TASK_STT, Lang.EN):           "STT",
    (S.TASK_AUDIO_EVENTS, Lang.HE):  "אירועי שמע",
    (S.TASK_AUDIO_EVENTS, Lang.EN):  "Audio Events",

    # --- Audio player ---
    (S.PLAYER_BACK, Lang.HE):        "⏪",
    (S.PLAYER_BACK, Lang.EN):        "⏪",
    (S.PLAYER_PLAY, Lang.HE):        "▶",
    (S.PLAYER_PLAY, Lang.EN):        "▶",
    (S.PLAYER_PAUSE, Lang.HE):       "⏸",
    (S.PLAYER_PAUSE, Lang.EN):       "⏸",
    (S.PLAYER_FORWARD, Lang.HE):     "⏩",
    (S.PLAYER_FORWARD, Lang.EN):     "⏩",
    (S.PLAYER_VOLUME, Lang.HE):      "🔈",
    (S.PLAYER_VOLUME, Lang.EN):      "🔈",
    (S.PLAYER_PREV_EVENT, Lang.HE):  "אירוע קודם \u25C0",
    (S.PLAYER_PREV_EVENT, Lang.EN):  "Previous event \u25C0",
    (S.PLAYER_NEXT_EVENT, Lang.HE):  "\u25B6 אירוע הבא",
    (S.PLAYER_NEXT_EVENT, Lang.EN):  "\u25B6 Next event",

    # --- Report table ---
    (S.COL_TIME, Lang.HE):           "זמן",
    (S.COL_TIME, Lang.EN):           "Time",
    (S.COL_TYPE, Lang.HE):           "סוג",
    (S.COL_TYPE, Lang.EN):           "Type",
    (S.COL_DETAILS, Lang.HE):        "פרטים",
    (S.COL_DETAILS, Lang.EN):        "Details",
    (S.COL_CONFIDENCE, Lang.HE):     "ביטחון",
    (S.COL_CONFIDENCE, Lang.EN):     "Confidence",
    (S.COL_PLAY, Lang.HE):           "נגן",
    (S.COL_PLAY, Lang.EN):           "Play",
    (S.FILTER_SEARCH, Lang.HE):      "חיפוש...",
    (S.FILTER_SEARCH, Lang.EN):      "Search...",
    (S.FILTER_SELECT_ALL, Lang.HE):  "בחר הכל",
    (S.FILTER_SELECT_ALL, Lang.EN):  "Select all",
    (S.FILTER_CLEAR_ALL, Lang.HE):   "נקה בחירה",
    (S.FILTER_CLEAR_ALL, Lang.EN):   "Clear all",

    # --- Transcript ---
    (S.TRANSCRIPT_TITLE, Lang.HE):   "תמליל",
    (S.TRANSCRIPT_TITLE, Lang.EN):   "Transcript",
    (S.TRANSCRIPT_EMPTY, Lang.HE):   "אין תמליל זמין",
    (S.TRANSCRIPT_EMPTY, Lang.EN):   "No transcript available",
    (S.TRANSCRIPT_SEARCH, Lang.HE):  "חיפוש בתמליל...",
    (S.TRANSCRIPT_SEARCH, Lang.EN):  "Search transcript...",
    (S.TRANSCRIPT_NO_MATCHES, Lang.HE): "0/0",
    (S.TRANSCRIPT_NO_MATCHES, Lang.EN): "0/0",
    (S.TRANSCRIPT_DOWNLOAD, Lang.HE): "הורדת התמליל",
    (S.TRANSCRIPT_DOWNLOAD, Lang.EN): "Download transcript",
    (S.TRANSCRIPT_DOWNLOAD_TITLE, Lang.HE): "שמירת תמליל",
    (S.TRANSCRIPT_DOWNLOAD_TITLE, Lang.EN): "Save transcript",
    (S.TRANSCRIPT_DOWNLOAD_EMPTY, Lang.HE): "אין תמליל לשמירה.",
    (S.TRANSCRIPT_DOWNLOAD_EMPTY, Lang.EN): "No transcript to save.",
    (S.EXPORT_DIALOG_TITLE, Lang.HE): "הורדת תמליל",
    (S.EXPORT_DIALOG_TITLE, Lang.EN): "Download transcript",
    (S.EXPORT_FORMAT_LABEL, Lang.HE): "פורמט:",
    (S.EXPORT_FORMAT_LABEL, Lang.EN): "Format:",
    (S.EXPORT_FORMAT_TXT, Lang.HE): "טקסט (‎.txt)",
    (S.EXPORT_FORMAT_TXT, Lang.EN): "Text (.txt)",
    (S.EXPORT_FORMAT_DOCX, Lang.HE): "מסמך Word ‏(‎.docx)",
    (S.EXPORT_FORMAT_DOCX, Lang.EN): "Word document (.docx)",
    (S.EXPORT_INCLUDE_TIMESTAMPS, Lang.HE): "כלול חותמות זמן",
    (S.EXPORT_INCLUDE_TIMESTAMPS, Lang.EN): "Include timestamps",
    (S.EXPORT_INCLUDE_EVENTS, Lang.HE): "כלול אירועי שמע",
    (S.EXPORT_INCLUDE_EVENTS, Lang.EN): "Include audio events",
    (S.EXPORT_OK, Lang.HE): "שמירה",
    (S.EXPORT_OK, Lang.EN): "Save",
    (S.EXPORT_CANCEL, Lang.HE): "ביטול",
    (S.EXPORT_CANCEL, Lang.EN): "Cancel",
    (S.EXPORT_SAVED, Lang.HE): "התמליל נשמר בהצלחה.",
    (S.EXPORT_SAVED, Lang.EN): "Transcript saved successfully.",
    (S.EXPORT_FAILED, Lang.HE): "שמירת התמליל נכשלה:",
    (S.EXPORT_FAILED, Lang.EN): "Failed to save transcript:",
    (S.EXPORT_NAME_THOROUGH, Lang.HE): "תמלול יסודי",
    (S.EXPORT_NAME_THOROUGH, Lang.EN): "תמלול יסודי",
    (S.EXPORT_NAME_FAST, Lang.HE): "תמלול מהיר",
    (S.EXPORT_NAME_FAST, Lang.EN): "תמלול מהיר",

    # --- Google Drive ---
    (S.GOOGLE_ACCOUNT_TITLE, Lang.HE): "חשבון Google",
    (S.GOOGLE_ACCOUNT_TITLE, Lang.EN): "Google account",
    (S.GOOGLE_SIGN_IN, Lang.HE): "התחבר עם Google",
    (S.GOOGLE_SIGN_IN, Lang.EN): "Sign in with Google",
    (S.GOOGLE_SIGN_OUT, Lang.HE): "נתק חשבון",
    (S.GOOGLE_SIGN_OUT, Lang.EN): "Disconnect account",
    (S.GOOGLE_CONNECTED_AS, Lang.HE): "מחובר כ־",
    (S.GOOGLE_CONNECTED_AS, Lang.EN): "Signed in as ",
    (S.GOOGLE_NOT_CONNECTED, Lang.HE): "לא מחובר ל‑Google Drive",
    (S.GOOGLE_NOT_CONNECTED, Lang.EN): "Not connected to Google Drive",
    (S.GOOGLE_SCOPE_EXPLANATION, Lang.HE):
        "האפליקציה תוכל ליצור ולערוך רק קבצים שהיא עצמה יצרה. "
        "אין לה גישה לשאר הקבצים בדרייב שלך.",
    (S.GOOGLE_SCOPE_EXPLANATION, Lang.EN):
        "The app can create and edit only the files it creates itself. "
        "It has no access to the rest of your Drive.",
    (S.GOOGLE_BROWSER_HINT, Lang.HE):
        "ייפתח דפדפן לאישור ההתחברות. הסיסמה שלך אף פעם לא עוברת דרך האפליקציה.",
    (S.GOOGLE_BROWSER_HINT, Lang.EN):
        "A browser window will open for consent. Your password never passes "
        "through this app.",
    (S.GOOGLE_AUTH_FAILED, Lang.HE): "ההתחברות ל‑Google נכשלה:",
    (S.GOOGLE_AUTH_FAILED, Lang.EN): "Google sign-in failed:",
    (S.GOOGLE_AUTH_TIMEOUT, Lang.HE): "ההתחברות ל‑Google לא הושלמה בזמן.",
    (S.GOOGLE_AUTH_TIMEOUT, Lang.EN): "Google sign-in was not completed in time.",
    (S.GOOGLE_SESSION_EXPIRED, Lang.HE):
        "ההרשאה ל‑Google פגה. יש להתחבר מחדש.",
    (S.GOOGLE_SESSION_EXPIRED, Lang.EN):
        "The Google authorization expired. Please sign in again.",
    (S.GOOGLE_KEYRING_UNAVAILABLE, Lang.HE):
        "לא ניתן לשמור את פרטי ההתחברות במאגר האישורים של Windows, "
        "ולכן ההעלאה ל‑Google Drive מושבתת.",
    (S.GOOGLE_KEYRING_UNAVAILABLE, Lang.EN):
        "Credentials cannot be stored in Windows Credential Manager, so "
        "Google Drive upload is disabled.",
    (S.GOOGLE_NOT_CONFIGURED, Lang.HE):
        "ההעלאה ל‑Google Drive אינה מוגדרת בגרסה הזו.",
    (S.GOOGLE_NOT_CONFIGURED, Lang.EN):
        "Google Drive upload is not configured in this build.",
    (S.GOOGLE_SIGNING_IN, Lang.HE): "מתחבר ל‑Google…",
    (S.GOOGLE_SIGNING_IN, Lang.EN): "Signing in to Google…",
    (S.GOOGLE_AUTH_DENIED, Lang.HE): "לא ניתן אישור גישה ל‑Google Drive.",
    (S.GOOGLE_AUTH_DENIED, Lang.EN): "Access to Google Drive was declined.",
    (S.GOOGLE_MISSING_SCOPE, Lang.HE):
        "החיבור הצליח אך לא ניתנה גישה ל‑Drive. יש להוסיף את הרשאת ה‑Drive במסך ההסכמה ולהתחבר שוב.",
    (S.GOOGLE_MISSING_SCOPE, Lang.EN):
        "Signed in, but Drive access was not granted. Add the Drive scope on the "
        "OAuth consent screen, then sign in again.",
    (S.GOOGLE_MENU_ACCOUNT, Lang.HE): "חשבון Google…",
    (S.GOOGLE_MENU_ACCOUNT, Lang.EN): "Google account…",
    (S.GOOGLE_MENU_UPLOAD_CURRENT, Lang.HE): "העלה את התמליל הנוכחי…",
    (S.GOOGLE_MENU_UPLOAD_CURRENT, Lang.EN): "Upload the current transcript…",
    (S.GOOGLE_MENU_OPEN_FOLDER, Lang.HE): "פתח את תיקיית Transcriptions בדרייב",
    (S.GOOGLE_MENU_OPEN_FOLDER, Lang.EN): "Open the Transcriptions folder in Drive",
    (S.GOOGLE_MENU_ASK_EVERY_TIME, Lang.HE): "שאל לפני כל העלאה",
    (S.GOOGLE_MENU_ASK_EVERY_TIME, Lang.EN): "Ask before every upload",
    (S.TRANSCRIPT_UPLOAD_DRIVE, Lang.HE): "העלאת התמליל ל‑Google Drive",
    (S.TRANSCRIPT_UPLOAD_DRIVE, Lang.EN): "Upload the transcript to Google Drive",
    (S.UPLOAD_DIALOG_TITLE, Lang.HE): "העלאה ל‑Google Drive",
    (S.UPLOAD_DIALOG_TITLE, Lang.EN): "Upload to Google Drive",
    (S.UPLOAD_OK, Lang.HE): "העלה",
    (S.UPLOAD_OK, Lang.EN): "Upload",
    (S.UPLOAD_NAME_LABEL, Lang.HE): "שם הקובץ:",
    (S.UPLOAD_NAME_LABEL, Lang.EN): "File name:",
    (S.UPLOAD_TARGET_LABEL, Lang.HE): "יעד:",
    (S.UPLOAD_TARGET_LABEL, Lang.EN): "Destination:",
    (S.UPLOAD_IN_PROGRESS, Lang.HE): "מעלה את התמליל ל‑Google Drive…",
    (S.UPLOAD_IN_PROGRESS, Lang.EN): "Uploading the transcript to Google Drive…",
    (S.UPLOAD_SUCCESS, Lang.HE): "התמליל הועלה ל‑Google Drive.",
    (S.UPLOAD_SUCCESS, Lang.EN): "The transcript was uploaded to Google Drive.",
    (S.UPLOAD_FAILED, Lang.HE): "ההעלאה ל‑Google Drive נכשלה:",
    (S.UPLOAD_FAILED, Lang.EN): "Upload to Google Drive failed:",
    (S.UPLOAD_RETRY, Lang.HE): "נסה שוב",
    (S.UPLOAD_RETRY, Lang.EN): "Retry",
    (S.UPLOAD_OPEN_IN_DOCS, Lang.HE): "פתח ב‑Google Docs",
    (S.UPLOAD_OPEN_IN_DOCS, Lang.EN): "Open in Google Docs",
    (S.UPLOAD_EXISTING_TITLE, Lang.HE): "התמליל כבר הועלה",
    (S.UPLOAD_EXISTING_TITLE, Lang.EN): "Already uploaded",
    (S.UPLOAD_EXISTING_QUESTION, Lang.HE):
        "התמליל הזה כבר הועלה ל‑Google Drive. מה לעשות?",
    (S.UPLOAD_EXISTING_QUESTION, Lang.EN):
        "This transcript was already uploaded to Google Drive. What now?",
    (S.UPLOAD_REPLACE_EXISTING, Lang.HE): "עדכן את המסמך הקיים",
    (S.UPLOAD_REPLACE_EXISTING, Lang.EN): "Update the existing document",
    (S.UPLOAD_CREATE_NEW, Lang.HE): "צור מסמך חדש",
    (S.UPLOAD_CREATE_NEW, Lang.EN): "Create a new document",
    (S.UPLOAD_NO_NETWORK, Lang.HE): "אין חיבור לאינטרנט.",
    (S.UPLOAD_NO_NETWORK, Lang.EN): "No internet connection.",

    # --- Detection type labels ---
    (S.DT_PROFANITY, Lang.HE):       "ניבול פה",
    (S.DT_PROFANITY, Lang.EN):       "Profanity",
    (S.DT_SHOUT, Lang.HE):           "צעקה",
    (S.DT_SHOUT, Lang.EN):           "Shout",
    (S.DT_SCREAM, Lang.HE):          "צרחה",
    (S.DT_SCREAM, Lang.EN):          "Scream",
    (S.DT_CRY, Lang.HE):             "בכי",
    (S.DT_CRY, Lang.EN):             "Cry",
    (S.DT_WAIL, Lang.HE):            "יללה",
    (S.DT_WAIL, Lang.EN):            "Wail",
    (S.DT_BABY_CRY, Lang.HE):        "בכי תינוק",
    (S.DT_BABY_CRY, Lang.EN):        "Baby cry",
    (S.DT_LAUGHTER, Lang.HE):        "צחוק",
    (S.DT_LAUGHTER, Lang.EN):        "Laughter",
    (S.DT_VOLUME_SPIKE, Lang.HE):    "עוצמה חריגה",
    (S.DT_VOLUME_SPIKE, Lang.EN):    "Volume spike",

    # --- Pipeline progress ---
    (S.PIPE_LOADED_CACHE, Lang.HE):       "נטען מקובץ מטמון!",
    (S.PIPE_LOADED_CACHE, Lang.EN):       "Loaded from cache!",
    (S.PIPE_STARTING, Lang.HE):           "מתחיל ניתוח...",
    (S.PIPE_STARTING, Lang.EN):           "Starting analysis...",
    (S.PIPE_LOADING_STT, Lang.HE):        "טוען מודל זיהוי דיבור...",
    (S.PIPE_LOADING_STT, Lang.EN):        "Loading speech recognition model...",
    (S.PIPE_STT_LOADED, Lang.HE):         "מודל דיבור נטען. טוען מודל אירועי שמע...",
    (S.PIPE_STT_LOADED, Lang.EN):         "STT model loaded. Loading audio events model...",
    (S.PIPE_ALL_MODELS_LOADED, Lang.HE):  "כל המודלים נטענו. מריץ ניתוח...",
    (S.PIPE_ALL_MODELS_LOADED, Lang.EN):  "All models loaded. Running analysis...",
    (S.PIPE_PARALLEL, Lang.HE):           "מריץ זיהוי דיבור ואירועי שמע במקביל...",
    (S.PIPE_PARALLEL, Lang.EN):           "Running speech & audio event detection in parallel...",
    (S.PIPE_PROFANITY_SEARCH, Lang.HE):   "מחפש ניבול פה...",
    (S.PIPE_PROFANITY_SEARCH, Lang.EN):   "Searching for profanity...",
    (S.PIPE_DONE, Lang.HE):               "הניתוח הושלם!",
    (S.PIPE_DONE, Lang.EN):               "Analysis complete!",
    (S.PIPE_STT_START, Lang.HE):          "סורק את ההקלטה לזיהוי קטעי דיבור...",
    (S.PIPE_STT_START, Lang.EN):          "Scanning audio for speech segments...",
    (S.PIPE_EVENTS_START, Lang.HE):       "מתחיל זיהוי אירועים...",
    (S.PIPE_EVENTS_START, Lang.EN):       "Starting event detection...",

    # --- STT progress ---
    (S.STT_TRANSCRIBING, Lang.HE):   "מתמלל...",
    (S.STT_TRANSCRIBING, Lang.EN):   "Transcribing...",
    (S.STT_GAP_FILL, Lang.HE):       "משלים פערים...",
    (S.STT_GAP_FILL, Lang.EN):       "Filling gaps...",
    (S.STT_STARTING, Lang.HE):       "מתחיל תמלול...",
    (S.STT_STARTING, Lang.EN):       "Starting transcription...",
    (S.STT_DOWNLOADING, Lang.HE):    "מוריד מודל תמלול (~1.5 GB)...",
    (S.STT_DOWNLOADING, Lang.EN):    "Downloading STT model (~1.5 GB)...",
    (S.TOXICITY_DOWNLOADING, Lang.HE): "מוריד מודל זיהוי ניבול פה (~700 MB)...",
    (S.TOXICITY_DOWNLOADING, Lang.EN): "Downloading toxicity model (~700 MB)...",

    # --- Audio events progress ---
    (S.AE_LOADING_AUDIO, Lang.HE):   "טוען שמע...",
    (S.AE_LOADING_AUDIO, Lang.EN):   "Loading audio...",
    (S.AE_CHECKING_VOLUME, Lang.HE): "בודק עוצמת קול...",
    (S.AE_CHECKING_VOLUME, Lang.EN): "Checking volume levels...",
    (S.AE_ANALYSING_PANNS, Lang.HE): "מנתח תוצאות PANNs...",
    (S.AE_ANALYSING_PANNS, Lang.EN): "Analysing PANNs results...",
    (S.AE_LOADING_MODEL, Lang.HE):   "טוען מודל PANNs...",
    (S.AE_LOADING_MODEL, Lang.EN):   "Loading PANNs model into memory...",

    # --- Sensitivity panel ---
    # --- Partial analysis warning ---
    (S.PARTIAL_WARNING, Lang.HE):    "⚠ ניתוח בתהליך — התוצאות המוצגות חלקיות",
    (S.PARTIAL_WARNING, Lang.EN):    "⚠ Analysis in progress — results shown are partial",
    (S.AI_PROFANITY_UNAVAILABLE, Lang.HE): "⚠ מודל AI לזיהוי ניבול פה לא זמין — זיהוי מילים בלבד",
    (S.AI_PROFANITY_UNAVAILABLE, Lang.EN): "⚠ AI profanity model unavailable — word-list detection only",

    (S.SENSITIVITY_TITLE, Lang.HE):  "⚙ רגישות",
    (S.SENSITIVITY_TITLE, Lang.EN):  "⚙ Sensitivity",
    (S.SENSITIVITY_LOW, Lang.HE):    "לא רגיש",
    (S.SENSITIVITY_LOW, Lang.EN):    "Low",
    (S.SENSITIVITY_HIGH, Lang.HE):   "רגיש מאוד",
    (S.SENSITIVITY_HIGH, Lang.EN):   "Very sensitive",

    # --- New-model check ---
    (S.CHECK_MODELS, Lang.HE):           "\u27F3 מודלים",
    (S.CHECK_MODELS, Lang.EN):           "\u27F3 Models",
    (S.CHECK_MODELS_TOOLTIP, Lang.HE):   "בדוק אם קיימים מודלי תמלול עבריים חדשים",
    (S.CHECK_MODELS_TOOLTIP, Lang.EN):   "Check for new Hebrew transcription models",
    (S.CHECK_MODELS_TITLE, Lang.HE):     "בדיקת מודלים חדשים",
    (S.CHECK_MODELS_TITLE, Lang.EN):     "Check for new models",
    (S.CHECK_MODELS_CHECKING, Lang.HE):  "בודק מודלים חדשים...",
    (S.CHECK_MODELS_CHECKING, Lang.EN):  "Checking for new models...",
    (S.CHECK_MODELS_NONE, Lang.HE):      "לא נמצאו מודלי תמלול עבריים חדשים.",
    (S.CHECK_MODELS_NONE, Lang.EN):      "No new Hebrew transcription models found.",
    (S.CHECK_MODELS_FOUND, Lang.HE):     "נמצאו מודלים עבריים חדשים:\n\n{list}",
    (S.CHECK_MODELS_FOUND, Lang.EN):     "New Hebrew models found:\n\n{list}",
    (S.CHECK_MODELS_FAILED, Lang.HE):    "בדיקת המודלים נכשלה:\n{msg}",
    (S.CHECK_MODELS_FAILED, Lang.EN):    "Model check failed:\n{msg}",
}
# fmt: on


def tr(key: str) -> str:
    """Return the localised string for *key* in the current language."""
    return _STRINGS.get((key, current_lang), key)
