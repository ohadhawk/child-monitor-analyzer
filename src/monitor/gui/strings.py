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
}
# fmt: on


def tr(key: str) -> str:
    """Return the localised string for *key* in the current language."""
    return _STRINGS.get((key, current_lang), key)
