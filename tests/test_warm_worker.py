"""
Regression tests for the warm analysis worker.

These lock in the behaviour that motivated the persistent worker: models must
be loaded once per session rather than once per file, and stopping an analysis
must no longer rely on killing the process.
"""

from __future__ import annotations

import multiprocessing
import queue as _queue_mod
import threading
import time

import pytest

from monitor import cancellation
from monitor.analysis_worker import (
    JOB_ANALYZE,
    JOB_SHUTDOWN,
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_FINISHED,
    MSG_READY,
    ModelPool,
    _run_job,
    _watch_cancellation,
    run_worker,
)


# ======================================================================
# Cancellation token
# ======================================================================

@pytest.fixture(autouse=True)
def _reset_cancellation():
    cancellation.begin_job()
    yield
    cancellation.begin_job()


def test_token_starts_clear():
    assert cancellation.is_cancelled() is False
    cancellation.raise_if_cancelled()  # must not raise


def test_request_cancel_raises_at_checkpoint():
    cancellation.request_cancel()
    assert cancellation.is_cancelled() is True
    with pytest.raises(cancellation.AnalysisCancelled):
        cancellation.raise_if_cancelled()


def test_begin_job_clears_previous_cancel():
    cancellation.request_cancel()
    cancellation.begin_job()
    assert cancellation.is_cancelled() is False
    cancellation.raise_if_cancelled()


def test_cancellation_is_visible_across_threads():
    """The pipeline cancels from a watchdog thread while work runs elsewhere."""
    seen = []

    def worker() -> None:
        for _ in range(200):
            try:
                cancellation.raise_if_cancelled()
            except cancellation.AnalysisCancelled:
                seen.append("cancelled")
                return
            time.sleep(0.005)
        seen.append("finished")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    cancellation.request_cancel()
    t.join(timeout=5)
    assert seen == ["cancelled"]


# ======================================================================
# Cancellation checkpoints are wired into the long-running loops
# ======================================================================

@pytest.mark.parametrize(
    "module_name",
    ["monitor.pipeline", "monitor.stt", "monitor.audio_events"],
)
def test_long_running_modules_have_cancellation_checkpoints(module_name):
    """Without these, a cancelled job would run to completion."""
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    assert "raise_if_cancelled" in source, (
        f"{module_name} has no cancellation checkpoint; a stopped analysis "
        f"would keep the worker busy until it finishes."
    )


# ======================================================================
# Watchdog: cancel counter -> cancellation token
# ======================================================================

def test_watchdog_cancels_the_running_job():
    cancel_upto = multiprocessing.Value("q", 0)
    state = {"job_id": 7}
    stop = threading.Event()
    t = threading.Thread(
        target=_watch_cancellation, args=(cancel_upto, state, stop), daemon=True,
    )
    t.start()
    try:
        cancel_upto.value = 7
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not cancellation.is_cancelled():
            time.sleep(0.02)
        assert cancellation.is_cancelled() is True
    finally:
        stop.set()
        t.join(timeout=2)


def test_watchdog_never_cancels_a_newer_job():
    """A cancel for job 3 must not abort job 4, which was queued afterwards."""
    cancel_upto = multiprocessing.Value("q", 3)
    state = {"job_id": 4}
    stop = threading.Event()
    t = threading.Thread(
        target=_watch_cancellation, args=(cancel_upto, state, stop), daemon=True,
    )
    t.start()
    try:
        time.sleep(0.5)
        assert cancellation.is_cancelled() is False
    finally:
        stop.set()
        t.join(timeout=2)


def test_watchdog_ignores_idle_worker():
    cancel_upto = multiprocessing.Value("q", 99)
    state = {"job_id": None}
    stop = threading.Event()
    t = threading.Thread(
        target=_watch_cancellation, args=(cancel_upto, state, stop), daemon=True,
    )
    t.start()
    try:
        time.sleep(0.5)
        assert cancellation.is_cancelled() is False
    finally:
        stop.set()
        t.join(timeout=2)


# ======================================================================
# ModelPool: the actual fix
# ======================================================================

class _FakeSTT:
    instances = 0

    def __init__(self, model_name: str) -> None:
        type(self).instances += 1
        self.model_name = model_name


class _FakeDetector:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1


class _FakeProfanity:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1


class _FakePipeline:
    def __init__(self, *, stt, audio_events, profanity) -> None:
        self.stt = stt
        self.audio_events = audio_events
        self.profanity = profanity


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the three heavyweight model classes with counting fakes."""
    import monitor.audio_events as audio_events_mod
    import monitor.pipeline as pipeline_mod
    import monitor.profanity as profanity_mod
    import monitor.stt as stt_mod

    _FakeSTT.instances = 0
    _FakeDetector.instances = 0
    _FakeProfanity.instances = 0

    monkeypatch.setattr(stt_mod, "HebrewSTT", _FakeSTT)
    monkeypatch.setattr(audio_events_mod, "AudioEventDetector", _FakeDetector)
    monkeypatch.setattr(profanity_mod, "ProfanityDetector", _FakeProfanity)
    monkeypatch.setattr(pipeline_mod, "AnalysisPipeline", _FakePipeline)
    return {"stt": _FakeSTT, "events": _FakeDetector, "profanity": _FakeProfanity}


def test_pool_loads_models_once_for_repeated_jobs(fake_models):
    """This is the regression: file 2 must not pay the model load cost again."""
    pool = ModelPool()
    first = pool.pipeline_for("thorough")
    second = pool.pipeline_for("thorough")

    assert fake_models["stt"].instances == 1
    assert fake_models["events"].instances == 1
    assert fake_models["profanity"].instances == 1
    assert first.stt is second.stt
    assert first.audio_events is second.audio_events
    assert first.profanity is second.profanity


def test_pool_reuses_detectors_when_stt_model_changes(fake_models):
    """Switching fast/thorough must not throw away the PANNs + toxicity models."""
    pool = ModelPool()
    thorough = pool.pipeline_for("thorough")
    fast = pool.pipeline_for("fast")

    assert fake_models["stt"].instances == 2
    assert thorough.stt is not fast.stt
    assert fake_models["events"].instances == 1
    assert fake_models["profanity"].instances == 1
    assert thorough.audio_events is fast.audio_events
    assert thorough.profanity is fast.profanity


def test_pool_uses_the_expected_model_for_each_key(fake_models):
    from monitor.stt import DEFAULT_MODEL, TURBO_MODEL

    pool = ModelPool()
    assert pool.pipeline_for("thorough").stt.model_name == DEFAULT_MODEL
    assert pool.pipeline_for("fast").stt.model_name == TURBO_MODEL
    assert pool.pipeline_for("none").stt is None


def test_pool_keeps_only_one_stt_engine_resident(fake_models):
    """Holding both engines would cost ~1.5 GB of extra RAM."""
    pool = ModelPool()
    pool.pipeline_for("thorough")
    first = pool._stt
    pool.pipeline_for("fast")
    assert pool._stt is not first
    assert pool._stt_model_name.endswith("turbo-ct2")


def test_pool_returns_a_fresh_pipeline_each_time(fake_models):
    """AnalysisPipeline holds no per-file state; a new one avoids stale state."""
    pool = ModelPool()
    assert pool.pipeline_for("thorough") is not pool.pipeline_for("thorough")


# ======================================================================
# _run_job message protocol
# ======================================================================

class _StubPool:
    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline
        self.calls = 0

    def pipeline_for(self, stt_model_key):
        self.calls += 1
        return self._pipeline


class _StubPipeline:
    def __init__(self, report=None, exc=None) -> None:
        self._report = report
        self._exc = exc
        self.analyzed: list = []

    def analyze(self, audio_path, **kwargs):
        self.analyzed.append(audio_path)
        if self._exc is not None:
            raise self._exc
        kwargs["on_progress"](50, "half way")
        return self._report


class _StubReport:
    def to_dict(self):
        return {"detections": []}


def _drain(queue) -> list:
    out = []
    while True:
        try:
            out.append(queue.get(timeout=1))
        except _queue_mod.Empty:
            return out


def test_run_job_tags_every_message_with_the_job_id():
    q = multiprocessing.Queue()
    pool = _StubPool(_StubPipeline(report=_StubReport()))
    cancel_upto = multiprocessing.Value("q", 0)

    _run_job(
        {"type": JOB_ANALYZE, "job_id": 42, "audio_path": "a.wav",
         "stt_model_key": "thorough"},
        q, pool, cancel_upto, {"job_id": None},
    )
    messages = _drain(q)
    assert messages, "worker produced no messages"
    assert all(m["job_id"] == 42 for m in messages)
    assert messages[-1]["type"] == MSG_FINISHED


def test_run_job_reports_failures_without_killing_the_worker():
    q = multiprocessing.Queue()
    pool = _StubPool(_StubPipeline(exc=RuntimeError("boom")))
    cancel_upto = multiprocessing.Value("q", 0)
    state = {"job_id": None}

    _run_job(
        {"type": JOB_ANALYZE, "job_id": 1, "audio_path": "a.wav"},
        q, pool, cancel_upto, state,
    )
    messages = _drain(q)
    assert messages[-1]["type"] == MSG_ERROR
    assert "boom" in messages[-1]["msg"]
    assert "RuntimeError" in messages[-1]["traceback"]
    assert state["job_id"] is None  # worker is free for the next job


def test_run_job_reports_cancellation():
    q = multiprocessing.Queue()
    pool = _StubPool(_StubPipeline(exc=cancellation.AnalysisCancelled()))
    cancel_upto = multiprocessing.Value("q", 0)

    _run_job(
        {"type": JOB_ANALYZE, "job_id": 5, "audio_path": "a.wav"},
        q, pool, cancel_upto, {"job_id": None},
    )
    messages = _drain(q)
    assert messages[-1]["type"] == MSG_CANCELLED
    assert messages[-1]["job_id"] == 5


def test_run_job_skips_a_job_cancelled_while_queued():
    q = multiprocessing.Queue()
    pipeline = _StubPipeline(report=_StubReport())
    pool = _StubPool(pipeline)
    cancel_upto = multiprocessing.Value("q", 9)

    _run_job(
        {"type": JOB_ANALYZE, "job_id": 9, "audio_path": "a.wav"},
        q, pool, cancel_upto, {"job_id": None},
    )
    messages = _drain(q)
    assert [m["type"] for m in messages] == [MSG_CANCELLED]
    assert pipeline.analyzed == []
    assert pool.calls == 0


def test_run_job_clears_a_stale_cancel_before_starting():
    """A cancel aimed at job 1 must not abort job 2."""
    q = multiprocessing.Queue()
    pipeline = _StubPipeline(report=_StubReport())
    pool = _StubPool(pipeline)
    cancel_upto = multiprocessing.Value("q", 1)
    cancellation.request_cancel()  # left over from job 1

    _run_job(
        {"type": JOB_ANALYZE, "job_id": 2, "audio_path": "b.wav"},
        q, pool, cancel_upto, {"job_id": None},
    )
    messages = _drain(q)
    assert messages[-1]["type"] == MSG_FINISHED
    assert pipeline.analyzed == ["b.wav"]


# ======================================================================
# Worker loop lifecycle
# ======================================================================

def _run_worker_loop(monkeypatch, jobs, pool):
    """Drive run_worker's loop in-thread with plain queues.

    Avoids spawning a real subprocess (which would re-import torch and, on
    Windows, re-enter pytest) while still exercising the loop itself.
    """
    import monitor.analysis_worker as worker_mod

    monkeypatch.setattr(worker_mod, "_setup_child_logging", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "ModelPool", lambda: pool)

    job_q: _queue_mod.Queue = _queue_mod.Queue()
    result_q: _queue_mod.Queue = _queue_mod.Queue()
    for job in jobs:
        job_q.put(job)
    job_q.put({"type": JOB_SHUTDOWN})

    cancel_upto = multiprocessing.Value("q", 0)
    thread = threading.Thread(
        target=run_worker, args=(job_q, result_q, cancel_upto, None), daemon=True,
    )
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "worker loop did not exit on shutdown"
    return _drain(result_q)


def test_worker_announces_readiness_and_exits_on_shutdown(monkeypatch):
    messages = _run_worker_loop(monkeypatch, [], _StubPool(_StubPipeline()))
    assert [m["type"] for m in messages] == [MSG_READY]


def test_worker_handles_many_jobs_with_one_model_pool(monkeypatch):
    """The regression: two files, one pool, one set of model loads."""
    pool = _StubPool(_StubPipeline(report=_StubReport()))
    jobs = [
        {"type": JOB_ANALYZE, "job_id": 1, "audio_path": "a.wav",
         "stt_model_key": "thorough"},
        {"type": JOB_ANALYZE, "job_id": 2, "audio_path": "b.wav",
         "stt_model_key": "thorough"},
    ]
    messages = _run_worker_loop(monkeypatch, jobs, pool)

    finished = [m for m in messages if m["type"] == MSG_FINISHED]
    assert [m["job_id"] for m in finished] == [1, 2]
    assert pool._pipeline.analyzed == ["a.wav", "b.wav"]


def test_worker_keeps_running_after_a_failed_job(monkeypatch):
    class _FlakyPool(_StubPool):
        def pipeline_for(self, stt_model_key):
            self.calls += 1
            if self.calls == 1:
                return _StubPipeline(exc=RuntimeError("first one blew up"))
            return self._pipeline

    pool = _FlakyPool(_StubPipeline(report=_StubReport()))
    jobs = [
        {"type": JOB_ANALYZE, "job_id": 1, "audio_path": "a.wav"},
        {"type": JOB_ANALYZE, "job_id": 2, "audio_path": "b.wav"},
    ]
    messages = _run_worker_loop(monkeypatch, jobs, pool)
    terminal = [m["type"] for m in messages if m["type"] in (MSG_ERROR, MSG_FINISHED)]
    assert terminal == [MSG_ERROR, MSG_FINISHED]


def test_worker_ignores_unknown_job_types(monkeypatch):
    pool = _StubPool(_StubPipeline(report=_StubReport()))
    jobs = [
        {"type": "nonsense", "job_id": 1},
        {"type": JOB_ANALYZE, "job_id": 2, "audio_path": "a.wav"},
    ]
    messages = _run_worker_loop(monkeypatch, jobs, pool)
    assert [m["type"] for m in messages if m["type"] == MSG_FINISHED]


# ======================================================================
# GUI wiring
# ======================================================================

def test_main_window_does_not_spawn_a_process_per_file():
    """The old code called multiprocessing.Process inside _start_analysis."""
    import ast
    import inspect

    from monitor.gui import main_window

    tree = ast.parse(inspect.getsource(main_window))
    spawning = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "Process"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert spawning == {"_ensure_worker"}, (
        f"multiprocessing.Process is created in {sorted(spawning)}; only "
        f"_ensure_worker may start the (single, reused) worker."
    )


def test_main_window_does_not_terminate_the_worker_to_stop_a_job():
    """Killing the worker is what forced the reload; only shutdown may do it."""
    import ast
    import inspect

    from monitor.gui import main_window

    tree = ast.parse(inspect.getsource(main_window))
    terminating = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr in ("terminate", "kill")
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert terminating <= {"_shutdown_worker"}, (
        f"the worker is terminated in {sorted(terminating)}; that discards the "
        f"loaded models."
    )
