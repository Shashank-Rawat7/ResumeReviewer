"""Request-scoped latency, token, retry and failure metrics for the ResumeReviewer pipeline.

WHAT IT RECORDS (per request, one structured JSON line on the `resume_reviewer.metrics` logger)
  per stage (summary, credentials, e1, e2, e3): status (ok / failed / not_run), wall-clock latency, input / output /
  total tokens (and reasoning tokens only when the response itself reports them), finish reason, retries, and for a
  failed stage the exception CLASS NAME and HTTP status code. Request level: endpoint latency, pipeline latency,
  totals, and whether the production normalisation of the missing-skills result succeeded.

WHAT IT NEVER RECORDS: resume text, job-description text, prompts, model output, exception messages, API keys,
  user names or e-mail addresses. The callback never reads prompt or output text; it reads only usage numbers.

DESIGN RULES
  * All state belongs to one RequestMetrics object created per request and passed down explicitly. Nothing is kept
    on the shared parser or model objects, and there is no registry of active requests. Concurrent requests
    therefore cannot see each other's tokens or retries.
  * Metrics never change behaviour: callbacks are passive, nothing is retried or re-ordered, and every public method
    swallows its own errors. A metrics bug can never fail a request.
  * A missing number is null, never zero. Retry counts are reported only when they can be trusted (see below).

RETRY COUNTING (best effort, null rather than misleading)
  The Groq SDK retries internally and reports a retry only as an INFO log line on the logger `groq._base_client`.
  One filter, installed once on that logger, counts those lines. A ContextVar holds the stage that is running in the
  current thread / task, so a retry line is credited to exactly that request and stage. Lines that cannot be
  credited (no running stage, or a stage that already closed) increment one process-level counter; a request whose
  window saw that counter move reports retries as null. To make the INFO lines visible to the filter the logger's
  own level may have to be lowered to INFO; the filter then drops every record the original threshold would have
  dropped, so the logger's visible output is unchanged. Handlers, propagation and other loggers are never touched.

Compatible with Python 3.9 (the Docker image). Standard library plus an optional langchain_core import.
"""
import contextvars
import importlib.util
import json
import logging
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

try:  # production always has langchain_core; without it token collection is simply unavailable
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where langchain_core is not installed
    _CallbackBase = object
    LANGCHAIN_AVAILABLE = False

RECORD_VERSION = 1
METRICS_LOGGER_NAME = "resume_reviewer.metrics"
STAGE_NAMES = ("summary", "credentials", "e1", "e2", "e3")
RETRY_LOGGER_NAME = "groq._base_client"
RETRY_MESSAGE_PREFIX = "Retrying request"
RETRY_SOURCE_MARKER = 'log.info("Retrying request to'

REASON_UNAVAILABLE = "tracking_unavailable"
REASON_DISRUPTED = "tracking_disrupted"
REASON_UNATTRIBUTED = "unattributed_retry_event"

_LOCK = threading.RLock()                      # guards stage records and the unattributed counter
_UNATTRIBUTED = 0                              # retry events that could not be credited to a stage
_CURRENT_STAGE = contextvars.ContextVar("resume_reviewer_current_stage", default=None)
_INSTALL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

def _count(value):
    """A usable token count: a non-negative int (not a bool), else None."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _seconds_to_ms(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(round(value * 1000))
    return None


def _ms_since(start):
    return int(round((time.perf_counter() - start) * 1000))


# ---------------------------------------------------------------------------------------------------------------
# Token usage from a LangChain LLMResult
# ---------------------------------------------------------------------------------------------------------------

def extract_usage(result) -> Optional[Dict[str, Any]]:
    """Usage numbers of ONE model call, or None when the response carries none.

    Source priority (one source per call, so nothing is counted twice):
      1. usage_metadata of the first generation's message (input_tokens / output_tokens / total_tokens). When a call
         returns several generations the provider reports one usage for the call and the library copies it onto each
         message, so the first one is used rather than summing copies.
      2. otherwise result.llm_output["token_usage"] (prompt_tokens / completion_tokens / total_tokens).
    Input and output must both be valid counts, or the call counts as having no usage.
    reasoning_tokens is filled only if the response reports it; it is never added to or subtracted from output."""
    generation = None
    for group in getattr(result, "generations", None) or []:
        for candidate in group or []:
            generation = candidate
            break
        if generation is not None:
            break
    message = getattr(generation, "message", None)
    info = getattr(generation, "generation_info", None)
    meta = getattr(message, "response_metadata", None)
    finish = None
    for holder in (info, meta):
        if isinstance(holder, dict) and isinstance(holder.get("finish_reason"), str):
            finish = holder["finish_reason"]
            break

    raw = None
    llm_output = getattr(result, "llm_output", None)
    if isinstance(llm_output, dict) and isinstance(llm_output.get("token_usage"), dict):
        raw = llm_output["token_usage"]

    usage_in = usage_out = usage_total = reasoning = source = None
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict) and _count(um.get("input_tokens")) is not None and _count(um.get("output_tokens")) is not None:
        usage_in, usage_out, usage_total = um["input_tokens"], um["output_tokens"], _count(um.get("total_tokens"))
        source = "usage_metadata"
        details = um.get("output_token_details")
        if isinstance(details, dict):
            reasoning = _count(details.get("reasoning"))
    elif raw is not None and _count(raw.get("prompt_tokens")) is not None and _count(raw.get("completion_tokens")) is not None:
        usage_in, usage_out, usage_total = raw["prompt_tokens"], raw["completion_tokens"], _count(raw.get("total_tokens"))
        source = "llm_output"
    if source is None:
        return {"has_usage": False, "finish_reason": finish, "provider_ms": _seconds_to_ms((raw or {}).get("total_time"))}
    if usage_total is None:
        usage_total = usage_in + usage_out
    if reasoning is None and raw is not None:
        details = raw.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = _count(details.get("reasoning_tokens"))
    return {"has_usage": True, "input": usage_in, "output": usage_out, "total": usage_total, "reasoning": reasoning,
            "source": source, "finish_reason": finish, "provider_ms": _seconds_to_ms((raw or {}).get("total_time"))}


# ---------------------------------------------------------------------------------------------------------------
# Stage record and callback
# ---------------------------------------------------------------------------------------------------------------

class StageRecord:
    """Mutable record of one pipeline stage. Mutated only under _LOCK."""

    def __init__(self, name):
        self.name = name
        self.status = "not_run"
        self.latency_ms = None
        self.error_class = None
        self.status_code = None
        self.retries = 0
        self.closed = False
        self.llm_calls = 0
        self.llm_calls_with_usage = 0
        self.finish_reason = None
        self._input = self._output = self._total = 0
        self._reasoning = 0
        self._reasoning_complete = True
        self._provider_ms = 0
        self._provider_complete = True
        self._sources = set()

    def add_call(self, usage):
        with _LOCK:
            self.llm_calls += 1
            usage = usage or {"has_usage": False}
            if usage.get("finish_reason"):
                self.finish_reason = usage["finish_reason"]
            if usage.get("provider_ms") is None:
                self._provider_complete = False
            else:
                self._provider_ms += usage["provider_ms"]
            if not usage.get("has_usage"):
                return
            self.llm_calls_with_usage += 1
            self._input += usage["input"]
            self._output += usage["output"]
            self._total += usage["total"]
            self._sources.add(usage["source"])
            if usage.get("reasoning") is None:
                self._reasoning_complete = False
            else:
                self._reasoning += usage["reasoning"]

    def tokens_known(self):
        return self.llm_calls > 0 and self.llm_calls_with_usage == self.llm_calls

    def as_dict(self, retries_trusted):
        known = self.tokens_known()
        ran = self.status in ("ok", "failed")
        return {
            "status": self.status,
            "latency_ms": self.latency_ms,
            "input_tokens": self._input if known else None,
            "output_tokens": self._output if known else None,
            "total_tokens": self._total if known else None,
            "reasoning_tokens": self._reasoning if known and self._reasoning_complete else None,
            "llm_calls": self.llm_calls,
            "llm_calls_with_usage": self.llm_calls_with_usage,
            "token_source": (sorted(self._sources)[0] if len(self._sources) == 1 else
                             ("mixed" if self._sources else None)),
            "provider_total_ms": self._provider_ms if self.llm_calls and self._provider_complete else None,
            "finish_reason": self.finish_reason,
            "retries": self.retries if (retries_trusted and ran) else None,
            "error_class": self.error_class,
            "status_code": self.status_code,
        }


class _UsageHandler(_CallbackBase):
    """Passive LangChain callback bound to ONE stage. It reads usage numbers only, never prompt or output text."""
    raise_error = False

    def __init__(self, record):
        super().__init__()
        self._record = record

    def on_chat_model_start(self, serialized, messages, **kwargs):
        return None                      # defined so LangChain does not fall back to on_llm_start with prompt text

    def on_llm_start(self, serialized, prompts, **kwargs):
        return None

    def on_llm_end(self, response, **kwargs):
        try:
            self._record.add_call(extract_usage(response))
        except Exception:
            pass


class _StageContext:
    """`with metrics.stage("e1") as stage:` times the block, records its outcome and exposes stage.config (the
    LangChain config carrying this stage's callback, or None). The block's exception always propagates unchanged."""

    def __init__(self, record):
        self._record = record
        self.config = None
        self._start = None
        self._token = None

    def __enter__(self):
        try:
            if LANGCHAIN_AVAILABLE:
                self.config = {"callbacks": [_UsageHandler(self._record)]}
            self._token = _CURRENT_STAGE.set(self._record)
            self._start = time.perf_counter()
        except Exception:
            self.config = None
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            record = self._record
            if self._start is not None:
                record.latency_ms = _ms_since(self._start)
            with _LOCK:
                record.closed = True
                if exc_type is None:
                    record.status = "ok"
                else:
                    record.status = "failed"
                    record.error_class = exc_type.__name__
                    code = getattr(exc, "status_code", None)
                    record.status_code = code if isinstance(code, int) and not isinstance(code, bool) else None
            if self._token is not None:
                try:
                    _CURRENT_STAGE.reset(self._token)
                except ValueError:
                    _CURRENT_STAGE.set(None)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------------------------------------------
# Retry tracking: one filter on the SDK logger, installed once
# ---------------------------------------------------------------------------------------------------------------

def unattributed_count():
    with _LOCK:
        return _UNATTRIBUTED


def _record_retry_event():
    global _UNATTRIBUTED
    stage = _CURRENT_STAGE.get()
    with _LOCK:
        if stage is not None and not stage.closed:
            stage.retries += 1
        else:
            _UNATTRIBUTED += 1


class _RetryFilter(logging.Filter):
    """Counts SDK retry lines and keeps the logger's visible output exactly as it was.

    If we had to lower the logger's level to INFO, every record the ORIGINAL threshold would have rejected is
    dropped here (after counting), so no INFO/DEBUG line becomes visible that was not visible before. If the
    application changes the logger's level after installation it has taken control and nothing is dropped."""
    _rr_retry_filter = True

    def __init__(self, logger, original_level, raised, applied_level):
        super().__init__()
        self.logger = logger
        self._rr_original_level = original_level      # the logger's own level before install (maybe NOTSET)
        self._rr_raised = raised
        self._rr_applied_level = applied_level

    def _threshold(self):
        if self._rr_original_level != logging.NOTSET:
            return self._rr_original_level
        parent = self.logger.parent
        return parent.getEffectiveLevel() if parent is not None else logging.WARNING

    def filter(self, record):
        try:
            if (record.levelno == logging.INFO and isinstance(record.msg, str)
                    and record.msg.startswith(RETRY_MESSAGE_PREFIX)):
                _record_retry_event()
            if (self._rr_raised and self.logger.level == self._rr_applied_level
                    and record.levelno < self._threshold()):
                return False
        except Exception:
            pass
        return True


class _TrackingState:
    available = False
    logger = None
    filter = None


_STATE = _TrackingState()


def _probe_sdk():
    """True when the installed Groq SDK still emits the retry line this design counts. Reads the module's source
    text without importing it."""
    try:
        spec = importlib.util.find_spec("groq._base_client")
        if spec is None or not spec.origin:
            return False
        with open(spec.origin, "r", encoding="utf-8") as handle:
            text = handle.read()
        return RETRY_SOURCE_MARKER in text and "logging.getLogger(__name__)" in text
    except Exception:
        return False


def install_retry_tracking(logger_name=RETRY_LOGGER_NAME, probe=None):
    """Install the retry filter once. Safe to call repeatedly (also after a module reload). Returns True when retry
    tracking is available. Never touches handlers, propagation or any other logger."""
    try:
        with _INSTALL_LOCK:
            logger = logging.getLogger(logger_name)
            existing = [f for f in logger.filters if getattr(f, "_rr_retry_filter", False)]
            if existing and type(existing[0]) is _RetryFilter and len(existing) == 1 and _STATE.filter is existing[0]:
                return _STATE.available
            if not (probe or _probe_sdk)():
                _STATE.available, _STATE.logger, _STATE.filter = False, logger, None
                return False
            if existing:                                  # stale filter (module reload): inherit its saved state
                original, raised, applied = (existing[0]._rr_original_level, existing[0]._rr_raised,
                                             existing[0]._rr_applied_level)
                for old in existing:
                    logger.removeFilter(old)
            else:
                raised = logger.getEffectiveLevel() > logging.INFO
                original, applied = logger.level, logging.INFO
                if raised:
                    logger.setLevel(logging.INFO)
            new = _RetryFilter(logger, original, raised, applied)
            logger.addFilter(new)
            _STATE.available, _STATE.logger, _STATE.filter = True, logger, new
            return True
    except Exception:
        _STATE.available = False
        return False


def uninstall_retry_tracking():
    """Remove the filter and restore the logger's level if this module raised it (used by tests and shutdown)."""
    try:
        with _INSTALL_LOCK:
            logger, current = _STATE.logger, _STATE.filter
            if logger is not None:
                for old in [f for f in logger.filters if getattr(f, "_rr_retry_filter", False)]:
                    if old._rr_raised and logger.level == old._rr_applied_level:
                        logger.setLevel(old._rr_original_level)
                    logger.removeFilter(old)
            _STATE.available, _STATE.logger, _STATE.filter = False, None, None
    except Exception:
        pass


def _tracking_problem():
    """None when retry counting can be trusted right now, else the reason it cannot."""
    state = _STATE
    if not state.available or state.logger is None or state.filter is None:
        return REASON_UNAVAILABLE
    logger = state.logger
    if state.filter not in logger.filters or logger.disabled or not logger.isEnabledFor(logging.INFO):
        return REASON_DISRUPTED
    return None


# ---------------------------------------------------------------------------------------------------------------
# The request collector
# ---------------------------------------------------------------------------------------------------------------

class _PipelineTimer:
    def __init__(self, owner):
        self._owner = owner
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._owner._pipeline_ms = _ms_since(self._start)
        except Exception:
            pass
        return False


class RequestMetrics:
    """Metrics of ONE request. Create it at the start of the request and pass it down; never share it."""

    def __init__(self, request_id=None, session_id=None, model=None):
        try:
            self.request_id = request_id or uuid.uuid4().hex
            self.session_id = session_id if isinstance(session_id, int) and not isinstance(session_id, bool) else None
            self.model = model if isinstance(model, str) else None
            self._t0 = time.perf_counter()
            self._stages = dict((name, StageRecord(name)) for name in STAGE_NAMES)
            self._pipeline_ms = None
            self._normalization_ok = None
            self._unattributed_at_start = unattributed_count()
            self._problem_at_start = _tracking_problem()
        except Exception:
            self.request_id, self.session_id, self.model = None, None, None
            self._t0, self._stages, self._pipeline_ms, self._normalization_ok = time.perf_counter(), {}, None, None
            self._unattributed_at_start, self._problem_at_start = 0, REASON_UNAVAILABLE

    def stage(self, name):
        try:
            record = self._stages.get(name)
            if record is None:
                record = self._stages[name] = StageRecord(name)
            return _StageContext(record)
        except Exception:
            return _StageContext(StageRecord(str(name)))

    def pipeline(self):
        return _PipelineTimer(self)

    def set_normalization(self, prioritized):
        """Record the normalisation status from the parser's own result (a dict or its JSON string)."""
        try:
            data = json.loads(prioritized) if isinstance(prioritized, str) else prioritized
            flag = data.get("normalization_ok") if isinstance(data, dict) else None
            self._normalization_ok = flag if isinstance(flag, bool) else None
        except Exception:
            self._normalization_ok = None

    def _retry_problem(self):
        problem = self._problem_at_start or _tracking_problem()
        if problem is None and unattributed_count() != self._unattributed_at_start:
            problem = REASON_UNATTRIBUTED
        return problem

    def finish(self, outcome="ok"):
        """The final record (a dict), or None if it could not be built."""
        try:
            problem = self._retry_problem()
            trusted = problem is None
            stages = dict((name, rec.as_dict(trusted)) for name, rec in self._stages.items())
            ran = [s for s in stages.values() if s["status"] in ("ok", "failed")]
            complete = bool(ran) and all(s["input_tokens"] is not None and s["output_tokens"] is not None for s in ran)
            totals = {
                "input_tokens": sum(s["input_tokens"] for s in ran) if complete else None,
                "output_tokens": sum(s["output_tokens"] for s in ran) if complete else None,
                "total_tokens": sum(s["total_tokens"] for s in ran) if complete else None,
                "tokens_complete": complete,
                "retries": sum(s["retries"] for s in ran) if (trusted and ran) else None,
                "failed_stages": sum(1 for s in stages.values() if s["status"] == "failed"),
                "not_run_stages": sum(1 for s in stages.values() if s["status"] == "not_run"),
            }
            return {
                "record_version": RECORD_VERSION, "request_id": self.request_id, "session_id": self.session_id,
                "model": self.model, "outcome": outcome if isinstance(outcome, str) else "error",
                "request_ms": _ms_since(self._t0), "pipeline_ms": self._pipeline_ms, "stages": stages,
                "totals": totals, "retries_unavailable_reason": problem, "normalization_ok": self._normalization_ok,
            }
        except Exception:
            return None

    def emit(self, outcome="ok"):
        """Build the record and log it as one JSON line. Never raises."""
        try:
            record = self.finish(outcome)
            if record is not None:
                logging.getLogger(METRICS_LOGGER_NAME).info(json.dumps(record, sort_keys=True, separators=(",", ":")))
            return record
        except Exception:
            return None


class _NullStage:
    config = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class NullMetrics:
    """Same interface as RequestMetrics, records nothing. Use when no collector is wanted."""

    def stage(self, name):
        return _NullStage()

    def pipeline(self):
        return _NullStage()

    def set_normalization(self, prioritized):
        return None

    def finish(self, outcome="ok"):
        return None

    def emit(self, outcome="ok"):
        return None


NULL_METRICS = NullMetrics()


# ---------------------------------------------------------------------------------------------------------------
# The dedicated metrics logger
# ---------------------------------------------------------------------------------------------------------------

def configure_metrics_logger(stream=None):
    """Make the dedicated logger emit bare JSON lines on stdout, once. Only the `resume_reviewer.metrics` logger is
    configured (level INFO, its own handler, propagate False so a later root configuration cannot duplicate lines).
    Idempotent, also after a module reload."""
    try:
        logger = logging.getLogger(METRICS_LOGGER_NAME)
        for old in [h for h in logger.handlers if getattr(h, "_rr_metrics_handler", False)]:
            if stream is None and type(old) is logging.StreamHandler:
                return logger
            logger.removeHandler(old)
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._rr_metrics_handler = True
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
    except Exception:
        return None
