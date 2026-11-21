"""Tests for the observability feature (src/metrics.py, its use in src/parser.py and backend/main.py).

Run from the project root:  python -m unittest tests.test_metrics -v
Synthetic data only. No network, no API key, no live model, no resume file. Nothing here reads or writes the
generated src/*.txt files: the parser is built with ResumeParser.__new__ and its file-saving hook is replaced.
Tests that need LangChain's callback machinery are skipped when langchain_core is not installed.
"""
import asyncio
import contextvars
import gc
import importlib
import importlib.util
import io
import json
import logging
import os
import re
import threading
import unittest
import weakref
from itertools import count
from types import SimpleNamespace
from unittest import mock

try:
    from tests.test_bullet_verification import load_parser
except ImportError:  # pragma: no cover - run as a plain script
    from test_bullet_verification import load_parser

from src import metrics as m

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAS_LANGCHAIN = m.LANGCHAIN_AVAILABLE
RESUME = "SENTINEL_RESUME_TEXT python sql"
JD = "SENTINEL_JD_TEXT"
SECRET = "gsk_SENTINEL_KEY"
_ids = count(1)


def response(usage_metadata=None, token_usage=None, finish=None, generations=1):
    """A stand-in for a LangChain LLMResult (extract_usage only uses getattr)."""
    meta = {"finish_reason": finish} if finish else {}
    gens = [SimpleNamespace(message=SimpleNamespace(usage_metadata=usage_metadata, response_metadata=meta),
                            generation_info=None) for _ in range(generations)]
    return SimpleNamespace(generations=[gens], llm_output={"token_usage": token_usage} if token_usage is not None else None)


def usage(i, o, **extra):
    data = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
    data.update(extra)
    return response(usage_metadata=data, finish="stop")


def feed(request, stage, resp):
    """What the callback does at the end of a model call."""
    request._stages[stage].add_call(m.extract_usage(resp))


class _Capture(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append((record.levelno, record.getMessage()))


# ---------------------------------------------------------------------------------------------------------------
class TokenExtraction(unittest.TestCase):
    def test_usage_metadata_is_used(self):
        u = m.extract_usage(usage(10, 5))
        self.assertEqual((u["input"], u["output"], u["total"], u["source"]), (10, 5, 15, "usage_metadata"))
        self.assertEqual(u["finish_reason"], "stop")

    def test_llm_output_only(self):
        u = m.extract_usage(response(token_usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
                                                  "total_time": 0.5, "completion_tokens_details": {"reasoning_tokens": 2}}))
        self.assertEqual((u["input"], u["output"], u["total"], u["source"]), (7, 3, 10, "llm_output"))
        self.assertEqual((u["reasoning"], u["provider_ms"]), (2, 500))

    def test_both_present_counts_once(self):
        u = m.extract_usage(response(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                                     token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}))
        self.assertEqual((u["input"], u["output"], u["source"]), (10, 5, "usage_metadata"))

    def test_missing_usage(self):
        self.assertFalse(m.extract_usage(response())["has_usage"])
        self.assertFalse(m.extract_usage(SimpleNamespace())["has_usage"])
        self.assertFalse(m.extract_usage(None)["has_usage"])

    def test_invalid_values_are_not_usage(self):
        for bad in ("10", -1, True, 1.5, None):
            self.assertFalse(m.extract_usage(response(usage_metadata={"input_tokens": bad, "output_tokens": 2}))["has_usage"])
        u = m.extract_usage(response(usage_metadata={"input_tokens": "x", "output_tokens": 2},
                                     token_usage={"prompt_tokens": 4, "completion_tokens": 1}))
        self.assertEqual((u["input"], u["source"], u["total"]), (4, "llm_output", 5))

    def test_total_is_derived_when_absent(self):
        self.assertEqual(m.extract_usage(response(usage_metadata={"input_tokens": 3, "output_tokens": 4}))["total"], 7)

    def test_reasoning_only_when_reported_and_not_mixed_into_output(self):
        none = m.extract_usage(usage(10, 5))
        self.assertIsNone(none["reasoning"])
        some = m.extract_usage(usage(10, 5, output_token_details={"reasoning": 4}))
        self.assertEqual((some["reasoning"], some["output"]), (4, 5))

    def test_several_generations_use_the_first_not_a_sum(self):
        three = response(usage_metadata={"input_tokens": 10, "output_tokens": 5}, generations=3)
        self.assertEqual(m.extract_usage(three)["input"], 10)

    def test_stage_aggregation_and_null_rules(self):
        r = m.RequestMetrics()
        with r.stage("e1"):
            feed(r, "e1", usage(10, 5))
            feed(r, "e1", usage(1, 2))
        with r.stage("e2"):
            feed(r, "e2", usage(4, 4))
            feed(r, "e2", response())            # one call without usage
        with r.stage("e3"):
            pass                                 # ran, no model call recorded
        rec = r.finish()
        e1, e2, e3 = rec["stages"]["e1"], rec["stages"]["e2"], rec["stages"]["e3"]
        self.assertEqual((e1["input_tokens"], e1["output_tokens"], e1["llm_calls"]), (11, 7, 2))
        self.assertIsNone(e2["input_tokens"])
        self.assertEqual((e2["llm_calls"], e2["llm_calls_with_usage"]), (2, 1))
        self.assertIsNone(e3["input_tokens"])
        self.assertFalse(rec["totals"]["tokens_complete"])
        self.assertIsNone(rec["totals"]["input_tokens"])

    def test_totals_when_complete(self):
        r = m.RequestMetrics()
        for name, (i, o) in (("summary", (1, 2)), ("credentials", (3, 4)), ("e1", (5, 6)), ("e2", (7, 8)), ("e3", (9, 10))):
            with r.stage(name):
                feed(r, name, usage(i, o))
        t = r.finish()["totals"]
        self.assertEqual((t["input_tokens"], t["output_tokens"], t["total_tokens"], t["tokens_complete"]), (25, 30, 55, True))
        self.assertEqual(t["not_run_stages"], 0)


# ---------------------------------------------------------------------------------------------------------------
class StageSemantics(unittest.TestCase):
    def test_ok_failed_not_run_and_exception_propagates(self):
        r = m.RequestMetrics()
        with r.stage("summary"):
            pass

        class Boom(Exception):
            status_code = 429

        with self.assertRaises(Boom):
            with r.stage("e1"):
                raise Boom("secret message " + SECRET)
        rec = r.finish()
        self.assertEqual(rec["stages"]["summary"]["status"], "ok")
        e1 = rec["stages"]["e1"]
        self.assertEqual((e1["status"], e1["error_class"], e1["status_code"]), ("failed", "Boom", 429))
        self.assertEqual(rec["stages"]["e2"]["status"], "not_run")
        self.assertNotIn(SECRET, json.dumps(rec))
        self.assertNotIn("secret message", json.dumps(rec))
        self.assertEqual((rec["totals"]["failed_stages"], rec["totals"]["not_run_stages"]), (1, 3))
        self.assertIsInstance(e1["latency_ms"], int)

    def test_non_integer_status_code_is_ignored(self):
        r = m.RequestMetrics()
        e = ValueError("x")
        e.status_code = "429 too many"
        with self.assertRaises(ValueError):
            with r.stage("e1"):
                raise e
        self.assertIsNone(r.finish()["stages"]["e1"]["status_code"])

    def test_context_var_reset_after_exception(self):
        r = m.RequestMetrics()
        with self.assertRaises(KeyError):
            with r.stage("e1"):
                raise KeyError("x")
        self.assertIsNone(m._CURRENT_STAGE.get())

    def test_pipeline_timer_and_record_is_json(self):
        r = m.RequestMetrics(session_id=3, model="mod")
        with r.pipeline():
            pass
        rec = r.finish("http_404")
        json.dumps(rec)
        self.assertEqual((rec["session_id"], rec["model"], rec["outcome"]), (3, "mod", "http_404"))
        self.assertIsInstance(rec["pipeline_ms"], int)
        self.assertGreaterEqual(rec["request_ms"], 0)

    def test_session_id_must_be_an_int(self):
        self.assertIsNone(m.RequestMetrics(session_id="alice@example.com").session_id)

    def test_normalization_status(self):
        r = m.RequestMetrics()
        self.assertIsNone(r.finish()["normalization_ok"])
        r.set_normalization(json.dumps({"normalization_ok": False}))
        self.assertIs(r.finish()["normalization_ok"], False)
        r.set_normalization({"normalization_ok": True})
        self.assertIs(r.finish()["normalization_ok"], True)
        r.set_normalization("not json")
        self.assertIsNone(r.finish()["normalization_ok"])

    def test_null_metrics_matches_the_public_interface(self):
        public = lambda o: {n for n in dir(o) if not n.startswith("_")}
        self.assertTrue(public(m.RequestMetrics) <= public(m.NullMetrics))
        with m.NULL_METRICS.stage("e1") as st:
            self.assertIsNone(st.config)
        self.assertIsNone(m.NULL_METRICS.emit())

    def test_emit_writes_one_json_line_and_never_raises(self):
        stream = io.StringIO()
        logger = logging.getLogger(m.METRICS_LOGGER_NAME)
        saved = (list(logger.handlers), logger.level, logger.propagate)
        try:
            m.configure_metrics_logger(stream)
            r = m.RequestMetrics(session_id=1, model="x")
            self.assertIsInstance(r.emit("ok"), dict)
            lines = stream.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["request_id"], r.request_id)
            self.assertFalse(logger.propagate)
        finally:
            logger.handlers[:], logger.level, logger.propagate = saved[0], saved[1], saved[2]
        broken = m.RequestMetrics()
        broken._stages = None
        self.assertIsNone(broken.emit())

    def test_configure_is_idempotent(self):
        logger = logging.getLogger(m.METRICS_LOGGER_NAME)
        saved = (list(logger.handlers), logger.level, logger.propagate)
        try:
            logger.handlers[:] = []
            m.configure_metrics_logger()
            m.configure_metrics_logger()
            self.assertEqual(len([h for h in logger.handlers if getattr(h, "_rr_metrics_handler", False)]), 1)
        finally:
            logger.handlers[:], logger.level, logger.propagate = saved[0], saved[1], saved[2]


# ---------------------------------------------------------------------------------------------------------------
class TrackingCase(unittest.TestCase):
    def setUp(self):
        n = next(_ids)
        self.parent = logging.getLogger("rrtest%d" % n)
        self.parent.setLevel(logging.WARNING)
        self.parent.propagate = False
        self.name = "rrtest%d.sdk" % n
        self.log = logging.getLogger(self.name)
        self.seen = []
        self.capture = _Capture(self.seen)
        self.log.addHandler(self.capture)
        self.addCleanup(m.uninstall_retry_tracking)
        self.addCleanup(self.log.removeHandler, self.capture)
        self.addCleanup(logging.disable, logging.NOTSET)

    def install(self, probe=lambda: True):
        return m.install_retry_tracking(self.name, probe=probe)

    def retry(self):
        self.log.info("Retrying request to %s in %f seconds", "https://api.example/secret?key=1", 0.5)

    def one_stage(self, n=1, name="e1"):
        r = m.RequestMetrics()
        with r.stage(name):
            for _ in range(n):
                self.retry()
        return r.finish()


class RetryCounting(TrackingCase):
    def test_retries_are_attributed_to_the_running_stage(self):
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            self.retry()
            self.retry()
        with r.stage("e2"):
            pass
        rec = r.finish()
        self.assertEqual((rec["stages"]["e1"]["retries"], rec["stages"]["e2"]["retries"]), (2, 0))
        self.assertIsNone(rec["stages"]["e3"]["retries"])
        self.assertEqual(rec["totals"]["retries"], 2)
        self.assertIsNone(rec["retries_unavailable_reason"])

    def test_inherited_warning_threshold_is_preserved(self):
        self.assertEqual(self.log.level, logging.NOTSET)
        self.install()
        self.assertEqual(self.log.level, logging.INFO)
        r = m.RequestMetrics()
        with r.stage("e1"):
            self.retry()
            self.log.info("unrelated info")
            self.log.debug("unrelated debug")
            self.log.warning("a warning")
            self.log.error("an error")
        self.assertEqual([lvl for lvl, _ in self.seen], [logging.WARNING, logging.ERROR])
        self.assertEqual(r.finish()["stages"]["e1"]["retries"], 1)

    def test_explicit_warning_error_original_levels(self):
        for original, expected in ((logging.WARNING, [30, 40]), (logging.ERROR, [40])):
            with self.subTest(level=original):
                del self.seen[:]
                m.uninstall_retry_tracking()
                self.log.setLevel(original)
                self.install()
                r = m.RequestMetrics()
                with r.stage("e1"):
                    self.retry()
                    self.log.info("unrelated")
                    self.log.warning("w")
                    self.log.error("e")
                self.assertEqual([lvl for lvl, _ in self.seen], expected)
                self.assertEqual(r.finish()["stages"]["e1"]["retries"], 1)
                m.uninstall_retry_tracking()
                self.assertEqual(self.log.level, original)

    def test_original_info_leaves_visibility_unchanged(self):
        self.log.setLevel(logging.INFO)
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            self.retry()
            self.log.info("unrelated info")
            self.log.debug("debug")
        self.assertEqual(len(self.seen), 2)               # retry line and unrelated info stay visible
        self.assertEqual(r.finish()["stages"]["e1"]["retries"], 1)
        self.assertEqual(self.log.level, logging.INFO)

    def test_original_debug_leaves_visibility_unchanged(self):
        self.log.setLevel(logging.DEBUG)
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            self.retry()
            self.log.debug("debug")
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(r.finish()["stages"]["e1"]["retries"], 1)

    def test_only_the_sdk_logger_level_changes(self):
        root = logging.getLogger()
        before = (root.level, list(root.handlers), self.parent.level, list(self.parent.handlers),
                  self.log.propagate, list(self.log.handlers))
        sibling = logging.getLogger(self.parent.name + ".other")
        sibling_level = sibling.level
        self.install()
        after = (root.level, list(root.handlers), self.parent.level, list(self.parent.handlers),
                 self.log.propagate, list(self.log.handlers))
        self.assertEqual(before, after)
        self.assertEqual(sibling.level, sibling_level)
        self.assertEqual(len(self.log.filters), 1)

    def test_application_changing_the_level_later_stops_dropping(self):
        self.install()
        self.log.setLevel(logging.DEBUG)
        self.log.info("now visible")
        self.log.debug("also visible")
        self.assertEqual(len(self.seen), 2)

    def test_install_twice_installs_one_filter_and_counts_once(self):
        self.assertTrue(self.install())
        self.assertTrue(self.install())
        self.assertEqual(len([f for f in self.log.filters if getattr(f, "_rr_retry_filter", False)]), 1)
        self.assertEqual(self.one_stage()["stages"]["e1"]["retries"], 1)

    def test_reload_does_not_duplicate_the_filter_or_the_count(self):
        self.install()
        importlib.reload(m)
        self.assertTrue(self.install())
        self.assertEqual(len([f for f in self.log.filters if getattr(f, "_rr_retry_filter", False)]), 1)
        self.assertEqual(self.one_stage()["stages"]["e1"]["retries"], 1)
        m.uninstall_retry_tracking()
        self.assertEqual(self.log.level, logging.NOTSET)

    def test_root_handlers_do_not_double_count(self):
        root_seen = []
        handler = _Capture(root_seen)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        self.parent.propagate = True
        self.install()
        self.assertEqual(self.one_stage()["stages"]["e1"]["retries"], 1)

    def test_probe_failure_means_unavailable_and_no_changes(self):
        self.assertFalse(self.install(probe=lambda: False))
        self.assertEqual(self.log.level, logging.NOTSET)
        self.assertEqual(self.log.filters, [])
        rec = self.one_stage()
        self.assertEqual(rec["retries_unavailable_reason"], m.REASON_UNAVAILABLE)
        self.assertIsNone(rec["stages"]["e1"]["retries"])
        self.assertIsNone(rec["totals"]["retries"])

    def test_real_probe_matches_installed_sdk(self):
        try:
            present = importlib.util.find_spec("groq") is not None
        except Exception:
            present = False
        self.assertEqual(m._probe_sdk(), present)

    def test_health_check_cases(self):
        def disable_logger():
            self.log.disabled = True

        def global_disable():
            logging.disable(logging.CRITICAL)

        def remove_filter():
            for f in list(self.log.filters):
                self.log.removeFilter(f)

        def lower_level():
            self.log.setLevel(logging.WARNING)

        for action in (disable_logger, global_disable, remove_filter, lower_level):
            with self.subTest(action=action.__name__):
                m.uninstall_retry_tracking()
                self.log.disabled = False
                logging.disable(logging.NOTSET)
                self.log.setLevel(logging.NOTSET)
                self.install()
                r = m.RequestMetrics()
                with r.stage("e1"):
                    self.retry()
                    action()
                rec = r.finish()
                self.assertEqual(rec["retries_unavailable_reason"], m.REASON_DISRUPTED)
                self.assertIsNone(rec["stages"]["e1"]["retries"])
                self.assertIsNone(rec["totals"]["retries"])
                logging.disable(logging.NOTSET)
                self.log.disabled = False
                # a request that starts while tracking is disrupted is also null
                action()
                self.assertEqual(m.RequestMetrics().finish()["retries_unavailable_reason"], m.REASON_DISRUPTED)

    def test_events_outside_a_stage_make_the_request_null(self):
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            self.retry()
        self.retry()                                      # between stages, same request
        with r.stage("e2"):
            pass
        rec = r.finish()
        self.assertEqual(rec["retries_unavailable_reason"], m.REASON_UNATTRIBUTED)
        self.assertIsNone(rec["stages"]["e1"]["retries"])
        self.assertIsNone(rec["totals"]["retries"])

    def test_events_before_a_request_do_not_affect_it(self):
        self.install()
        self.retry()                                      # outside any request
        before = m.unattributed_count()
        rec = self.one_stage(2)
        self.assertEqual(rec["stages"]["e1"]["retries"], 2)
        self.assertEqual(m.unattributed_count(), before)
        self.assertIsNone(rec["retries_unavailable_reason"])

    def test_event_in_a_closed_stage_is_unattributed(self):
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            ctx = contextvars.copy_context()
        ctx.run(self.retry)
        self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_UNATTRIBUTED)

    def test_thread_without_context_is_unattributed_but_copied_context_is_attributed(self):
        self.install()
        r = m.RequestMetrics()
        with r.stage("e1"):
            t = threading.Thread(target=lambda: contextvars.Context().run(self.retry))
            t.start()
            t.join()
        self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_UNATTRIBUTED)
        r = m.RequestMetrics()
        with r.stage("e1"):
            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: ctx.run(self.retry))
            t.start()
            t.join()
        rec = r.finish()
        self.assertEqual(rec["stages"]["e1"]["retries"], 1)

    def test_unattributed_counter_is_thread_safe(self):
        self.install()
        before = m.unattributed_count()

        def noise():
            for _ in range(500):
                contextvars.Context().run(self.retry)

        threads = [threading.Thread(target=noise) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(m.unattributed_count() - before, 4000)

    def test_concurrent_requests_keep_exact_per_request_numbers(self):
        self.install()
        n = 6
        barrier, overlap = threading.Barrier(n), threading.Barrier(n)
        out = {}
        before = m.unattributed_count()

        def work(i):
            r = m.RequestMetrics(session_id=i)
            barrier.wait()
            with r.stage("e1"):
                for _ in range(i + 1):
                    self.retry()
                overlap.wait()
                feed(r, "e1", usage(10 * (i + 1), i + 1))
            out[i] = r.finish()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i in range(n):
            e1 = out[i]["stages"]["e1"]
            self.assertEqual((e1["retries"], e1["input_tokens"], e1["output_tokens"]), (i + 1, 10 * (i + 1), i + 1))
            self.assertIsNone(out[i]["retries_unavailable_reason"])
        self.assertEqual(m.unattributed_count(), before)

    def test_contextless_noise_nulls_overlapping_requests_only(self):
        self.install()
        a, b = m.RequestMetrics(), m.RequestMetrics()
        with a.stage("e1"):
            with b.stage("e1"):
                t = threading.Thread(target=lambda: contextvars.Context().run(self.retry))
                t.start()
                t.join()
        for r in (a, b):
            self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_UNATTRIBUTED)
        later = self.one_stage(1)
        self.assertEqual(later["stages"]["e1"]["retries"], 1)

    def test_asyncio_tasks_and_to_thread_keep_context(self):
        self.install()

        async def one(i):
            r = m.RequestMetrics(session_id=i)
            with r.stage("e1"):
                await asyncio.sleep(0)
                for _ in range(i + 1):
                    self.retry()
                await asyncio.sleep(0)
                await asyncio.to_thread(self.retry)      # copies the current context
            return r.finish()

        async def main():
            return await asyncio.gather(*[one(i) for i in range(5)])

        for i, rec in enumerate(asyncio.run(main())):
            self.assertEqual(rec["stages"]["e1"]["retries"], i + 2)
            self.assertIsNone(rec["retries_unavailable_reason"])

    def test_no_registry_of_requests(self):
        self.install()
        r = m.RequestMetrics()
        ref = weakref.ref(r)
        with r.stage("e1"):
            self.retry()
        r.finish()
        del r
        gc.collect()
        self.assertIsNone(ref())

    def test_reason_precedence(self):
        self.install()
        r = m.RequestMetrics()
        self.retry()                                      # unattributed
        self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_UNATTRIBUTED)
        self.log.disabled = True
        self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_DISRUPTED)
        m.uninstall_retry_tracking()
        self.assertEqual(r.finish()["retries_unavailable_reason"], m.REASON_UNAVAILABLE)

    def test_url_and_message_arguments_are_never_formatted_or_stored(self):
        self.install()
        with mock.patch.object(logging.LogRecord, "getMessage", side_effect=AssertionError("must not format")):
            r = m.RequestMetrics()
            logger_filter = [f for f in self.log.filters if getattr(f, "_rr_retry_filter", False)][0]
            rec = logging.LogRecord(self.name, logging.INFO, "f", 1, "Retrying request to %s in %f seconds",
                                    ("https://secret", 1.0), None)
            with r.stage("e1"):
                logger_filter.filter(rec)
        self.assertEqual(r.finish()["stages"]["e1"]["retries"], 1)


# ---------------------------------------------------------------------------------------------------------------
class SpyChain:
    """Stands in for `prompt | llm` and for the summarize chain; records every invoke."""

    def __init__(self, world, name):
        self.world, self.name = world, name

    def invoke(self, payload, config=None):
        w = self.world
        w.calls.append((self.name, payload, config))
        step = w.script.get(self.name, {})
        if config and config.get("callbacks"):
            for callback in config["callbacks"]:
                callback.on_llm_end(usage(*step.get("tokens", (10, 5))))
        if step.get("error") is not None:
            raise step["error"]
        content = step.get("content", self.name + " output")
        if self.name == "summary":
            return {"output_text": content}
        return SimpleNamespace(content=content)


class SpyPrompt:
    def __init__(self, world, name):
        self.world, self.name = world, name

    def __or__(self, other):
        return SpyChain(self.world, self.name)


class World:
    def __init__(self, script=None):
        self.script = script or {}
        self.calls = []
        self.saved = []


def build_parser(mod, world):
    p = mod.ResumeParser.__new__(mod.ResumeParser)
    p.llm = object()
    p.resume_prompt = "resume prompt"
    p.credentials_prompt = SpyPrompt(world, "credentials")
    p.job_prompt = SpyPrompt(world, "e1")
    p.missing_skills_prompt = SpyPrompt(world, "e2")
    p.ats_bullet_prompt = SpyPrompt(world, "e3")
    p.preprocess_resume = lambda path: [SimpleNamespace(page_content=RESUME)]
    p.save_text_to_file = lambda path, text: world.saved.append(path)   # never writes src/*.txt
    return p


def run_parse(mod, world, metrics="omit"):
    parser = build_parser(mod, world)
    fake_chain = lambda **kwargs: SpyChain(world, "summary")
    with mock.patch.object(mod, "load_summarize_chain", fake_chain), mock.patch.object(mod.logging, "exception"):
        if metrics == "omit":
            return parser.parse_resume("x.pdf", JD)
        return parser.parse_resume("x.pdf", JD, metrics=metrics)


class FailingCollector:
    def stage(self, name):
        raise RuntimeError("collector broken")

    def set_normalization(self, value):
        raise RuntimeError("collector broken")


class ParserIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_parser()

    def setUp(self):
        self.addCleanup(m.uninstall_retry_tracking)

    def test_identical_six_tuple_and_calls_with_every_kind_of_collector(self):
        results, call_logs = [], []
        for metrics in ("omit", None, m.NULL_METRICS, m.RequestMetrics(), FailingCollector()):
            world = World()
            results.append(run_parse(self.mod, world, metrics))
            call_logs.append([(name, payload) for name, payload, _ in world.calls])
        self.assertEqual(len(results[0]), 6)
        for other in results[1:]:
            self.assertEqual(other, results[0])
        for other in call_logs[1:]:
            self.assertEqual(other, call_logs[0])
        self.assertEqual([n for n, _ in call_logs[0]], ["summary", "credentials", "e1", "e2", "e3"])

    def test_inputs_are_unchanged(self):
        world = World()
        run_parse(self.mod, world, m.RequestMetrics())
        by_name = {n: p for n, p, _ in world.calls}
        self.assertEqual(by_name["credentials"], {"resume_content": RESUME})
        self.assertEqual(by_name["e1"], {"job_description": JD})
        self.assertEqual(set(by_name["e2"]), {"job_description", "job_skills", "resume_content"})
        self.assertEqual(by_name["e3"], {"job_description": JD, "resume_content": RESUME})

    def test_no_collector_means_config_none(self):
        for metrics in ("omit", None, m.NULL_METRICS, FailingCollector()):
            world = World()
            run_parse(self.mod, world, metrics)
            self.assertTrue(all(config is None for _, _, config in world.calls), metrics)

    def test_collector_passes_a_callback_config_per_stage(self):
        if not HAS_LANGCHAIN:
            self.skipTest("langchain_core not installed")
        world = World()
        run_parse(self.mod, world, m.RequestMetrics())
        handlers = [config["callbacks"][0] for _, _, config in world.calls]
        self.assertEqual(len({id(h) for h in handlers}), 5)

    def test_all_stages_ok_with_tokens(self):
        if not HAS_LANGCHAIN:
            self.skipTest("langchain_core not installed")
        world = World({"e1": {"tokens": (100, 7)}})
        r = m.RequestMetrics(session_id=1, model="m")
        with r.pipeline():
            run_parse(self.mod, world, r)
        rec = r.finish()
        self.assertTrue(all(s["status"] == "ok" for s in rec["stages"].values()))
        self.assertEqual((rec["stages"]["e1"]["input_tokens"], rec["stages"]["e1"]["output_tokens"]), (100, 7))
        self.assertTrue(rec["totals"]["tokens_complete"])
        self.assertEqual(rec["totals"]["input_tokens"], 140)
        self.assertIs(rec["normalization_ok"], False)      # the fake E2 text is not valid JSON

    def test_e1_failure_skips_e2_but_e3_still_runs(self):
        world = World({"e1": {"error": RuntimeError("boom " + SECRET + RESUME)}})
        r = m.RequestMetrics()
        result = run_parse(self.mod, world, r)
        rec = r.finish()
        self.assertEqual(result[3], "Error generating job skills.")
        s = rec["stages"]
        self.assertEqual((s["e1"]["status"], s["e1"]["error_class"]), ("failed", "RuntimeError"))
        self.assertEqual(s["e2"]["status"], "not_run")
        self.assertEqual(s["e3"]["status"], "ok")
        self.assertEqual([n for n, _, _ in world.calls], ["summary", "credentials", "e1", "e3"])
        self.assertIsNone(rec["normalization_ok"])
        self.assertNotIn(SECRET, json.dumps(rec))

    def test_credentials_failure_is_swallowed_as_before(self):
        world = World({"credentials": {"error": ValueError("x")}})
        r = m.RequestMetrics()
        self.assertEqual(run_parse(self.mod, world, r)[1], "Error extracting credentials.")
        self.assertEqual(r.finish()["stages"]["credentials"]["status"], "failed")

    def test_summary_failure_propagates_and_later_stages_not_run(self):
        world = World({"summary": {"error": RuntimeError("down")}})
        r = m.RequestMetrics()
        with self.assertRaises(self.mod.ResumeParserError):
            run_parse(self.mod, world, r)
        s = r.finish()["stages"]
        self.assertEqual(s["summary"]["status"], "failed")
        self.assertTrue(all(s[n]["status"] == "not_run" for n in ("credentials", "e1", "e2", "e3")))

    def test_e3_failure_propagates(self):
        class Rate(Exception):
            status_code = 429

        world = World({"e3": {"error": Rate("slow down")}})
        r = m.RequestMetrics()
        with self.assertRaises(Rate):
            run_parse(self.mod, world, r)
        e3 = r.finish()["stages"]["e3"]
        self.assertEqual((e3["status"], e3["error_class"], e3["status_code"]), ("failed", "Rate", 429))

    def test_privacy_sentinels_never_reach_the_record_or_the_log(self):
        world = World({"e2": {"error": RuntimeError(SECRET + RESUME + JD)}})
        stream = io.StringIO()
        logger = logging.getLogger(m.METRICS_LOGGER_NAME)
        saved = (list(logger.handlers), logger.level, logger.propagate)
        try:
            m.configure_metrics_logger(stream)
            r = m.RequestMetrics(session_id=9, model="m")
            with r.pipeline():
                run_parse(self.mod, world, r)
            r.emit("ok")
        finally:
            logger.handlers[:], logger.level, logger.propagate = saved
        text = stream.getvalue()
        for secret in (SECRET, "SENTINEL_RESUME_TEXT", "SENTINEL_JD_TEXT", "python sql"):
            self.assertNotIn(secret, text)
        self.assertEqual(len(text.splitlines()), 1)

    def test_saving_is_replaced_so_no_generated_file_is_written(self):
        world = World()
        run_parse(self.mod, world, m.RequestMetrics())
        self.assertEqual(len(world.saved), 6)      # paths were only recorded by the stand-in, nothing was written


@unittest.skipUnless(HAS_LANGCHAIN, "langchain_core not installed")
class RealCallbackFlow(unittest.TestCase):
    """The same callbacks driven by a real LangChain chat model (offline, scripted)."""

    @staticmethod
    def make_chat(items):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class ScriptedChat(BaseChatModel):
            script: list

            @property
            def _llm_type(self):
                return "scripted"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                item = self.script.pop(0)
                if item.get("error"):
                    raise item["error"]
                message = AIMessage(content="out", usage_metadata=item.get("usage_metadata"),
                                    response_metadata={"finish_reason": "stop"})
                return ChatResult(generations=[ChatGeneration(message=message)],
                                  llm_output=item.get("llm_output"))

        return ScriptedChat(script=list(items))

    def test_usage_metadata_flows_through_a_runnable_chain(self):
        from langchain_core.runnables import RunnableLambda
        llm = self.make_chat([{"usage_metadata": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}])
        chain = RunnableLambda(lambda d: "prompt text") | llm
        r = m.RequestMetrics()
        with r.stage("e1") as st:
            chain.invoke({"x": 1}, config=st.config)
        e1 = r.finish()["stages"]["e1"]
        self.assertEqual((e1["input_tokens"], e1["output_tokens"], e1["token_source"], e1["finish_reason"]),
                         (12, 3, "usage_metadata", "stop"))

    def test_llm_output_token_usage_is_used_when_no_usage_metadata(self):
        from langchain_core.runnables import RunnableLambda
        llm = self.make_chat([{"llm_output": {"token_usage": {"prompt_tokens": 8, "completion_tokens": 2}}}])
        r = m.RequestMetrics()
        with r.stage("e1") as st:
            (RunnableLambda(lambda d: "p") | llm).invoke({}, config=st.config)
        e1 = r.finish()["stages"]["e1"]
        self.assertEqual((e1["input_tokens"], e1["output_tokens"], e1["token_source"]), (8, 2, "llm_output"))

    def test_no_usage_gives_null_not_zero(self):
        from langchain_core.runnables import RunnableLambda
        llm = self.make_chat([{}])
        r = m.RequestMetrics()
        with r.stage("e1") as st:
            (RunnableLambda(lambda d: "p") | llm).invoke({}, config=st.config)
        rec = r.finish()
        self.assertIsNone(rec["stages"]["e1"]["input_tokens"])
        self.assertEqual(rec["stages"]["e1"]["llm_calls"], 1)
        self.assertFalse(rec["totals"]["tokens_complete"])

    def test_handler_ignores_prompt_text(self):
        handler = m._UsageHandler(m.StageRecord("e1"))
        self.assertIsNone(handler.on_chat_model_start({}, [[SimpleNamespace(content=RESUME)]]))
        self.assertIsNone(handler.on_llm_start({}, [RESUME]))
        self.assertNotIn("SENTINEL", repr(vars(handler)))

    def test_concurrent_requests_get_their_own_tokens(self):
        from langchain_core.runnables import RunnableLambda
        n, barrier, out = 5, threading.Barrier(5), {}

        def work(i):
            llm = self.make_chat([{"usage_metadata": {"input_tokens": 100 + i, "output_tokens": i, "total_tokens": 100 + 2 * i}}])
            r = m.RequestMetrics()
            barrier.wait()
            with r.stage("e1") as st:
                (RunnableLambda(lambda d: "p") | llm).invoke({}, config=st.config)
            out[i] = r.finish()["stages"]["e1"]

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i in range(n):
            self.assertEqual((out[i]["input_tokens"], out[i]["output_tokens"]), (100 + i, i))


# ---------------------------------------------------------------------------------------------------------------
class SourceGuards(unittest.TestCase):
    @staticmethod
    def read(*parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_main_uses_metrics_without_touching_the_response_or_save(self):
        source = self.read("backend", "main.py")
        self.assertIn("metrics=request_metrics", source)
        self.assertEqual(source.count("install_retry_tracking()"), 1)
        self.assertEqual(source.count("request_metrics.emit(outcome)"), 1)
        self.assertRegex(source, r"finally:\s*\n\s*request_metrics\.emit\(outcome\)")
        self.assertIn('session.data = {', source)
        self.assertEqual(len(re.findall(r"clean_resume_analysis\(", source)), 2)   # definition + one call, unchanged
        call = source[source.index("resume_analysis = clean_resume_analysis("):]
        call = call[:call.index(")\n") + 1]
        self.assertNotIn("metrics", call)
        self.assertIn("        return resume_analysis\n", source.replace("\r\n", "\n"))

    def test_parser_does_not_import_the_metrics_module(self):
        source = self.read("src", "parser.py")
        self.assertNotRegex(source, r"(?m)^\s*(from|import)\s+(src\.)?metrics")
        self.assertEqual(source.count("with _metrics_stage(metrics,"), 5)
        for name in ("summary", "credentials", "e1", "e2", "e3"):
            self.assertIn('_metrics_stage(metrics, "%s")' % name, source)

    def test_metrics_module_has_no_file_output_and_no_forbidden_reads(self):
        source = self.read("src", "metrics.py")
        self.assertNotRegex(source, r"open\([^)]*['\"][wa]b?['\"]")
        for forbidden in (".env", "resume_text.txt", "METRICS_JSONL_PATH", "RotatingFileHandler", "FileHandler"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
