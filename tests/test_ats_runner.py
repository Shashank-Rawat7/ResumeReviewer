"""Offline tests for the ATS evaluation runner (eval/ats/run_ats_eval.py) and the shared dataset module.

Run from the project root:  python -m unittest tests.test_ats_runner -v
Every model call is a FAKE: no network, no provider import, no API key, no .env, no protected data file. Results go
to temporary folders; the real eval/ats/results folder is never written by these tests.
"""
import ast
import builtins
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATS_DIR = os.path.join(ROOT, "eval", "ats")
PROTECTED = ("resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
             "missing_skills.txt", "ats_optimized_bullets.txt")


def _load(name, filename):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(ATS_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def read_text(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def append_text(path, text):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write(text)


runner = _load("ats_runner", "run_ats_eval.py")
dataset = runner.dataset
checks = runner.checks

FIXED_TIME = runner.datetime(2026, 10, 27, 13, 0, 0, tzinfo=runner.timezone.utc)
FORBIDDEN_PREFIXES = ("langchain", "langchain_groq", "groq", "dotenv", "yaml", "src")


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.5
        return self.value


class Sleeps:
    def __init__(self):
        self.seconds = []

    def __call__(self, seconds):
        self.seconds.append(seconds)


class FakeStatusError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(FakeStatusError):
    def __init__(self, message="rate limited"):
        super().__init__(message, 429)


def output_for(case, extra_first=(), fenced=False):
    sections = [{"section": s["heading"], "bullets": list(s["bullets"])} for s in case["resume"]["sections"]]
    sections[0]["bullets"].extend(extra_first)
    text = json.dumps({"optimized_bullets": sections})
    return "```json\n" + text + "\n```" if fenced else text


class FakeClient:
    """script: {case_id: [item, ...]} consumed one item per ATTEMPT. An item is a response text (str), a
    ModelResponse, or an Exception instance to raise. Records every call it receives."""

    def __init__(self, cases, script, default=None):
        self.by_resume = {c["resume_text"]: c["id"] for c in cases}
        self.script = {k: list(v) for k, v in script.items()}
        self.default = default
        self.calls = []
        self.overrides_received = []
        self.info = {"model": "openai/gpt-oss-20b", "temperature": 0.2, "max_tokens": 8192}

    def __call__(self, job_description, resume_content):
        case_id = self.by_resume[resume_content]
        self.calls.append(case_id)
        queue = self.script.get(case_id)
        item = queue.pop(0) if queue else self.default
        if callable(item):
            item = item(case_id)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, runner.ModelResponse):
            return item
        return runner.ModelResponse(item, {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})

    def factory(self, max_tokens_override=None):
        self.overrides_received.append(max_tokens_override)
        info = dict(self.info)
        if max_tokens_override is not None and "effective_max_tokens" not in info:
            info["effective_max_tokens"] = max_tokens_override  # what the real builder reports
        return self, info


class Workspace(unittest.TestCase):
    """A temporary results root; helpers to plan, run and read artifacts."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = os.path.join(self.temp.name, "results")
        self.dev_cases = dataset.load_cases("dev")
        self.holdout_cases = dataset.load_cases("holdout")
        self.sleeps = Sleeps()
        self.clock = FakeClock()
        self.out_lines = []

    def out(self, line):
        self.out_lines.append(line)

    def main(self, argv, client=None, **kwargs):
        factory = (client.factory if client else (lambda **kw: (_ for _ in ()).throw(AssertionError("live client built"))))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = runner.main(argv, results_root=self.results, client_factory=factory, sleep=self.sleeps,
                               clock=self.clock, now=lambda: FIXED_TIME, out=self.out, **kwargs)
        return code, stderr.getvalue()

    def plan(self, run_id="dev1", split="dev", repeats=None, extra=()):
        argv = ["plan", "--split", split, "--run-id", run_id] + list(extra)
        if repeats:
            argv += ["--repeats", str(repeats)]
        code, err = self.main(argv)
        self.assertEqual(code, 0, err)
        return run_id

    def run_cli(self, run_id, client, calls, extra=()):
        return self.main(["run", "--plan", run_id, "--live", "--confirm-calls", str(calls)] + list(extra), client)

    def read(self, run_id, name):
        with open(os.path.join(self.results, run_id, name), encoding="utf-8", newline="") as handle:
            return handle.read()

    def jsonl(self, run_id, name):
        return [json.loads(line) for line in self.read(run_id, name).splitlines()]

    def completed_dev_run(self, run_id="dev1", repeats=1):
        self.plan(run_id, repeats=repeats)
        client = FakeClient(self.dev_cases, {}, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))
        code, err = self.run_cli(run_id, client, 9 * repeats)
        self.assertEqual(code, 0, err)
        return run_id


# ---------------------------------------------------------------------------------------------------------------
class DatasetAndFingerprints(unittest.TestCase):
    def test_splits_sizes_and_order(self):
        dev, hold, every = (dataset.load_cases(s) for s in ("dev", "holdout", "all"))
        self.assertEqual((len(dev), len(hold), len(every)), (9, 3, 12))
        self.assertEqual([c["id"] for c in every], sorted(c["id"] for c in every))
        self.assertEqual({c["id"] for c in hold}, {"c02_strong_ml_pipeline", "c06_absent_terraform_aws",
                                                    "c10_inflation_assisted"})

    def test_rendering_and_stored_text(self):
        for case in dataset.load_cases("all"):
            self.assertEqual(case["resume_text"], dataset.render_resume(case["resume"]))

    def test_vocabulary_loader_returns_only_terms(self):
        terms = dataset.load_vocabulary()
        self.assertNotIn("meta", terms)
        self.assertEqual(set(terms), set(json.loads(read_text(dataset.VOCAB_PATH))["terms"]))

    def test_fingerprints_stable_and_sensitive(self):
        self.assertEqual(dataset.dataset_fingerprint("dev"), dataset.dataset_fingerprint("dev"))
        with tempfile.TemporaryDirectory() as folder:
            copy_dir = os.path.join(folder, "cases")
            shutil.copytree(dataset.CASES_DIR, copy_dir)
            before = dataset.dataset_fingerprint("dev", copy_dir)["sha256"]
            with open(os.path.join(copy_dir, "c01_strong_task_api.json"), "a", encoding="utf-8") as handle:
                handle.write(" ")
            self.assertNotEqual(dataset.dataset_fingerprint("dev", copy_dir)["sha256"], before)


class Helpers(unittest.TestCase):
    def test_rate_always_shows_the_denominator(self):
        self.assertEqual(runner.rate(3, 18), "3/18 (16.7%)")
        self.assertEqual(runner.rate(0, 0), "0/0 (n/a)")

    def test_csv_safe_neutralises_formulas(self):
        for risky in ("=SUM(1)", "+1", "-2", "@cmd", "\tx", "\rx"):
            self.assertTrue(runner.csv_safe(risky).startswith("'"), risky)
        self.assertEqual(runner.csv_safe("Built a thing"), "Built a thing")
        self.assertEqual(runner.csv_safe(None), "")
        self.assertEqual(runner.csv_safe(5), "5")

    def test_sanitize_message_removes_keys_and_limits_length(self):
        text = runner.sanitize_message("failed with key gsk_AbCdEf123456 and Bearer abc.def-ghi and "
                                       "api_key=SECRETVALUE and " + "x" * 400)
        for leaked in ("gsk_AbCdEf123456", "abc.def-ghi", "SECRETVALUE"):
            self.assertNotIn(leaked, text)
        self.assertLessEqual(len(text), 200)
        self.assertNotIn("\n", runner.sanitize_message("a\nb"))

    def test_classify_exception(self):
        self.assertEqual(runner.classify_exception(RateLimitError()), ("rate_limit", 429))
        self.assertEqual(runner.classify_exception(FakeStatusError("x", 401)), ("auth", 401))
        self.assertEqual(runner.classify_exception(FakeStatusError("x", 503)), ("server_error", 503))
        self.assertEqual(runner.classify_exception(FakeStatusError("x", 400)), ("api_error", 400))
        self.assertEqual(runner.classify_exception(TimeoutError("t")), ("transport", None))
        self.assertEqual(runner.classify_exception(ConnectionError("c")), ("transport", None))
        self.assertEqual(runner.classify_exception(ValueError("v")), ("api_error", None))

    def test_extract_prompt_block_and_example_numbers(self):
        yaml_text = runner.read_source_file("prompts.yaml")
        block = runner.extract_prompt_block(yaml_text)
        self.assertIn("{job_description}", block)
        self.assertIn("{resume_content}", block)
        self.assertIn("JSON Output Format Example", block)
        self.assertTrue(checks.example_numbers_from_prompt(block))
        other = runner.extract_prompt_block("a: |\n  one\nb: |\n  two\n", "a")
        self.assertEqual(other, "one\n")
        with self.assertRaises(runner.Refused):
            runner.extract_prompt_block("a: |\n  one\n", "missing")

    def test_expected_model_name(self):
        self.assertEqual(runner.expected_model_name('X = 1\n    GROQ_MODEL = "m/1"\n'), "m/1")
        with self.assertRaises(runner.Refused):
            runner.expected_model_name("nothing")

    def test_usage_extraction(self):
        class A:
            usage_metadata = {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11, "extra": 1}

        class B:
            usage_metadata = None
            response_metadata = {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

        class C:
            usage_metadata = {"input_tokens": True}

        self.assertEqual(runner._usage_from_message(A()), {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11})
        self.assertEqual(runner._usage_from_message(B()), {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
        self.assertIsNone(runner._usage_from_message(C()))
        self.assertIsNone(runner._usage_from_message(object()))


class GuardedFiles(unittest.TestCase):
    def test_only_two_source_files_may_be_read(self):
        for name in PROTECTED + ("anything.txt", "recommender.py", "../.env"):
            with mock.patch.object(builtins, "open", side_effect=AssertionError("opened")):
                with self.assertRaises(runner.Refused, msg=name):
                    runner.read_source_file(name)

    def test_run_dir_refuses_paths_outside_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            run = runner.RunDir(folder, "r1")
            run.make()
            for bad in ("../x.txt", "/etc/x", "raw/../../x", "..\\x"):
                with self.assertRaises(runner.Refused, msg=bad):
                    run.write_text(bad, "x")
            for bad_id in ("../r", "a/b", "", ".hidden", "x" * 200):
                with self.assertRaises(runner.Refused, msg=bad_id):
                    runner.RunDir(folder, bad_id)
            self.assertEqual(os.listdir(folder), ["r1"])

    def test_run_dir_make_refuses_an_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            runner.RunDir(folder, "r1").make()
            with self.assertRaises(runner.Refused):
                runner.RunDir(folder, "r1").make()

    def test_writes_are_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            run = runner.RunDir(folder, "r1")
            run.make()
            run.write_text("a.txt", "v1")
            with mock.patch.object(os, "replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    run.write_text("a.txt", "v2")
            self.assertEqual(run.read_text("a.txt"), "v1")
            run.write_text("a.txt", "v3")
            self.assertEqual(run.read_text("a.txt"), "v3")
            self.assertEqual([n for n in os.listdir(run.path) if n.startswith(".tmp")], [])

    def test_text_is_written_verbatim_without_newline_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = runner.RunDir(folder, "r1")
            run.make()
            run.write_text("raw/x.txt", "a\r\nb\n\n")
            with open(os.path.join(run.path, "raw", "x.txt"), "rb") as handle:
                self.assertEqual(handle.read(), b"a\r\nb\n\n")


# ---------------------------------------------------------------------------------------------------------------
class Planning(Workspace):
    def test_default_plan_has_18_calls_and_excludes_holdouts(self):
        code, err = self.main(["plan", "--run-id", "p1"])
        self.assertEqual((code, err), (0, ""))
        plan = json.loads(self.read("p1", "plan.json"))
        self.assertEqual((plan["split"], plan["repeats"], plan["planned_calls"]), ("dev", 2, 18))
        self.assertEqual(len(plan["case_ids"]), 9)
        self.assertEqual(len(plan["schedule"]), 18)
        holdout_ids = {c["id"] for c in self.holdout_cases}
        self.assertFalse(holdout_ids & set(plan["case_ids"]))
        self.assertFalse(holdout_ids & {i["case_id"] for i in plan["schedule"]})
        meta = json.loads(self.read("p1", "metadata.json"))
        self.assertEqual((meta["status"], meta["live_calls_made"]), ("planned", 0))
        self.assertTrue(plan["token_estimates"]["ARE_ESTIMATES_NOT_MEASURED_USAGE"])
        self.assertEqual(plan["pacing"]["minimum_wall_seconds"], 17 * 40)
        self.assertEqual(set(plan["fingerprints"]), {"dataset", "vocabulary", "checker", "prompt"})
        self.assertEqual(sorted(os.listdir(os.path.join(self.results, "p1"))), ["metadata.json", "plan.json"])
        text = "\n".join(self.out_lines)
        self.assertIn("planned model calls: 18", text)
        self.assertIn("ESTIMATED tokens (not measured)", text)
        self.assertIn("--confirm-calls 18", text)

    def test_schedule_is_case_major_and_deterministic(self):
        self.plan("a")
        self.plan("b")
        first = json.loads(self.read("a", "plan.json"))
        second = json.loads(self.read("b", "plan.json"))
        self.assertEqual(first["schedule"], second["schedule"])
        self.assertEqual(first["schedule"][:3], [{"case_id": self.dev_cases[0]["id"], "repeat": 1},
                                                 {"case_id": self.dev_cases[0]["id"], "repeat": 2},
                                                 {"case_id": self.dev_cases[1]["id"], "repeat": 1}])
        self.assertEqual(first["fingerprints"], second["fingerprints"])

    def test_default_run_id_comes_from_split_and_time(self):
        code, _ = self.main(["plan"])
        self.assertEqual(code, 0)
        self.assertEqual(os.listdir(self.results), ["ats_dev_20261027T130000Z"])

    def test_all_split_is_not_offered(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.make_parser().parse_args(["plan", "--split", "all"])
        with self.assertRaises(runner.Refused):
            runner.build_plan("all", 2)

    def test_repeat_bounds(self):
        for bad in (0, 6, -1):
            code, err = self.main(["plan", "--run-id", "r", "--repeats", str(bad)])
            self.assertEqual(code, 2)
            self.assertIn("repeats", err)
        self.assertFalse(os.path.exists(self.results))

    def test_existing_run_id_is_refused_and_untouched(self):
        self.plan("p1")
        before = self.read("p1", "plan.json")
        code, err = self.main(["plan", "--run-id", "p1"])
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual(self.read("p1", "plan.json"), before)

    def test_holdout_plan_requires_explicit_split_and_a_frozen_dev_run(self):
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "h1"])
        self.assertEqual(code, 2)
        self.assertIn("--frozen-from", err)
        self.assertFalse(os.path.exists(os.path.join(self.results, "h1")))
        code, err = self.main(["plan", "--run-id", "d", "--frozen-from", "x"])
        self.assertEqual(code, 2)
        self.assertIn("holdout plans only", err)

    def test_holdout_plan_after_completed_dev_run(self):
        self.completed_dev_run("dev1")
        self.plan("h1", split="holdout", extra=("--frozen-from", "dev1"))
        plan = json.loads(self.read("h1", "plan.json"))
        self.assertEqual((plan["split"], plan["planned_calls"]), ("holdout", 6))
        self.assertEqual(set(plan["case_ids"]), {c["id"] for c in self.holdout_cases})
        self.assertEqual(plan["frozen_from"]["run_id"], "dev1")
        self.assertIn(runner.HOLDOUT_PHRASE, "\n".join(self.out_lines))

    def test_holdout_plan_refused_when_dev_run_is_not_complete(self):
        self.plan("dev1")
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "h1", "--frozen-from", "dev1"])
        self.assertEqual(code, 2)
        self.assertIn("not a completed development run", err)

    def test_holdout_plan_refused_when_configuration_changed_since_dev_run(self):
        self.completed_dev_run("dev1")
        with tempfile.TemporaryDirectory() as folder:
            changed = os.path.join(folder, "checks.py")
            shutil.copy(dataset.CHECKS_PATH, changed)
            with open(changed, "a", encoding="utf-8") as handle:
                handle.write("\n# changed\n")
            with mock.patch.object(dataset, "CHECKS_PATH", changed):
                code, err = self.main(["plan", "--split", "holdout", "--run-id", "h1", "--frozen-from", "dev1"])
        self.assertEqual(code, 2)
        self.assertIn("not frozen", err)

    def test_holdout_plan_refused_for_a_dev_run_that_is_actually_holdout(self):
        self.completed_dev_run("dev1")
        self.plan("h1", split="holdout", extra=("--frozen-from", "dev1"))
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "h2", "--frozen-from", "h1"])
        self.assertEqual(code, 2)


class DryRunIsolation(Workspace):
    def test_plan_works_with_network_provider_imports_and_environment_blocked(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in FORBIDDEN_PREFIXES:
                raise ImportError("blocked in dry-run test: " + name)
            return real_import(name, *args, **kwargs)

        def guard(key):
            # the standard library reads locale variables (argparse/gettext); the key and secrets must stay unread
            if any(word in str(key).upper() for word in ("KEY", "TOKEN", "GROQ", "SECRET", "PASSWORD")):
                raise AssertionError("secret-like environment variable accessed: " + str(key))

        class Forbidden(object):
            def get(self, key, default=None):
                guard(key)
                return default

            def __getitem__(self, key):
                guard(key)
                raise KeyError(key)

            def __contains__(self, key):
                guard(key)
                return False

            def __getattr__(self, name):
                raise AssertionError("environment attribute used: " + name)

        def forbidden_getenv(key, default=None):
            guard(key)
            return default

        def no_network(*a, **k):
            raise AssertionError("network used")

        before = set(sys.modules)
        with mock.patch.object(socket, "socket", no_network), \
                mock.patch.object(socket, "create_connection", no_network), \
                mock.patch.object(socket, "getaddrinfo", no_network), \
                mock.patch.object(builtins, "__import__", guarded_import), \
                mock.patch.object(os, "getenv", forbidden_getenv), \
                mock.patch.object(os, "environ", Forbidden()):
            code, err = self.main(["plan", "--run-id", "dry"])
        self.assertEqual((code, err), (0, ""))
        new = {m for m in set(sys.modules) - before if m.split(".")[0] in FORBIDDEN_PREFIXES}
        self.assertEqual(new, set())
        self.assertTrue(os.path.exists(os.path.join(self.results, "dry", "plan.json")))

    def test_summarize_and_label_sheet_do_not_build_a_client(self):
        self.completed_dev_run("dev1")
        for command in ("summarize", "label-sheet"):
            code, err = self.main([command, "dev1"])
            self.assertEqual((code, err), (0, ""), command)


# ---------------------------------------------------------------------------------------------------------------
class LiveSafeguards(Workspace):
    def assert_refused(self, argv, fragment, client=None):
        calls_before = len(client.calls) if client else 0
        code, err = self.main(argv, client)
        self.assertEqual(code, 2, err)
        self.assertIn(fragment, err)
        if client:
            self.assertEqual(len(client.calls), calls_before)

    def good_client(self):
        return FakeClient(self.dev_cases, {}, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))

    def test_refused_without_live_flag(self):
        self.plan("p1")
        client = self.good_client()
        self.assert_refused(["run", "--plan", "p1", "--confirm-calls", "18"], "--live", client)
        self.assertFalse(os.path.exists(os.path.join(self.results, "p1", "calls.jsonl")))

    def test_refused_without_confirm_calls(self):
        self.plan("p1")
        self.assert_refused(["run", "--plan", "p1", "--live"], "--confirm-calls is required", self.good_client())

    def test_refused_when_confirm_calls_does_not_match(self):
        self.plan("p1")
        for wrong in ("17", "19", "0", "9"):
            self.assert_refused(["run", "--plan", "p1", "--live", "--confirm-calls", wrong], "does not match", self.good_client())

    def test_refused_for_unknown_plan(self):
        self.assert_refused(["run", "--plan", "nope", "--live", "--confirm-calls", "18"], "no run directory")

    def test_refused_when_plan_json_is_tampered(self):
        self.plan("p1")
        path = os.path.join(self.results, "p1", "plan.json")
        plan = json.loads(read_text(path))
        plan["planned_calls"] = 19
        write_text(path, json.dumps(plan))
        self.assert_refused(["run", "--plan", "p1", "--live", "--confirm-calls", "19"], "stale", self.good_client())

    def test_refused_when_prompt_changed(self):
        self.plan("p1")
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "src"))
            for name in ("prompts.yaml", "parser.py"):
                shutil.copy(os.path.join(ROOT, "src", name), os.path.join(folder, "src", name))
            path = os.path.join(folder, "src", "prompts.yaml")
            text = read_text(path).replace("Never invent", "Never fabricate", 1)
            write_text(path, text)
            client = self.good_client()
            code, err = self.main(["run", "--plan", "p1", "--live", "--confirm-calls", "18"], client,
                                  source_root=folder)
        self.assertEqual(code, 2)
        self.assertIn("fingerprint:prompt", err)
        self.assertEqual(client.calls, [])

    def test_refused_when_checker_vocabulary_or_dataset_changed(self):
        self.plan("p1")
        with tempfile.TemporaryDirectory() as folder:
            changed = os.path.join(folder, "checks.py")
            shutil.copy(dataset.CHECKS_PATH, changed)
            append_text(changed, "\n# x\n")
            vocab = os.path.join(folder, "v.json")
            shutil.copy(dataset.VOCAB_PATH, vocab)
            append_text(vocab, "\n")
            cases_copy = os.path.join(folder, "cases")
            shutil.copytree(dataset.CASES_DIR, cases_copy)
            append_text(os.path.join(cases_copy, "c01_strong_task_api.json"), " ")
            for label, kwargs, patch_checker in (("fingerprint:checker", {}, True),
                                                 ("fingerprint:vocabulary", {"vocab_path": vocab}, False),
                                                 ("fingerprint:dataset", {"cases_dir": cases_copy}, False)):
                client = self.good_client()
                context = mock.patch.object(dataset, "CHECKS_PATH", changed) if patch_checker else contextlib.nullcontext()
                with context:
                    code, err = self.main(["run", "--plan", "p1", "--live", "--confirm-calls", "18"], client, **kwargs)
                self.assertEqual(code, 2, label)
                self.assertIn(label, err)
                self.assertEqual(client.calls, [], label)

    def test_a_holdout_file_change_does_not_stale_a_dev_plan(self):
        self.plan("p1")
        with tempfile.TemporaryDirectory() as folder:
            cases_copy = os.path.join(folder, "cases")
            shutil.copytree(dataset.CASES_DIR, cases_copy)
            append_text(os.path.join(cases_copy, "c02_strong_ml_pipeline.json"), " ")
            run = runner.open_run_dir(self.results, "p1")
            plan = run.read_json("plan.json")
            self.assertEqual(runner.check_plan_is_current(plan, run, self.results, cases_dir=cases_copy), [])

    def test_live_client_model_must_match_the_plan(self):
        self.plan("p1")
        client = self.good_client()
        client.info = {"model": "some/other-model"}
        self.assert_refused(["run", "--plan", "p1", "--live", "--confirm-calls", "18"], "plan expects", client)

    def test_missing_client_is_not_built_before_every_check_passes(self):
        self.plan("p1")
        with mock.patch.object(runner, "build_live_client", side_effect=AssertionError("built")):
            for argv in (["run", "--plan", "p1"], ["run", "--plan", "p1", "--live"],
                         ["run", "--plan", "p1", "--live", "--confirm-calls", "3"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.main(argv, results_root=self.results), 2)

    def test_a_plan_runs_only_once(self):
        self.plan("p1", repeats=1)
        client = self.good_client()
        self.assertEqual(self.run_cli("p1", client, 9)[0], 0)
        count = len(client.calls)
        self.assert_refused(["run", "--plan", "p1", "--live", "--confirm-calls", "9"], "already been executed", client)
        self.assertEqual(len(client.calls), count)

    def test_holdout_run_needs_the_extra_phrase(self):
        self.completed_dev_run("dev1")
        self.plan("h1", split="holdout", extra=("--frozen-from", "dev1"))
        client = FakeClient(self.holdout_cases, {}, default=lambda cid: output_for(next(c for c in self.holdout_cases if c["id"] == cid)))
        self.assert_refused(["run", "--plan", "h1", "--live", "--confirm-calls", "6"], "--confirm-holdout", client)
        self.assert_refused(["run", "--plan", "h1", "--live", "--confirm-calls", "6", "--confirm-holdout", "yes"],
                            "--confirm-holdout", client)
        self.assertFalse(os.path.exists(os.path.join(self.results, "h1", "calls.jsonl")))
        code, err = self.main(["run", "--plan", "h1", "--live", "--confirm-calls", "6", "--confirm-holdout",
                               runner.HOLDOUT_PHRASE], client)
        self.assertEqual((code, err), (0, ""))
        calls = self.jsonl("h1", "calls.jsonl")
        self.assertEqual(len(calls), 6)
        self.assertEqual({c["split"] for c in calls}, {"holdout"})
        self.assertEqual({c["case_id"] for c in calls}, {c["id"] for c in self.holdout_cases})

    def test_holdout_run_is_refused_if_the_dev_run_is_no_longer_valid(self):
        self.completed_dev_run("dev1")
        self.plan("h1", split="holdout", extra=("--frozen-from", "dev1"))
        meta_path = os.path.join(self.results, "dev1", "metadata.json")
        meta = json.loads(read_text(meta_path))
        meta["status"] = "aborted"
        write_text(meta_path, json.dumps(meta))
        client = FakeClient(self.holdout_cases, {})
        self.assert_refused(["run", "--plan", "h1", "--live", "--confirm-calls", "6", "--confirm-holdout",
                             runner.HOLDOUT_PHRASE], "stale", client)

    def test_dev_run_never_contains_holdout_cases(self):
        self.plan("p1", repeats=1)
        client = self.good_client()
        self.assertEqual(self.run_cli("p1", client, 9)[0], 0)
        holdout_ids = {c["id"] for c in self.holdout_cases}
        self.assertFalse(holdout_ids & set(client.calls))
        self.assertEqual(len(client.calls), 9)


# ---------------------------------------------------------------------------------------------------------------
class Execution(Workspace):
    """Nine dev cases, one repeat, with a scripted fixture covering every response and error kind."""

    def setUp(self):
        super().setUp()
        ids = [c["id"] for c in self.dev_cases]
        self.ids = ids
        case = {c["id"]: c for c in self.dev_cases}
        self.fenced_original = output_for(case[ids[1]], fenced=True)
        contaminated = output_for(case[ids[0]], extra_first=(
            "Built a REST API serving 5,000 requests per day.",
            "Added Redis caching to the task service."))
        self.script = {
            ids[0]: [contaminated],                                  # ok, with flagged bullets
            ids[1]: [self.fenced_original],                          # ok, fenced JSON
            ids[2]: ["this is not json"],                            # parse failure
            ids[3]: [""],                                            # empty response
            ids[4]: ['{"optimized_bullets": "oops"}'],               # valid JSON, wrong schema
            ids[5]: ['{"optimized_bullets": []}'],                   # schema invalid: empty list
            ids[6]: [FakeStatusError("bad request key gsk_SECRET123456789012345678901234", 400)],   # api_error, no retry
            ids[7]: [RateLimitError("429 Bearer tok_abc"), output_for(case[ids[7]])],                # retried, then ok
            ids[8]: [TimeoutError("t1"), TimeoutError("t2")],        # transport twice -> api_error
        }
        self.client = FakeClient(self.dev_cases, self.script)
        self.plan("e1", repeats=1)
        self.code, self.err = self.run_cli("e1", self.client, 9, ("--retry-backoff", "7"))

    def test_run_completed_and_exit_code(self):
        self.assertEqual((self.code, self.err), (0, ""))
        meta = json.loads(self.read("e1", "metadata.json"))
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["attempted_calls"], 9)
        self.assertEqual(meta["not_attempted"], [])
        self.assertEqual(meta["live_calls_made"], 11)  # 9 scheduled calls + 1 rate-limit retry + 1 transport retry

    def test_call_statuses(self):
        calls = {c["case_id"]: c for c in self.jsonl("e1", "calls.jsonl")}
        got = [calls[i]["status"] for i in self.ids]
        self.assertEqual(got, ["ok", "ok", "parse_failed", "empty_response", "schema_invalid", "schema_invalid",
                               "api_error", "ok", "api_error"])

    def test_calls_are_in_schedule_order_with_unique_ids(self):
        calls = self.jsonl("e1", "calls.jsonl")
        self.assertEqual([c["case_id"] for c in calls], self.ids)
        self.assertEqual(len({c["call_id"] for c in calls}), 9)
        self.assertEqual({c["split"] for c in calls}, {"dev"})
        self.assertEqual([c["repeat"] for c in calls], [1] * 9)

    def test_parse_failures_are_recorded_and_never_retried(self):
        self.assertEqual(self.client.calls.count(self.ids[2]), 1)
        self.assertEqual(self.client.calls.count(self.ids[3]), 1)
        self.assertEqual(self.client.calls.count(self.ids[4]), 1)
        calls = {c["case_id"]: c for c in self.jsonl("e1", "calls.jsonl")}
        self.assertEqual(len(calls[self.ids[2]]["attempts"]), 1)
        self.assertTrue(calls[self.ids[2]]["production_would_500"])
        self.assertTrue(calls[self.ids[3]]["production_would_500"])
        self.assertFalse(calls[self.ids[4]]["production_would_500"])
        self.assertEqual(calls[self.ids[4]]["schema_status"], "wrong_shape")
        self.assertEqual(calls[self.ids[5]]["schema_status"], "empty")

    def test_raw_responses_are_saved_verbatim(self):
        raw_dir = os.path.join(self.results, "e1", "raw")
        self.assertEqual(sorted(os.listdir(raw_dir)), sorted(f"{i}__r01.txt" for i in self.ids[:6] + [self.ids[7]]))
        self.assertEqual(self.read("e1", f"raw/{self.ids[1]}__r01.txt"), self.fenced_original)
        self.assertEqual(self.read("e1", f"raw/{self.ids[2]}__r01.txt"), "this is not json")
        self.assertEqual(self.read("e1", f"raw/{self.ids[3]}__r01.txt"), "")
        calls = {c["case_id"]: c for c in self.jsonl("e1", "calls.jsonl")}
        self.assertEqual(calls[self.ids[1]]["raw_path"], f"raw/{self.ids[1]}__r01.txt")
        self.assertIsNone(calls[self.ids[6]]["raw_path"])

    def test_api_failures_and_retry_accounting(self):
        calls = {c["case_id"]: c for c in self.jsonl("e1", "calls.jsonl")}
        bad = calls[self.ids[6]]
        self.assertEqual((bad["status"], bad["error_kind"], bad["http_status"], len(bad["attempts"])),
                         ("api_error", "api_error", 400, 1))
        retried = calls[self.ids[7]]
        self.assertEqual([a["outcome"] for a in retried["attempts"]], ["rate_limit", "response"])
        self.assertEqual(retried["status"], "ok")
        self.assertTrue(retried["response_received"])
        transport = calls[self.ids[8]]
        self.assertEqual([a["outcome"] for a in transport["attempts"]], ["transport", "transport"])
        self.assertEqual((transport["status"], transport["error_kind"]), ("api_error", "transport"))
        self.assertFalse(transport["response_received"])
        self.assertEqual(self.sleeps.seconds.count(7), 2)  # one backoff before each retry
        self.assertEqual(self.client.calls.count(self.ids[7]), 2)
        self.assertEqual(self.client.calls.count(self.ids[8]), 2)

    def test_pacing_between_scheduled_calls(self):
        self.assertEqual(self.sleeps.seconds.count(40), 8)  # 9 calls -> 8 gaps

    def test_no_secret_reaches_any_artifact(self):
        for folder, _, names in os.walk(os.path.join(self.results, "e1")):
            for name in names:
                text = read_text(os.path.join(folder, name))
                for secret in ("gsk_SECRET", "tok_abc"):
                    self.assertNotIn(secret, text, name)
        bad = {c["case_id"]: c for c in self.jsonl("e1", "calls.jsonl")}[self.ids[6]]
        self.assertIn("[REDACTED]", bad["error_message"])

    def test_usage_and_latency_are_recorded(self):
        first = self.jsonl("e1", "calls.jsonl")[0]
        self.assertEqual(first["usage"], {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        self.assertIsNotNone(first["latency_s"])

    def test_evaluations_only_for_validated_outputs(self):
        evaluations = self.jsonl("e1", "evaluations.jsonl")
        self.assertEqual([e["case_id"] for e in evaluations], [self.ids[0], self.ids[1], self.ids[7]])
        self.assertEqual(evaluations[0]["evaluation"]["bullets_total"], 11)
        self.assertEqual(evaluations[0]["evaluation"]["schema"]["status"], "ok")

    def test_checker_findings_use_the_case_configuration(self):
        first = self.jsonl("e1", "evaluations.jsonl")[0]["evaluation"]
        self.assertEqual(first["numbers"]["bullets_with_invented"], 1)
        self.assertEqual(first["absent_skills"]["bullets_with_leak"], 1)
        self.assertEqual(first["technologies"]["bullets_with_unsupported"], 1)

    def test_summary_denominator_arithmetic(self):
        code, err = self.main(["summarize", "e1"])
        self.assertEqual((code, err), (0, ""))
        summary = json.loads(self.read("e1", "summary.json"))
        counts = summary["counts"]
        self.assertEqual(counts["planned_calls"], 9)
        self.assertEqual(counts["attempted_calls"], 9)
        self.assertEqual(counts["not_attempted_calls"], 0)
        self.assertEqual(counts["responses_received"], 7)
        self.assertEqual(counts["api_failures"], 2)
        self.assertEqual(counts["empty_responses"], 1)
        self.assertEqual(counts["json_parse_failures"], 1)
        self.assertEqual(counts["schema_invalid_responses"], 2)
        self.assertEqual(counts["outputs_eligible_for_bullet_checks"], 3)
        self.assertEqual(counts["bullets_checked"], 11 + 5 + 7)
        self.assertEqual(counts["model_call_attempts_including_retries"], 11)
        rates = summary["rates"]
        self.assertEqual(rates["production_failures_over_responses_received"], "2/7 (28.6%)")
        self.assertEqual(rates["api_failures_over_attempted_calls"], "2/9 (22.2%)")
        self.assertEqual(rates["schema_invalid_over_responses_received"], "2/7 (28.6%)")
        self.assertEqual(rates["outputs_eligible_over_responses_received"], "3/7 (42.9%)")

    def test_d3_and_d4_are_reported_separately_never_summed(self):
        self.main(["summarize", "e1"])
        checkers = json.loads(self.read("e1", "summary.json"))["checkers"]
        self.assertEqual(checkers["D2_invented_numbers"]["rate"], "1/23 (4.3%)")
        self.assertEqual(checkers["D3_absent_skills"]["rate"], "1/23 (4.3%)")
        self.assertEqual(checkers["D4_technologies_advisory"]["rate"], "1/23 (4.3%)")  # the SAME bullet as D3
        self.assertNotIn("combined", json.dumps(checkers).lower())
        report = self.read("e1", "report.md")
        self.assertIn("D3 declared absent skills | 1/23", report)
        self.assertIn("D4 technologies not in resume (advisory) | 1/23", report)
        self.assertIn("must not be added", report)

    def test_report_states_limits_and_split(self):
        self.main(["summarize", "e1"])
        report = self.read("e1", "report.md")
        for fragment in ("DEVELOPMENT", "synthetic", "selection effect", "Production failures (would be HTTP 500) | 2/7",
                         "API failures | 2/9", "Bullets reviewed: 0/23"):
            self.assertIn(fragment, report)
        self.assertNotIn("HOLDOUT", report)

    def test_summaries_are_deterministic(self):
        self.main(["summarize", "e1"])
        first = (self.read("e1", "summary.json"), self.read("e1", "report.md"))
        self.main(["summarize", "e1"])
        self.assertEqual(first, (self.read("e1", "summary.json"), self.read("e1", "report.md")))

    def test_label_sheet_rows_columns_and_blank_human_fields(self):
        text = self.read("e1", "bullets.csv")
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(list(rows[0].keys()), list(runner.CSV_COLUMNS))
        self.assertEqual(len(rows), 23)
        self.assertEqual([r["case_id"] for r in rows[:11]], [self.ids[0]] * 11)
        for row in rows:
            for column in runner.HUMAN_COLUMNS:
                self.assertEqual(row[column], "")
        flagged = [r for r in rows if r["checker_findings"]]
        self.assertTrue(any("D2 invented numbers: 5,000" in r["checker_findings"] for r in flagged))
        self.assertTrue(any("D3 absent skills: Redis" in r["checker_findings"] for r in flagged))
        redis = next(r for r in rows if "Redis caching" in r["rewritten_bullet"])
        self.assertEqual(redis["d3_absent_skill_leaks"], "Redis")
        self.assertEqual(redis["d4_unsupported_technologies"], "Redis")

    def test_label_sheet_traces_to_source_sections_without_pretending(self):
        rows = list(csv.DictReader(io.StringIO(self.read("e1", "bullets.csv"))))
        case = self.dev_cases[0]
        first = rows[0]
        self.assertEqual(first["matched_source_heading"], case["resume"]["sections"][0]["heading"])
        self.assertEqual(first["source_bullets"], " || ".join(case["resume"]["sections"][0]["bullets"]))
        unmatched = runner._matched_heading("Totally unrelated heading", case)
        self.assertEqual(unmatched, "")

    def test_label_sheet_refuses_to_overwrite_human_labels_but_regenerates_blank_sheets(self):
        path = os.path.join(self.results, "e1", "bullets.csv")
        text = read_text(path)
        self.assertEqual(self.main(["label-sheet", "e1"])[0], 0)  # blank sheet: regenerated, identical
        self.assertEqual(read_text(path), text)
        rows = list(csv.DictReader(io.StringIO(text)))
        for row in rows[:3]:
            row["supported_by_source_evidence"] = "S"
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(runner.CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        write_text(path, out.getvalue())
        code, err = self.main(["label-sheet", "e1"])
        self.assertEqual(code, 2)
        self.assertIn("human labels", err)
        self.assertEqual(read_text(path), out.getvalue())
        self.main(["summarize", "e1"])
        summary = json.loads(self.read("e1", "summary.json"))
        self.assertEqual(summary["counts"]["bullets_reviewed_by_human"], 3)
        self.assertEqual(summary["rates"]["bullets_reviewed_over_bullets_checked"], "3/23 (13.0%)")

    def test_per_case_and_repeat_consistency_are_raw_counts(self):
        self.main(["summarize", "e1"])
        summary = json.loads(self.read("e1", "summary.json"))
        self.assertEqual(summary["repeat_consistency_raw_counts"]["cases_where_all_repeats_ok"], "3/9")
        first = summary["per_case"][0]
        self.assertEqual(first["d3_flagged_by_repeat"], {"1": "1/11"})
        self.assertEqual(first["statuses_by_repeat"], {"1": "ok"})


class ExecutionRules(Workspace):
    def fresh(self, script, repeats=1, **options):
        self.plan("x", repeats=repeats)
        client = FakeClient(self.dev_cases, script, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))
        extra = []
        for key, value in options.items():
            extra += [f"--{key.replace('_', '-')}", str(value)]
        code, err = self.run_cli("x", client, 9 * repeats, extra)
        return client, code, err

    def test_abort_after_consecutive_api_failures(self):
        ids = [c["id"] for c in self.dev_cases]
        errors = {i: [FakeStatusError("down", 400)] for i in ids[:3]}
        client, code, err = self.fresh(errors, abort_after=3)
        self.assertEqual(code, 0)
        meta = json.loads(self.read("x", "metadata.json"))
        self.assertEqual((meta["status"], meta["attempted_calls"]), ("aborted", 3))
        self.assertIn("3 consecutive", meta["aborted_reason"])
        self.assertEqual([i["case_id"] for i in meta["not_attempted"]], ids[3:])
        self.assertEqual(len(self.jsonl("x", "calls.jsonl")), 3)
        self.main(["summarize", "x"])
        counts = json.loads(self.read("x", "summary.json"))["counts"]
        self.assertEqual((counts["planned_calls"], counts["attempted_calls"], counts["not_attempted_calls"]), (9, 3, 6))

    def test_non_consecutive_failures_do_not_abort(self):
        ids = [c["id"] for c in self.dev_cases]
        errors = {ids[0]: [FakeStatusError("x", 400)], ids[2]: [FakeStatusError("x", 400)], ids[4]: [FakeStatusError("x", 400)]}
        self.fresh(errors, abort_after=2)
        self.assertEqual(json.loads(self.read("x", "metadata.json"))["status"], "complete")
        self.assertEqual(len(self.jsonl("x", "calls.jsonl")), 9)

    def test_authentication_errors_abort_immediately_without_retry(self):
        ids = [c["id"] for c in self.dev_cases]
        client, _, _ = self.fresh({ids[0]: [FakeStatusError("nope", 401)]}, max_retries=3)
        meta = json.loads(self.read("x", "metadata.json"))
        self.assertEqual(meta["status"], "aborted")
        self.assertEqual(meta["aborted_reason"], "authentication error")
        self.assertEqual(client.calls, [ids[0]])

    def test_retries_are_bounded_and_configurable(self):
        ids = [c["id"] for c in self.dev_cases]
        client, _, _ = self.fresh({ids[0]: [RateLimitError()] * 10}, max_retries=0)
        self.assertEqual(client.calls.count(ids[0]), 1)
        self.setUp()
        client, _, _ = self.fresh({ids[0]: [RateLimitError()] * 10}, max_retries=3, abort_after=9)
        self.assertEqual(client.calls.count(ids[0]), 4)
        first = self.jsonl("x", "calls.jsonl")[0]
        self.assertEqual(len(first["attempts"]), 4)
        self.assertEqual(first["error_kind"], "rate_limit")

    def test_server_errors_are_recorded_distinctly_and_not_retried(self):
        ids = [c["id"] for c in self.dev_cases]
        client, _, _ = self.fresh({ids[0]: [FakeStatusError("boom", 503)]}, max_retries=3)
        first = self.jsonl("x", "calls.jsonl")[0]
        self.assertEqual((first["error_kind"], first["http_status"], len(first["attempts"])), ("server_error", 503, 1))

    def test_raw_response_is_saved_before_parsing(self):
        ids = [c["id"] for c in self.dev_cases]
        seen = {}

        def exploding(*args, **kwargs):
            seen["raw_existed"] = os.path.exists(os.path.join(self.results, "x", "raw", f"{ids[0]}__r01.txt"))
            raise RuntimeError("checker crashed")

        with mock.patch.object(checks, "evaluate_output", exploding):
            self.fresh({})
        self.assertTrue(seen["raw_existed"])
        first = self.jsonl("x", "calls.jsonl")[0]
        self.assertEqual((first["status"], first["evaluation_error"]), ("evaluation_error", "RuntimeError"))
        self.assertTrue(first["response_received"])
        self.assertTrue(os.path.exists(os.path.join(self.results, "x", "raw", f"{ids[0]}__r01.txt")))
        self.assertEqual(self.jsonl("x", "evaluations.jsonl") if os.path.exists(os.path.join(self.results, "x", "evaluations.jsonl")) else [], [])
        self.main(["summarize", "x"])
        self.assertEqual(json.loads(self.read("x", "summary.json"))["counts"]["checker_errors_response_saved"], 9)

    def test_non_string_content_is_recorded_as_a_parse_failure(self):
        ids = [c["id"] for c in self.dev_cases]
        self.fresh({ids[0]: [runner.ModelResponse([{"type": "text", "text": "x"}])]})
        first = self.jsonl("x", "calls.jsonl")[0]
        self.assertEqual(first["status"], "parse_failed")
        self.assertTrue(first["production_would_500"])
        self.assertIn("type", self.read("x", f"raw/{ids[0]}__r01.txt"))

    def test_whitespace_only_response_is_an_empty_response(self):
        ids = [c["id"] for c in self.dev_cases]
        self.fresh({ids[0]: ["  \n  "]})
        first = self.jsonl("x", "calls.jsonl")[0]
        self.assertEqual(first["status"], "empty_response")
        self.assertEqual(self.read("x", f"raw/{ids[0]}__r01.txt"), "  \n  ")

    def test_two_repeats_each_get_their_own_raw_file(self):
        self.fresh({}, repeats=2)
        names = sorted(os.listdir(os.path.join(self.results, "x", "raw")))
        self.assertEqual(len(names), 18)
        self.assertIn(f"{self.dev_cases[0]['id']}__r01.txt", names)
        self.assertIn(f"{self.dev_cases[0]['id']}__r02.txt", names)

    def test_identical_runs_produce_identical_artifacts(self):
        digests = []
        for run_id in ("x", "y"):
            self.clock = FakeClock()
            self.plan(run_id, repeats=1)
            client = FakeClient(self.dev_cases, {}, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))
            self.assertEqual(self.run_cli(run_id, client, 9)[0], 0)
            digests.append({n: self.read(run_id, n) for n in ("calls.jsonl", "evaluations.jsonl", "bullets.csv")})
        for name in digests[0]:
            self.assertEqual(digests[0][name].replace('"x"', '"y"'), digests[1][name], name)

    def test_formula_like_bullets_are_neutralised_in_the_csv(self):
        ids = [c["id"] for c in self.dev_cases]
        case = self.dev_cases[0]
        risky = ["=SUM(A1:A9)", "+1 trick", "-2 trick", "@HYPERLINK(1)"]
        self.fresh({ids[0]: [output_for(case, extra_first=risky)]})
        rows = list(csv.DictReader(io.StringIO(self.read("x", "bullets.csv"))))
        cells = [r["rewritten_bullet"] for r in rows if r["case_id"] == ids[0]]
        for original in risky:
            self.assertIn("'" + original, cells)
            self.assertNotIn(original, cells)
        for line in self.read("x", "bullets.csv").splitlines()[1:]:
            self.assertNotRegex(line, r"(^|,)[=+@]")  # no unquoted formula starts

    def test_csv_neutralisation_covers_every_cell(self):
        row = {column: "=1" for column in runner.CSV_COLUMNS}
        text = runner.rows_to_csv([row])
        parsed = list(csv.reader(io.StringIO(text)))[1]
        self.assertTrue(all(cell == "'=1" for cell in parsed))


# ---------------------------------------------------------------------------------------------------------------
class Restrictions(Workspace):
    def test_runner_source_never_references_parse_resume_or_imports_providers_at_top_level(self):
        source = read_text(os.path.join(ATS_DIR, "run_ats_eval.py"))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "parse_resume")
            if isinstance(node, ast.Attribute):
                self.assertNotEqual(node.attr, "parse_resume")
        allowed = {"argparse", "ast", "csv", "io", "importlib", "json", "math", "os", "re", "statistics", "sys",
                   "textwrap", "time", "datetime"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue({a.name.split(".")[0] for a in node.names} <= allowed, ast.dump(node))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], allowed, ast.dump(node))
        lazy = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "src.parser"]
        self.assertEqual(len(lazy), 1)
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_live_client")
        self.assertIn(lazy[0], list(ast.walk(builder)))

    def test_dataset_module_is_standard_library_only(self):
        tree = ast.parse(read_text(os.path.join(ATS_DIR, "dataset.py")))
        allowed = {"copy", "glob", "hashlib", "importlib", "json", "os", "re", "sys"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue({a.name.split(".")[0] for a in node.names} <= allowed)
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], allowed)

    def test_whole_workflow_never_opens_protected_files_and_writes_only_inside_results(self):
        opened = []
        real_open = builtins.open

        def spying_open(file, mode="r", *args, **kwargs):
            opened.append((os.path.abspath(os.fspath(file)) if not isinstance(file, int) else str(file), mode))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch.object(builtins, "open", spying_open), mock.patch.object(io, "open", spying_open):
            self.completed_dev_run("w1")
            self.main(["summarize", "w1"])
            self.main(["label-sheet", "w1"])
            self.plan("w2", split="holdout", extra=("--frozen-from", "w1"))
        names = [os.path.basename(p) for p, _ in opened]
        for protected in PROTECTED:
            self.assertNotIn(protected, names)
        src_reads = {os.path.basename(p) for p, _ in opened if os.sep + "src" + os.sep in p}
        self.assertLessEqual(src_reads, {"prompts.yaml", "parser.py"})
        self.assertFalse(any(n == ".env" for n in names))
        results = os.path.realpath(self.results)
        temp = os.path.realpath(self.temp.name)
        for path, mode in opened:
            if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
                real = os.path.realpath(path)
                self.assertTrue(real.startswith(results + os.sep) or real.startswith(temp + os.sep) or "__pycache__" in real,
                                f"write outside the results folder: {path}")

    def test_real_results_folder_is_untouched_by_these_tests(self):
        real_results = os.path.join(ATS_DIR, "results")
        before = sorted(os.listdir(real_results)) if os.path.isdir(real_results) else None
        self.completed_dev_run("z1")
        after = sorted(os.listdir(real_results)) if os.path.isdir(real_results) else None
        self.assertEqual(before, after)

    def test_planning_makes_no_model_calls(self):
        with mock.patch.object(runner, "build_live_client", side_effect=AssertionError("live client built")):
            code = runner.main(["plan", "--run-id", "np"], results_root=self.results, now=lambda: FIXED_TIME,
                               out=self.out)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.read("np", "metadata.json"))["live_calls_made"], 0)

    def test_protected_file_names_cannot_be_read_even_with_a_valid_looking_root(self):
        for name in PROTECTED:
            with self.assertRaises(runner.Refused):
                runner.read_source_file(name, ROOT)


# ---------------------------------------------------------------------------------------------------------------
# Hardening: reproducibility metadata, plan invalidation, provider retry discovery, live-setting verification
# ---------------------------------------------------------------------------------------------------------------
ENV = {"packages": {"langchain": "0.3.1", "langchain-core": "0.3.2", "langchain-groq": "0.2.0", "groq": "0.11.0"},
       "provider_retry": {"langchain_groq_default_max_retries": 2, "groq_sdk_default_max_retries": 2,
                          "method": "test fixture"},
       "python": "3.10.0"}


def changed_env(**changes):
    env = json.loads(json.dumps(ENV))
    for key, value in changes.items():
        if key == "package":
            env["packages"]["langchain-groq"] = value
        elif key == "retry":
            env["provider_retry"]["langchain_groq_default_max_retries"] = value
    return env


class SourceCopy(object):
    """A temporary root holding copies of ONLY src/prompts.yaml and src/parser.py, optionally edited."""

    def __init__(self, test, replacements=(), append_to_parser=""):
        self.temp = tempfile.TemporaryDirectory()
        test.addCleanup(self.temp.cleanup)
        os.makedirs(os.path.join(self.temp.name, "src"))
        for name in ("prompts.yaml", "parser.py"):
            shutil.copy(os.path.join(ROOT, "src", name), os.path.join(self.temp.name, "src", name))
        path = os.path.join(self.temp.name, "src", "parser.py")
        text = read_text(path)
        for old, new in replacements:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        write_text(path, text + append_to_parser)
        self.root = self.temp.name


class HardeningMetadata(Workspace):
    def test_plan_records_code_fingerprints_settings_packages_and_retry(self):
        code, err = self.main(["plan", "--run-id", "m1"], environment=ENV)
        self.assertEqual((code, err), (0, ""))
        plan = json.loads(self.read("m1", "plan.json"))
        self.assertEqual(plan["plan_schema_version"], 3)
        code_fp = plan["code_fingerprints"]
        self.assertEqual(code_fp["runner"], dataset.sha256_file(runner.RUNNER_PATH))
        self.assertEqual(code_fp["dataset_module"], dataset.sha256_file(runner.DATASET_MODULE_PATH))
        self.assertEqual(code_fp["parser_py"], dataset.sha256_bytes(runner.read_source_file("parser.py").encode("utf-8")))
        for value in code_fp.values():
            self.assertRegex(value, r"^[0-9a-f]{64}$")
        settings = plan["model_settings"]
        self.assertEqual(settings["model"], plan["model_expected"])
        for key in ("temperature", "max_tokens", "reasoning_effort"):
            self.assertNotEqual(settings[key], "unknown", key)
        self.assertEqual(plan["packages"], ENV["packages"])
        self.assertEqual(plan["provider_retry"], ENV["provider_retry"])
        self.assertEqual(plan["python_version"], "3.10.0")
        self.assertIn("NOT guarantee", plan["call_count_meaning"])
        meta = json.loads(self.read("m1", "metadata.json"))
        for key in ("code_fingerprints", "model_settings", "packages", "provider_retry", "plan_schema_version"):
            self.assertEqual(meta[key], plan[key], key)
        text = "\n".join(self.out_lines)
        self.assertIn("hidden provider retries", text)
        self.assertIn("langchain-groq=2", text)

    def test_exact_settings_from_a_known_parser_source(self):
        source = SourceCopy(self, replacements=[("temperature=0.2", "temperature=0.7")])
        code, _ = self.main(["plan", "--run-id", "m2"], environment=ENV, source_root=source.root)
        self.assertEqual(code, 0)
        settings = json.loads(self.read("m2", "plan.json"))["model_settings"]
        self.assertEqual((settings["temperature"], settings["max_tokens"], settings["reasoning_effort"]),
                         (0.7, 8192, "low"))
        self.assertEqual(settings["max_retries"], runner.NOT_SET_IN_PARSER)

    def test_no_secret_or_environment_contents_reach_the_artifacts(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_TESTSECRET1234567890", "OTHER_SECRET": "hunter2"}):
            self.main(["plan", "--run-id", "m3"])
        for name in ("plan.json", "metadata.json"):
            text = self.read("m3", name)
            self.assertNotIn("gsk_TESTSECRET", text)
            self.assertNotIn("hunter2", text)
            self.assertNotIn("GROQ_API_KEY", text)

    def test_default_environment_discovery_is_offline_and_imports_nothing(self):
        before = set(sys.modules)
        env = runner.discover_environment()
        new = {m for m in set(sys.modules) - before if m.split(".")[0] in FORBIDDEN_PREFIXES}
        self.assertEqual(new, set())
        self.assertEqual(set(env), {"packages", "provider_retry", "python"})
        self.assertEqual(set(env["packages"]), set(runner.PROVIDER_PACKAGES))


class HardeningInvalidation(Workspace):
    def good_client(self):
        return FakeClient(self.dev_cases, {}, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))

    def refused_labels(self, plan_kwargs, run_kwargs, plan_env=ENV, run_env=ENV):
        code, err = self.main(["plan", "--run-id", "h"], environment=plan_env, **plan_kwargs)
        self.assertEqual(code, 0, err)
        client = self.good_client()
        code, err = self.main(["run", "--plan", "h", "--live", "--confirm-calls", "18"], client,
                              environment=run_env, **run_kwargs)
        self.assertEqual(client.calls, [])
        return code, err

    def test_control_unchanged_configuration_is_accepted(self):
        self.main(["plan", "--run-id", "h"], environment=ENV)
        run = runner.open_run_dir(self.results, "h")
        self.assertEqual(runner.check_plan_is_current(run.read_json("plan.json"), run, self.results, environment=ENV), [])

    def test_changed_runner_code_rejects_the_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            changed = os.path.join(folder, "run_ats_eval.py")
            shutil.copy(runner.RUNNER_PATH, changed)
            append_text(changed, "\n# edit\n")
            self.main(["plan", "--run-id", "h"], environment=ENV)
            with mock.patch.object(runner, "RUNNER_PATH", changed):
                code, err = self.main(["run", "--plan", "h", "--live", "--confirm-calls", "18"], self.good_client(),
                                      environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("code:runner", err)

    def test_changed_dataset_module_rejects_the_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            changed = os.path.join(folder, "dataset.py")
            shutil.copy(runner.DATASET_MODULE_PATH, changed)
            append_text(changed, "\n# edit\n")
            self.main(["plan", "--run-id", "h"], environment=ENV)
            with mock.patch.object(runner, "DATASET_MODULE_PATH", changed):
                code, err = self.main(["run", "--plan", "h", "--live", "--confirm-calls", "18"], self.good_client(),
                                      environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("code:dataset_module", err)

    def test_parser_comment_change_rejects_only_through_the_parser_fingerprint(self):
        source = SourceCopy(self, append_to_parser="\n# harmless comment\n")
        code, err = self.refused_labels({}, {"source_root": source.root})
        self.assertEqual(code, 2)
        self.assertIn("code:parser_py", err)
        self.assertNotIn("model_settings", err)

    def test_changed_model_settings_reject_the_plan(self):
        for old, new in (("temperature=0.2", "temperature=0.9"), ("max_tokens=8192", "max_tokens=4096"),
                         ('reasoning_effort="low"', 'reasoning_effort="high"')):
            with self.subTest(change=new):
                self.setUp()
                source = SourceCopy(self, replacements=[(old, new)])
                code, err = self.refused_labels({}, {"source_root": source.root})
                self.assertEqual(code, 2)
                self.assertIn("model_settings", err)
                self.assertIn("code:parser_py", err)

    def test_changed_package_versions_reject_the_plan(self):
        code, err = self.refused_labels({}, {}, run_env=changed_env(package="0.9.9"))
        self.assertEqual(code, 2)
        self.assertIn("packages", err)

    def test_changed_provider_retry_configuration_rejects_the_plan(self):
        code, err = self.refused_labels({}, {}, run_env=changed_env(retry=0))
        self.assertEqual(code, 2)
        self.assertIn("provider_retry", err)

    def test_plan_from_the_old_format_is_rejected(self):
        self.main(["plan", "--run-id", "h"], environment=ENV)
        path = os.path.join(self.results, "h", "plan.json")
        plan = json.loads(read_text(path))
        for key in ("plan_schema_version", "code_fingerprints", "model_settings", "packages", "provider_retry"):
            plan.pop(key)
        write_text(path, json.dumps(plan))
        client = self.good_client()
        code, err = self.main(["run", "--plan", "h", "--live", "--confirm-calls", "18"], client, environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("plan_schema_version", err)
        self.assertEqual(client.calls, [])

    def test_holdout_plan_requires_unchanged_model_settings_and_environment(self):
        self.plan("dev1", repeats=1)
        client = self.good_client()
        self.assertEqual(self.main(["run", "--plan", "dev1", "--live", "--confirm-calls", "9"], client)[0], 0)
        source = SourceCopy(self, replacements=[("temperature=0.2", "temperature=0.9")])
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold", "--frozen-from", "dev1"],
                              source_root=source.root)
        self.assertEqual(code, 2)
        self.assertIn("not frozen", err)
        self.assertIn("model_settings", err)
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold2", "--frozen-from", "dev1"],
                              environment=changed_env(package="9.9.9"))
        self.assertEqual(code, 2)
        self.assertIn("packages", err)
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold3", "--frozen-from", "dev1"])
        self.assertEqual((code, err), (0, ""))


class SettingsDiscovery(unittest.TestCase):
    PARSER = ('GROQ_MODEL = "m/x"\nllm = ChatGroq(model=GROQ_MODEL, temperature=0.3, max_tokens=100, '
              'reasoning_effort="medium")\n')

    def test_parse_llm_settings_reads_literals(self):
        got = runner.parse_llm_settings(self.PARSER)
        self.assertEqual((got["model"], got["temperature"], got["max_tokens"], got["reasoning_effort"]),
                         ("m/x", 0.3, 100, "medium"))
        self.assertEqual(got["max_retries"], runner.NOT_SET_IN_PARSER)
        self.assertEqual(got["request_timeout"], runner.NOT_SET_IN_PARSER)

    def test_explicit_retry_and_timeout_settings_are_recorded(self):
        got = runner.parse_llm_settings('GROQ_MODEL = "m"\nChatGroq(max_retries=5, timeout=30)\n')
        self.assertEqual((got["max_retries"], got["request_timeout"]), (5, 30))

    def test_missing_or_non_literal_settings_are_unknown_not_guessed(self):
        got = runner.parse_llm_settings('GROQ_MODEL = "m"\nChatGroq(temperature=TEMP, max_tokens=int("3"))\n')
        self.assertEqual((got["temperature"], got["max_tokens"], got["reasoning_effort"]),
                         ("unknown", "unknown", "unknown"))
        none_at_all = runner.parse_llm_settings('GROQ_MODEL = "m"\nx = 1\n')
        self.assertEqual(none_at_all["temperature"], "unknown")

    def test_unparseable_parser_source_gives_unknown_settings(self):
        got = runner.parse_llm_settings('GROQ_MODEL = "m"\ndef broken(:\n')
        self.assertEqual((got["model"], got["temperature"]), ("m", "unknown"))

    def test_missing_model_name_is_refused(self):
        with self.assertRaises(runner.Refused):
            runner.parse_llm_settings("ChatGroq(temperature=1)\n")

    def test_package_versions_handle_missing_and_failing_lookups(self):
        def lookup(name):
            if name == "groq":
                raise runner.importlib_metadata.PackageNotFoundError(name)
            if name == "langchain":
                raise RuntimeError("broken metadata")
            return "1.2.3"
        got = runner.package_versions(lookup)
        self.assertEqual(got, {"langchain": "unknown", "langchain-core": "1.2.3", "langchain-groq": "1.2.3",
                               "groq": "not installed"})

    def fake_finder(self, folder, **layout):
        """layout: package -> (file name, text). Returns a find_spec stand-in; no real import happens."""
        for package, (filename, text) in layout.items():
            os.makedirs(os.path.join(folder, package), exist_ok=True)
            write_text(os.path.join(folder, package, filename), text)

        def find_spec(name):
            if name in layout:
                return type("Spec", (), {"submodule_search_locations": [os.path.join(folder, name)]})()
            return None
        return find_spec

    def test_retry_defaults_are_read_from_source_text_without_importing(self):
        with tempfile.TemporaryDirectory() as folder:
            finder = self.fake_finder(
                folder,
                langchain_groq=("chat_models.py", "class X:\n    max_retries: int = 2\n    other: int = 9\n"),
                groq=("_constants.py", "DEFAULT_TIMEOUT = 1\nDEFAULT_MAX_RETRIES = 3\n"))
            before = set(sys.modules)
            got = runner.discover_provider_retry(finder)
            self.assertEqual(set(sys.modules), before)
        self.assertEqual((got["langchain_groq_default_max_retries"], got["groq_sdk_default_max_retries"]), (2, 3))
        self.assertIn("not verified at runtime", got["method"])

    def test_annotated_and_commented_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            finder = self.fake_finder(folder, langchain_groq=(
                "chat_models.py", "    max_retries: Optional[int] = 5  # retries\n"))
            self.assertEqual(runner.discover_provider_retry(finder)["langchain_groq_default_max_retries"], 5)

    def test_unreadable_or_absent_information_is_unknown_or_not_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            finder = self.fake_finder(folder, langchain_groq=("chat_models.py", "no setting here\n"))
            got = runner.discover_provider_retry(finder)
            self.assertEqual(got["langchain_groq_default_max_retries"], "unknown")
            self.assertEqual(got["groq_sdk_default_max_retries"], "not installed")
        self.assertEqual(runner.discover_provider_retry(lambda name: None)["groq_sdk_default_max_retries"],
                         "not installed")

        def exploding(name):
            raise ImportError("boom")
        self.assertEqual(runner.discover_provider_retry(exploding)["groq_sdk_default_max_retries"], "unknown")
        empty = lambda name: type("Spec", (), {"submodule_search_locations": None})()
        self.assertEqual(runner.discover_provider_retry(empty)["langchain_groq_default_max_retries"], "unknown")

    def test_discovery_runs_inside_the_offline_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            finder = self.fake_finder(
                folder,
                langchain_groq=("chat_models.py", "    max_retries: int = 4\n"),
                groq=("_constants.py", "DEFAULT_MAX_RETRIES = 4\n"))
            env = runner.discover_environment(find_spec=finder, version_lookup=lambda n: "9.9")
        self.assertEqual(env["provider_retry"]["langchain_groq_default_max_retries"], 4)
        self.assertEqual(env["packages"]["groq"], "9.9")


class LiveSettingVerification(Workspace):
    def run_with(self, info, source_root=None, env=ENV):
        self.main(["plan", "--run-id", "v"], environment=env, **({"source_root": source_root} if source_root else {}))
        client = FakeClient(self.dev_cases, {}, default=lambda cid: output_for(next(c for c in self.dev_cases if c["id"] == cid)))
        client.info = info
        code, err = self.main(["run", "--plan", "v", "--live", "--confirm-calls", "18"], client, environment=env,
                              **({"source_root": source_root} if source_root else {}))
        return client, code, err

    BASE = {"model": "openai/gpt-oss-20b", "temperature": 0.2, "max_tokens": 8192, "reasoning_effort": "low",
            "max_retries": 2}

    def test_matching_client_runs_and_has_nothing_unverified(self):
        client, code, err = self.run_with(dict(self.BASE))
        self.assertEqual((code, err), (0, ""))
        meta = json.loads(self.read("v", "metadata.json"))
        self.assertEqual(meta["unverified_settings"], [])
        self.assertEqual(meta["client"]["max_retries"], 2)
        self.assertEqual(len(client.calls), 18)

    def test_each_mismatch_is_refused_before_any_call(self):
        for key, value in (("temperature", 0.9), ("max_tokens", 1), ("reasoning_effort", "high"), ("max_retries", 0)):
            with self.subTest(key=key):
                self.setUp()
                info = dict(self.BASE)
                info[key] = value
                client, code, err = self.run_with(info)
                self.assertEqual(code, 2)
                self.assertIn("differs from the plan", err)
                self.assertEqual(client.calls, [])
                self.assertFalse(os.path.exists(os.path.join(self.results, "v", "calls.jsonl")))

    def test_parser_configured_retries_are_compared_instead_of_the_library_default(self):
        source = SourceCopy(self, replacements=[("temperature=0.2", "temperature=0.2, max_retries=1")])
        client, code, err = self.run_with(dict(self.BASE), source_root=source.root)  # live reports 2, parser says 1
        self.assertEqual(code, 2)
        self.assertIn("provider max_retries (plan 1, live client 2)", err)
        self.assertEqual(client.calls, [])

    def test_unknown_values_are_recorded_as_unverified_not_guessed(self):
        info = {"model": "openai/gpt-oss-20b", "temperature": None}
        client, code, err = self.run_with(info)
        self.assertEqual((code, err), (0, ""))
        unverified = json.loads(self.read("v", "metadata.json"))["unverified_settings"]
        self.assertEqual(unverified, ["temperature", "max_tokens", "reasoning_effort", "max_retries"])
        self.assertIn("could not be verified", "\n".join(self.out_lines))

    def test_unknown_plan_retry_default_is_unverified_when_live_reports_a_number(self):
        env = json.loads(json.dumps(ENV))
        env["provider_retry"]["langchain_groq_default_max_retries"] = "not installed"
        client, code, err = self.run_with(dict(self.BASE), env=env)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(self.read("v", "metadata.json"))["unverified_settings"], ["max_retries"])


class HardeningRestrictions(unittest.TestCase):
    BANNED = {"subprocess", "socket", "sqlite3", "urllib", "http", "requests", "sqlalchemy", "psycopg2", "dotenv",
              "yaml", "shutil"}

    def imports(self, path):
        tree = ast.parse(read_text(path))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [(a.name, node) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                found.append((node.module, node))
        return tree, found

    def test_no_process_network_database_or_environment_file_imports(self):
        for path in (os.path.join(ATS_DIR, "run_ats_eval.py"), os.path.join(ATS_DIR, "dataset.py")):
            tree, found = self.imports(path)
            for name, node in found:
                if name == "src.parser":
                    continue  # the single lazy live-only import, checked in Restrictions
                self.assertNotIn(name.split(".")[0], self.BANNED, name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                    self.assertNotIn(node.attr, ("system", "popen", "spawnl", "execv", "fork"), node.attr)

    def test_runner_never_references_environment_files_or_git_and_docker(self):
        source = read_text(os.path.join(ATS_DIR, "run_ats_eval.py"))
        tree = ast.parse(source)
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and len(n.value) < 60]  # code strings, not the long docstrings
        for text in strings:
            self.assertNotEqual(text.strip(), ".env")
            self.assertNotRegex(text, r"^(git|docker)( |$)")
        reads_env = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "environ"]
        self.assertEqual(reads_env, [])  # only os.getenv("GROQ_API_KEY") in the live builder, never environ
        getenv_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "getenv"]
        self.assertEqual(len(getenv_calls), 1)
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_live_client")
        self.assertIn(getenv_calls[0], list(ast.walk(builder)))


# ---------------------------------------------------------------------------------------------------------------
# Evaluation-only max_tokens override, truncation handling and frozen pacing
# ---------------------------------------------------------------------------------------------------------------
TRUNCATED_JSON = '{"optimized_bullets": [{"section": "Experience", "bullets": ["Built a'


def truncated(text=TRUNCATED_JSON):
    return runner.ModelResponse(text, {"input_tokens": 100, "output_tokens": 4096, "total_tokens": 4196}, "length")


def finished(text, reason="stop"):
    return runner.ModelResponse(text, {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}, reason)


class OverrideBase(Workspace):
    def good(self, cid):
        return output_for(next(c for c in self.dev_cases if c["id"] == cid))

    def client(self, script=None):
        return FakeClient(self.dev_cases, script or {}, default=self.good)

    def edit_json(self, run_id, name, change):
        path = os.path.join(self.results, run_id, name)
        data = json.loads(read_text(path))
        change(data)
        write_text(path, json.dumps(data))

    def plan_with_override(self, run_id="o1", extra=("--eval-max-tokens", "--pace-seconds", "65"), repeats=None):
        argv = ["plan", "--run-id", run_id] + list(extra)
        if repeats:
            argv += ["--repeats", str(repeats)]
        code, err = self.main(argv, environment=ENV)
        self.assertEqual((code, err), (0, ""))
        return run_id


class OverridePlanning(OverrideBase):
    def test_flag_without_a_value_means_4096_and_is_recorded_separately_from_production(self):
        self.plan_with_override("o1")
        plan = json.loads(self.read("o1", "plan.json"))
        self.assertEqual(plan["model_settings"]["max_tokens"], 8192)  # production, parsed from src/parser.py
        self.assertEqual(plan["evaluation_overrides"], {"max_tokens": 4096})
        self.assertEqual(plan["effective_settings"]["max_tokens"], 4096)
        self.assertIn("override", plan["effective_settings"]["max_tokens_source"])
        self.assertEqual(plan["pacing"], {"pace_seconds": 65, "minimum_wall_seconds": 17 * 65})
        self.assertEqual((plan["planned_calls"], len(plan["case_ids"]), plan["repeats"]), (18, 9, 2))
        est = plan["token_estimates"]
        self.assertEqual(est["largest_request_tokens_estimate"], est["largest_input_tokens_estimate"] + 4096)
        meta = json.loads(self.read("o1", "metadata.json"))
        for key in ("model_settings", "evaluation_overrides", "effective_settings", "pacing"):
            self.assertEqual(meta[key], plan[key], key)
        text = "\n".join(self.out_lines)
        self.assertIn("EVALUATION-ONLY OVERRIDE: max_tokens=4096", text)
        self.assertIn("src/parser.py is not changed", text)
        self.assertNotIn("pacing is under 60 s", text)

    def test_explicit_value_and_the_no_override_default(self):
        self.plan_with_override("o2", extra=("--eval-max-tokens", "3000"))
        plan = json.loads(self.read("o2", "plan.json"))
        self.assertEqual((plan["evaluation_overrides"], plan["effective_settings"]["max_tokens"]),
                         ({"max_tokens": 3000}, 3000))
        self.main(["plan", "--run-id", "o3"], environment=ENV)
        plan = json.loads(self.read("o3", "plan.json"))
        self.assertEqual(plan["evaluation_overrides"], {})
        self.assertEqual(plan["effective_settings"]["max_tokens"], 8192)
        self.assertIn("production", plan["effective_settings"]["max_tokens_source"])
        self.assertIn("pacing is under 60 s", "\n".join(self.out_lines))

    def test_unsafe_override_values_are_refused_before_anything_is_written(self):
        for value, fragment in (("100", "at least 512"), ("0", "at least 512"), ("-5", "at least 512"),
                                ("9000", "above the production max_tokens 8192")):
            with self.subTest(value=value):
                code, err = self.main(["plan", "--run-id", "bad", "--eval-max-tokens", value], environment=ENV)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)
                self.assertFalse(os.path.exists(os.path.join(self.results, "bad")))

    def test_override_may_not_exceed_a_lower_production_value(self):
        source = SourceCopy(self, replacements=[("max_tokens=8192", "max_tokens=2048")])
        code, err = self.main(["plan", "--run-id", "bad", "--eval-max-tokens", "3000"], environment=ENV,
                              source_root=source.root)
        self.assertEqual(code, 2)
        self.assertIn("above the production max_tokens 2048", err)

    def test_planning_leaves_production_files_alone_and_matches_their_fingerprint(self):
        paths = [os.path.join(ROOT, "src", "parser.py"), os.path.join(ROOT, "src", "prompts.yaml")]
        before = [read_text(p) for p in paths]
        self.plan_with_override("o4")
        self.assertEqual([read_text(p) for p in paths], before)
        plan = json.loads(self.read("o4", "plan.json"))
        self.assertEqual(plan["code_fingerprints"]["parser_py"],
                         dataset.sha256_bytes(runner.read_source_file("parser.py").encode("utf-8")))
        self.assertEqual(runner.parse_llm_settings(runner.read_source_file("parser.py"))["max_tokens"], 8192)

    def test_offline_planning_builds_no_client_reads_no_key_and_opens_no_socket(self):
        seen_getenv = []
        real_getenv = os.getenv

        def spy(name, default=None):
            seen_getenv.append(name)
            return real_getenv(name, default)

        before = set(sys.modules)
        with mock.patch.object(runner, "build_live_client", side_effect=AssertionError("client built")), \
                mock.patch.object(socket, "socket", side_effect=AssertionError("socket opened")), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_NOT_A_REAL_KEY_FOR_TESTS"}), \
                mock.patch.object(os, "getenv", spy):
            code, err = self.main(["plan", "--run-id", "o5", "--eval-max-tokens"], environment=ENV)
        self.assertEqual((code, err), (0, ""))
        self.assertNotIn("GROQ_API_KEY", seen_getenv)
        new = {m for m in set(sys.modules) - before if m.split(".")[0] in FORBIDDEN_PREFIXES}
        self.assertEqual(new, set())
        self.assertNotIn("gsk_NOT_A_REAL_KEY", self.read("o5", "plan.json") + self.read("o5", "metadata.json"))


class OverrideFreshness(OverrideBase):
    def stale_labels(self, run_id):
        run = runner.open_run_dir(self.results, run_id)
        return runner.check_plan_is_current(run.read_json("plan.json"), run, self.results, environment=ENV)

    def refused_run(self, run_id, extra=()):
        client = self.client()
        code, err = self.main(["run", "--plan", run_id, "--live", "--confirm-calls", "18"] + list(extra), client,
                              environment=ENV)
        self.assertEqual(client.calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.results, run_id, "calls.jsonl")))
        return code, err

    def test_control_unchanged_plan_with_an_override_is_current(self):
        self.plan_with_override("f1")
        self.assertEqual(self.stale_labels("f1"), [])

    def test_changing_the_override_in_the_plan_is_detected(self):
        self.plan_with_override("f2")
        self.edit_json("f2", "plan.json", lambda p: p["evaluation_overrides"].update(max_tokens=2048))  # effective stale
        labels = self.stale_labels("f2")
        self.assertIn("effective_settings", labels)
        self.assertIn("metadata:evaluation_overrides", labels)

        def consistent(p):
            p["evaluation_overrides"] = {"max_tokens": 2048}
            p["effective_settings"]["max_tokens"] = 2048

        self.edit_json("f2", "plan.json", consistent)
        labels = self.stale_labels("f2")
        self.assertIn("metadata:evaluation_overrides", labels)  # plan.json no longer equals the planned copy
        code, err = self.refused_run("f2")
        self.assertEqual(code, 2)
        self.assertIn("the plan is stale", err)

    def test_removing_the_override_is_detected(self):
        self.plan_with_override("f3")
        self.edit_json("f3", "plan.json", lambda p: p.pop("evaluation_overrides"))
        self.assertIn("effective_settings", self.stale_labels("f3"))

        def cleared(p):
            p["evaluation_overrides"] = {}
            p["effective_settings"] = runner.effective_settings_for(p["model_settings"], None)

        self.edit_json("f3", "plan.json", cleared)
        self.assertIn("metadata:evaluation_overrides", self.stale_labels("f3"))
        code, err = self.refused_run("f3")
        self.assertEqual(code, 2)
        self.assertIn("evaluation_overrides", err)

    def test_a_parser_change_to_the_production_cap_invalidates_an_override_plan(self):
        self.plan_with_override("f4")
        source = SourceCopy(self, replacements=[("max_tokens=8192", "max_tokens=6000")])
        run = runner.open_run_dir(self.results, "f4")
        labels = runner.check_plan_is_current(run.read_json("plan.json"), run, self.results, environment=ENV,
                                              source_root=source.root)
        for expected in ("model_settings", "code:parser_py"):
            self.assertIn(expected, labels)
        self.assertNotIn("effective_settings", labels)  # the override, not the production cap, is what is used
        lower = SourceCopy(self, replacements=[("max_tokens=8192", "max_tokens=2048")])  # below the 4096 override
        labels = runner.check_plan_is_current(run.read_json("plan.json"), run, self.results, environment=ENV,
                                              source_root=lower.root)
        self.assertEqual(len(labels), 1)
        self.assertIn("above the production max_tokens 2048", labels[0])

    def test_pacing_is_frozen_in_the_plan(self):
        self.plan_with_override("f5")
        self.edit_json("f5", "plan.json", lambda p: p["pacing"].update(pace_seconds=10))
        labels = self.stale_labels("f5")
        self.assertIn("pacing", labels)
        self.assertIn("metadata:pacing", labels)
        self.edit_json("f5", "plan.json", lambda p: p["pacing"].update(pace_seconds=10, minimum_wall_seconds=170))
        self.assertIn("metadata:pacing", self.stale_labels("f5"))

    def test_a_different_run_time_pace_is_refused_and_the_plan_pace_is_used(self):
        self.plan_with_override("f6")
        code, err = self.refused_run("f6", extra=("--pace-seconds", "10"))
        self.assertEqual(code, 2)
        self.assertIn("frozen in the plan", err)
        client = self.client()
        code, err = self.main(["run", "--plan", "f6", "--live", "--confirm-calls", "18", "--pace-seconds", "65"],
                              client, environment=ENV)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.sleeps.seconds.count(65), 17)
        self.assertEqual(len(client.calls), 18)

    def test_old_format_plans_are_rejected(self):
        self.plan_with_override("f7")

        def downgrade(p):
            p["plan_schema_version"] = 2
            for key in ("evaluation_overrides", "effective_settings"):
                p.pop(key)

        self.edit_json("f7", "plan.json", downgrade)
        self.assertIn("plan_schema_version", self.stale_labels("f7"))
        code, err = self.refused_run("f7")
        self.assertEqual(code, 2)
        self.assertIn("plan_schema_version", err)

    def test_dev_fingerprints_and_holdout_isolation_still_apply_with_an_override(self):
        self.plan_with_override("f8", repeats=1)
        plan = json.loads(self.read("f8", "plan.json"))
        self.assertEqual(plan["split"], "dev")
        self.assertFalse({"c02_strong_ml_pipeline", "c06_absent_terraform_aws", "c10_inflation_assisted"}
                         & set(plan["case_ids"]))
        for key in ("dataset", "vocabulary", "checker", "prompt"):
            self.assertRegex(plan["fingerprints"][key], r"^[0-9a-f]{64}$")
        self.assertRegex(plan["code_fingerprints"]["runner"], r"^[0-9a-f]{64}$")

    def test_holdout_needs_the_same_frozen_override_and_still_needs_its_phrase(self):
        self.plan_with_override("dev1", extra=("--eval-max-tokens",), repeats=1)
        client = self.client()
        self.assertEqual(self.main(["run", "--plan", "dev1", "--live", "--confirm-calls", "9"], client,
                                   environment=ENV)[0], 0)
        self.assertEqual(client.overrides_received, [4096])
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold", "--frozen-from", "dev1"],
                              environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("not frozen", err)
        self.assertIn("evaluation_overrides", err)
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold2", "--frozen-from", "dev1",
                               "--eval-max-tokens", "3000"], environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("evaluation_overrides", err)
        code, err = self.main(["plan", "--split", "holdout", "--run-id", "hold3", "--frozen-from", "dev1",
                               "--eval-max-tokens"], environment=ENV)
        self.assertEqual((code, err), (0, ""))
        blocked = self.client()
        code, err = self.main(["run", "--plan", "hold3", "--live", "--confirm-calls", "6"], blocked, environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("--confirm-holdout", err)
        self.assertEqual(blocked.calls, [])


class OverrideLiveSettings(OverrideBase):
    def run_override(self, info_changes=None, run_id="l1"):
        self.plan_with_override(run_id, extra=("--eval-max-tokens",), repeats=1)
        client = self.client()
        client.info.update(info_changes or {})
        code, err = self.main(["run", "--plan", run_id, "--live", "--confirm-calls", "9"], client, environment=ENV)
        return client, code, err

    def test_the_run_passes_the_plans_override_to_the_client_builder(self):
        client, code, err = self.run_override()
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(client.overrides_received, [4096])
        meta = json.loads(self.read("l1", "metadata.json"))
        self.assertEqual(meta["client"]["effective_max_tokens"], 4096)
        self.assertEqual(meta["client"]["max_tokens"], 8192)  # the production client keeps its setting
        self.assertIn("effective max_tokens: 4096", "\n".join(self.out_lines))

    def test_without_an_override_the_builder_receives_none(self):
        self.plan("l2", repeats=1)
        client = self.client()
        self.assertEqual(self.run_cli("l2", client, 9)[0], 0)
        self.assertEqual(client.overrides_received, [None])

    def test_a_client_that_ignores_the_override_is_refused_before_any_call(self):
        client, code, err = self.run_override({"effective_max_tokens": 8192})
        self.assertEqual(code, 2)
        self.assertIn("effective max_tokens (plan 4096, live client 8192)", err)
        self.assertEqual(client.calls, [])

    def test_a_client_that_does_not_report_its_effective_cap_is_refused(self):
        client, code, err = self.run_override({"effective_max_tokens": None})
        self.assertEqual(code, 2)
        self.assertIn("did not report it", err)
        self.assertEqual(client.calls, [])

    def test_a_mutated_production_client_is_refused(self):
        client, code, err = self.run_override({"max_tokens": 4096})  # production attribute lowered
        self.assertEqual(code, 2)
        self.assertIn("max_tokens (plan 8192, live client 4096)", err)
        self.assertEqual(client.calls, [])


class FakeMessage(object):
    def __init__(self, content="{}", response_metadata=None, usage_metadata=None):
        self.content = content
        self.response_metadata = response_metadata
        self.usage_metadata = usage_metadata


class FakeLLM(object):
    """Mimics the wrapper's request building: bound kwargs are merged OVER the instance defaults per request."""
    instances = []

    def __init__(self):
        self.model_name, self.temperature, self.max_tokens = "openai/gpt-oss-20b", 0.2, 8192
        self.reasoning_effort, self.max_retries = "low", 2
        self.requests = []
        self.reply = FakeMessage("{}", {"finish_reason": "stop"},
                                 {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})

    def bind(self, **kwargs):
        return FakeBound(self, kwargs)

    def invoke(self, inputs, **kwargs):
        self.requests.append(dict({"max_tokens": self.max_tokens}, **kwargs))
        return self.reply


class FakeBound(object):
    def __init__(self, llm, kwargs):
        self.llm, self.kwargs = llm, kwargs

    def invoke(self, inputs):
        return self.llm.invoke(inputs, **self.kwargs)


class FakePrompt(object):
    def __or__(self, other):
        prompt = self

        class Chain(object):
            def invoke(self, inputs):
                return other.invoke(inputs)

        return Chain()


class FakeResumeParser(object):
    def __init__(self):
        self.llm = FakeLLM()
        self.ats_bullet_prompt = FakePrompt()
        FakeLLM.instances.append(self.llm)


class BuiltClientWiring(unittest.TestCase):
    """build_live_client against a FAKE src.parser module: no provider, no key, no network."""

    def build(self, override):
        import types
        pkg, mod = types.ModuleType("src"), types.ModuleType("src.parser")
        pkg.__path__ = []
        mod.ResumeParser = FakeResumeParser
        with mock.patch.dict(sys.modules, {"src": pkg, "src.parser": mod}), \
                mock.patch.dict(os.environ, {"GROQ_API_KEY": "dummy-not-a-secret"}):
            return runner.build_live_client(max_tokens_override=override)

    def test_override_reaches_the_request_and_production_object_is_not_mutated(self):
        call, info = self.build(4096)
        llm = FakeLLM.instances[-1]
        response = call("jd", "resume")
        self.assertEqual(llm.requests[-1]["max_tokens"], 4096)  # the REQUEST parameter, not just metadata
        self.assertEqual(llm.max_tokens, 8192)
        self.assertEqual((info["max_tokens"], info["effective_max_tokens"], info["max_tokens_override"]),
                         (8192, 4096, 4096))
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})

    def test_without_an_override_the_request_keeps_the_production_cap(self):
        call, info = self.build(None)
        call("jd", "resume")
        self.assertEqual(FakeLLM.instances[-1].requests[-1]["max_tokens"], 8192)
        self.assertEqual((info["max_tokens"], info["effective_max_tokens"], info["max_tokens_override"]),
                         (8192, 8192, None))

    def test_one_clients_override_does_not_leak_into_the_next(self):
        self.build(4096)
        call, info = self.build(None)
        call("jd", "resume")
        self.assertEqual(FakeLLM.instances[-1].requests[-1]["max_tokens"], 8192)

    def test_length_finish_reason_is_passed_through_and_a_missing_one_is_not_invented(self):
        call, _ = self.build(4096)
        FakeLLM.instances[-1].reply = FakeMessage("{", {"finish_reason": "length"})
        self.assertEqual(call("jd", "resume").finish_reason, "length")
        FakeLLM.instances[-1].reply = FakeMessage("{", {})
        self.assertIsNone(call("jd", "resume").finish_reason)
        FakeLLM.instances[-1].reply = FakeMessage("{", None)
        self.assertIsNone(call("jd", "resume").finish_reason)

    def test_the_builder_applies_the_cap_with_bind_and_never_assigns_it_to_the_client(self):
        tree = ast.parse(read_text(os.path.join(ATS_DIR, "run_ats_eval.py")))
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_live_client")
        for node in ast.walk(builder):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and target.attr == "max_tokens")
        binds = [n for n in ast.walk(builder) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "bind"]
        self.assertEqual(len(binds), 1)
        self.assertEqual([k.arg for k in binds[0].keywords], ["max_tokens"])

    def test_the_cap_value_is_never_hardcoded_outside_the_documented_default(self):
        tree = ast.parse(read_text(os.path.join(ATS_DIR, "run_ats_eval.py")))
        literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 4096]
        self.assertEqual(len(literals), 1)  # DEFAULT_EVAL_MAX_TOKENS only
        self.assertEqual(runner.DEFAULT_EVAL_MAX_TOKENS, 4096)

    def test_finish_reason_helper(self):
        helper = runner._finish_reason_from_message
        self.assertEqual(helper(FakeMessage(response_metadata={"finish_reason": "length"})), "length")
        for bad in ({"finish_reason": None}, {"finish_reason": ""}, {"finish_reason": 3}, {}, None):
            self.assertIsNone(helper(FakeMessage(response_metadata=bad)))
        self.assertIsNone(helper(object()))


class TruncationHandling(OverrideBase):
    def run_plan(self, script, run_id="t1", extra=(), repeats=2):
        self.plan_with_override(run_id, extra=("--eval-max-tokens",), repeats=repeats)
        client = self.client(script)
        calls = 9 * repeats
        code, err = self.main(["run", "--plan", run_id, "--live", "--confirm-calls", str(calls)] + list(extra),
                              client, environment=ENV)
        return client, code, err

    def test_known_truncation_gets_its_own_status_and_the_raw_text_is_preserved(self):
        client, code, err = self.run_plan({"c01_strong_task_api": [truncated(), self.good("c01_strong_task_api")]})
        self.assertEqual((code, err), (0, ""))
        first = self.jsonl("t1", "calls.jsonl")[0]
        self.assertEqual((first["status"], first["truncated"], first["finish_reason"]), ("truncated", True, "length"))
        self.assertTrue(first["production_would_500"])  # the cut-off JSON would not parse in production either
        self.assertIn("output cut off", first["problems"][0])
        self.assertEqual(self.read("t1", first["raw_path"]), TRUNCATED_JSON)
        evaluated = {row["call_id"] for row in self.jsonl("t1", "evaluations.jsonl")}
        self.assertNotIn(first["call_id"], evaluated)

    def test_truncation_with_empty_or_valid_text_is_still_truncated(self):
        valid = self.good("c01_strong_task_api")
        client, code, err = self.run_plan({"c01_strong_task_api": [truncated(""), truncated(valid)]})
        records = self.jsonl("t1", "calls.jsonl")
        self.assertEqual([r["status"] for r in records[:2]], ["truncated", "truncated"])
        self.assertTrue(records[0]["production_would_500"])
        self.assertNotEqual(records[1]["production_would_500"], True)  # parseable text: nothing invented

    def test_parse_failures_keep_their_classification_when_not_truncated(self):
        script = {"c01_strong_task_api": [finished("not json"), finished("", "stop")],
                  "c03_vague_backend_intern": [runner.ModelResponse("not json"), finished('{"x": 1}')]}
        client, code, err = self.run_plan(script)
        by_id = {r["call_id"]: r for r in self.jsonl("t1", "calls.jsonl")}
        self.assertEqual(by_id["c01_strong_task_api__r01"]["status"], "parse_failed")
        self.assertFalse(by_id["c01_strong_task_api__r01"]["truncated"])
        self.assertEqual(by_id["c01_strong_task_api__r01"]["finish_reason"], "stop")
        self.assertEqual(by_id["c01_strong_task_api__r02"]["status"], "empty_response")
        unknown = by_id["c03_vague_backend_intern__r01"]
        self.assertEqual((unknown["status"], unknown["finish_reason"], unknown["finish_reason_available"]),
                         ("parse_failed", None, False))
        self.assertEqual(by_id["c03_vague_backend_intern__r02"]["status"], "schema_invalid")

    def test_two_consecutive_truncations_abort_the_run(self):
        client, code, err = self.run_plan({"c01_strong_task_api": [truncated(), truncated()]})
        self.assertEqual(code, 0)
        meta = json.loads(self.read("t1", "metadata.json"))
        self.assertEqual(meta["status"], "aborted")
        self.assertIn("2 consecutive truncated", meta["aborted_reason"])
        self.assertEqual((meta["attempted_calls"], len(meta["not_attempted"])), (2, 16))
        self.assertEqual(len(client.calls), 2)
        self.assertIn("ABORTED", "\n".join(self.out_lines))
        self.assertEqual(meta["execution_config"]["abort_after_truncated"], 2)

    def test_non_consecutive_truncations_do_not_abort(self):
        script = {"c01_strong_task_api": [truncated(), self.good("c01_strong_task_api")],
                  "c03_vague_backend_intern": [truncated(), self.good("c03_vague_backend_intern")]}
        client, code, err = self.run_plan(script)
        meta = json.loads(self.read("t1", "metadata.json"))
        self.assertEqual((meta["status"], meta["attempted_calls"]), ("complete", 18))
        self.assertIsNone(meta["aborted_reason"])

    def test_the_truncation_limit_is_configurable_and_validated(self):
        client, code, err = self.run_plan({"c01_strong_task_api": [truncated()]}, extra=("--abort-after-truncated", "1"))
        meta = json.loads(self.read("t1", "metadata.json"))
        self.assertEqual((meta["status"], meta["attempted_calls"]), ("aborted", 1))
        self.setUp()
        self.plan_with_override("t2")
        blocked = self.client()
        code, err = self.main(["run", "--plan", "t2", "--live", "--confirm-calls", "18", "--abort-after-truncated", "0"],
                              blocked, environment=ENV)
        self.assertEqual(code, 2)
        self.assertIn("at least 1", err)
        self.assertEqual(blocked.calls, [])

    def test_api_failures_and_truncations_have_independent_abort_counters(self):
        script = {"c01_strong_task_api": [truncated(), RateLimitError(), RateLimitError()]}
        client, code, err = self.run_plan(script)
        meta = json.loads(self.read("t1", "metadata.json"))
        self.assertEqual(meta["status"], "complete")  # a failed API call in between resets the truncation streak

    def test_summary_and_report_use_correct_denominators(self):
        script = {"c01_strong_task_api": [truncated()],
                  "c03_vague_backend_intern": [finished("not json")],
                  "c04_vague_data_scripts": [RateLimitError(), RateLimitError()]}
        self.plan_with_override("s1", extra=("--eval-max-tokens",), repeats=1)
        client = self.client(script)
        self.assertEqual(self.main(["run", "--plan", "s1", "--live", "--confirm-calls", "9"], client,
                                   environment=ENV)[0], 0)
        self.assertEqual(self.main(["summarize", "s1"])[0], 0)
        summary = json.loads(self.read("s1", "summary.json"))
        counts, rates = summary["counts"], summary["rates"]
        self.assertEqual((counts["planned_calls"], counts["attempted_calls"], counts["not_attempted_calls"]), (9, 9, 0))
        self.assertEqual((counts["responses_received"], counts["truncated_responses"], counts["api_failures"],
                          counts["json_parse_failures"]), (8, 1, 1, 1))
        self.assertEqual(rates["truncated_over_responses_received"], "1/8 (12.5%)")
        self.assertEqual(rates["production_failures_over_responses_received"], "1/7 (14.3%)")  # truncated excluded
        self.assertEqual(rates["api_failures_over_attempted_calls"], "1/9 (11.1%)")
        self.assertEqual(rates["outputs_eligible_over_responses_received"], "6/8 (75.0%)")
        self.assertEqual(summary["finish_reasons"], {"length": 1, "not reported": 6, "stop": 1})
        self.assertEqual(summary["configuration"]["effective_max_tokens"], 4096)
        self.assertEqual(summary["configuration"]["production_max_tokens"], 8192)
        report = read_text(os.path.join(self.results, "s1", "report.md"))
        for fragment in ("Truncated (output cap reached) | 1/8", "cap-limited", "Evaluation-only output cap: max_tokens=4096",
                         "Production failures (would be HTTP 500) | 1/7", "r1: truncated",
                         "Finish reason was not reported for 6 response(s)"):
            self.assertIn(fragment, report)

    def test_aborted_run_reports_unattempted_calls_in_the_summary(self):
        self.run_plan({"c01_strong_task_api": [truncated(), truncated()]}, run_id="s2")
        self.main(["summarize", "s2"])
        summary = json.loads(self.read("s2", "summary.json"))
        self.assertEqual((summary["counts"]["attempted_calls"], summary["counts"]["not_attempted_calls"],
                          summary["counts"]["truncated_responses"]), (2, 16, 2))
        self.assertIn("Run aborted: 2 consecutive truncated", read_text(os.path.join(self.results, "s2", "report.md")))

    def test_truncated_calls_do_not_enter_the_label_sheet(self):
        self.run_plan({"c01_strong_task_api": [truncated(), self.good("c01_strong_task_api")]})
        sheet = self.read("t1", "bullets.csv")
        self.assertNotIn("c01_strong_task_api__r01", sheet)
        self.assertIn("c01_strong_task_api__r02", sheet)

    def test_statuses_list_includes_truncated(self):
        self.assertIn("truncated", runner.CALL_STATUSES)


class OverrideRestrictions(unittest.TestCase):
    def test_production_configuration_is_unchanged(self):
        parser_text = read_text(os.path.join(ROOT, "src", "parser.py"))
        settings = runner.parse_llm_settings(parser_text)
        self.assertEqual((settings["model"], settings["temperature"], settings["max_tokens"],
                          settings["reasoning_effort"]), ("openai/gpt-oss-20b", 0.2, 8192, "low"))

    def test_no_new_imports_beyond_the_standard_library_and_the_single_lazy_parser_import(self):
        tree = ast.parse(read_text(os.path.join(ATS_DIR, "run_ats_eval.py")))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertEqual(names, {"argparse", "ast", "csv", "io", "importlib", "json", "math", "os", "re", "statistics",
                                 "sys", "textwrap", "time", "datetime", "src"})


class RealWrapperRequest(unittest.TestCase):
    """OPTIONAL. Runs only where langchain-groq and groq are installed. It builds the REAL ChatGroq with a dummy
    key in a child process (so this process never imports a provider), replaces the HTTP-level create() with a
    recorder BEFORE any request, and blocks sockets. It never contacts Groq."""

    CHILD = r'''
import json, socket, sys, warnings
from importlib import metadata
warnings.simplefilter("ignore")


def blocked(*args, **kwargs):
    raise RuntimeError("network blocked by the test")


socket.socket.connect = blocked
socket.create_connection = blocked
from unittest import mock
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
llm = ChatGroq(model="openai/gpt-oss-20b", api_key="dummy-not-a-secret", temperature=0.2, max_tokens=8192,
               reasoning_effort="low")
seen = []


def fake_create(**kwargs):
    seen.append(kwargs)
    return {"choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}


with mock.patch.object(llm.client, "create", side_effect=fake_create):
    production = PromptTemplate.from_template("{a}") | llm
    evaluation = PromptTemplate.from_template("{a}") | llm.bind(max_tokens=4096)
    messages = [production.invoke({"a": "x"}), evaluation.invoke({"a": "x"}), production.invoke({"a": "x"})]
keep = ("max_tokens", "reasoning_effort", "model", "temperature")
print(json.dumps({"version": metadata.version("langchain-groq"),
                  "requests": [{k: s.get(k) for k in keep} for s in seen],
                  "finish": [m.response_metadata.get("finish_reason") for m in messages],
                  "llm_max_tokens": llm.max_tokens}))
'''

    @unittest.skipIf(importlib.util.find_spec("langchain_groq") is None or importlib.util.find_spec("groq") is None,
                     "langchain-groq / groq not installed in this Python: the real-wrapper check did NOT run")
    def test_bound_cap_changes_the_outgoing_request_and_production_stays_at_8192(self):
        import subprocess
        env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper()
               and "GROQ" not in k.upper()}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as folder:
            done = subprocess.run([sys.executable, "-c", self.CHILD], cwd=folder, env=env, capture_output=True,
                                  text=True, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        result = json.loads(done.stdout.strip().splitlines()[-1])
        sizes = [request["max_tokens"] for request in result["requests"]]
        self.assertEqual(sizes, [8192, 4096, 8192], "installed langchain-groq " + result["version"])
        for request in result["requests"]:
            self.assertEqual((request["model"], request["temperature"], request["reasoning_effort"]),
                             ("openai/gpt-oss-20b", 0.2, "low"))
        self.assertEqual(result["finish"], ["length", "length", "length"])
        self.assertEqual(result["llm_max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
