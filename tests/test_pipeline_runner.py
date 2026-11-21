"""Tests for eval/pipeline/runner.py (Step 6, phase 1): plan, gated live run, summarize.

EVERY model in this file is a scripted fake. No test calls Groq, uses the network, reads a secret file, imports a
production module or touches a protected resume artifact. The "API key" is a dummy string that is checked never to
reach an artifact. The real dataset files are READ (they are what the plan hashes) and a test confirms they are
unchanged; everything the runner writes goes to a temporary folder.
"""
import ast
import hashlib
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
PIPELINE = os.path.join(ROOT, "eval", "pipeline")
RUNNER_PATH = os.path.join(PIPELINE, "runner.py")


def _load(name, path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("pipeline_runner", RUNNER_PATH)
cli = runner.cli
score = cli.score
stages_module = runner.stages_module
dataset_module = runner.dataset_module

DUMMY_KEY = "sk-test-dummy-not-a-real-key"
FIXED_NOW = "2026-01-01T00:00:00Z"
FIXED_ENV = {"python": "test", "platform": "test", "packages": {}}
DATASET = dataset_module.load_dataset()
IDS = list(DATASET["ids"])
TEMPLATES = runner.load_prompt_templates(runner.ROOT)
PREFIX = {stage: TEMPLATES[stage].split("{", 1)[0][:60] for stage in runner.STAGES}
E1_MARKER = "ZZ_E1_MARKER_ZZ"
E1_REPLY = "```json\n" + json.dumps({category: ([E1_MARKER] if i == 0 else [])
                                      for i, category in enumerate(score.E1_CATEGORIES)}) + "\n```"
E2_REPLY = '{"crucial": [], "good_to_have": [], "optional": []}'
E3_REPLY = '```json\n{"optimized_bullets": [{"section": "Experience", "bullets": ["Built data pipelines."]}]}\n```'


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def dataset_hashes():
    found = {}
    for folder in (dataset_module.DEFAULT_SCENARIOS_DIR, dataset_module.DEFAULT_EXPECTED_DIR):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                with open(os.path.join(folder, name), "rb") as handle:
                    found[os.path.join(os.path.basename(folder), name)] = hashlib.sha256(handle.read()).hexdigest()
    return found


class TransportError(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 503


class PermanentError(Exception):
    status_code = 400


class APITimeoutError(Exception):
    """Matched by class name, like the provider library's timeout error."""


class FakeResponse:
    def __init__(self, content, finish_reason="stop", model_name="openai/gpt-oss-20b"):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason, "model_name": model_name}


class ScriptedModel:
    """A fake model. It recognises the stage by the prompt's opening text and the scenario by its job description.

    faults: {(stage, scenario_id): [item, ...]}; each invoke for that key consumes one item: an exception instance is
    raised, None (or an exhausted list) answers normally, a FakeResponse is returned as is.
    fail_after: after that many invocations every call raises PermanentError."""

    def __init__(self, faults=None, replies=None, fail_after=None):
        self.faults = {key: list(value) for key, value in (faults or {}).items()}
        self.replies = {"e1": E1_REPLY, "e2": E2_REPLY, "e3": E3_REPLY}
        self.replies.update(replies or {})
        self.fail_after = fail_after
        self.invocations = []          # (stage, scenario_id, prompt)

    def identify(self, prompt):
        stage = [s for s in runner.STAGES if prompt.startswith(PREFIX[s])]
        sid = [i for i in IDS if DATASET["scenarios"][i]["inputs"]["job_description"] in prompt]
        assert len(stage) == 1 and len(sid) == 1, (stage, sid)
        return stage[0], sid[0]

    def invoke(self, prompt):
        stage, sid = self.identify(prompt)
        self.invocations.append((stage, sid, prompt))
        if self.fail_after is not None and len(self.invocations) > self.fail_after:
            raise PermanentError("scripted permanent failure")
        queue = self.faults.get((stage, sid))
        item = queue.pop(0) if queue else None
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(self.replies[stage])

    @property
    def sequence(self):
        return [(stage, sid) for stage, sid, _ in self.invocations]


def expected_sequence(repeats):
    return [(stage, sid) for _ in range(repeats) for sid in IDS for stage in runner.STAGES]


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pipeline_runner_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.results = os.path.join(self.tmp, "results")
        self.before = dataset_hashes()

    def tearDown(self):
        self.assertEqual(dataset_hashes(), self.before)

    def run_main(self, *argv, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        code = runner.main(list(argv), stdout=out, stderr=err, **kwargs)
        return code, out.getvalue(), err.getvalue()

    def make_plan(self, repeats=1, run_id="t1", root=None, **extra):
        argv = ["plan", "--repeats", str(repeats), "--run-id", run_id, "--results-dir", self.results]
        for key, value in extra.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        if root:
            argv += ["--root", root]
        code, out, err = self.run_main(*argv)
        self.assertEqual(code, 0, err)
        return runner.load_plan(self.results, run_id)

    def go_live(self, plan, model, confirm=None, results=None, root=None, key=DUMMY_KEY):
        sleeps, factory_calls = [], []

        def factory(settings, api_key):
            factory_calls.append((settings, api_key))
            return model

        confirm = plan["schedule"]["planned_calls"] if confirm is None else confirm
        argv = ["run", "--plan", plan["run_id"], "--live", "--confirm-calls", str(confirm),
                "--results-dir", results or self.results]
        if root:
            argv += ["--root", root]
        environment = {k: v for k, v in os.environ.items() if k != runner.API_KEY_VARIABLE}
        if key is not None:
            environment[runner.API_KEY_VARIABLE] = key
        with mock.patch.dict(os.environ, environment, clear=True):
            code, out, err = self.run_main(*argv, model_factory=factory, sleep=sleeps.append,
                                           now=lambda: FIXED_NOW, environment=FIXED_ENV)
        return code, out, err, sleeps, factory_calls

    def run_dir(self, run_id="t1"):
        return os.path.join(self.results, run_id)

    def read_json(self, name, run_id="t1"):
        with open(os.path.join(self.run_dir(run_id), name), encoding="utf-8") as handle:
            return json.load(handle)

    def read_calls(self, run_id="t1"):
        with open(os.path.join(self.run_dir(run_id), "calls.jsonl"), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class HelperFunctions(unittest.TestCase):
    def test_clean_response_matches_the_production_substitution(self):
        import re
        for text in ('```json\n{"a": 1}\n```', '  {"a": 1}  ', '```json{"a":1}```', "plain", "", '```\n{"a":1}\n```'):
            self.assertEqual(runner.clean_response(text), re.sub(r"^```json|```$", "", text.strip()).strip(), text)
        self.assertEqual(runner.clean_response('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_render_prompt_is_str_format(self):
        for stage in runner.STAGES:
            values = {name: f"<{name}>" for name in runner.PROMPT_VARIABLES[stage]}
            self.assertEqual(runner.render_prompt(TEMPLATES[stage], **values), TEMPLATES[stage].format(**values))

    def test_prompt_prefixes_identify_the_stage(self):
        self.assertEqual(len(set(PREFIX.values())), 3)
        for prefix in PREFIX.values():
            self.assertTrue(prefix.strip())

    def test_model_settings_come_from_the_text_of_parser_py(self):
        text = ("class C:\n    M = 'some/model'\n"
                "llm = ChatGroq(model=C.M, api_key=key, temperature=0.5, max_tokens=10, reasoning_effort='low')\n")
        settings = runner.parse_model_settings(text)
        self.assertEqual(settings, {"max_tokens": 10, "model": "some/model", "reasoning_effort": "low", "temperature": 0.5})
        self.assertNotIn("api_key", settings)

    def test_model_settings_errors(self):
        with self.assertRaises(runner.PlanError):
            runner.parse_model_settings("x = 1\n")
        with self.assertRaises(runner.PlanError):
            runner.parse_model_settings("ChatGroq(model='m', temperature=t, max_tokens=1, reasoning_effort='low')\n")
        with self.assertRaises(runner.PlanError):
            runner.parse_model_settings("ChatGroq(model='m', temperature=0.1)\n")

    def test_transport_failure_classification(self):
        for error in (TransportError(), ServerError(), APITimeoutError(), TimeoutError(), ConnectionError()):
            self.assertTrue(runner.is_transport_failure(error), type(error).__name__)
        for error in (PermanentError(), ValueError("x"), KeyError("k")):
            self.assertFalse(runner.is_transport_failure(error), type(error).__name__)

    def test_hashing_ignores_line_ending_style(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        a, b = os.path.join(folder, "a"), os.path.join(folder, "b")
        with open(a, "wb") as handle:
            handle.write(b"x\r\ny\r\n")
        with open(b, "wb") as handle:
            handle.write(b"x\ny\n")
        self.assertEqual(runner.sha256_file(a), runner.sha256_file(b))

    def test_aggregate_across_repeats_shows_each_repeat_and_the_range_only(self):
        def report(rate, available=True):
            metric = {"metric": "m", "numerator": None if not available else int(rate * 4), "denominator": 4,
                      "rate": rate if available else None, "available": available, "direction": "higher_is_better",
                      "unavailable_reason": None if available else "why", "per_scenario": []}
            return {"score": {"aggregate": {"e1": [metric]}}, "e3": {"aggregate": {"metrics": [dict(metric, metric="s")]}}}

        rows = runner.aggregate_across_repeats({1: report(0.5), 2: report(0.25), 3: report(None, available=False)})
        e1 = [r for r in rows if r["stage"] == "e1"][0]
        self.assertEqual((e1["rate_min"], e1["rate_max"], e1["repeats_available"]), (0.25, 0.5, 2))
        self.assertEqual([p["available"] for p in e1["per_repeat"]], [True, True, False])
        self.assertEqual(e1["per_repeat"][2]["unavailable_reason"], "why")
        self.assertNotIn("numerator", e1)             # nothing is pooled across repeats
        self.assertTrue([r for r in rows if r["stage"] == "e3"][0]["screening_only"])

    def test_summarize_calls_counts_logical_calls_retries_and_skips(self):
        lines = [
            {"repeat": 1, "scenario_id": "a", "stage": "e1", "attempt": 1, "status": "error"},
            {"repeat": 1, "scenario_id": "a", "stage": "e1", "attempt": 2, "status": "ok", "finish_reason": "stop"},
            {"repeat": 1, "scenario_id": "a", "stage": "e2", "attempt": 0, "status": "skipped"},
            {"repeat": 1, "scenario_id": "a", "stage": "e3", "attempt": 1, "status": "error"},
            {"repeat": 1, "scenario_id": "b", "stage": "e3", "attempt": 1, "status": "ok", "finish_reason": "length"},
        ]
        table = runner.summarize_calls(lines)
        self.assertEqual((table["e1"]["logical_calls_ok"], table["e1"]["retried_attempts"], table["e1"]["attempts"]), (1, 1, 2))
        self.assertEqual(table["e2"]["skipped"], 1)
        self.assertEqual((table["e3"]["logical_calls_ok"], table["e3"]["logical_calls_failed"]), (1, 1))
        self.assertEqual(table["e3"]["finish_reasons"], {"length": 1})


class Plan(RunnerCase):
    def test_default_plan_has_three_repeats_pacing_and_ninety_calls(self):
        plan = runner.build_plan()
        self.assertEqual(plan["parameters"], {"repeats": 3, "pace_seconds": 40, "max_retries": 1,
                                              "retry_backoff_seconds": 60, "abort_after_consecutive_failures": 3})
        self.assertEqual(len(IDS), 10)
        self.assertEqual(plan["schedule"]["planned_calls"], 90)
        self.assertEqual(plan["schedule"]["planned_calls_by_stage"], {"e1": 30, "e2": 30, "e3": 30})
        self.assertEqual(plan["schedule"]["max_requests_with_runner_retry"], 180)
        self.assertEqual(plan["schedule"]["minimum_pacing_seconds"], 40 * 89)

    def test_the_plan_records_the_production_model_settings(self):
        model = runner.build_plan(repeats=1)["model"]
        self.assertEqual(model, {"max_tokens": 8192, "model": "openai/gpt-oss-20b", "reasoning_effort": "low",
                                 "temperature": 0.2})

    def test_the_plan_records_the_client_retry_override_separately_from_production_settings(self):
        plan = runner.build_plan(repeats=1)
        self.assertEqual(plan["evaluation_overrides"], {"max_retries": 0, "request_timeout": 120})
        self.assertNotIn("max_retries", plan["model"])           # production sets none; the override is not hidden in it
        self.assertEqual(plan["parameters"]["max_retries"], 1)   # the RUNNER's own retry policy is unchanged
        self.assertIn("max_retries=0", " ".join(plan["conditions"]))
        self.assertIn("CONFIGURED name", " ".join(plan["conditions"]))
        self.assertIn("provider-library retries are disabled", plan["schedule"]["max_requests_note"])

    def test_the_plan_command_states_that_client_retries_are_disabled(self):
        code, out, _ = self.run_main("plan", "--run-id", "t1", "--results-dir", self.results)
        self.assertEqual(code, 0)
        self.assertIn("max_retries=0", out)
        self.assertIn("max HTTP requests if every call is retried once: 180", out)

    def test_the_plan_is_deterministic_and_carries_no_timestamp_or_path(self):
        first, second = runner.build_plan(), runner.build_plan()
        self.assertEqual(first, second)
        text = runner.canonical_json(first)
        self.assertNotIn(ROOT, text)
        self.assertNotIn(self.tmp, text)
        self.assertIsNone(__import__("re").search(r"20\d\d-\d\d-\d\dT", text))
        self.assertTrue(first["run_id"].startswith("pipeline_") and len(first["run_id"]) == len("pipeline_") + 12)
        self.assertEqual(first["plan_hash"], runner._plan_hash(first))
        self.assertNotEqual(first["plan_hash"], runner.build_plan(repeats=2)["plan_hash"])

    def test_scenario_and_label_files_are_hashed_and_scenarios_stay_unfrozen(self):
        plan = runner.build_plan(repeats=1)
        for sid in IDS:
            for folder, key in ((dataset_module.DEFAULT_SCENARIOS_DIR, "scenario_file_sha256"),
                                (dataset_module.DEFAULT_EXPECTED_DIR, "label_file_sha256")):
                with open(os.path.join(folder, sid + ".json"), "rb") as handle:
                    digest = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
                self.assertEqual(plan["dataset"][key][sid], digest, (key, sid))
        self.assertEqual(plan["dataset"]["label_frozen_sha256"], DATASET["verified_hashes"])
        self.assertIs(plan["dataset"]["scenarios_frozen"], False)

    def test_prompts_sources_and_code_are_hashed(self):
        plan = runner.build_plan(repeats=1)
        for stage in runner.STAGES:
            entry = plan["prompts"][stage]
            self.assertEqual(entry["template_sha256"], runner.sha256_text(TEMPLATES[stage]))
            self.assertEqual(entry["variables"], list(runner.PROMPT_VARIABLES[stage]))
        self.assertEqual(sorted(plan["source_files"]), ["frontend/display.py", "src/parser.py", "src/prompts.yaml"])
        self.assertIn("runner.py", plan["code"])
        self.assertIn("cli.py", plan["code"])
        self.assertTrue(all(len(v) == 64 for v in plan["code"].values()))

    def test_plan_command_writes_only_the_run_folder_and_prints_the_call_count(self):
        code, out, err = self.run_main("plan", "--run-id", "t1", "--results-dir", self.results)
        self.assertEqual(code, 0, err)
        self.assertIn("planned_calls: 90", out)
        self.assertIn("NOT executed", out)
        self.assertIn("run --plan t1 --live --confirm-calls 90", out)
        self.assertEqual(os.listdir(self.results), ["t1"])
        self.assertEqual(os.listdir(self.run_dir()), ["plan.json"])
        plan = runner.load_plan(self.results, "t1")
        self.assertEqual(plan["schedule"]["planned_calls"], 90)

    def test_an_identical_plan_is_idempotent_and_a_different_one_is_refused(self):
        self.make_plan(repeats=1)
        self.assertEqual(self.run_main("plan", "--repeats", "1", "--run-id", "t1", "--results-dir", self.results)[0], 0)
        code, _, err = self.run_main("plan", "--repeats", "2", "--run-id", "t1", "--results-dir", self.results)
        self.assertEqual(code, 1)
        self.assertIn("different plan already exists", err)

    def test_invalid_parameters_are_usage_errors_and_write_nothing(self):
        for extra in (["--repeats", "0"], ["--repeats", "6"], ["--max-retries", "2"], ["--pace-seconds", "-1"],
                      ["--abort-after", "0"], ["--run-id", "../evil"], ["--run-id", ".hidden"]):
            code, _, err = self.run_main("plan", "--results-dir", self.results, *extra)
            self.assertEqual(code, 2, extra)
        self.assertFalse(os.path.exists(self.results))

    def _copied_root(self):
        root = os.path.join(self.tmp, "root")
        for parts in runner.SOURCE_FILES:
            os.makedirs(os.path.join(root, parts[0]), exist_ok=True)
            shutil.copy(os.path.join(runner.ROOT, *parts), os.path.join(root, *parts))
        return root

    def test_a_prompt_with_the_wrong_placeholders_is_refused(self):
        root = self._copied_root()
        path = os.path.join(root, "src", "prompts.yaml")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("{job_skills}", text)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("{job_skills}", "{job_skillz}"))
        code, _, err = self.run_main("plan", "--run-id", "t1", "--results-dir", self.results, "--root", root)
        self.assertEqual(code, 1)
        self.assertIn("placeholders", err)

    def test_a_plan_can_be_made_from_a_copied_root_and_matches_the_real_one(self):
        root = self._copied_root()
        self.assertEqual(runner.build_plan(repeats=1, run_id="t1", root=root),
                         runner.build_plan(repeats=1, run_id="t1"))


class LiveGate(RunnerCase):
    def setUp(self):
        super().setUp()
        self.plan = self.make_plan(repeats=1)

    def assertRefused(self, code, expected, factory_calls):
        self.assertEqual(code, expected)
        self.assertEqual(factory_calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.run_dir(), "calls.jsonl")))

    def test_run_requires_the_live_flag(self):
        factory_calls = []
        code, _, err = self.run_main("run", "--plan", "t1", "--confirm-calls", "30", "--results-dir", self.results,
                                     model_factory=lambda *a: factory_calls.append(a))
        self.assertRefused(code, 2, factory_calls)
        self.assertIn("--live", err)

    def test_run_requires_the_exact_call_count(self):
        for confirm in (None, 29, 31, 0):
            argv = ["run", "--plan", "t1", "--live", "--results-dir", self.results]
            if confirm is not None:
                argv += ["--confirm-calls", str(confirm)]
            factory_calls = []
            code, _, err = self.run_main(*argv, model_factory=lambda *a: factory_calls.append(a))
            self.assertRefused(code, 2, factory_calls)
            self.assertIn("planned_calls", err)

    def test_a_missing_plan_is_an_error_before_any_call(self):
        factory_calls = []
        code, _, _ = self.run_main("run", "--plan", "nope", "--live", "--confirm-calls", "30", "--results-dir",
                                   self.results, model_factory=lambda *a: factory_calls.append(a))
        self.assertEqual((code, factory_calls), (1, []))

    def test_an_edited_plan_file_is_detected_by_its_own_hash(self):
        path = os.path.join(self.run_dir(), "plan.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["parameters"]["pace_seconds"] = 0
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, _, err, sleeps, factory_calls = self.go_live(self.plan, ScriptedModel())
        self.assertRefused(code, 1, factory_calls)
        self.assertIn("plan_hash", err)

    def test_a_consistent_plan_that_no_longer_matches_the_code_is_refused(self):
        path = os.path.join(self.run_dir(), "plan.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["code"]["score.py"] = "0" * 64
        data["plan_hash"] = runner._plan_hash(data)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, _, err, sleeps, factory_calls = self.go_live(runner.load_plan(self.results, "t1"), ScriptedModel())
        self.assertRefused(code, 1, factory_calls)
        self.assertIn("code.score.py", err)

    def test_a_tampered_client_override_is_detected_before_any_call(self):
        path = os.path.join(self.run_dir(), "plan.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["evaluation_overrides"]["max_retries"] = 2
        data["plan_hash"] = runner._plan_hash(data)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, _, err, _, factory_calls = self.go_live(runner.load_plan(self.results, "t1"), ScriptedModel())
        self.assertRefused(code, 1, factory_calls)
        self.assertIn("evaluation_overrides.max_retries", err)

    def test_changed_prompts_or_source_files_refuse_the_run(self):
        root = os.path.join(self.tmp, "root")
        for parts in runner.SOURCE_FILES:
            os.makedirs(os.path.join(root, parts[0]), exist_ok=True)
            shutil.copy(os.path.join(runner.ROOT, *parts), os.path.join(root, *parts))
        plan = self.make_plan(run_id="t2", root=root)
        with open(os.path.join(root, "src", "prompts.yaml"), "a", encoding="utf-8") as handle:
            handle.write("\n# edited after planning\n")
        code, _, err, _, factory_calls = self.go_live(plan, ScriptedModel(), root=root)
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("source_files.src/prompts.yaml", err)

    def test_a_changed_scenario_file_refuses_the_run(self):
        scenarios = os.path.join(self.tmp, "scenarios")
        shutil.copytree(dataset_module.DEFAULT_SCENARIOS_DIR, scenarios)
        plan = runner.build_plan(repeats=1, run_id="t3", scenarios_dir=scenarios)
        runner.validate_frozen(plan, scenarios_dir=scenarios)                 # unchanged: passes
        with open(os.path.join(scenarios, IDS[0] + ".json"), "a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaises(runner.PlanError) as caught:
            runner.validate_frozen(plan, scenarios_dir=scenarios)
        self.assertIn("dataset.scenario_file_sha256." + IDS[0], str(caught.exception))

    def test_a_missing_api_key_is_refused_before_a_client_exists(self):
        code, _, err, _, factory_calls = self.go_live(self.plan, ScriptedModel(), key=None)
        self.assertRefused(code, 2, factory_calls)
        self.assertIn(runner.API_KEY_VARIABLE, err)

    def test_a_run_folder_that_already_holds_results_is_refused(self):
        with open(os.path.join(self.run_dir(), "calls.jsonl"), "w", encoding="utf-8") as handle:
            handle.write("")
        code, _, err, _, factory_calls = self.go_live(self.plan, ScriptedModel())
        self.assertEqual((code, factory_calls), (2, []))
        self.assertIn("already holds results", err)

    def test_a_valid_request_builds_one_client_with_the_frozen_settings(self):
        model = ScriptedModel()
        code, out, err, _, factory_calls = self.go_live(self.plan, model)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(factory_calls[0][0], dict(self.plan["model"], **self.plan["evaluation_overrides"]))
        self.assertEqual(factory_calls[0][0]["request_timeout"], 120)
        self.assertEqual(factory_calls[0][1], DUMMY_KEY)
        self.assertIn("summarize t1", out)

    def test_the_api_key_never_reaches_an_artifact_or_the_output(self):
        code, out, err, _, _ = self.go_live(self.plan, ScriptedModel())
        self.assertNotIn(DUMMY_KEY, out + err)
        for name in os.listdir(self.run_dir()):
            with open(os.path.join(self.run_dir(), name), encoding="utf-8") as handle:
                self.assertNotIn(DUMMY_KEY, handle.read(), name)


class Execution(RunnerCase):
    def run_full(self, repeats=3, model=None, run_id="t1", **plan_extra):
        plan = self.make_plan(repeats=repeats, run_id=run_id, **plan_extra)
        model = model or ScriptedModel()
        code, out, err, sleeps, _ = self.go_live(plan, model)
        return plan, model, code, sleeps

    def test_three_repeats_make_ninety_calls_in_the_planned_order(self):
        plan, model, code, sleeps = self.run_full()
        self.assertEqual(code, 0)
        self.assertEqual(len(model.invocations), 90)
        self.assertEqual(model.sequence, expected_sequence(3))

    def test_pacing_is_applied_between_logical_calls_only(self):
        _, _, _, sleeps = self.run_full()
        self.assertEqual(sleeps, [40] * 89)

    def test_e2_receives_the_cleaned_e1_output_and_e3_never_sees_e1_or_e2(self):
        _, model, _, _ = self.run_full(repeats=1)
        cleaned = runner.clean_response(E1_REPLY)
        for stage, sid, prompt in model.invocations:
            inputs = DATASET["scenarios"][sid]["inputs"]
            jd, resume = inputs["job_description"], inputs["resume_text"]
            if stage == "e1":
                self.assertEqual(prompt, runner.render_prompt(TEMPLATES["e1"], job_description=jd))
            elif stage == "e2":
                self.assertEqual(prompt, runner.render_prompt(TEMPLATES["e2"], job_description=jd, job_skills=cleaned,
                                                              resume_content=resume))
                self.assertIn(E1_MARKER, prompt)
            else:
                self.assertEqual(prompt, runner.render_prompt(TEMPLATES["e3"], job_description=jd, resume_content=resume))
                self.assertNotIn(E1_MARKER, prompt)

    def test_outputs_files_follow_the_cli_contract_and_keep_raw_and_normalised_e2(self):
        self.run_full()
        production = stages_module.load_parser()
        cleaned = runner.clean_response(E1_REPLY)
        for number in (1, 2, 3):
            data = self.read_json(f"outputs_rep{number}.json")
            self.assertEqual(data["format_version"], cli.OUTPUTS_VERSION)
            self.assertEqual(sorted(data["outputs"]), IDS)
            for sid, entry in data["outputs"].items():
                self.assertEqual(sorted(entry), sorted(cli.KNOWN_KEYS))
                self.assertEqual(entry["e1"], cleaned)
                self.assertEqual(entry["e2_raw"], E2_REPLY)
                self.assertEqual(entry["e3_raw"], runner.clean_response(E3_REPLY))
                inputs = DATASET["scenarios"][sid]["inputs"]
                direct = production.build_prioritized_missing_skills(E2_REPLY, inputs["job_description"], cleaned,
                                                                     inputs["resume_text"])[0]
                self.assertEqual(entry["e2_normalized"], json.loads(json.dumps(direct)))
                self.assertIsInstance(entry["e2_normalized"]["normalization_ok"], bool)

    def test_calls_log_has_one_line_per_attempt_and_no_response_text(self):
        self.run_full()
        calls = self.read_calls()
        self.assertEqual(len(calls), 90)
        self.assertEqual([c["sequence"] for c in calls], list(range(1, 91)))
        self.assertTrue(all(c["status"] == "ok" and c["attempt"] == 1 for c in calls))
        self.assertTrue(all(c["finish_reason"] == "stop" and c["configured_model_name"] == "openai/gpt-oss-20b"
                            for c in calls))
        self.assertTrue(all("model_name" not in c for c in calls))
        with open(os.path.join(self.run_dir(), "calls.jsonl"), encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in (E1_MARKER, "optimized_bullets", "crucial"):
            self.assertNotIn(forbidden, text)
        self.assertEqual({c["stage"] for c in calls}, {"e1", "e2", "e3"})

    def test_manifest_records_status_models_counts_and_output_hashes(self):
        plan, _, _, _ = self.run_full()
        manifest = self.read_json("manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["completed_repeats"], [1, 2, 3])
        self.assertEqual(manifest["plan_hash"], plan["plan_hash"])
        self.assertEqual(manifest["configured_model_names"], ["openai/gpt-oss-20b"])
        self.assertNotIn("observed_model_names", manifest)
        self.assertIn("not an identifier verified or reported independently by the provider", manifest["model_name_note"])
        self.assertEqual(manifest["calls"]["e1"], {"logical_calls": 30, "attempts": 30, "ok": 30, "failed": 0, "skipped": 0})
        self.assertEqual(manifest["finish_reasons"]["e3"], {"stop": 30})
        self.assertEqual(manifest["environment"], FIXED_ENV)
        self.assertEqual((manifest["started_at"], manifest["finished_at"]), (FIXED_NOW, FIXED_NOW))
        for name, digest in manifest["outputs_sha256"].items():
            self.assertEqual(runner.sha256_file(os.path.join(self.run_dir(), name)), digest)

    def test_one_transport_failure_is_retried_once_with_backoff(self):
        model = ScriptedModel(faults={("e1", IDS[0]): [TransportError()]})
        _, _, code, sleeps = self.run_full(repeats=1, model=model)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.invocations), 31)
        self.assertEqual(sleeps.count(60), 1)
        self.assertEqual(sleeps.count(40), 29)
        first = [c for c in self.read_calls() if c["scenario_id"] == IDS[0] and c["stage"] == "e1"]
        self.assertEqual([(c["attempt"], c["status"]) for c in first], [(1, "error"), (2, "ok")])
        self.assertEqual((first[0]["error_class"], first[0]["status_code"], first[0]["retryable"]), ("TransportError", 429, True))
        self.assertIsNotNone(self.read_json("outputs_rep1.json")["outputs"][IDS[0]]["e1"])

    def test_a_second_transport_failure_is_final_and_e3_still_runs(self):
        model = ScriptedModel(faults={("e1", IDS[0]): [TransportError(), ServerError()]})
        _, _, code, _ = self.run_full(repeats=1, model=model)
        self.assertEqual(code, 0)
        mine = [(s, st) for st, s, _ in model.invocations if s == IDS[0]]
        self.assertEqual(mine, [(IDS[0], "e1"), (IDS[0], "e1"), (IDS[0], "e3")])     # two attempts, no E2, then E3
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[0]]
        self.assertEqual((entry["e1"], entry["e2_raw"], entry["e2_normalized"]), (None, None, None))
        self.assertEqual(entry["e3_raw"], runner.clean_response(E3_REPLY))
        log = [c for c in self.read_calls() if c["scenario_id"] == IDS[0]]
        self.assertIn(("e2", "skipped"), [(c["stage"], c["status"]) for c in log])
        manifest = self.read_json("manifest.json")
        self.assertEqual((manifest["calls"]["e1"]["failed"], manifest["calls"]["e2"]["skipped"]), (1, 1))

    def test_a_permanent_error_is_not_retried(self):
        model = ScriptedModel(faults={("e2", IDS[1]): [PermanentError("SECRET-MESSAGE-TEXT")]})
        _, _, code, sleeps = self.run_full(repeats=1, model=model)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.invocations), 30)
        self.assertNotIn(60, sleeps)
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[1]]
        self.assertEqual((entry["e2_raw"], entry["e2_normalized"]), (None, None))
        self.assertIsNotNone(entry["e1"])
        self.assertIsNotNone(entry["e3_raw"])                     # E2 failed, E3 unaffected
        with open(os.path.join(self.run_dir(), "calls.jsonl"), encoding="utf-8") as handle:
            self.assertNotIn("SECRET-MESSAGE-TEXT", handle.read())

    def test_with_max_retries_zero_a_transport_failure_is_final(self):
        model = ScriptedModel(faults={("e3", IDS[2]): [TransportError()]})
        _, _, code, sleeps = self.run_full(repeats=1, model=model, max_retries=0)
        self.assertEqual(len(model.invocations), 30)
        self.assertNotIn(60, sleeps)
        self.assertIsNone(self.read_json("outputs_rep1.json")["outputs"][IDS[2]]["e3_raw"])

    def test_non_text_content_is_a_failure_and_is_not_retried(self):
        model = ScriptedModel(faults={("e1", IDS[0]): [FakeResponse(["not", "text"])]})
        self.run_full(repeats=1, model=model)
        self.assertEqual(len(model.invocations), 30 - 1)          # E2 skipped, no retry
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[0]]
        self.assertIsNone(entry["e1"])
        self.assertIn("invalid_content", [c["status"] for c in self.read_calls() if c["scenario_id"] == IDS[0]])

    def test_an_answer_that_is_not_json_is_a_result_not_a_failure(self):
        model = ScriptedModel(replies={"e1": "this is not json"})
        self.run_full(repeats=1, model=model)
        self.assertEqual(len(model.invocations), 30)              # nothing retried, E2 ran with the junk
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[0]]
        self.assertEqual(entry["e1"], "this is not json")
        self.assertIsNotNone(entry["e2_normalized"])

    def test_an_e3_failure_leaves_e1_and_e2_intact(self):
        model = ScriptedModel(faults={("e3", IDS[0]): [PermanentError()]})
        self.run_full(repeats=1, model=model)
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[0]]
        self.assertIsNone(entry["e3_raw"])
        self.assertIsNotNone(entry["e1"])
        self.assertIsNotNone(entry["e2_normalized"])

    def test_e3_prompts_are_identical_whatever_happens_to_e1_and_e2(self):
        healthy = ScriptedModel()
        self.run_full(repeats=1, model=healthy, run_id="ok")
        broken = ScriptedModel(faults={("e1", sid): [PermanentError()] for sid in IDS[:5]}
                               | {("e2", sid): [PermanentError()] for sid in IDS[5:]})
        self.run_full(repeats=1, model=broken, run_id="bad")
        self.assertEqual([p for st, _, p in healthy.invocations if st == "e3"],
                         [p for st, _, p in broken.invocations if st == "e3"])
        outputs = self.read_json("outputs_rep1.json", "bad")["outputs"]
        self.assertTrue(all(entry["e3_raw"] is not None for entry in outputs.values()))

    def test_the_run_aborts_after_consecutive_failures_and_makes_no_more_calls(self):
        model = ScriptedModel(fail_after=0)
        plan, _, code, _ = self.run_full(repeats=1, model=model)
        self.assertEqual(code, 1)
        self.assertEqual(len(model.invocations), 3)               # e1 fails, e3 fails, next e1 fails: abort
        manifest = self.read_json("manifest.json")
        self.assertEqual((manifest["status"], manifest["completed_repeats"]), ("aborted", []))
        self.assertIn("3 consecutive failed calls", manifest["abort_reason"])
        self.assertFalse(os.path.exists(os.path.join(self.run_dir(), "outputs_rep1.json")))

    def test_completed_repeats_survive_an_abort_in_a_later_repeat(self):
        model = ScriptedModel(fail_after=30)
        plan, _, code, _ = self.run_full(repeats=3, model=model)
        self.assertEqual(code, 1)
        manifest = self.read_json("manifest.json")
        self.assertEqual((manifest["status"], manifest["completed_repeats"]), ("aborted", [1]))
        self.assertTrue(os.path.exists(os.path.join(self.run_dir(), "outputs_rep1.json")))
        self.assertFalse(os.path.exists(os.path.join(self.run_dir(), "outputs_rep2.json")))

    def test_a_success_between_failures_resets_the_abort_counter(self):
        faults = {("e1", sid): [PermanentError()] for sid in IDS}          # every E1 fails, E3 always succeeds
        model = ScriptedModel(faults=faults)
        _, _, code, _ = self.run_full(repeats=1, model=model)
        self.assertEqual(code, 0)
        self.assertEqual(self.read_json("manifest.json")["status"], "completed")

    def test_two_identical_runs_write_byte_identical_artifacts(self):
        plan = runner.build_plan(repeats=2, run_id="same")
        copies = []
        for label in ("a", "b"):
            results = os.path.join(self.tmp, label)
            runner.write_plan(plan, results)
            code, _, _, _, _ = self.go_live(plan, ScriptedModel(), results=results)
            self.assertEqual(code, 0)
            folder = os.path.join(results, "same")
            copies.append({name: read_bytes(os.path.join(folder, name)) for name in sorted(os.listdir(folder))})
        self.assertEqual(copies[0], copies[1])
        self.assertEqual(sorted(copies[0]), ["calls.jsonl", "manifest.json", "outputs_rep1.json", "outputs_rep2.json",
                                              "plan.json"])


class SummarizeThroughTheRealCli(RunnerCase):
    def setUp(self):
        super().setUp()
        self.plan = self.make_plan(repeats=3)
        faults = {("e3", IDS[0]): [PermanentError(), PermanentError(), PermanentError()]}   # once per repeat
        self.model = ScriptedModel(faults=faults)
        self.assertEqual(self.go_live(self.plan, self.model)[0], 0)

    def summarize(self, *extra):
        return self.run_main("summarize", "t1", "--results-dir", self.results, *extra)

    def test_summarize_writes_cli_reports_review_rows_and_the_summary(self):
        code, out, err = self.summarize()
        self.assertEqual(code, 0, err)
        names = set(os.listdir(self.run_dir()))
        for n in (1, 2, 3):
            self.assertLessEqual({f"report_rep{n}.json", f"review_rows_rep{n}.json"}, names)
        self.assertLessEqual({"summary.json", "summary.md"}, names)

    def test_each_report_equals_a_direct_cli_run_on_the_same_outputs_file(self):
        self.summarize()
        for n in (1, 2, 3):
            outputs = os.path.join(self.run_dir(), f"outputs_rep{n}.json")
            out, err = io.StringIO(), io.StringIO()
            self.assertEqual(cli.main(["--outputs", outputs], stdout=out, stderr=err), 0)
            self.assertEqual(self.read_json(f"report_rep{n}.json"), json.loads(out.getvalue()))

    def test_a_failed_call_is_reported_as_missing_never_as_clean(self):
        self.summarize()
        report = self.read_json("report_rep1.json")
        first = [s for s in report["e3"]["scenarios"] if s["scenario_id"] == IDS[0]][0]["e3"]
        self.assertEqual(first["descriptive"]["output_status"], "output_missing")
        self.assertTrue(all(not m["available"] for m in first["metrics"]))
        self.assertIn(IDS[0], report["input"]["missing_stage_outputs"]["e3"])

    def test_summary_numbers_are_the_cli_numbers(self):
        self.summarize()
        summary = self.read_json("summary.json")
        reports = {n: self.read_json(f"report_rep{n}.json") for n in (1, 2, 3)}
        self.assertEqual(summary["repeats"], [1, 2, 3])
        self.assertEqual(summary["plan_hash"], self.plan["plan_hash"])
        stages = {row["stage"] for row in summary["metrics"]}
        self.assertEqual(stages, {"e1", "e2", "course", "e3"})
        for row in summary["metrics"]:
            self.assertEqual(row["screening_only"], row["stage"] == "e3")
            for entry in row["per_repeat"]:
                report = reports[entry["repeat"]]
                metrics = (report["e3"]["aggregate"]["metrics"] if row["stage"] == "e3"
                           else report["score"]["aggregate"][row["stage"]])
                direct = [m for m in metrics if m["metric"] == row["metric"]][0]
                self.assertEqual((entry["numerator"], entry["denominator"], entry["available"]),
                                 (direct["numerator"], direct["denominator"], direct["available"]))
                self.assertEqual(entry["per_scenario"], direct["per_scenario"])
        self.assertEqual(summary["call_outcomes"]["e3"]["logical_calls_failed"], 3)    # IDS[0] in each repeat
        self.assertEqual(summary["call_outcomes"]["e3"]["logical_calls_ok"], 27)
        self.assertEqual(summary["configured_model_names"], ["openai/gpt-oss-20b"])
        self.assertNotIn("observed_model_names", summary)

    def test_summary_is_deterministic_and_refuses_to_overwrite_without_force(self):
        self.assertEqual(self.summarize()[0], 0)
        first = {n: read_bytes(os.path.join(self.run_dir(), n)) for n in ("summary.json", "summary.md",
                                                                           "report_rep1.json")}
        code, _, err = self.summarize()
        self.assertEqual(code, 1)
        self.assertEqual(self.summarize("--force")[0], 0)
        second = {n: read_bytes(os.path.join(self.run_dir(), n)) for n in first}
        self.assertEqual(first, second)

    def test_markdown_shows_the_plan_hash_repeat_columns_and_screen_labels(self):
        self.summarize()
        with open(os.path.join(self.run_dir(), "summary.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn(self.plan["plan_hash"], text)
        self.assertIn("| rep 1 | rep 2 | rep 3 |", text)
        self.assertIn("(screen)", text)
        self.assertIn("configured model: openai/gpt-oss-20b", text)
        self.assertIn("not independently verified by the provider", text)
        self.assertNotIn("returned by the provider", text)
        self.assertIn("## Call outcomes", text)
        self.assertIn("## Limitations", text)

    def test_summarize_needs_outputs_and_a_matching_manifest(self):
        fresh = self.make_plan(repeats=1, run_id="empty")
        code, _, err = self.run_main("summarize", "empty", "--results-dir", self.results)
        self.assertEqual(code, 1)
        self.assertIn("nothing to summarize", err)
        path = os.path.join(self.run_dir(), "manifest.json")
        data = self.read_json("manifest.json")
        data["plan_hash"] = "0" * 64
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, _, err = self.summarize()
        self.assertEqual(code, 1)
        self.assertIn("different plan", err)

    def test_summarize_makes_no_model_call(self):
        factory_calls = []
        with mock.patch.object(runner, "make_groq_model", side_effect=AssertionError("model client created")):
            self.assertEqual(self.summarize()[0], 0)
        self.assertEqual(factory_calls, [])


class IsolationAndSafety(RunnerCase):
    @classmethod
    def setUpClass(cls):
        with open(RUNNER_PATH, encoding="utf-8") as handle:
            cls.source = handle.read()
        cls.tree = ast.parse(cls.source)

    def test_top_level_imports_are_standard_library_only(self):
        allowed = {"argparse", "ast", "hashlib", "importlib", "json", "os", "platform", "re", "string", "sys", "time",
                   "datetime"}
        found = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                found |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                found.add((node.module or "").split(".")[0])
        self.assertLessEqual(found, allowed, found - allowed)

    def test_the_model_client_is_imported_only_inside_make_groq_model(self):
        inside = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        names = [a.name for a in child.names] if isinstance(child, ast.Import) else [child.module]
                        inside.setdefault(node.name, set()).update(names)
        self.assertEqual(inside.get("make_groq_model"), {"langchain_groq"})
        self.assertEqual(inside.get("load_prompt_templates"), {"yaml"})
        self.assertEqual(inside.get("gather_environment"), {"importlib"})
        self.assertEqual(set(inside), {"make_groq_model", "load_prompt_templates", "gather_environment"})

    def test_no_forbidden_names_appear_in_the_source(self):
        lowered = self.source.lower()
        for token in ("dotenv", "backend", "resumeparser", "parse_resume", "subprocess", "socket", "urllib", "holdout",
                      "resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                      "missing_skills.txt", "ats_optimized_bullets.txt", "import requests", "os.getenv", "open(\".env"):
            self.assertNotIn(token, lowered, token)

    def test_the_key_is_read_from_the_environment_only(self):
        reads = [n for n in ast.walk(self.tree) if isinstance(n, ast.Attribute) and n.attr == "environ"]
        self.assertEqual(len(reads), 1)

    def test_plan_run_and_summarize_use_no_network_and_import_no_production_module(self):
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network used")):
            plan = self.make_plan(repeats=1)
            self.assertEqual(self.go_live(plan, ScriptedModel())[0], 0)
            self.assertEqual(self.run_main("summarize", "t1", "--results-dir", self.results)[0], 0)
        for name in ("backend", "backend.main", "src.parser", "parser", "langchain_groq", "groq"):
            self.assertNotIn(name, sys.modules, name)

    def test_nothing_is_written_outside_the_results_folder(self):
        before = dataset_hashes()
        plan = self.make_plan(repeats=1)
        self.go_live(plan, ScriptedModel())
        self.run_main("summarize", "t1", "--results-dir", self.results)
        self.assertEqual(os.listdir(self.tmp), ["results"])
        self.assertEqual(dataset_hashes(), before)
        self.assertEqual(len(before), 20)

    def test_make_groq_model_passes_the_production_arguments_plus_max_retries_zero_and_the_timeout(self):
        recorded = {}

        class RecordingChatGroq:
            def __init__(self, **kwargs):
                recorded.update(kwargs)

        fake_module = type(sys)("langchain_groq")
        fake_module.ChatGroq = RecordingChatGroq
        plan = runner.build_plan(repeats=1)
        settings = dict(plan["model"], **plan["evaluation_overrides"])
        with mock.patch.dict(sys.modules, {"langchain_groq": fake_module}):
            runner.make_groq_model(settings, DUMMY_KEY)
        self.assertEqual(recorded, {"model": "openai/gpt-oss-20b", "api_key": DUMMY_KEY, "temperature": 0.2,
                                    "max_tokens": 8192, "reasoning_effort": "low", "max_retries": 0, "timeout": 120})
        self.assertNotIn("langchain_groq", sys.modules)              # the stand-in is gone again
        with mock.patch.dict(sys.modules, {"langchain_groq": fake_module}):
            with self.assertRaises(KeyError):                          # the override cannot silently go missing
                runner.make_groq_model(dict(plan["model"]), DUMMY_KEY)

    def test_no_observed_model_wording_remains(self):
        self.assertNotIn("observed_model", self.source)
        self.assertNotIn("returned by the provider", self.source)

    def test_the_real_make_groq_model_is_not_used_by_these_tests(self):
        plan = self.make_plan(repeats=1)
        with mock.patch.object(runner, "make_groq_model", side_effect=AssertionError("real client requested")):
            code, _, err, _, _ = self.go_live(plan, ScriptedModel())    # an injected factory takes its place
        self.assertEqual(code, 0, err)


class TimeoutBehaviour(RunnerCase):
    def test_the_plan_records_the_request_timeout_as_an_override_not_a_production_setting(self):
        plan = runner.build_plan(repeats=1)
        self.assertEqual(plan["evaluation_overrides"]["request_timeout"], 120)
        self.assertEqual(plan["evaluation_overrides"]["max_retries"], 0)
        self.assertNotIn("request_timeout", plan["model"])
        self.assertIn("request_timeout", " ".join(plan["conditions"]))

    def test_the_timeout_is_part_of_the_plan_hash(self):
        base = runner.build_plan(repeats=1)
        with mock.patch.object(runner, "EVALUATION_OVERRIDES", {"max_retries": 0, "request_timeout": 90}):
            other = runner.build_plan(repeats=1)
        self.assertNotEqual(base["plan_hash"], other["plan_hash"])
        self.assertNotEqual(base["run_id"], other["run_id"])

    def test_the_plan_command_prints_the_timeout(self):
        code, out, _ = self.run_main("plan", "--run-id", "t1", "--results-dir", self.results)
        self.assertEqual(code, 0)
        self.assertIn("client request timeout: 120 s", out)
        self.assertIn("max_retries=0", out)

    def test_a_tampered_timeout_is_detected_before_any_call(self):
        plan = self.make_plan(repeats=1)
        path = os.path.join(self.run_dir(), "plan.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["evaluation_overrides"]["request_timeout"] = 600
        data["plan_hash"] = runner._plan_hash(data)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        code, _, err, _, factory_calls = self.go_live(runner.load_plan(self.results, "t1"), ScriptedModel())
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("evaluation_overrides.request_timeout", err)

    def test_the_factory_receives_the_timeout_and_the_zero_client_retries(self):
        plan = self.make_plan(repeats=1)
        _, _, _, _, factory_calls = self.go_live(plan, ScriptedModel())
        self.assertEqual((factory_calls[0][0]["request_timeout"], factory_calls[0][0]["max_retries"]), (120, 0))

    def test_a_timeout_is_retried_once_after_the_backoff_and_can_succeed(self):
        plan = self.make_plan(repeats=1)
        model = ScriptedModel(faults={("e2", IDS[0]): [APITimeoutError()]})
        code, _, _, sleeps, _ = self.go_live(plan, model)
        self.assertEqual(code, 0)
        self.assertIn(60, sleeps)
        lines = [l for l in self.read_calls() if l["scenario_id"] == IDS[0] and l["stage"] == "e2"]
        self.assertEqual([(l["attempt"], l["status"]) for l in lines], [(1, "error"), (2, "ok")])
        self.assertEqual((lines[0]["error_class"], lines[0]["retryable"]), ("APITimeoutError", True))
        self.assertIsNotNone(self.read_json("outputs_rep1.json")["outputs"][IDS[0]]["e2_raw"])

    def test_two_timeouts_are_a_final_failure_and_e3_still_runs(self):
        plan = self.make_plan(repeats=1)
        model = ScriptedModel(faults={("e1", IDS[0]): [APITimeoutError(), APITimeoutError()]})
        code, _, _, _, _ = self.go_live(plan, model)
        self.assertEqual(code, 0)
        entry = self.read_json("outputs_rep1.json")["outputs"][IDS[0]]
        self.assertEqual((entry["e1"], entry["e2_raw"]), (None, None))
        self.assertIsNotNone(entry["e3_raw"])
        lines = [l for l in self.read_calls() if l["scenario_id"] == IDS[0]]
        self.assertEqual([(l["stage"], l["attempt"], l["status"]) for l in lines],
                         [("e1", 1, "error"), ("e1", 2, "error"), ("e2", 0, "skipped"), ("e3", 1, "ok")])


class ContinuationCase(RunnerCase):
    def make_source(self, run_id="src", fail_after=30):
        """A source run interrupted inside repeat 2 (a permanent failure after the 30 calls of repeat 1)."""
        plan = self.make_plan(repeats=3, run_id=run_id)
        code = self.go_live(plan, ScriptedModel(fail_after=fail_after))[0]
        self.assertEqual(code, 1)
        return plan

    def make_continuation(self, run_id="cont", source="src", **extra):
        return self.make_plan(repeats=2, run_id=run_id, continues_from=source, **extra)

    def folder_bytes(self, run_id):
        folder = self.run_dir(run_id)
        return {name: read_bytes(os.path.join(folder, name)) for name in sorted(os.listdir(folder))}

    def rewrite(self, run_id, name, edit):
        path = os.path.join(self.run_dir(run_id), name)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        edit(data)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def plan_refused(self, source="src", repeats=2, **extra):
        argv = ["plan", "--repeats", str(repeats), "--run-id", "refused", "--results-dir", self.results,
                "--continues-from", source]
        for key, value in extra.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        code, out, err = self.run_main(*argv)
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists(self.run_dir("refused")))
        return code, err


class ContinuationPlan(ContinuationCase):
    def setUp(self):
        super().setUp()
        self.source = self.make_source()
        self.cont = self.make_continuation()

    def test_the_plan_covers_exactly_repeats_two_and_three_with_sixty_calls(self):
        self.assertEqual((self.cont["parameters"]["repeats"], self.cont["parameters"]["first_repeat"]), (2, 2))
        self.assertEqual(self.cont["schedule"]["planned_calls"], 60)
        self.assertEqual(self.cont["schedule"]["planned_calls_by_stage"], {"e1": 20, "e2": 20, "e3": 20})
        self.assertEqual(self.cont["schedule"]["max_requests_with_runner_retry"], 120)
        self.assertEqual(self.cont["schedule"]["minimum_pacing_seconds"], 40 * 59)

    def test_the_plan_records_the_source_identity_and_file_hashes(self):
        block = self.cont["continues"]
        folder = self.run_dir("src")
        self.assertEqual(block["run_id"], "src")
        self.assertEqual(block["plan_hash"], self.source["plan_hash"])
        self.assertEqual(block["outputs_rep1_sha256"], runner.sha256_file(os.path.join(folder, "outputs_rep1.json")))
        self.assertEqual(block["calls_jsonl_sha256"], runner.sha256_file(os.path.join(folder, "calls.jsonl")))
        lines = self.read_calls("src")
        self.assertEqual(block["excluded_source_log_lines"], len([l for l in lines if l["repeat"] != 1]))
        self.assertGreater(block["excluded_source_log_lines"], 0)

    def test_the_plan_is_deterministic(self):
        first = runner.build_plan(repeats=2, run_id="cont", continues_from="src", results_dir=self.results)
        second = runner.build_plan(repeats=2, run_id="cont", continues_from="src", results_dir=self.results)
        self.assertEqual(first, second)
        self.assertEqual(first, self.cont)
        automatic = runner.build_plan(repeats=2, continues_from="src", results_dir=self.results)
        self.assertEqual(automatic, runner.build_plan(repeats=2, continues_from="src", results_dir=self.results))
        self.assertRegex(automatic["run_id"], r"^pipeline_[0-9a-f]{12}$")
        self.assertEqual(automatic["plan_hash"], runner._plan_hash(automatic))

    def test_the_timeout_difference_is_disclosed_in_the_plan(self):
        text = " ".join(self.cont["conditions"] + self.cont["limitations"])
        self.assertIn("set no explicit client request timeout", text)
        self.assertIn("request_timeout=120 s", text)
        self.assertIn("repeats 2-3", text)

    def test_only_the_runner_file_the_timeout_and_the_description_differ_from_the_source(self):
        runner._check_compatible(self.source, self.cont, "src")          # raises on any other difference
        for section in ("dataset", "model", "prompts", "source_files"):
            self.assertEqual(self.source[section], self.cont[section])

    def test_the_plan_command_prints_the_continuation_and_the_exact_live_command(self):
        code, out, _ = self.run_main("plan", "--repeats", "2", "--run-id", "cont2", "--results-dir", self.results,
                                     "--continues-from", "src")
        self.assertEqual(code, 0)
        self.assertIn("continues run src", out)
        self.assertIn("repeats 2-3", out)
        self.assertIn("run --plan cont2 --live --confirm-calls 60", out)

    def test_only_two_repeats_are_allowed(self):
        for repeats in (1, 3):
            code, err = self.plan_refused(repeats=repeats)
            self.assertEqual(code, 2)
            self.assertIn("exactly 2 repeats", err)

    def test_the_source_must_hold_a_complete_repeat_one_only(self):
        self.make_source("short", fail_after=10)                          # repeat 1 never finished
        code, err = self.plan_refused(source="short")
        self.assertEqual(code, 1)
        self.assertIn("repeat 1 only", err)

    def test_a_source_with_two_complete_repeats_is_refused(self):
        full = self.make_plan(repeats=2, run_id="full")
        self.assertEqual(self.go_live(full, ScriptedModel())[0], 0)
        code, err = self.plan_refused(source="full")
        self.assertEqual(code, 1)
        self.assertIn("repeat 1 only", err)

    def test_a_source_whose_log_lacks_a_repeat_one_call_is_refused(self):
        path = os.path.join(self.run_dir("src"), "calls.jsonl")
        lines = read_bytes(path).decode("utf-8").splitlines(keepends=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("".join(lines[1:]))
        code, err = self.plan_refused()
        self.assertEqual(code, 1)
        self.assertIn("not complete in its call log", err)

    def test_a_source_output_with_a_missing_field_is_refused(self):
        self.rewrite("src", "outputs_rep1.json", lambda d: d["outputs"][IDS[3]].update(e2_raw=None))
        code, err = self.plan_refused()
        self.assertEqual(code, 1)
        self.assertIn("incomplete", err)
        self.assertIn(IDS[3], err)

    def test_an_edited_source_plan_is_refused(self):
        self.rewrite("src", "plan.json", lambda d: d["parameters"].update(pace_seconds=0))
        code, err = self.plan_refused()
        self.assertEqual(code, 1)
        self.assertIn("plan_hash", err)

    def test_a_missing_source_is_refused(self):
        code, err = self.plan_refused(source="nope")
        self.assertEqual(code, 1)

    def test_changed_pacing_makes_the_continuation_incompatible(self):
        code, err = self.plan_refused(pace_seconds=30)
        self.assertEqual(code, 1)
        self.assertIn("incompatible", err)
        self.assertIn("parameters.pace_seconds", err)

    def test_changed_prompts_make_the_continuation_incompatible(self):
        root = os.path.join(self.tmp, "root")
        for parts in runner.SOURCE_FILES:
            os.makedirs(os.path.join(root, parts[0]), exist_ok=True)
            shutil.copy(os.path.join(runner.ROOT, *parts), os.path.join(root, *parts))
        with open(os.path.join(root, "src", "prompts.yaml"), "a", encoding="utf-8") as handle:
            handle.write("\n# edited\n")
        argv = ["plan", "--repeats", "2", "--run-id", "cont", "--results-dir", self.results, "--continues-from", "src",
                "--root", root]
        code, _, err = self.run_main(*argv)
        self.assertEqual(code, 1)
        self.assertIn("incompatible", err)
        self.assertIn("source_files.src/prompts.yaml", err)

    def test_changed_model_settings_scoring_code_or_dataset_make_it_incompatible(self):
        for section, edit in (("model", lambda p: p["model"].update(temperature=0.9)),
                              ("code", lambda p: p["code"].update({"score.py": "0" * 64})),
                              ("dataset", lambda p: p["dataset"]["scenario_file_sha256"].update({IDS[0]: "0" * 64}))):
            changed = json.loads(json.dumps(self.cont))
            edit(changed)
            with self.assertRaises(runner.PlanError) as caught:
                runner._check_compatible(self.source, changed, "src")
            self.assertIn("incompatible", str(caught.exception))
            self.assertIn(section, str(caught.exception))

    def test_a_changed_client_retry_override_makes_it_incompatible(self):
        changed = json.loads(json.dumps(self.cont))
        changed["evaluation_overrides"]["max_retries"] = 1
        with self.assertRaises(runner.PlanError):
            runner._check_compatible(self.source, changed, "src")

    def test_validate_frozen_needs_the_results_directory_for_a_continuation(self):
        with self.assertRaises(runner.PlanError):
            runner.validate_frozen(self.cont)
        runner.validate_frozen(self.cont, results_dir=self.results)


class ContinuationRun(ContinuationCase):
    def setUp(self):
        super().setUp()
        self.source = self.make_source()
        self.cont = self.make_continuation()
        self.source_before = self.folder_bytes("src")

    def test_a_source_file_changed_after_planning_refuses_the_run_before_any_call(self):
        with open(os.path.join(self.run_dir("src"), "outputs_rep1.json"), "a", encoding="utf-8") as handle:
            handle.write(" ")
        code, _, err, _, factory_calls = self.go_live(self.cont, ScriptedModel())
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("continues.outputs_rep1_sha256", err)
        self.assertFalse(os.path.exists(os.path.join(self.run_dir("cont"), "calls.jsonl")))

    def test_a_changed_source_call_log_refuses_the_run(self):
        with open(os.path.join(self.run_dir("src"), "calls.jsonl"), "a", encoding="utf-8") as handle:
            handle.write("\n")
        with open(os.path.join(self.run_dir("src"), "calls.jsonl"), "rb") as handle:
            self.assertTrue(handle.read())
        before = runner.sha256_file(os.path.join(self.run_dir("src"), "calls.jsonl"))
        self.assertNotEqual(before, self.cont["continues"]["calls_jsonl_sha256"])
        code, _, err, _, factory_calls = self.go_live(self.cont, ScriptedModel())
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("continues.calls_jsonl_sha256", err)

    def test_a_source_plan_replaced_by_another_valid_plan_refuses_the_run(self):
        other = runner.build_plan(repeats=2, run_id="src")      # valid and compatible, but not the recorded plan
        with open(os.path.join(self.run_dir("src"), "plan.json"), "w", encoding="utf-8") as handle:
            handle.write(cli.to_json(other))
        code, _, err, _, factory_calls = self.go_live(self.cont, ScriptedModel())
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("continues.plan_hash", err)

    def test_an_edited_continuation_plan_is_refused(self):
        self.rewrite("cont", "plan.json", lambda d: d["continues"].update(run_id="elsewhere"))
        code, _, err, _, factory_calls = self.go_live(self.cont, ScriptedModel())
        self.assertEqual((code, factory_calls), (1, []))
        self.assertIn("plan_hash", err)

    def test_a_continuation_folder_that_already_holds_results_is_refused(self):
        with open(os.path.join(self.run_dir("cont"), "outputs_rep2.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        code, _, err, _, factory_calls = self.go_live(self.cont, ScriptedModel())
        self.assertEqual((code, factory_calls), (2, []))
        self.assertIn("already holds results", err)

    def test_the_run_makes_sixty_calls_for_repeats_two_and_three(self):
        model = ScriptedModel()
        code, out, err, sleeps, factory_calls = self.go_live(self.cont, model)
        self.assertEqual(code, 0, err)
        self.assertEqual(model.sequence, expected_sequence(2))
        self.assertEqual(len(model.invocations), 60)
        self.assertEqual(sleeps, [40] * 59)
        self.assertEqual(factory_calls[0][0]["request_timeout"], 120)
        names = set(os.listdir(self.run_dir("cont")))
        self.assertLessEqual({"outputs_rep2.json", "outputs_rep3.json", "calls.jsonl", "manifest.json"}, names)
        self.assertNotIn("outputs_rep1.json", names)
        self.assertEqual({l["repeat"] for l in self.read_calls("cont")}, {2, 3})
        manifest = self.read_json("manifest.json", "cont")
        self.assertEqual((manifest["status"], manifest["completed_repeats"]), ("completed", [2, 3]))
        self.assertIn("summarize cont", out)

    def test_the_source_run_is_byte_identical_after_the_run_and_the_summary(self):
        self.assertEqual(self.go_live(self.cont, ScriptedModel())[0], 0)
        self.assertEqual(self.run_main("summarize", "cont", "--results-dir", self.results)[0], 0)
        self.assertEqual(self.folder_bytes("src"), self.source_before)

    def test_nothing_is_written_outside_the_two_run_folders(self):
        self.go_live(self.cont, ScriptedModel())
        self.run_main("summarize", "cont", "--results-dir", self.results)
        self.assertEqual(os.listdir(self.tmp), ["results"])
        self.assertEqual(sorted(os.listdir(self.results)), ["cont", "src"])


class ContinuationSummary(ContinuationCase):
    def setUp(self):
        super().setUp()
        self.source = self.make_source()
        self.cont = self.make_continuation()
        self.assertEqual(self.go_live(self.cont, ScriptedModel())[0], 0)
        self.source_before = self.folder_bytes("src")

    def summarize(self, *extra):
        return self.run_main("summarize", "cont", "--results-dir", self.results, *extra)

    def test_all_three_repeats_are_reported_in_the_continuation_folder(self):
        code, out, err = self.summarize()
        self.assertEqual(code, 0, err)
        names = set(os.listdir(self.run_dir("cont")))
        for n in (1, 2, 3):
            self.assertLessEqual({f"report_rep{n}.json", f"review_rows_rep{n}.json"}, names)
        self.assertEqual(self.read_json("summary.json", "cont")["repeats"], [1, 2, 3])
        self.assertNotIn("report_rep1.json", os.listdir(self.run_dir("src")))
        self.assertEqual(self.folder_bytes("src"), self.source_before)

    def test_repeat_one_is_the_cli_result_of_the_original_outputs_and_repeats_two_and_three_of_the_new_run(self):
        self.summarize()
        for n, folder in ((1, "src"), (2, "cont"), (3, "cont")):
            out, err = io.StringIO(), io.StringIO()
            outputs = os.path.join(self.run_dir(folder), f"outputs_rep{n}.json")
            self.assertEqual(cli.main(["--outputs", outputs], stdout=out, stderr=err), 0)
            self.assertEqual(self.read_json(f"report_rep{n}.json", "cont"), json.loads(out.getvalue()))
        summary = self.read_json("summary.json", "cont")
        self.assertEqual(summary["outputs_sha256"]["outputs_rep1.json"],
                         runner.sha256_file(os.path.join(self.run_dir("src"), "outputs_rep1.json")))
        self.assertEqual(summary["continuation"]["repeat_sources"], {"1": "src", "2": "cont", "3": "cont"})
        for row in summary["metrics"]:
            self.assertEqual([e["repeat"] for e in row["per_repeat"]], [1, 2, 3])

    def test_the_incomplete_repeat_two_calls_of_the_source_are_excluded(self):
        source_lines = self.read_calls("src")
        self.assertTrue([l for l in source_lines if l["repeat"] == 2 and l["status"] == "error"])
        self.summarize()
        summary = self.read_json("summary.json", "cont")
        outcomes = summary["call_outcomes"]
        for stage in runner.STAGES:
            self.assertEqual((outcomes[stage]["logical_calls_ok"], outcomes[stage]["logical_calls_failed"],
                              outcomes[stage]["skipped"]), (30, 0, 0), stage)
            self.assertEqual((outcomes[stage]["attempts"], outcomes[stage]["retried_attempts"]), (30, 0), stage)
        self.assertEqual(summary["continuation"]["excluded_source_log_lines"],
                         len([l for l in source_lines if l["repeat"] != 1]))

    def test_the_timeout_difference_is_disclosed_in_json_and_markdown(self):
        self.summarize()
        summary = self.read_json("summary.json", "cont")
        self.assertEqual(summary["continuation"]["client_request_timeout"],
                         {"1": "none set (library default)", "2": "120 s", "3": "120 s"})
        with open(os.path.join(self.run_dir("cont"), "summary.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("## Provenance and conditions", text)
        self.assertIn("repeat 1: none set (library default); repeat 2: 120 s; repeat 3: 120 s", text)
        self.assertIn("| rep 1 | rep 2 | rep 3 |", text)
        self.assertIn(self.source["plan_hash"], text)
        self.assertIn("set no explicit client request timeout", " ".join(summary["conditions"]))

    def test_the_summary_is_deterministic_and_refuses_to_overwrite_without_force(self):
        self.assertEqual(self.summarize()[0], 0)
        names = ("summary.json", "summary.md", "report_rep1.json", "review_rows_rep1.json")
        first = {n: read_bytes(os.path.join(self.run_dir("cont"), n)) for n in names}
        self.assertEqual(self.summarize()[0], 1)
        self.assertEqual(self.summarize("--force")[0], 0)
        self.assertEqual(first, {n: read_bytes(os.path.join(self.run_dir("cont"), n)) for n in names})

    def test_a_changed_source_file_refuses_the_summary(self):
        with open(os.path.join(self.run_dir("src"), "outputs_rep1.json"), "a", encoding="utf-8") as handle:
            handle.write(" ")
        code, _, err = self.summarize()
        self.assertEqual(code, 1)
        self.assertIn("no longer matches", err)
        self.assertFalse(os.path.exists(os.path.join(self.run_dir("cont"), "summary.json")))

    def test_a_continuation_folder_holding_outputs_for_repeat_one_is_refused(self):
        shutil.copy(os.path.join(self.run_dir("src"), "outputs_rep1.json"), os.path.join(self.run_dir("cont"),
                                                                                      "outputs_rep1.json"))
        code, _, err = self.summarize()
        self.assertEqual(code, 1)
        self.assertIn("must not hold outputs_rep1.json", err)

    def test_planning_and_summarizing_a_continuation_use_no_model_and_no_network(self):
        with mock.patch.object(runner, "make_groq_model", side_effect=AssertionError("model client created")), \
                mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network used")):
            self.assertEqual(self.run_main("plan", "--repeats", "2", "--run-id", "cont3", "--results-dir", self.results,
                                           "--continues-from", "src")[0], 0)
            self.assertEqual(self.summarize()[0], 0)
        for name in ("backend", "backend.main", "langchain_groq", "groq"):
            self.assertNotIn(name, sys.modules, name)

    def test_a_continuation_cannot_be_the_source_of_another(self):
        code, _, err = self.run_main("plan", "--repeats", "2", "--run-id", "chain", "--results-dir", self.results,
                                     "--continues-from", "cont")
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(self.run_dir("chain")))


if __name__ == "__main__":
    unittest.main()
