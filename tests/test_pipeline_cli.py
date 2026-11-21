"""Tests for eval/pipeline/cli.py (Step 5): the offline evaluation command line.

Offline: no network, no model, no .env, no protected resume artifact, no production import.
The real dataset is READ (it is the thing the CLI verifies); every file the CLI writes goes to a temporary folder.
Outputs used here are tiny synthetic fixtures, not model results, so no test measures model performance.
"""
import ast
import contextlib
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
CLI_PATH = os.path.join(PIPELINE, "cli.py")


def _load(name, filename):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, os.path.join(PIPELINE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = _load("pipeline_cli", "cli.py")
dataset_module = cli.dataset_module
score = cli.score
e3 = cli.e3


def dataset_hashes():
    found = {}
    for folder in (dataset_module.DEFAULT_SCENARIOS_DIR, dataset_module.DEFAULT_EXPECTED_DIR):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                with open(os.path.join(folder, name), "rb") as handle:
                    found[os.path.join(os.path.basename(folder), name)] = hashlib.sha256(handle.read()).hexdigest()
    return found


DATASET = dataset_module.load_dataset()
IDS = list(DATASET["ids"])
E3_GOOD = json.dumps({"optimized_bullets": [{"section": "Experience", "bullets": ["Built data pipelines."]}]})


def sample_outputs():
    """Two scenarios with outputs (one usable, one unusable E3); the rest have none."""
    tiers = score._tiers()
    return {
        IDS[0]: {"e1": {category: [] for category in score.E1_CATEGORIES},
                 "e2_normalized": dict({tier: [] for tier in tiers}, normalization_ok=True),
                 "e3_raw": E3_GOOD},
        IDS[1]: {"e3_raw": "this is not json"},
    }


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pipeline_cli_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write(self, name, content, binary=False):
        target = self.path(name)
        with open(target, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as handle:
            handle.write(content)
        return target

    def write_outputs(self, outputs=None, name="outputs.json", wrapped=True):
        outputs = sample_outputs() if outputs is None else outputs
        body = {"format_version": cli.OUTPUTS_VERSION, "outputs": outputs} if wrapped else outputs
        return self.write(name, json.dumps(body))

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(list(argv), stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    def report_of(self, *argv):
        code, out, err = self.run_cli(*argv)
        return code, json.loads(out), err


class TemplateAndValidate(CliTestCase):
    def test_validate_dataset_succeeds_on_the_real_dataset(self):
        code, out, _ = self.run_cli("--validate-dataset")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(data["dataset_ok"])
        self.assertEqual(data["ids"], IDS)
        self.assertEqual(data["verified_hashes"], DATASET["verified_hashes"])

    def test_template_lists_every_scenario_with_every_key_null(self):
        code, out, _ = self.run_cli("--template")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(data["format_version"], cli.OUTPUTS_VERSION)
        self.assertEqual(sorted(data["outputs"]), IDS)
        for entry in data["outputs"].values():
            self.assertEqual(entry, {"e1": None, "e2_normalized": None, "e2_raw": None, "e3_raw": None})

    def test_the_template_round_trips_as_an_all_missing_run(self):
        _, template, _ = self.run_cli("--template")
        outputs = self.write("template.json", template)
        code, report, _ = self.report_of("--outputs", outputs)
        self.assertEqual(code, 0)
        self.assertEqual(report["input"]["scenario_ids_without_entry"], [])
        for stage in cli.STAGES:
            self.assertEqual(report["input"]["missing_stage_outputs"][stage], IDS)
        self.assertEqual(report["e3"]["descriptive"]["output_status_distribution"], {"output_missing": len(IDS)})
        for metric in report["e3"]["aggregate"]["metrics"]:
            self.assertFalse(metric["available"], metric["metric"])

    def test_output_flags_are_a_usage_error_with_the_dataset_modes(self):
        for mode in ("--validate-dataset", "--template"):
            code, _, err = self.run_cli(mode, "--output-report", self.path("r.json"))
            self.assertEqual(code, 2, mode)
            self.assertFalse(os.path.exists(self.path("r.json")))

    def test_exactly_one_mode_is_required(self):
        self.assertEqual(self.run_cli()[0], 2)
        self.assertEqual(self.run_cli("--template", "--validate-dataset")[0], 2)
        self.assertEqual(self.run_cli("--outputs", self.path("x.json"), "--template")[0], 2)

    def test_a_tampered_label_stops_the_run_with_exit_1(self):
        scenarios, expected = self.path("scenarios"), self.path("expected")
        shutil.copytree(dataset_module.DEFAULT_SCENARIOS_DIR, scenarios, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(dataset_module.DEFAULT_EXPECTED_DIR, expected, ignore=shutil.ignore_patterns("__pycache__"))
        victim = os.path.join(expected, sorted(f for f in os.listdir(expected) if f.endswith(".json"))[0])
        with open(victim, "ab") as handle:
            handle.write(b" ")
        outputs = self.write_outputs()
        for argv in (("--validate-dataset",), ("--outputs", outputs), ("--template",)):
            code, out, err = self.run_cli(*argv, "--scenarios-dir", scenarios, "--expected-dir", expected)
            self.assertEqual(code, 1, argv)
            self.assertEqual(out, "")
            self.assertIn("hashes to", err)

    def test_the_untampered_copy_still_passes(self):
        scenarios, expected = self.path("scenarios"), self.path("expected")
        shutil.copytree(dataset_module.DEFAULT_SCENARIOS_DIR, scenarios, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(dataset_module.DEFAULT_EXPECTED_DIR, expected, ignore=shutil.ignore_patterns("__pycache__"))
        self.assertEqual(self.run_cli("--validate-dataset", "--scenarios-dir", scenarios, "--expected-dir", expected)[0], 0)


class ReadingTheOutputsFile(CliTestCase):
    def assertExit1(self, outputs_path, fragment):
        code, out, err = self.run_cli("--outputs", outputs_path)
        self.assertEqual(code, 1, err)
        self.assertEqual(out, "")
        self.assertIn(fragment, err)

    def test_a_missing_file(self):
        self.assertExit1(self.path("nope.json"), "cannot be read")

    def test_invalid_json(self):
        self.assertExit1(self.write("a.json", "{not json"), "not valid JSON")

    def test_duplicate_keys(self):
        self.assertExit1(self.write("a.json", '{"%s": {}, "%s": {}}' % (IDS[0], IDS[0])), "duplicate key")

    def test_a_bom(self):
        self.assertExit1(self.write("a.json", b"\xef\xbb\xbf{}", binary=True), "BOM")

    def test_a_non_object_top_level(self):
        self.assertExit1(self.write("a.json", "[1, 2]"), "not a JSON object")

    def test_a_non_dict_scenario_entry(self):
        self.assertExit1(self.write("a.json", json.dumps({IDS[0]: ["x"]})), "must be a JSON object")
        self.assertExit1(self.write_outputs({IDS[0]: "x"}, "b.json"), "must be a JSON object")

    def test_a_wrong_format_version(self):
        self.assertExit1(self.write("a.json", json.dumps({"format_version": "v0", "outputs": {}})), "format_version")

    def test_a_wrapper_without_an_outputs_object(self):
        body = json.dumps({"format_version": cli.OUTPUTS_VERSION, "outputs": []})
        self.assertExit1(self.write("a.json", body), "'outputs' must be")

    def test_an_unexpected_wrapper_key(self):
        body = json.dumps({"format_version": cli.OUTPUTS_VERSION, "outputs": {}, "extra": 1})
        self.assertExit1(self.write("a.json", body), "unexpected top-level key")

    def test_the_bare_form_gives_the_same_scores_as_the_wrapped_form(self):
        _, wrapped, _ = self.report_of("--outputs", self.write_outputs(wrapped=True, name="w.json"))
        _, bare, _ = self.report_of("--outputs", self.write_outputs(wrapped=False, name="b.json"))
        self.assertEqual((wrapped["input"]["envelope"], bare["input"]["envelope"]), ("wrapped", "bare"))
        self.assertEqual(wrapped["score"], bare["score"])
        self.assertEqual(wrapped["e3"], bare["e3"])

    def test_the_file_digest_is_reported(self):
        target = self.write_outputs()
        with open(target, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(self.report_of("--outputs", target)[1]["input"]["outputs_file_sha256"], digest)


class UnknownIdsAndKeys(CliTestCase):
    def outputs_with_unknown(self):
        outputs = sample_outputs()
        outputs["zz_not_a_scenario"] = {"e3_raw": E3_GOOD, "e1": {}}
        return outputs

    def test_unknown_ids_are_warnings_and_are_not_scored(self):
        code, report, _ = self.report_of("--outputs", self.write_outputs(self.outputs_with_unknown()))
        self.assertEqual(code, 0)
        self.assertEqual(report["input"]["unknown_ids_ignored"], ["zz_not_a_scenario"])
        self.assertTrue(any("zz_not_a_scenario" in w and "ignored" in w for w in report["warnings"]))
        listed = [s["scenario_id"] for s in report["score"]["scenarios"]] + \
                 [s["scenario_id"] for s in report["e3"]["scenarios"]]
        self.assertNotIn("zz_not_a_scenario", listed)
        self.assertEqual(report["score"]["unknown_output_ids"], [])
        self.assertEqual(report["e3"]["unknown_output_ids"], [])

    def test_unknown_ids_do_not_change_any_score(self):
        _, with_unknown, _ = self.report_of("--outputs", self.write_outputs(self.outputs_with_unknown(), "u.json"))
        _, without, _ = self.report_of("--outputs", self.write_outputs(name="k.json"))
        self.assertEqual(with_unknown["score"], without["score"])
        self.assertEqual(with_unknown["e3"], without["e3"])

    def test_an_unrecognised_key_is_a_warning_and_is_ignored(self):
        outputs = sample_outputs()
        outputs[IDS[0]]["e4"] = "whatever"
        code, report, _ = self.report_of("--outputs", self.write_outputs(outputs))
        self.assertEqual(code, 0)
        self.assertTrue(any(IDS[0] in w and "'e4'" in w for w in report["warnings"]))

    def test_strict_turns_warnings_into_exit_3(self):
        target = self.write_outputs(self.outputs_with_unknown())
        self.assertEqual(self.run_cli("--outputs", target)[0], 0)
        self.assertEqual(self.run_cli("--outputs", target, "--strict")[0], 3)


class StageSelection(CliTestCase):
    def forbid(self, *names):
        stack = contextlib.ExitStack()
        mocks = {}
        for owner, name in names:
            mocks[name] = stack.enter_context(mock.patch.object(owner, name, side_effect=AssertionError(name + " ran")))
        self.addCleanup(stack.close)
        return mocks

    def test_unknown_or_empty_stage_is_a_usage_error(self):
        target = self.write_outputs()
        self.assertEqual(self.run_cli("--outputs", target, "--stages", "e1,e9")[0], 2)
        self.assertEqual(self.run_cli("--outputs", target, "--stages", ",")[0], 2)

    def test_stages_are_canonical_and_deduplicated(self):
        self.assertEqual(cli.parse_stages("e3, e1,e3"), ("e1", "e3"))

    def test_only_e3_runs_no_other_scoring_function(self):
        self.forbid((score, "score_e1"), (score, "score_e2"), (score, "score_course"), (score, "score_dataset"))
        code, report, err = self.report_of("--outputs", self.write_outputs(), "--stages", "e3")
        self.assertEqual(code, 0, err)
        self.assertIsNone(report["score"])
        self.assertEqual(report["stages_run"], ["e3"])
        self.assertEqual(list(report["input"]["missing_stage_outputs"]), ["e3"])
        self.assertIsNotNone(report["e3"])

    def test_only_e1_runs_neither_e2_course_nor_e3(self):
        self.forbid((score, "score_e2"), (score, "score_course"), (e3, "score_e3_dataset"), (e3, "score_e3"))
        code, report, err = self.report_of("--outputs", self.write_outputs(), "--stages", "e1")
        self.assertEqual(code, 0, err)
        self.assertIsNone(report["e3"])
        self.assertEqual(list(report["score"]["aggregate"]), ["e1"])
        for scenario in report["score"]["scenarios"]:
            self.assertEqual(list(scenario["stages"]), ["e1"])

    def test_only_course_runs_neither_e1_nor_e2_scoring(self):
        self.forbid((score, "score_e1"), (score, "score_e2"), (e3, "score_e3_dataset"))
        code, report, err = self.report_of("--outputs", self.write_outputs(), "--stages", "course")
        self.assertEqual(code, 0, err)
        self.assertEqual(list(report["score"]["aggregate"]), ["course"])
        self.assertEqual(report["score"]["warnings"], [])

    def test_e2_without_course_does_not_run_course(self):
        self.forbid((score, "score_course"), (score, "score_e1"))
        code, report, err = self.report_of("--outputs", self.write_outputs(), "--stages", "e2")
        self.assertEqual(code, 0, err)
        self.assertEqual(list(report["score"]["aggregate"]), ["e2"])

    def test_e3_does_not_read_e1_or_e2_data(self):
        outputs = {sid: {"e3_raw": entry.get("e3_raw")} for sid, entry in sample_outputs().items()}
        poisoned = {sid: dict(entry, e1="garbage", e2_normalized=12345) for sid, entry in outputs.items()}
        _, clean, _ = self.report_of("--outputs", self.write_outputs(outputs, "a.json"), "--stages", "e3")
        _, other, _ = self.report_of("--outputs", self.write_outputs(poisoned, "b.json"), "--stages", "e3")
        self.assertEqual(clean["e3"], other["e3"])

    def test_the_embedded_reports_equal_a_direct_call_with_no_rescoring(self):
        outputs = sample_outputs()
        _, report, _ = self.report_of("--outputs", self.write_outputs(outputs))
        direct_score = json.loads(cli.to_json(score.score_dataset(DATASET, outputs)))
        direct_e3 = json.loads(cli.to_json(e3.score_e3_dataset(DATASET, outputs)))
        self.assertEqual(report["score"], direct_score)
        self.assertEqual(report["e3"], direct_e3)

    def test_a_subset_report_equals_the_matching_part_of_the_full_report(self):
        target = self.write_outputs()
        _, full, _ = self.report_of("--outputs", target)
        _, subset, _ = self.report_of("--outputs", target, "--stages", "e2")
        self.assertEqual(subset["score"]["aggregate"]["e2"], full["score"]["aggregate"]["e2"])
        self.assertEqual([s["stages"]["e2"] for s in subset["score"]["scenarios"]],
                         [s["stages"]["e2"] for s in full["score"]["scenarios"]])

    def test_a_scorer_crash_is_reported_and_the_other_group_still_runs(self):
        with mock.patch.object(score, "score_e1", side_effect=RuntimeError("boom")):
            code, report, err = self.report_of("--outputs", self.write_outputs())
        self.assertEqual(code, 1)
        self.assertIsNone(report["score"])
        self.assertIsNotNone(report["e3"])
        self.assertEqual(report["errors"][0]["stages"], ["e1", "e2", "course"])
        self.assertIn("RuntimeError: boom", report["errors"][0]["error"])


class OutputFiles(CliTestCase):
    def listing(self):
        return sorted(os.listdir(self.tmp))

    def test_without_output_flags_the_report_goes_to_stdout_and_nothing_is_written(self):
        target = self.write_outputs()
        before = self.listing()
        code, out, err = self.run_cli("--outputs", target)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["cli_report_version"], cli.CLI_REPORT_VERSION)
        self.assertIn("stages run", err)
        self.assertEqual(self.listing(), before)

    def test_with_an_output_report_stdout_is_the_summary(self):
        target, report_path = self.write_outputs(), self.path("report.json")
        code, out, _ = self.run_cli("--outputs", target, "--output-report", report_path)
        self.assertEqual(code, 0)
        self.assertIn("e3.bullets_with_unsupported_number", out)
        self.assertIn("[screen only]", out)
        with open(report_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["stages_run"], list(cli.STAGES))

    def test_an_existing_file_is_refused_without_force_and_left_untouched(self):
        target, report_path = self.write_outputs(), self.write("report.json", "KEEP")
        code, _, err = self.run_cli("--outputs", target, "--output-report", report_path)
        self.assertEqual(code, 2)
        self.assertIn("--force", err)
        with open(report_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "KEEP")
        self.assertEqual(self.run_cli("--outputs", target, "--output-report", report_path, "--force")[0], 0)
        with open(report_path, encoding="utf-8") as handle:
            self.assertIn("cli_report_version", handle.read())

    def test_a_refused_target_means_no_file_is_written_at_all(self):
        target, rows = self.write_outputs(), self.path("rows.json")
        existing = self.write("report.json", "KEEP")
        code, _, _ = self.run_cli("--outputs", target, "--output-report", existing, "--review-rows", rows)
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(rows))

    def test_it_never_writes_into_the_dataset_folders_or_over_the_outputs_file(self):
        target = self.write_outputs()
        before = dataset_hashes()
        inside = os.path.join(dataset_module.DEFAULT_EXPECTED_DIR, "report_should_not_exist.json")
        for flag in ("--output-report", "--review-rows"):
            code, _, err = self.run_cli("--outputs", target, flag, inside, "--force")
            self.assertEqual(code, 2, flag)
            self.assertIn("dataset folders", err)
        self.assertFalse(os.path.exists(inside))
        existing = os.path.join(dataset_module.DEFAULT_SCENARIOS_DIR, sorted(os.listdir(
            dataset_module.DEFAULT_SCENARIOS_DIR))[0])
        self.assertEqual(self.run_cli("--outputs", target, "--output-report", existing, "--force")[0], 2)
        self.assertEqual(self.run_cli("--outputs", target, "--output-report", target, "--force")[0], 2)
        self.assertEqual(dataset_hashes(), before)

    def test_a_directory_target_and_identical_targets_are_usage_errors(self):
        target = self.write_outputs()
        self.assertEqual(self.run_cli("--outputs", target, "--output-report", self.tmp)[0], 2)
        same = self.path("same.json")
        self.assertEqual(self.run_cli("--outputs", target, "--output-report", same, "--review-rows", same)[0], 2)
        self.assertFalse(os.path.exists(same))

    def test_review_rows_need_the_e3_stage(self):
        code, _, err = self.run_cli("--outputs", self.write_outputs(), "--stages", "e1", "--review-rows",
                                    self.path("rows.json"))
        self.assertEqual(code, 2)
        self.assertIn("e3", err)

    def test_review_rows_come_from_build_review_rows_and_are_blank_for_the_reviewer(self):
        rows_path = self.path("rows.json")
        code, _, _ = self.run_cli("--outputs", self.write_outputs(), "--review-rows", rows_path)
        self.assertEqual(code, 0)
        with open(rows_path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["format_version"], cli.ROWS_VERSION)
        expected = e3.build_review_rows(DATASET["labels"][IDS[0]], DATASET["scenarios"][IDS[0]], E3_GOOD, IDS[0])
        self.assertEqual(data["rows"], json.loads(json.dumps(expected)))
        self.assertEqual(len(data["rows"]), 1)           # IDS[1] has unusable output, the rest none: no rows
        self.assertIsNone(data["rows"][0]["human_verdict"])

    def test_the_review_rows_file_is_valid_even_when_there_are_no_rows(self):
        rows_path = self.path("rows.json")
        empty = {sid: {"e3_raw": None} for sid in IDS}
        self.assertEqual(self.run_cli("--outputs", self.write_outputs(empty), "--review-rows", rows_path)[0], 0)
        with open(rows_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["rows"], [])


class DeterminismAndStrictness(CliTestCase):
    def test_two_runs_give_byte_identical_reports(self):
        target = self.write_outputs()
        first = self.run_cli("--outputs", target)[1]
        second = self.run_cli("--outputs", target)[1]
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("}\n"))

    def test_the_report_does_not_depend_on_where_the_files_are(self):
        other = tempfile.mkdtemp(prefix="pipeline_cli_other_")
        self.addCleanup(shutil.rmtree, other, True)
        a = self.write_outputs(name="a.json")
        b = os.path.join(other, "completely_different_name.json")
        shutil.copy(a, b)
        self.assertEqual(self.run_cli("--outputs", a)[1], self.run_cli("--outputs", b)[1])
        self.assertNotIn(self.tmp, self.run_cli("--outputs", a)[1])

    def test_the_report_is_strict_json_and_round_trips(self):
        text = self.run_cli("--outputs", self.write_outputs())[1]

        def refuse(constant):
            raise AssertionError("non-JSON constant " + constant)

        data = json.loads(text, parse_constant=refuse)
        self.assertEqual(cli.to_json(data), text)
        self.assertEqual(list(data), sorted(data))

    def test_every_metric_keeps_its_numerator_denominator_and_availability(self):
        _, report, _ = self.report_of("--outputs", self.write_outputs())
        metrics = [m for stage in report["score"]["aggregate"].values() for m in stage] + \
                  report["e3"]["aggregate"]["metrics"]
        self.assertTrue(metrics)
        for metric in metrics:
            for key in ("numerator", "denominator", "available"):
                self.assertIn(key, metric, metric["metric"])

    def test_missing_outputs_are_reported_but_only_strict_fails_on_them(self):
        target = self.write_outputs()
        code, report, _ = self.report_of("--outputs", target)
        self.assertEqual(code, 0)
        self.assertIn(IDS[2], report["input"]["missing_stage_outputs"]["e1"])
        self.assertNotIn(IDS[0], report["input"]["missing_stage_outputs"]["e1"])
        self.assertTrue(any("stage 'e3' has no output" in w for w in report["warnings"]))
        self.assertEqual(self.run_cli("--outputs", target, "--strict")[0], 3)

    def test_a_complete_clean_input_passes_strict(self):
        tiers = score._tiers()
        complete = {sid: {"e1": {c: [] for c in score.E1_CATEGORIES},
                          "e2_normalized": dict({t: [] for t in tiers}, normalization_ok=True),
                          "e3_raw": json.dumps({"optimized_bullets": [{"section": "Experience", "bullets": []}]})}
                    for sid in IDS}
        code, report, err = self.report_of("--outputs", self.write_outputs(complete), "--stages", "e1")
        self.assertEqual(report["input"]["missing_stage_outputs"], {"e1": []})
        self.assertEqual(code, 0, err)
        self.assertEqual(self.run_cli("--outputs", self.path("outputs.json"), "--stages", "e1", "--strict")[0], 0)


class IsolationAndSafety(CliTestCase):
    @classmethod
    def setUpClass(cls):
        with open(CLI_PATH, encoding="utf-8") as handle:
            cls.source = handle.read()
        cls.tree = ast.parse(cls.source)

    def test_only_standard_library_imports(self):
        allowed = {"argparse", "hashlib", "importlib", "json", "os", "sys"}
        found = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                found |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                found.add((node.module or "").split(".")[0])
        self.assertLessEqual(found, allowed, found - allowed)

    def test_no_forbidden_names_appear_in_the_source(self):
        forbidden = ("backend", "ResumeParser", "parse_resume", "SkillRecommender", "subprocess", "socket", "urllib",
                     "requests", "dotenv", "os.environ", "getenv", "groq", "youtube", "langchain", "holdout",
                     "resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                     "missing_skills.txt", "ats_optimized_bullets.txt")
        lowered = self.source.lower()
        for token in forbidden:
            self.assertNotIn(token.lower(), lowered, token)

    def test_it_loads_only_sibling_evaluation_modules_by_path(self):
        loaded = [call.args[1].value for call in ast.walk(self.tree)
                  if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_load"
                  and len(call.args) > 1 and isinstance(call.args[1], ast.Constant)]
        self.assertEqual(sorted(loaded), ["dataset.py", "e3.py", "score.py"])

    def test_the_only_file_writes_are_the_two_opt_in_outputs(self):
        opens = [node for node in ast.walk(self.tree)
                 if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open"]
        self.assertEqual(len(opens), 1)   # write_text; reading goes through dataset.read_json_strict

    def test_a_full_run_touches_no_network(self):
        target = self.write_outputs()
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network used")):
            self.assertEqual(self.run_cli("--outputs", target)[0], 0)
            self.assertEqual(self.run_cli("--validate-dataset")[0], 0)

    def test_production_modules_are_not_imported_by_a_run(self):
        self.run_cli("--outputs", self.write_outputs())
        for name in ("backend", "backend.main", "src.parser", "parser", "langchain", "groq"):
            self.assertNotIn(name, sys.modules, name)

    def test_the_dataset_files_are_unchanged_by_every_kind_of_run(self):
        before = dataset_hashes()
        self.assertEqual(len(before), 20)
        target = self.write_outputs()
        self.run_cli("--outputs", target, "--output-report", self.path("r.json"), "--review-rows", self.path("w.json"))
        self.run_cli("--validate-dataset")
        self.run_cli("--template")
        self.run_cli("--outputs", target, "--strict", "--stages", "e3")
        self.assertEqual(dataset_hashes(), before)


if __name__ == "__main__":
    unittest.main()
