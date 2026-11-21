"""Offline tests for eval/pipeline/e3.py (whole-pipeline evaluation, Step 4: E3 screening).

Run from the project root:  python -B -m unittest tests.test_pipeline_e3 -v

Every E3 output scored here is hand-authored in this file. The tests prove how the SCREENS and the report treat supplied
text; they say nothing about what a live model writes, and a score of 0 flags never means the bullets are accurate.
Tests under SelfConsistency generate E3 text FROM the real labels and resumes. They check that the authored labels and the
screens agree with each other. They are NOT model-performance results.
No network, no .env, no protected resume artifact, no E1/E2 data, no production import.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE = os.path.join(ROOT, "eval", "pipeline")


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(PIPELINE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e3 = _load("pipeline_e3", "e3.py")
dataset_module = _load("pipeline_dataset", "dataset.py")

# ---------------------------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------------------------

SOURCE_BULLETS = ["Built 12 pipelines.", "Cut load time from 90 minutes.", "Processed 2 million rows."]
RESUME_TEXT = "EXPERIENCE\n" + "\n".join("- " + b for b in SOURCE_BULLETS) + "\nEDUCATION\nB.Tech 2021\n"


def make_label(scenario_id="t1", numbers=("12", "90", "2 million", "2021"), absent=True, guards=True):
    skills = [
        {"skill_id": "t_k8s", "canonical": "Kubernetes", "aliases": ["k8s"],
         "ats": {"absent_from_resume": absent, "prohibited_claim_terms": ["Kubernetes", "k8s"] if absent else []}},
        {"skill_id": "t_gq", "canonical": "GraphQL", "aliases": [],
         "ats": {"absent_from_resume": False,
                 "prohibited_claim_terms": ["designed GraphQL", "GraphQL APIs"] if guards else []}},
        {"skill_id": "t_py", "canonical": "Python", "aliases": [],
         "ats": {"absent_from_resume": False, "prohibited_claim_terms": []}},
    ]
    return {"scenario_id": scenario_id, "labels_version": "draft-1", "skills": skills, "ats": {
        "supported_numbers": list(numbers) if numbers is not None else None,
        "supported_claims": [{"id": "c1", "claim": "Built 12 pipelines."}, {"id": "c2", "claim": "Cut load time."}],
        "prohibited_claims": ["Any Kubernetes experience.", "Any number not listed."],
        "absent_skills": ["Kubernetes"] if absent else [],
        "acceptance_rule": "Acceptable if every claim maps to a supported claim and adds nothing prohibited."}}


def make_scenario(resume_text=RESUME_TEXT, bullets=SOURCE_BULLETS):
    return {"scenario_id": "t1", "inputs": {"job_description": "JD", "resume_text": resume_text, "resume_structure": {
        "header": ["X"], "sections": [{"heading": "EXPERIENCE", "entries": [
            {"title": "Engineer", "dates": "2021", "bullets": list(bullets)}]}]}}}


def raw(*groups, fenced=False):
    """groups: (section, [bullets]) pairs -> a raw E3 string."""
    text = json.dumps({"optimized_bullets": [{"section": s, "bullets": list(b)} for s, b in groups]})
    return "```json\n" + text + "\n```" if fenced else text


CLEAN = raw(("Experience - Engineer", ["Built 12 data pipelines.", "Reduced load time from 90 minutes by tuning queries."]))
NON_OK = {"output_missing": None, "invalid_json": "not json", "wrong_shape": '{"x": 1}', "empty": '{"optimized_bullets": []}'}


def score(raw_output, label=None, scenario=None):
    return e3.score_e3(label or make_label(), scenario or make_scenario(), raw_output)


def metrics_of(record):
    return {m["metric"]: m for m in record["metrics"]}


def make_dataset(*pairs):
    ids = [label["scenario_id"] for label, _s in pairs]
    return {"ids": ids, "labels": {l["scenario_id"]: l for l, _s in pairs},
            "scenarios": {l["scenario_id"]: s for l, s in pairs}, "verified_hashes": {}}


# ---------------------------------------------------------------------------------------------------------------
# Output status and non-ok outputs
# ---------------------------------------------------------------------------------------------------------------

class UnusableOutputsGetNoQualityScore(unittest.TestCase):
    def test_missing_invalid_wrong_shaped_and_empty_outputs_have_no_numerator_and_no_rate(self):
        for status, bad in NON_OK.items():
            with self.subTest(status=status):
                record = score(bad)
                self.assertEqual(record["status"], status)
                self.assertEqual(record["descriptive"]["output_status"], status)
                self.assertEqual(record["descriptive"]["bullets_total"], 0)
                self.assertEqual(len(record["metrics"]), 5)
                for metric in record["metrics"]:
                    self.assertIsNone(metric["numerator"], metric["metric"])
                    self.assertIsNone(metric["rate"], metric["metric"])
                    self.assertFalse(metric["available"], metric["metric"])
                    self.assertEqual(metric["unscorable_count"], 1)
                    self.assertIn("unavailable_reason", metric)

    def test_the_unscorable_reason_names_the_failure(self):
        self.assertEqual(metrics_of(score(None))["bullets_mentioning_absent_skill"]["unscorable_reasons"],
                         {"output_missing": 1})
        self.assertEqual(metrics_of(score("not json"))["bullets_mentioning_absent_skill"]["unscorable_reasons"],
                         {"output_invalid_json": 1})
        self.assertEqual(metrics_of(score('{"x": 1}'))["bullets_mentioning_absent_skill"]["unscorable_reasons"],
                         {"output_wrong_shape": 1})

    def test_production_would_500_is_recorded_for_unparseable_output_only(self):
        self.assertTrue(score("not json")["observations"]["production_would_500"])
        self.assertFalse(score('{"x": 1}')["observations"]["production_would_500"])

    def test_non_string_output_is_invalid_not_scored(self):
        self.assertEqual(score(["a"])["status"], "invalid_json")

    def test_one_bad_item_makes_the_output_wrong_shaped_and_unscored(self):
        bad = json.dumps({"optimized_bullets": [{"section": "A", "bullets": ["ok"]}, {"section": "B", "bullets": [3]}]})
        self.assertEqual(score(bad)["status"], "wrong_shape")

    def test_fenced_output_is_accepted_as_production_cleans_it(self):
        record = score(raw(("Experience", ["Built 12 pipelines."]), fenced=True))
        self.assertEqual(record["status"], "scored")


# ---------------------------------------------------------------------------------------------------------------
# Clean output and the meaning of zero
# ---------------------------------------------------------------------------------------------------------------

class CleanOutputAndZeroMeaning(unittest.TestCase):
    def test_a_clean_output_scores_zero_flags_with_numerators_denominators_and_availability(self):
        record = score(CLEAN)
        self.assertEqual(record["status"], "scored")
        for name, metric in metrics_of(record).items():
            with self.subTest(metric=name):
                self.assertTrue(metric["available"])
                self.assertEqual(metric["numerator"], 0)
                self.assertEqual(metric["denominator"], 1 if name == "scenario_has_flagged_bullet" else 2)
                self.assertEqual(metric["rate"], 0.0)
                self.assertEqual(metric["direction"], "lower_is_better")
                self.assertEqual(metric["unscorable_reasons"], {})

    def test_every_quality_metric_says_it_is_screening_and_what_zero_means(self):
        for metric in score(CLEAN)["metrics"]:
            self.assertTrue(metric["screening_only"])
            self.assertIn("not evidence", metric["zero_means"])
        record = score(CLEAN)
        self.assertIn("not factual verification", record["screening_notice"])
        report = e3.score_e3_dataset(make_dataset((make_label(), make_scenario())), {"t1": {"e3_raw": CLEAN}})
        self.assertIn("not factual verification", report["screening_notice"])
        for metric in report["aggregate"]["metrics"]:
            self.assertTrue(metric["screening_only"])
            self.assertIn("not evidence", metric["zero_means"])

    def test_descriptive_counts_are_kept_apart_from_the_quality_metrics(self):
        record = score(CLEAN)
        self.assertEqual(record["descriptive"], {
            "output_status": "ok", "bullets_total": 2, "source_bullets_total": 3, "identical_to_source_bullets": 0,
            "near_duplicate_pairs": 0, "unit_mismatch_advisory_bullets": 0})
        self.assertFalse({m["metric"] for m in record["metrics"]} & set(record["descriptive"]))

    def test_a_flagless_output_with_an_unsupported_claim_is_not_detected_and_the_report_says_so(self):
        """A blind spot, shown on purpose: an invented responsibility without a number or listed term scores clean."""
        record = score(raw(("Experience", ["Led the whole company migration single-handedly."])))
        self.assertEqual(metrics_of(record)["scenario_has_flagged_bullet"]["numerator"], 0)
        self.assertTrue(any("not detected" in text for text in record["limitations"]))


# ---------------------------------------------------------------------------------------------------------------
# Number screens
# ---------------------------------------------------------------------------------------------------------------

class NumberScreens(unittest.TestCase):
    def test_a_number_outside_the_authored_list_is_flagged_and_the_listed_ones_are_not(self):
        record = score(raw(("Experience", ["Built 12 pipelines.", "Cut cost by 40%.", "Processed 2 million rows in 2021."])))
        metric = metrics_of(record)["bullets_with_unsupported_number"]
        self.assertEqual((metric["numerator"], metric["denominator"]), (1, 3))
        self.assertEqual(record["bullets"][1]["unsupported_numbers"], ["40%"])
        self.assertEqual(record["bullets"][2]["unsupported_numbers"], [])

    def test_authored_list_is_primary_and_resume_text_is_the_cross_check(self):
        scenario = make_scenario(resume_text=RESUME_TEXT + "- Ran 7 services.\n")
        record = score(raw(("Experience", ["Ran 7 services."])), scenario=scenario)
        metrics = metrics_of(record)
        self.assertEqual(metrics["bullets_with_unsupported_number"]["numerator"], 1)         # not in the authored list
        self.assertEqual(metrics["bullets_with_number_not_in_resume_text"]["numerator"], 0)  # but the resume states it
        self.assertIn("PRIMARY", metrics["bullets_with_unsupported_number"]["note"])
        self.assertIn("CROSS-CHECK", metrics["bullets_with_number_not_in_resume_text"]["note"])

    def test_a_number_missing_from_both_is_flagged_by_both(self):
        metrics = metrics_of(score(raw(("Experience", ["Cut cost by 40%."]))))
        self.assertEqual((metrics["bullets_with_unsupported_number"]["numerator"],
                          metrics["bullets_with_number_not_in_resume_text"]["numerator"]), (1, 1))

    def test_equal_values_in_other_formats_are_not_flagged(self):
        record = score(raw(("Experience", ["Processed 2,000,000 rows.", "Processed 2M rows.", "Built 12.0 pipelines."])))
        self.assertEqual(metrics_of(record)["bullets_with_unsupported_number"]["numerator"], 0)

    def test_an_empty_authored_list_flags_every_number(self):
        label = make_label(numbers=())
        record = score(raw(("Experience", ["Built 12 pipelines."])), label=label)
        self.assertEqual(metrics_of(record)["bullets_with_unsupported_number"]["numerator"], 1)

    def test_an_unusable_authored_list_makes_the_primary_unscorable_not_zero(self):
        record = score(CLEAN, label=make_label(numbers=None))
        metric = metrics_of(record)["bullets_with_unsupported_number"]
        self.assertFalse(metric["available"])
        self.assertEqual(metric["unscorable_reasons"], {"supported_numbers_unavailable": 1})
        self.assertTrue(metrics_of(record)["bullets_with_number_not_in_resume_text"]["available"])

    def test_missing_resume_text_makes_only_the_cross_check_unscorable(self):
        scenario = make_scenario()
        del scenario["inputs"]["resume_text"]
        record = score(CLEAN, scenario=scenario)
        self.assertEqual(metrics_of(record)["bullets_with_number_not_in_resume_text"]["unscorable_reasons"],
                         {"resume_text_unavailable": 1})
        self.assertTrue(metrics_of(record)["bullets_with_unsupported_number"]["available"])

    def test_the_unit_mismatch_advisory_is_descriptive_not_a_quality_metric(self):
        label = make_label(numbers=("35",))
        scenario = make_scenario(resume_text=RESUME_TEXT + "- Saved 35 hours.\n")
        record = score(raw(("Experience", ["Cut effort by 35%."])), label=label, scenario=scenario)
        self.assertEqual(record["descriptive"]["unit_mismatch_advisory_bullets"], 1)   # units come from the resume text
        self.assertEqual(metrics_of(record)["bullets_with_unsupported_number"]["numerator"], 0)
        self.assertEqual(metrics_of(record)["bullets_with_number_not_in_resume_text"]["numerator"], 0)

    def test_a_bare_authored_number_does_not_make_a_unit_a_mismatch(self):
        record = score(raw(("Experience", ["Reduced load time from 90 minutes."])))  # authored "90" has no unit
        self.assertEqual(record["descriptive"]["unit_mismatch_advisory_bullets"], 0)


# ---------------------------------------------------------------------------------------------------------------
# Absent-skill and guard-phrase screens
# ---------------------------------------------------------------------------------------------------------------

class AbsentSkillAndGuardScreens(unittest.TestCase):
    def test_an_absent_skill_its_alias_and_a_negated_mention_are_all_flagged(self):
        record = score(raw(("Experience", ["Deployed services on Kubernetes.", "Ran jobs on k8s.",
                                            "Has no Kubernetes experience.", "Built 12 pipelines."])))
        metric = metrics_of(record)["bullets_mentioning_absent_skill"]
        self.assertEqual((metric["numerator"], metric["denominator"]), (3, 4))
        self.assertEqual(record["bullets"][0]["absent_skills_mentioned"], ["Kubernetes"])

    def test_matching_is_whole_word_and_case_insensitive(self):
        record = score(raw(("Experience", ["Used KUBERNETES daily.", "Studied kubernetesque patterns."])))
        self.assertEqual(metrics_of(record)["bullets_mentioning_absent_skill"]["numerator"], 1)

    def test_a_guard_phrase_on_a_non_absent_skill_is_flagged_separately(self):
        record = score(raw(("Projects", ["Designed GraphQL services for clients.", "Wrote GraphQL queries.",
                                          "Built 12 pipelines."])))
        metrics = metrics_of(record)
        self.assertEqual((metrics["bullets_with_guard_phrase"]["numerator"], metrics["bullets_with_guard_phrase"]["denominator"]),
                         (1, 3))
        self.assertEqual(metrics["bullets_mentioning_absent_skill"]["numerator"], 0)
        self.assertEqual(record["bullets"][0]["guard_terms_hit"], ["designed GraphQL"])

    def test_nothing_to_screen_for_is_unscorable_not_zero_over_n(self):
        record = score(CLEAN, label=make_label(absent=False, guards=False))
        for name, reason in (("bullets_mentioning_absent_skill", "no_absent_skills_defined"),
                             ("bullets_with_guard_phrase", "no_guard_terms_defined")):
            metric = metrics_of(record)[name]
            self.assertFalse(metric["available"], name)
            self.assertIsNone(metric["numerator"], name)
            self.assertEqual(metric["unscorable_reasons"], {reason: 1})
        self.assertEqual(metrics_of(record)["scenario_has_flagged_bullet"]["denominator"], 1)  # numbers still scored

    def test_with_no_scorable_primary_screen_the_scenario_level_metric_is_unscorable(self):
        label = make_label(numbers=None, absent=False, guards=False)
        metric = metrics_of(score(CLEAN, label=label))["scenario_has_flagged_bullet"]
        self.assertFalse(metric["available"])
        self.assertEqual(metric["unscorable_reasons"], {"no_primary_screen_scorable": 1})

    def test_build_screens_collects_aliases_and_prohibited_terms_of_absent_skills(self):
        screens = e3.build_screens(make_label())
        self.assertEqual(screens["absent"], [{"name": "Kubernetes", "aliases": ["k8s"], "skill_ids": ["t_k8s"]}])
        self.assertEqual(screens["guards"], [{"name": "designed GraphQL", "aliases": ["GraphQL APIs"],
                                              "skill_ids": ["t_gq"]}])
        self.assertEqual(screens["numbers_text"], "12\n90\n2 million\n2021")

    def test_an_absent_skill_listed_without_a_skill_label_is_still_screened(self):
        label = make_label()
        label["ats"]["absent_skills"] = ["Kubernetes", "Terraform"]
        record = score(raw(("Experience", ["Wrote Terraform modules."])), label=label)
        self.assertEqual(metrics_of(record)["bullets_mentioning_absent_skill"]["numerator"], 1)


# ---------------------------------------------------------------------------------------------------------------
# scenario_has_flagged_bullet and degenerate outputs
# ---------------------------------------------------------------------------------------------------------------

class ScenarioLevelAndDegenerate(unittest.TestCase):
    def test_any_flag_on_a_primary_screen_marks_the_scenario(self):
        metric = metrics_of(score(raw(("Experience", ["Built 12 pipelines.", "Used Kubernetes."]))))["scenario_has_flagged_bullet"]
        self.assertEqual((metric["numerator"], metric["denominator"]), (1, 1))
        self.assertEqual(metric["counts"]["flagged_bullets_by_screen"], {"absent_skill": 1, "guard_phrase": 0,
                                                                         "unsupported_number": 0})

    def test_the_cross_check_alone_does_not_mark_the_scenario(self):
        label = make_label(numbers=("12", "90", "2 million", "2021", "40"))
        record = score(raw(("Experience", ["Cut cost by 40%."])), label=label)
        self.assertEqual(metrics_of(record)["bullets_with_number_not_in_resume_text"]["numerator"], 1)
        self.assertEqual(metrics_of(record)["scenario_has_flagged_bullet"]["numerator"], 0)

    def test_a_verbatim_copy_scores_clean_but_is_warned_about(self):
        record = score(raw(("Experience", list(SOURCE_BULLETS))))
        self.assertEqual(metrics_of(record)["scenario_has_flagged_bullet"]["numerator"], 0)
        self.assertEqual(record["descriptive"]["identical_to_source_bullets"], 3)
        self.assertTrue(any(w.startswith("verbatim_copy") for w in record["warnings"]))

    def test_dropping_bullets_is_warned_about(self):
        record = score(raw(("Experience", ["Built 12 pipelines, rewritten."])))
        self.assertTrue(any(w.startswith("fewer_bullets_than_source: 1 bullets returned for 3") for w in record["warnings"]))

    def test_a_genuine_rewrite_has_no_warnings(self):
        self.assertEqual(score(raw(("Experience", ["Built 12 pipelines.", "Cut 90 minutes from loads.",
                                                    "Processed 2 million rows nightly."])))["warnings"], [])

    def test_near_duplicates_are_counted_descriptively(self):
        record = score(raw(("Experience", ["Built 12 data pipelines for sales.", "Built 12 data pipelines for sales!"])))
        self.assertEqual(record["descriptive"]["near_duplicate_pairs"], 1)


# ---------------------------------------------------------------------------------------------------------------
# Review rows
# ---------------------------------------------------------------------------------------------------------------

class ReviewRows(unittest.TestCase):
    def rows(self, raw_output=None):
        return e3.build_review_rows(make_label(), make_scenario(), raw_output or raw(
            ("Experience - Engineer", ["Built 12 pipelines on Kubernetes.", "Cut cost by 40%."])))

    def test_one_row_per_bullet_with_findings_claims_and_blank_human_fields(self):
        rows = self.rows()
        self.assertEqual([r["row_id"] for r in rows], ["t1:0", "t1:1"])
        first = rows[0]
        self.assertEqual(first["bullet"], "Built 12 pipelines on Kubernetes.")
        self.assertEqual(first["section"], "Experience - Engineer")
        self.assertEqual(first["automated_findings"]["absent_skills_mentioned"], ["Kubernetes"])
        self.assertEqual(rows[1]["automated_findings"]["unsupported_numbers"], ["40%"])
        self.assertEqual(first["authored_supported_claims"], [{"id": "c1", "claim": "Built 12 pipelines."},
                                                              {"id": "c2", "claim": "Cut load time."}])
        self.assertEqual(first["authored_prohibited_claims"], ["Any Kubernetes experience.", "Any number not listed."])
        self.assertEqual(first["acceptance_rule"], make_label()["ats"]["acceptance_rule"])
        self.assertIsNone(first["human_verdict"])
        self.assertIsNone(first["human_notes"])
        self.assertIn("screening signals", first["reviewer_note"])

    def test_non_ok_outputs_give_no_rows(self):
        for status, bad in NON_OK.items():
            with self.subTest(status=status):
                self.assertEqual(e3.build_review_rows(make_label(), make_scenario(), bad), [])

    def test_rows_are_deterministic_json_serialisable_and_write_nothing(self):
        before = set(os.listdir(PIPELINE))
        first, second = self.rows(), self.rows()
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(before, set(os.listdir(PIPELINE)))

    def test_the_rows_do_not_mutate_their_inputs(self):
        label, scenario = make_label(), make_scenario()
        snapshot = copy.deepcopy((label, scenario))
        e3.build_review_rows(label, scenario, CLEAN)
        self.assertEqual((label, scenario), snapshot)


# ---------------------------------------------------------------------------------------------------------------
# Dataset report: pooling, missing outputs, independence, determinism, JSON
# ---------------------------------------------------------------------------------------------------------------

class DatasetReport(unittest.TestCase):
    def two_scenarios(self):
        return make_dataset((make_label("a"), make_scenario()), (make_label("b"), make_scenario()))

    def test_pooled_rate_is_summed_numerators_over_summed_denominators_not_a_mean_of_rates(self):
        outputs = {"a": {"e3_raw": raw(("E", ["Deployed on Kubernetes.", "Built 12 pipelines."]))},          # 1/2 = 0.5
                   "b": {"e3_raw": raw(("E", ["Built 12 pipelines.", "Cut 90 minutes.", "Used 2 million rows.",
                                              "Wrote tests in 2021."]))}}                                      # 0/4 = 0.0
        report = e3.score_e3_dataset(self.two_scenarios(), outputs)
        pooled = {m["metric"]: m for m in report["aggregate"]["metrics"]}["bullets_mentioning_absent_skill"]
        self.assertEqual((pooled["numerator"], pooled["denominator"]), (1, 6))
        self.assertEqual(pooled["rate"], round(1 / 6, 6))
        self.assertNotEqual(pooled["rate"], 0.25)  # the mean of 0.5 and 0.0
        self.assertEqual([p["rate"] for p in pooled["per_scenario"]], [0.5, 0.0])
        scenario_level = {m["metric"]: m for m in report["aggregate"]["metrics"]}["scenario_has_flagged_bullet"]
        self.assertEqual((scenario_level["numerator"], scenario_level["denominator"]), (1, 2))

    def test_missing_and_unusable_outputs_are_excluded_from_the_sums_but_stay_visible(self):
        outputs = {"a": {"e3_raw": CLEAN}, "b": {"e3_raw": "not json"}, "zzz": {}}
        report = e3.score_e3_dataset(self.two_scenarios(), outputs)
        pooled = {m["metric"]: m for m in report["aggregate"]["metrics"]}["bullets_mentioning_absent_skill"]
        self.assertEqual((pooled["numerator"], pooled["denominator"]), (0, 2))
        self.assertEqual((pooled["scenarios_scored"], pooled["scenarios_without_usable_output"]), (1, 1))
        self.assertIsNone(pooled["per_scenario"][1]["rate"])
        self.assertEqual(report["descriptive"]["output_status_distribution"], {"invalid_json": 1, "ok": 1})
        self.assertEqual(report["unknown_output_ids"], ["zzz"])
        self.assertIn("output_invalid_json", pooled["unscorable_reasons"])

    def test_no_outputs_at_all_gives_no_available_metric(self):
        report = e3.score_e3_dataset(self.two_scenarios(), {})
        self.assertEqual(report["descriptive"]["output_status_distribution"], {"output_missing": 2})
        for metric in report["aggregate"]["metrics"]:
            self.assertFalse(metric["available"], metric["metric"])
            self.assertIsNone(metric["rate"], metric["metric"])
        self.assertEqual([s["output_supplied"] for s in report["scenarios"]], [False, False])

    def test_verbatim_copy_warnings_reach_the_report(self):
        outputs = {"a": {"e3_raw": raw(("E", list(SOURCE_BULLETS)))}}
        report = e3.score_e3_dataset(self.two_scenarios(), outputs)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertTrue(report["warnings"][0].startswith("e3:a: verbatim_copy"))

    def test_descriptive_totals_are_separate_from_the_metrics(self):
        outputs = {"a": {"e3_raw": CLEAN}, "b": {"e3_raw": CLEAN}}
        report = e3.score_e3_dataset(self.two_scenarios(), outputs)
        self.assertEqual(report["descriptive"]["totals"]["bullets_total"], 4)
        self.assertNotIn("bullets_total", {m["metric"] for m in report["aggregate"]["metrics"]})

    def test_e3_ignores_any_e1_or_e2_data(self):
        base = e3.score_e3_dataset(self.two_scenarios(), {"a": {"e3_raw": CLEAN}})
        extra = e3.score_e3_dataset(self.two_scenarios(), {"a": {"e3_raw": CLEAN, "e1": {"technical_skills": ["x"]},
                                                                 "e2_normalized": {"crucial": []}, "e2_raw": "{}"}})
        self.assertEqual(json.dumps(base), json.dumps(extra))

    def test_no_e1_or_e2_name_appears_in_the_code(self):
        with open(os.path.join(PIPELINE, "e3.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for forbidden in ("e1", "e2", "e1_output", "e2_normalized", "e2_raw", "normalized", "prioritized"):
            self.assertNotIn(forbidden, identifiers)
        for function in (e3.score_e3, e3.build_review_rows, e3.score_e3_dataset):
            self.assertFalse({"e1", "e2", "normalized"} & set(inspect.signature(function).parameters))

    def test_the_report_is_deterministic_independent_of_output_key_order_and_json_serialisable(self):
        outputs = {"a": {"e3_raw": CLEAN}, "b": {"e3_raw": raw(("E", ["Used Kubernetes in 2021."]))}}
        first = e3.score_e3_dataset(self.two_scenarios(), outputs)
        second = e3.score_e3_dataset(self.two_scenarios(), copy.deepcopy(dict(reversed(list(outputs.items())))))
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual([s["scenario_id"] for s in first["scenarios"]], ["a", "b"])
        self.assertEqual(first["report_version"], e3.REPORT_VERSION)

    def test_inputs_are_not_modified(self):
        dataset, outputs = self.two_scenarios(), {"a": {"e3_raw": CLEAN}}
        snapshot = copy.deepcopy((dataset, outputs))
        e3.score_e3_dataset(dataset, outputs)
        self.assertEqual((dataset, outputs), snapshot)

    def test_malformed_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            e3.score_e3_dataset(self.two_scenarios(), [])
        with self.assertRaises(ValueError):
            e3.score_e3_dataset(self.two_scenarios(), {"a": "text"})

    def test_every_metric_record_has_the_required_fields(self):
        required = {"stage", "metric", "numerator", "denominator", "rate", "available", "direction",
                    "unscorable_count", "unscorable_reasons", "note", "screening_only", "zero_means"}
        report = e3.score_e3_dataset(self.two_scenarios(), {"a": {"e3_raw": CLEAN}, "b": {"e3_raw": "{}"}})
        records = [m for s in report["scenarios"] for m in s["e3"]["metrics"]] + report["aggregate"]["metrics"]
        for metric in records:
            self.assertTrue(required <= set(metric), metric["metric"])
            if not metric["available"]:
                self.assertIn("unavailable_reason", metric)


# ---------------------------------------------------------------------------------------------------------------
# Reused helpers and isolation
# ---------------------------------------------------------------------------------------------------------------

def _hash_tree(directory):
    result = {}
    for folder in ("scenarios", "expected"):
        base = os.path.join(directory, folder)
        for name in sorted(os.listdir(base)):
            with open(os.path.join(base, name), "rb") as handle:
                result[f"{folder}/{name}"] = hashlib.sha256(handle.read()).hexdigest()
    return result


class ReusedHelpersAndIsolation(unittest.TestCase):
    def test_the_two_reused_score_helpers_still_have_the_shape_e3_relies_on(self):
        metric = e3._metric("e3", "m", 1, 4, "lower_is_better", unscorable={"x": 1}, note="n", counts={"k": 1})
        for key in ("stage", "metric", "numerator", "denominator", "rate", "available", "direction", "unscorable_count",
                    "unscorable_reasons", "note", "counts"):
            self.assertIn(key, metric)
        self.assertEqual(metric["rate"], 0.25)
        pooled = e3._aggregate_metric([("a", metric), ("b", e3._metric("e3", "m", 0, 4, "lower_is_better", note="n"))])
        self.assertEqual((pooled["numerator"], pooled["denominator"]), (1, 8))
        self.assertEqual([p["scenario_id"] for p in pooled["per_scenario"]], ["a", "b"])

    def test_the_module_imports_only_the_standard_library_it_needs(self):
        with open(os.path.join(PIPELINE, "e3.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module)
        self.assertEqual(found, {"importlib.util", "json", "os", "sys"})

    def test_the_module_opens_no_file_and_names_no_protected_artifact(self):
        with open(os.path.join(PIPELINE, "e3.py"), "r", encoding="utf-8") as handle:
            source = handle.read()
        calls = {n.func.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("open", calls)
        for forbidden in (".env", "resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                          "missing_skills.txt", "ats_optimized_bullets.txt", "parse_resume", "ResumeParser", "backend"):
            self.assertNotIn(forbidden, source)

    def test_scoring_makes_no_network_call(self):
        def refuse(*args, **kwargs):
            raise AssertionError("network access attempted")
        with mock.patch.object(socket.socket, "connect", refuse), mock.patch.object(socket, "create_connection", refuse), \
                mock.patch.object(socket, "getaddrinfo", refuse):
            e3.score_e3_dataset(make_dataset((make_label(), make_scenario())), {"t1": {"e3_raw": CLEAN}})
            e3.build_review_rows(make_label(), make_scenario(), CLEAN)


# ---------------------------------------------------------------------------------------------------------------
# Self-consistency of the real labels (NOT model performance)
# ---------------------------------------------------------------------------------------------------------------

class SelfConsistency(unittest.TestCase):
    """E3 text below is generated FROM the real resumes and labels. These tests show that the authored numbers, absent
    skills and guard terms agree with the resumes and that the screens can see a planted violation. They are fixtures:
    they say nothing about any model, and a pass here is not a quality result."""

    @classmethod
    def setUpClass(cls):
        cls.before = _hash_tree(PIPELINE)
        cls.dataset = dataset_module.load_dataset()

    def verbatim_output(self, scenario):
        groups = []
        for section in scenario["inputs"]["resume_structure"]["sections"]:
            for entry in section.get("entries") or []:
                groups.append((f"{section['heading']} - {entry['title']}", list(entry["bullets"])))
        return raw(*groups)

    def test_the_resume_bullets_themselves_trip_none_of_the_authored_screens(self):
        for sid in self.dataset["ids"]:
            with self.subTest(scenario=sid):
                record = e3.score_e3(self.dataset["labels"][sid], self.dataset["scenarios"][sid],
                                     self.verbatim_output(self.dataset["scenarios"][sid]))
                self.assertEqual(record["status"], "scored")
                metrics = metrics_of(record)
                screens = e3.build_screens(self.dataset["labels"][sid])
                scorable = {"bullets_with_unsupported_number": True, "bullets_with_number_not_in_resume_text": True,
                            "bullets_mentioning_absent_skill": bool(screens["absent"]),
                            "bullets_with_guard_phrase": bool(screens["guards"])}
                for name, expected_available in scorable.items():
                    # a screen with nothing authored to look for is unscorable, never a vacuous 0/N
                    self.assertEqual(metrics[name]["available"], expected_available, name)
                    if expected_available:
                        self.assertEqual(metrics[name]["numerator"], 0, (name, record["bullets"]))
                    else:
                        self.assertIsNone(metrics[name]["numerator"], name)
                self.assertEqual(record["descriptive"]["identical_to_source_bullets"], record["descriptive"]["bullets_total"])
                self.assertTrue(any(w.startswith("verbatim_copy") for w in record["warnings"]))

    def test_every_listed_absent_skill_has_a_matching_skill_label_with_terms(self):
        for sid in self.dataset["ids"]:
            with self.subTest(scenario=sid):
                label = self.dataset["labels"][sid]
                screens = e3.build_screens(label)
                by_name = {e["name"].casefold(): e for e in screens["absent"]}
                for name in label["ats"]["absent_skills"]:
                    self.assertTrue(by_name[name.casefold()]["skill_ids"], name)

    def test_a_planted_absent_skill_number_and_guard_term_are_each_detected_in_every_scenario(self):
        for sid in self.dataset["ids"]:
            with self.subTest(scenario=sid):
                label, scenario = self.dataset["labels"][sid], self.dataset["scenarios"][sid]
                screens = e3.build_screens(label)
                plants = [("absent", f" Experienced with {screens['absent'][0]['name']}.", "absent_skills_mentioned"),
                          ("number", " Increased revenue by 987654%.", "unsupported_numbers")]
                if screens["guards"]:
                    plants.append(("guard", f" Strong {screens['guards'][0]['name']}.", "guard_terms_hit"))
                bullets = [b for s in scenario["inputs"]["resume_structure"]["sections"]
                           for e in s.get("entries") or [] for b in e["bullets"]]
                for kind, suffix, field in plants:
                    record = e3.score_e3(label, scenario, raw(("Experience", [bullets[0] + suffix] + bullets[1:])))
                    self.assertTrue(record["bullets"][0][field], (kind, record["bullets"][0]))
                    self.assertEqual(metrics_of(record)["scenario_has_flagged_bullet"]["numerator"], 1, kind)

    def test_the_real_dataset_files_were_not_changed(self):
        self.assertEqual(self.before, _hash_tree(PIPELINE))


if __name__ == "__main__":
    unittest.main()
