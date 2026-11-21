"""Offline tests for eval/pipeline/score.py (whole-pipeline evaluation, Step 3).

Run from the project root:  python -B -m unittest tests.test_pipeline_score -v

Every output scored here is hand-authored in this file. The tests prove how the SCORER treats supplied data; they say
nothing about what a live model emits. The production logic the scorer reuses (skill key, eligibility rule) comes from
eval/pipeline/stages.py, which executes pure definitions read from the source text: no application import, no .env,
no protected resume artifact, no network. Real dataset files are only read, and their hashes are asserted unchanged.
"""
import ast
import copy
import hashlib
import importlib.util
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


stages = _load("pipeline_stages", "stages.py")
score = _load("pipeline_score", "score.py")
dataset_module = _load("pipeline_dataset", "dataset.py")

# ---------------------------------------------------------------------------------------------------------------
# Fixtures: a small hand-authored label and hand-authored outputs
# ---------------------------------------------------------------------------------------------------------------


def label_skill(skill_id, canonical, truth, acceptable, *, aliases=(), category="technical_skills",
                outcome_class="determinate", tier_set=None, tier_class=None, eligible=False, anchor=None,
                accepted=None):
    return {
        "skill_id": skill_id, "canonical": canonical, "aliases": list(aliases), "category": category,
        "authoring_truth": truth, "acceptable_outcomes": list(acceptable), "outcome_class": outcome_class,
        "jd": {"mentions": [{"anchor": anchor or f"{canonical} is required", "wording_class": "required"}],
               "expected_tier_set": tier_set,
               "tier_class": tier_class or ("not_applicable" if tier_set is None else
                                            "determinate" if len(tier_set) == 1 else "ambiguous")},
        "extraction": {"expected_in_jd_extraction": "yes", "accepted_forms": accepted or [canonical]},
        "course": {"eligible_expected": eligible},
    }


ABSENT_OK = ["ABSENT", "EMITTED:no_gap"]


def make_label(scenario_id="t1"):
    return {"scenario_id": scenario_id, "labels_version": "draft-1", "skills": [
        label_skill("t_py", "Python", "demonstrated", ABSENT_OK, aliases=["Python3"]),
        label_skill("t_dk", "Docker", "missing", ["EMITTED:missing"], tier_set=["crucial"], eligible=True),
        label_skill("t_k8", "Kubernetes", "missing", ["EMITTED:missing"], aliases=["k8s"],
                    tier_set=["good_to_have", "optional"], eligible=True),
        label_skill("t_tf", "Terraform", "partial", ["EMITTED:partial"], tier_set=["optional"], eligible=True),
        label_skill("t_gq", "GraphQL", "mention_only", ABSENT_OK + ["EMITTED:partial", "EMITTED:missing"],
                    outcome_class="ambiguous", tier_set=["crucial"], eligible=True),
    ]}


def make_dataset(*labels):
    ids = [label["scenario_id"] for label in labels]
    return {"ids": ids, "scenarios": {i: {} for i in ids}, "labels": {l["scenario_id"]: l for l in labels},
            "verified_hashes": {}}


def entry(skill, tier_quote, category="technical_skills", status="missing", found=True, related=True,
          listed=True, **extra):
    data = {"skill": skill, "category": category, "status": status, "reason": "Reason.", "jd_evidence": tier_quote,
            "evidence_found": found, "evidence_related": related, "in_extracted_list": listed}
    data.update(extra)
    return data


def normalized(crucial=(), good=(), optional=(), ok=True, diagnostics=()):
    return {"crucial": list(crucial), "good_to_have": list(good), "optional": list(optional),
            "normalization_ok": ok, "diagnostics": list(diagnostics)}


def perfect_normalized():
    return normalized(
        crucial=[entry("Docker", "Docker is required")],
        good=[entry("Kubernetes", "Kubernetes is required")],
        optional=[entry("Terraform", "Terraform is required", status="partial", resume_evidence_found=True)])


PERFECT_E1 = {"technical_skills": ["Python", "Docker", "Kubernetes", "Terraform", "GraphQL"],
              "soft_skills": [], "academic_qualifications": []}


def metrics_of(stage_record):
    return {m["metric"]: m for m in stage_record["metrics"]}


def e2_of(normalized_output, raw=None, label=None):
    return score.score_e2(label or make_label(), normalized_output, raw)


# ---------------------------------------------------------------------------------------------------------------
# 1. Perfect match
# ---------------------------------------------------------------------------------------------------------------

class PerfectMatch(unittest.TestCase):
    def test_e1_perfect_extraction_has_full_recall(self):
        record = score.score_e1(make_label(), PERFECT_E1)
        recall = metrics_of(record)["skill_recall"]
        self.assertEqual((recall["numerator"], recall["denominator"], recall["rate"]), (5, 5, 1.0))
        self.assertEqual(recall["counts"]["false_negatives"], 0)
        self.assertEqual(record["observations"]["unlabelled_predictions"], [])

    def test_e2_perfect_output_scores_fully_on_every_available_metric(self):
        metrics = metrics_of(e2_of(perfect_normalized()))
        for name in ("status_agreement", "gap_detection_recall", "status_exact_match_given_flagged",
                     "tier_agreement_determinate", "tier_agreement_ambiguous_set", "jd_evidence_found"):
            with self.subTest(metric=name):
                self.assertEqual(metrics[name]["rate"], 1.0, metrics[name])
        self.assertEqual((metrics["false_flag_rate"]["numerator"], metrics["false_flag_rate"]["denominator"]), (0, 1))
        self.assertEqual(metrics["false_flag_rate"]["rate"], 0.0)  # zero is a real value here, not "unavailable"
        self.assertTrue(metrics["false_flag_rate"]["available"])

    def test_course_perfect_output_conforms_to_the_rule_and_the_labels(self):
        metrics = metrics_of(score.score_course(make_label(), perfect_normalized()))
        for name in ("offered_options_match_rule", "skipped_count_match_rule", "label_eligibility_agreement"):
            self.assertEqual(metrics[name]["rate"], 1.0, metrics[name])


# ---------------------------------------------------------------------------------------------------------------
# 2-3. False positives, false negatives, empty predictions and empty ground truth
# ---------------------------------------------------------------------------------------------------------------

class FalsePositivesAndNegatives(unittest.TestCase):
    def test_a_missed_skill_is_a_false_negative_and_an_extra_one_is_listed_not_penalised(self):
        output = {"technical_skills": ["Python", "Docker", "Terraform", "GraphQL", "Rust"]}  # no Kubernetes, extra Rust
        record = score.score_e1(make_label(), output)
        recall = metrics_of(record)["skill_recall"]
        self.assertEqual((recall["numerator"], recall["denominator"]), (4, 5))
        self.assertEqual(recall["counts"]["true_positives"], 4)
        self.assertEqual(recall["counts"]["false_negatives"], 1)
        self.assertIsNone(recall["counts"]["false_positives"])  # unavailable, not zero
        self.assertEqual(recall["counts"]["unlabelled_predictions"], 1)
        self.assertEqual(record["observations"]["unlabelled_predictions"],
                         [{"category": "technical_skills", "item": "Rust"}])
        self.assertEqual([r["skill_id"] for r in record["skills"] if r["matched"] is False], ["t_k8"])
        self.assertEqual(record["observations"]["input_notes"],
                         ["category_missing:soft_skills", "category_missing:academic_qualifications"])

    def test_precision_and_f1_are_permanently_unavailable_with_a_reason(self):
        for output in (PERFECT_E1, {"technical_skills": []}, None):
            metrics = metrics_of(score.score_e1(make_label(), output))
            for name in ("precision", "f1"):
                with self.subTest(name=name, output=output):
                    self.assertFalse(metrics[name]["available"])
                    self.assertIsNone(metrics[name]["rate"])
                    self.assertTrue(metrics[name]["permanently_unavailable"])
                    self.assertIn("exhaustive", metrics[name]["unavailable_reason"])

    def test_a_skill_found_only_in_another_category_is_not_a_hit(self):
        output = {"technical_skills": ["Docker"], "soft_skills": ["Python"], "academic_qualifications": []}
        record = score.score_e1(make_label(), output)
        python = next(r for r in record["skills"] if r["skill_id"] == "t_py")
        self.assertFalse(python["matched"])
        self.assertEqual(python["found_in_other_category"], ["soft_skills"])

    def test_empty_predictions_give_recall_zero_which_is_available(self):
        recall = metrics_of(score.score_e1(make_label(), {"technical_skills": [], "soft_skills": [],
                                                           "academic_qualifications": []}))["skill_recall"]
        self.assertEqual((recall["numerator"], recall["denominator"], recall["rate"]), (0, 5, 0.0))
        self.assertTrue(recall["available"])

    def test_empty_ground_truth_is_unavailable_not_zero_and_not_one(self):
        empty = {"scenario_id": "t0", "labels_version": "draft-1", "skills": []}
        recall = metrics_of(score.score_e1(empty, PERFECT_E1))["skill_recall"]
        self.assertEqual(recall["denominator"], 0)
        self.assertIsNone(recall["rate"])
        self.assertFalse(recall["available"])
        for stage_metric in e2_of(normalized(), label=empty)["metrics"]:
            self.assertIsNone(stage_metric["rate"], stage_metric["metric"])

    def test_e2_unflagged_gap_lowers_recall_and_a_flagged_demonstrated_skill_is_a_false_flag(self):
        output = normalized(crucial=[entry("Python", "Python is required")],  # false flag
                            good=[entry("Kubernetes", "Kubernetes is required")])  # Docker, Terraform not flagged
        metrics = metrics_of(e2_of(output))
        self.assertEqual((metrics["gap_detection_recall"]["numerator"], metrics["gap_detection_recall"]["denominator"]),
                         (1, 3))
        self.assertEqual((metrics["false_flag_rate"]["numerator"], metrics["false_flag_rate"]["denominator"]), (1, 1))
        self.assertEqual(metrics["false_flag_rate"]["direction"], "lower_is_better")


# ---------------------------------------------------------------------------------------------------------------
# 4. Ambiguous tier sets
# ---------------------------------------------------------------------------------------------------------------

class AmbiguousTierSets(unittest.TestCase):
    def test_any_member_of_an_ambiguous_set_is_accepted_and_is_scored_separately(self):
        for tier_key in ("good", "optional"):
            with self.subTest(tier=tier_key):
                tiers = {"crucial": [entry("Docker", "Docker is required")],
                         tier_key: [entry("Kubernetes", "Kubernetes is required")]}
                metrics = metrics_of(e2_of(normalized(**tiers)))
                self.assertEqual(metrics["tier_agreement_ambiguous_set"]["numerator"], 1)

    def test_a_tier_outside_the_ambiguous_set_fails_without_touching_the_determinate_metric(self):
        output = normalized(crucial=[entry("Docker", "Docker is required"), entry("Kubernetes", "K8s is required")])
        metrics = metrics_of(e2_of(output))
        self.assertEqual((metrics["tier_agreement_ambiguous_set"]["numerator"],
                          metrics["tier_agreement_ambiguous_set"]["denominator"]), (0, 1))
        self.assertEqual((metrics["tier_agreement_determinate"]["numerator"],
                          metrics["tier_agreement_determinate"]["denominator"]), (1, 1))  # Docker ok; Terraform, GraphQL have no entry
        self.assertEqual(metrics["tier_agreement_determinate"]["unscorable_reasons"], {"no_emitted_entry": 2})

    def test_ambiguous_set_labels_are_not_in_the_determinate_denominator(self):
        metrics = metrics_of(e2_of(perfect_normalized()))
        self.assertEqual(metrics["tier_agreement_determinate"]["denominator"], 2)  # Docker, Terraform (GraphQL absent)
        self.assertEqual(metrics["tier_agreement_ambiguous_set"]["denominator"], 1)  # Kubernetes only


# ---------------------------------------------------------------------------------------------------------------
# 5. Missing / invalid / failed outputs
# ---------------------------------------------------------------------------------------------------------------

class MissingOutputs(unittest.TestCase):
    def test_a_missing_output_is_never_counted_correct(self):
        label = make_label()
        for record in (score.score_e1(label, None), score.score_e2(label, None), score.score_course(label, None)):
            self.assertEqual(record["status"], "output_missing")
            for metric in record["metrics"]:
                with self.subTest(stage=record["stage"], metric=metric["metric"]):
                    self.assertIsNone(metric["numerator"])
                    self.assertIsNone(metric["rate"])
                    self.assertFalse(metric["available"])

    def test_denominators_are_still_reported_for_a_missing_output(self):
        metrics = metrics_of(score.score_e2(make_label(), None))
        self.assertEqual(metrics["status_agreement"]["denominator"], 5)
        self.assertEqual(metrics["status_agreement"]["unscorable_reasons"], {"output_missing": 5})
        self.assertEqual(metrics["gap_detection_recall"]["denominator"], 3)
        self.assertEqual(metrics["false_flag_rate"]["denominator"], 1)

    def test_invalid_e1_output_is_reported_not_scored(self):
        for bad in ("not json", "[]", {"technical_skills": "Python"}, 7):
            with self.subTest(output=bad):
                record = score.score_e1(make_label(), bad)
                self.assertEqual(record["status"], "invalid_output")
                self.assertIsNone(metrics_of(record)["skill_recall"]["numerator"])

    def test_e1_accepts_json_text(self):
        record = score.score_e1(make_label(), json.dumps(PERFECT_E1))
        self.assertEqual(metrics_of(record)["skill_recall"]["rate"], 1.0)

    def test_invalid_e2_shapes_are_reported_not_scored(self):
        for bad in ({}, {"crucial": []}, "text", normalized(ok="yes"), dict(normalized(), optional="x")):
            with self.subTest(output=bad):
                self.assertEqual(e2_of(bad)["status"], "invalid_output")

    def test_a_failed_normalisation_scores_nothing_and_keeps_the_diagnostics(self):
        fallback = normalized(ok=False, diagnostics=["Comparison output is not valid JSON."])
        record = e2_of(fallback)
        self.assertEqual(record["status"], "normalization_failed")
        self.assertEqual(record["observations"]["diagnostics"], ["Comparison output is not valid JSON."])
        metrics = metrics_of(record)
        self.assertIsNone(metrics["status_agreement"]["numerator"])  # empty tiers are NOT read as "model found no gaps"
        self.assertEqual(metrics["status_agreement"]["unscorable_reasons"], {"normalization_failed": 5})

    def test_dataset_level_missing_and_unknown_outputs(self):
        report = score.score_dataset(make_dataset(make_label("a"), make_label("b")),
                                     {"a": {"e1": PERFECT_E1, "e2_normalized": perfect_normalized()}, "zzz": {}})
        self.assertEqual(report["unknown_output_ids"], ["zzz"])
        self.assertEqual([s["output_supplied"] for s in report["scenarios"]], [True, False])
        self.assertEqual(report["scenarios"][1]["stages"]["e2"]["status"], "output_missing")
        pooled = next(m for m in report["aggregate"]["e2"] if m["metric"] == "status_agreement")
        self.assertEqual((pooled["scenarios_scored"], pooled["scenarios_without_usable_output"]), (1, 1))
        self.assertEqual([p["scenario_id"] for p in pooled["per_scenario"]], ["a", "b"])
        self.assertIsNone(pooled["per_scenario"][1]["rate"])  # the failure stays visible next to the pooled figure

    def test_malformed_dataset_level_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            score.score_dataset(make_dataset(make_label()), [])
        with self.assertRaises(ValueError):
            score.score_dataset(make_dataset(make_label()), {"t1": "text"})


# ---------------------------------------------------------------------------------------------------------------
# 6. Missing versus partial
# ---------------------------------------------------------------------------------------------------------------

class MissingVersusPartial(unittest.TestCase):
    def test_a_wrong_status_fails_agreement_and_exact_match_but_still_counts_as_detected(self):
        output = normalized(crucial=[entry("Docker", "Docker is required", status="partial")],
                            good=[entry("Kubernetes", "Kubernetes is required")],
                            optional=[entry("Terraform", "Terraform is required", status="missing")])
        metrics = metrics_of(e2_of(output))
        self.assertEqual(metrics["gap_detection_recall"]["rate"], 1.0)
        self.assertEqual((metrics["status_agreement"]["numerator"], metrics["status_agreement"]["denominator"]), (2, 4))
        self.assertEqual((metrics["status_exact_match_given_flagged"]["numerator"],
                          metrics["status_exact_match_given_flagged"]["denominator"]), (1, 3))
        self.assertEqual(metrics["gap_detection_recall"]["counts"]["truth_vs_observed"],
                         {"missing": {"missing": 1, "partial": 1}, "partial": {"missing": 1}})

    def test_a_lenient_label_accepts_either_status_but_exact_match_does_not(self):
        label = make_label()
        label["skills"][1]["acceptable_outcomes"] = ["EMITTED:missing", "EMITTED:partial"]
        output = normalized(crucial=[entry("Docker", "Docker is required", status="partial")],
                            good=[entry("Kubernetes", "Kubernetes is required")],
                            optional=[entry("Terraform", "Terraform is required", status="partial")])
        metrics = metrics_of(e2_of(output, label=label))
        self.assertEqual(metrics["status_agreement"]["rate"], 1.0)
        self.assertEqual(metrics["status_exact_match_given_flagged"]["numerator"], 2)  # Docker partial != missing

    def test_an_unflagged_gap_is_unscorable_for_the_exact_match(self):
        metrics = metrics_of(e2_of(normalized(good=[entry("Kubernetes", "Kubernetes is required")])))
        self.assertEqual(metrics["status_exact_match_given_flagged"]["unscorable_reasons"], {"not_flagged": 2})


# ---------------------------------------------------------------------------------------------------------------
# 7. Invalid raw no_gap versus genuine normalised absence
# ---------------------------------------------------------------------------------------------------------------

def raw_text(**tiers):
    return json.dumps({"crucial": tiers.get("crucial", []), "good_to_have": tiers.get("good", []),
                       "optional": tiers.get("optional", [])})


def python_record(stage_record):
    return next(r for r in stage_record["skills"] if r["skill_id"] == "t_py")


class RawNoGapVersusGenuineAbsence(unittest.TestCase):
    base = None

    def test_genuine_absence_is_counted_correct_and_labelled_as_such(self):
        record = e2_of(perfect_normalized(), raw=raw_text(crucial=[entry("Docker", "Docker is required")]))
        python = python_record(record)
        self.assertEqual((python["observed_outcomes"], python["absence_provenance"]), (["ABSENT"], "genuine_absence"))
        self.assertTrue(python["status_counted_correct"])

    def test_a_rejected_raw_no_gap_is_not_credited_as_a_prediction_of_absence(self):
        raw = raw_text(crucial=[entry("Docker", "Docker is required"), entry("Python", "Python is required",
                                                                              status="no_gap")])
        record = e2_of(perfect_normalized(), raw=raw)
        python = python_record(record)
        self.assertEqual(python["observed_outcomes"], ["ABSENT"])          # normalised view: absent
        self.assertEqual(python["absence_provenance"], "rejected_raw_entry")  # raw view: a rejected entry
        self.assertEqual(python["raw_statuses_for_absence"], ["no_gap"])
        self.assertTrue(python["status_label_pass"])                          # the label itself accepts ABSENT
        self.assertFalse(python["status_counted_correct"])                    # ...but it is not counted correct
        self.assertIn("parser_dropped", python["interpretation"])
        metrics = metrics_of(record)
        self.assertEqual((metrics["status_agreement"]["numerator"], metrics["status_agreement"]["denominator"]), (3, 4))
        self.assertEqual(metrics["status_agreement"]["counts"]["absence_provenance"], {"rejected_raw_entry": 1})
        self.assertEqual(metrics["false_flag_rate"]["numerator"], 0)  # not a flag either

    def test_without_the_raw_string_provenance_is_unknown_and_says_so(self):
        python = python_record(e2_of(perfect_normalized()))
        self.assertEqual(python["absence_provenance"], "unknown")
        self.assertIn("provenance_unknown", python["interpretation"])
        self.assertEqual(python["status_counted_correct"], True)

    def test_unreadable_raw_leaves_provenance_unknown(self):
        record = e2_of(perfect_normalized(), raw="not json")
        self.assertEqual(python_record(record)["absence_provenance"], "unknown")
        self.assertEqual(record["observations"]["raw_reading"], "not_valid_json")

    def test_a_fenced_raw_string_is_read(self):
        raw = "```json\n" + raw_text(crucial=[entry("Python", "Python is required", status="no_gap")]) + "\n```"
        self.assertEqual(python_record(e2_of(perfect_normalized(), raw=raw))["absence_provenance"],
                         "rejected_raw_entry")

    def test_a_raw_entry_dropped_for_another_reason_is_not_a_genuine_absence(self):
        raw = raw_text(crucial=[entry("Python", "Python is required", status="missing")])  # valid raw, absent in normalised
        self.assertEqual(python_record(e2_of(perfect_normalized(), raw=raw))["absence_provenance"],
                         "dropped_raw_entry_other")

    def test_an_invalid_status_inside_the_normalised_output_is_an_anomaly_and_not_a_match(self):
        output = perfect_normalized()
        output["crucial"].append(entry("Python", "Python is required", status="no_gap"))
        record = e2_of(output)
        self.assertEqual(record["observations"]["anomalies"],
                         [{"type": "invalid_normalized_status", "tier": "crucial", "index": 1, "status": "no_gap"}])
        self.assertEqual(python_record(record)["observed_outcomes"], ["ABSENT"])

    def test_the_draft_token_cannot_be_satisfied_by_a_normalised_entry(self):
        label = make_label()
        label["skills"][0]["acceptable_outcomes"] = ["EMITTED:no_gap"]  # ABSENT not accepted at all
        python = python_record(e2_of(perfect_normalized(), label=label))
        self.assertFalse(python["status_counted_correct"])

    def test_end_to_end_through_the_production_normaliser(self):
        """A raw string with a no_gap entry is run through the real (extracted) normaliser, then scored."""
        parser = stages.load_parser()
        jd = "Requirements:\nPython is required\nDocker is required\n"
        helper = json.dumps({"technical_skills": ["Python", "Docker"], "soft_skills": [], "academic_qualifications": []})
        raw = raw_text(crucial=[entry("Docker", "Docker is required")],
                       good=[entry("Python", "Python is required", status="no_gap")])
        import logging
        logger = logging.getLogger(stages.PARSER_MODULE_NAME)
        logger.propagate = False
        self.addCleanup(setattr, logger, "propagate", True)
        result, _legacy = parser.build_prioritized_missing_skills(raw, jd, helper, None)
        record = e2_of(result, raw=raw)
        self.assertTrue(record["status"] == "scored")
        self.assertEqual(python_record(record)["absence_provenance"], "rejected_raw_entry")
        self.assertEqual(record["observations"]["diagnostics"], ["Dropped malformed entry in 'good_to_have': "
                                                                  "invalid status 'no_gap'."])


# ---------------------------------------------------------------------------------------------------------------
# 8. Unsupported evidence flags
# ---------------------------------------------------------------------------------------------------------------

class EvidenceSupport(unittest.TestCase):
    def test_unfound_jd_quotes_and_unfound_resume_quotes_are_counted_as_unsupported(self):
        output = normalized(
            crucial=[entry("Docker", "Docker is required", found=False, related=None)],
            good=[entry("Kubernetes", "Kubernetes is required")],
            optional=[entry("Terraform", "Terraform is required", status="partial", resume_evidence_found=False)])
        metrics = metrics_of(e2_of(output))
        self.assertEqual((metrics["jd_evidence_found"]["numerator"], metrics["jd_evidence_found"]["denominator"]), (2, 3))
        self.assertEqual((metrics["resume_evidence_quote_found"]["numerator"],
                          metrics["resume_evidence_quote_found"]["denominator"]), (0, 1))
        self.assertEqual(metrics["resume_evidence_quote_found"]["unscorable_reasons"],
                         {"resume_evidence_not_annotated": 2})

    def test_quote_agreement_with_the_authored_anchor_is_whole_phrase_and_case_insensitive(self):
        output = normalized(
            crucial=[entry("Docker", "- Hands-on DOCKER is required.  ")],   # contains the anchor
            good=[entry("Kubernetes", "something unrelated")],               # does not
            optional=[entry("Terraform", "Terraform")])                      # contained in the anchor
        metrics = metrics_of(e2_of(output))
        self.assertEqual((metrics["jd_quote_matches_authored_anchor"]["numerator"],
                          metrics["jd_quote_matches_authored_anchor"]["denominator"]), (2, 3))

    def test_a_non_boolean_evidence_flag_is_unscorable_not_correct(self):
        output = perfect_normalized()
        output["crucial"][0]["evidence_found"] = "yes"
        metrics = metrics_of(e2_of(output))
        self.assertEqual(metrics["jd_evidence_found"]["unscorable_reasons"], {"evidence_found_not_boolean": 1})
        self.assertEqual(metrics["jd_evidence_found"]["denominator"], 2)

    def test_entries_without_a_label_are_listed_not_scored(self):
        output = perfect_normalized()
        output["optional"].append(entry("Rust", "Rust is required"))
        observations = e2_of(output)["observations"]
        self.assertEqual(observations["unlabelled_entries"], [{"tier": "optional", "skill": "Rust", "status": "missing"}])


# ---------------------------------------------------------------------------------------------------------------
# 9. Evaluation-only aliases
# ---------------------------------------------------------------------------------------------------------------

class EvaluationOnlyAliases(unittest.TestCase):
    def test_production_key_does_not_equate_an_alias_with_its_canonical_name(self):
        key = stages.load_parser()._skill_key
        self.assertNotEqual(key("k8s"), key("Kubernetes"))
        self.assertEqual(key("Docker"), key("docker"))     # what production does do
        self.assertEqual(key("REST APIs"), key("rest api"))

    def test_the_scorer_matches_an_authored_alias_and_says_it_is_evaluation_only(self):
        output = normalized(crucial=[entry("Docker", "Docker is required")],
                            good=[entry("k8s", "Kubernetes is required")],
                            optional=[entry("Terraform", "Terraform is required", status="partial")])
        record = e2_of(output)
        kubernetes = next(r for r in record["skills"] if r["skill_id"] == "t_k8")
        self.assertEqual(kubernetes["observed_outcomes"], ["EMITTED:missing"])
        text = " ".join(record["matching_assumptions"])
        self.assertIn("evaluation aid only", text)
        self.assertIn("does not resolve aliases or acronyms", text)

    def test_e1_uses_only_the_authored_accepted_forms_and_never_substrings(self):
        label = make_label()
        output = {"technical_skills": ["Python scripting", "Docker Compose", "k8s", "Kubernetes", "Terraform"]}
        record = score.score_e1(label, output)
        matched = {r["skill_id"]: r["matched"] for r in record["skills"]}
        self.assertEqual(matched, {"t_py": False, "t_dk": False, "t_k8": True, "t_tf": True, "t_gq": False})
        self.assertIn("No substring, semantic or acronym matching", " ".join(record["matching_assumptions"]))
        self.assertIn("does not resolve aliases or acronyms", " ".join(record["matching_assumptions"]))

    def test_an_accepted_form_other_than_the_canonical_name_is_counted_as_such(self):
        label = make_label()
        label["skills"][1]["extraction"]["accepted_forms"] = ["Docker", "Docker Engine"]
        record = score.score_e1(label, {"technical_skills": ["Docker Engine"]})
        recall = metrics_of(record)["skill_recall"]
        self.assertEqual(recall["counts"]["matched_by_non_canonical_form"], 1)

    def test_case_and_punctuation_are_normalised_like_production(self):
        record = score.score_e1(make_label(), {"technical_skills": ["PYTHON", "d.o.c.k.e.r", "kubernetes!"]})
        self.assertEqual(metrics_of(record)["skill_recall"]["numerator"], 3)


# ---------------------------------------------------------------------------------------------------------------
# 10. Unscorable fields and denominator accounting
# ---------------------------------------------------------------------------------------------------------------

class DenominatorAccounting(unittest.TestCase):
    def test_every_label_in_scope_is_either_scored_or_counted_unscorable_with_a_reason(self):
        metrics = metrics_of(e2_of(perfect_normalized()))
        agreement = metrics["status_agreement"]
        self.assertEqual(agreement["denominator"] + agreement["unscorable_count"], 5)
        self.assertEqual(agreement["unscorable_reasons"], {"ambiguous_label_excluded": 1})

    def test_an_ambiguous_outcome_label_is_listed_with_what_was_observed_and_not_scored(self):
        output = perfect_normalized()
        output["crucial"].append(entry("GraphQL", "GraphQL is required"))
        record = e2_of(output)
        self.assertEqual(record["observations"]["ambiguous_labels"], [
            {"skill_id": "t_gq", "observed_outcomes": ["EMITTED:missing"], "absence_provenance": None}])
        self.assertEqual(metrics_of(record)["status_agreement"]["denominator"], 4)

    def test_unrecognised_outcome_class_is_unscorable(self):
        label = make_label()
        label["skills"][4]["outcome_class"] = "undecided"
        reasons = metrics_of(e2_of(perfect_normalized(), label=label))["status_agreement"]["unscorable_reasons"]
        self.assertEqual(reasons, {"unrecognised_outcome_class": 1})

    def test_labels_without_usable_forms_are_unscorable_for_e1(self):
        label = make_label()
        label["skills"][0]["extraction"]["accepted_forms"] = []
        recall = metrics_of(score.score_e1(label, PERFECT_E1))["skill_recall"]
        self.assertEqual((recall["numerator"], recall["denominator"]), (4, 4))
        self.assertEqual(recall["unscorable_reasons"], {"no_usable_accepted_form": 1})

    def test_mention_only_and_other_out_of_scope_truths_do_not_enter_recall_or_false_flag_denominators(self):
        metrics = metrics_of(e2_of(perfect_normalized()))
        self.assertEqual(metrics["gap_detection_recall"]["denominator"], 3)
        self.assertEqual(metrics["false_flag_rate"]["denominator"], 1)

    def test_pooled_denominators_are_sums_and_unscorable_reasons_are_merged(self):
        dataset = make_dataset(make_label("a"), make_label("b"))
        outputs = {sid: {"e1": PERFECT_E1, "e2_normalized": perfect_normalized()} for sid in ("a", "b")}
        report = score.score_dataset(dataset, outputs)
        pooled = {m["metric"]: m for m in report["aggregate"]["e2"]}
        self.assertEqual((pooled["status_agreement"]["numerator"], pooled["status_agreement"]["denominator"]), (8, 8))
        self.assertEqual(pooled["status_agreement"]["unscorable_reasons"], {"ambiguous_label_excluded": 2})
        self.assertEqual(pooled["status_agreement"]["counts"]["absence_provenance"], {"unknown": 2})
        e1_pooled = {m["metric"]: m for m in report["aggregate"]["e1"]}
        self.assertEqual(e1_pooled["skill_recall"]["denominator"], 10)
        self.assertTrue(e1_pooled["f1"]["permanently_unavailable"])
        self.assertIsNone(e1_pooled["f1"]["rate"])


# ---------------------------------------------------------------------------------------------------------------
# Strict agreement, degenerate outputs, pooling (Step 3 corrections)
# ---------------------------------------------------------------------------------------------------------------

class StrictAgreement(unittest.TestCase):
    def test_unknown_provenance_is_credited_by_the_qualified_metric_but_excluded_from_the_strict_one(self):
        record = e2_of(perfect_normalized())  # no raw string: Python's ABSENT has unknown provenance
        metrics = metrics_of(record)
        qualified, strict = metrics["status_agreement"], metrics["status_agreement_strict"]
        self.assertEqual((qualified["numerator"], qualified["denominator"]), (4, 4))
        self.assertEqual(qualified["counts"]["unknown_provenance_counted_correct"], 1)
        self.assertEqual((strict["numerator"], strict["denominator"]), (3, 3))
        self.assertEqual(strict["unscorable_reasons"],
                         {"absence_provenance_unknown": 1, "ambiguous_label_excluded": 1})
        self.assertIn("QUALIFIED", qualified["note"])
        self.assertIn("STRICT", strict["note"])

    def test_supplying_the_raw_string_makes_the_absence_scorable_again(self):
        raw = raw_text(crucial=[entry("Docker", "Docker is required")])
        strict = metrics_of(e2_of(perfect_normalized(), raw=raw))["status_agreement_strict"]
        self.assertEqual((strict["numerator"], strict["denominator"]), (4, 4))

    def test_a_rejected_raw_entry_still_fails_both_metrics(self):
        raw = raw_text(crucial=[entry("Python", "Python is required", status="no_gap")])
        metrics = metrics_of(e2_of(perfect_normalized(), raw=raw))
        for name in ("status_agreement", "status_agreement_strict"):
            self.assertEqual((metrics[name]["numerator"], metrics[name]["denominator"]), (3, 4), name)

    def test_a_missing_output_blocks_the_strict_metric_too(self):
        strict = metrics_of(score.score_e2(make_label(), None))["status_agreement_strict"]
        self.assertIsNone(strict["numerator"])
        self.assertEqual(strict["denominator"], 5)


class DegenerateOutputs(unittest.TestCase):
    def test_an_empty_but_valid_output_cannot_look_successful(self):
        genuine_raw = raw_text()
        record = e2_of(normalized(), raw=genuine_raw)
        metrics = metrics_of(record)
        self.assertEqual(record["status"], "scored")
        self.assertEqual((metrics["gap_detection_recall"]["numerator"], metrics["gap_detection_recall"]["denominator"]),
                         (0, 3))
        self.assertEqual(metrics["gap_detection_recall"]["rate"], 0.0)
        self.assertEqual(metrics["gap_detection_recall"]["counts"]["truth_vs_observed"],
                         {"missing": {"ABSENT": 2}, "partial": {"ABSENT": 1}})
        # the vacuous successes: no false flags and the demonstrated skill's ABSENT is accepted
        self.assertEqual(metrics["false_flag_rate"]["rate"], 0.0)
        self.assertEqual((metrics["status_agreement"]["numerator"], metrics["status_agreement"]["denominator"]), (1, 4))
        self.assertEqual(metrics["status_agreement"]["rate"], 0.25)
        self.assertTrue(any(w.startswith("empty_comparison") for w in record["observations"]["warnings"]))
        self.assertIn("gap_detection_recall", metrics["false_flag_rate"]["note"])

    def test_an_empty_output_without_raw_is_strictly_zero_not_credited(self):
        metrics = metrics_of(e2_of(normalized()))
        strict = metrics["status_agreement_strict"]
        self.assertEqual((strict["numerator"], strict["denominator"]), (0, 3))
        self.assertEqual(strict["rate"], 0.0)

    def test_flagging_every_labelled_skill_is_caught_by_the_false_flag_rate(self):
        output = normalized(crucial=[entry("Python", "Python is required"), entry("Docker", "Docker is required")],
                            good=[entry("Kubernetes", "Kubernetes is required"),
                                  entry("Terraform", "Terraform is required")],
                            optional=[entry("GraphQL", "GraphQL is required")])
        record = e2_of(output)
        metrics = metrics_of(record)
        self.assertEqual(metrics["gap_detection_recall"]["rate"], 1.0)  # high recall alone would look good
        self.assertEqual((metrics["false_flag_rate"]["numerator"], metrics["false_flag_rate"]["denominator"]), (1, 1))
        self.assertEqual(metrics["false_flag_rate"]["rate"], 1.0)
        self.assertEqual(metrics["false_flag_rate"]["direction"], "lower_is_better")
        # Python (demonstrated) and Terraform (partial, emitted as missing) are wrong; Docker, Kubernetes right
        self.assertEqual((metrics["status_agreement"]["numerator"], metrics["status_agreement"]["denominator"]), (2, 4))
        self.assertEqual(record["observations"]["warnings"], [])

    def test_the_dataset_report_surfaces_an_empty_output_as_a_warning(self):
        report = score.score_dataset(make_dataset(make_label("a"), make_label("b")),
                                     {"a": {"e1": PERFECT_E1, "e2_normalized": perfect_normalized()},
                                      "b": {"e1": PERFECT_E1, "e2_normalized": normalized()}})
        self.assertEqual(len(report["warnings"]), 1)
        self.assertTrue(report["warnings"][0].startswith("e2:b: empty_comparison"))
        pooled = {m["metric"]: m for m in report["aggregate"]["e2"]}
        self.assertEqual(pooled["gap_detection_recall"]["per_scenario"][1]["rate"], 0.0)  # visible, not averaged away


class PooledRates(unittest.TestCase):
    def test_pooled_rate_is_summed_numerators_over_summed_denominators_not_a_mean_of_rates(self):
        small = make_label("b_small")
        small["skills"] = small["skills"][:2]  # Python, Docker: 2 labelled skills
        dataset = make_dataset(make_label("a_large"), small)  # 5 labelled skills
        outputs = {"a_large": {"e1": PERFECT_E1},                         # 5/5 = 1.0
                   "b_small": {"e1": {"technical_skills": ["Python"]}}}   # 1/2 = 0.5
        report = score.score_dataset(dataset, outputs)
        pooled = next(m for m in report["aggregate"]["e1"] if m["metric"] == "skill_recall")
        self.assertEqual((pooled["numerator"], pooled["denominator"]), (6, 7))
        self.assertEqual(pooled["rate"], round(6 / 7, 6))
        self.assertNotEqual(pooled["rate"], 0.75)  # the mean of 1.0 and 0.5
        self.assertEqual([p["rate"] for p in pooled["per_scenario"]], [1.0, 0.5])


# ---------------------------------------------------------------------------------------------------------------
# Course eligibility
# ---------------------------------------------------------------------------------------------------------------

class CourseEligibilityScoring(unittest.TestCase):
    def test_an_entry_whose_quote_was_not_found_is_skipped_and_disagrees_with_a_label_that_expects_it(self):
        output = perfect_normalized()
        output["crucial"][0]["evidence_found"] = False
        record = score.score_course(make_label(), output)
        self.assertEqual(record["observations"]["skipped"], 1)
        self.assertEqual(record["observations"]["offered"], ["Kubernetes", "Terraform"])
        metrics = metrics_of(record)
        self.assertEqual(metrics["offered_options_match_rule"]["rate"], 1.0)  # the production rule was followed
        self.assertEqual((metrics["label_eligibility_agreement"]["numerator"],
                          metrics["label_eligibility_agreement"]["denominator"]), (3, 4))  # Docker expected, not offered

    def test_non_technical_entries_are_not_offered_or_counted(self):
        output = normalized(crucial=[entry("Communication", "Communication is required", category="soft_skills")])
        record = score.score_course(make_label(), output)
        self.assertEqual((record["observations"]["offered"], record["observations"]["skipped"]), ([], 0))
        self.assertEqual(metrics_of(record)["skipped_count_match_rule"]["rate"], 1.0)

    def test_unusable_comparison_output_follows_the_production_contract(self):
        for bad in (normalized(ok=False), normalized(crucial=[entry("Docker", "q", status="no_gap")])):
            with self.subTest(output=bad):
                record = score.score_course(make_label(), bad)
                self.assertFalse(record["observations"]["comparison_output_usable"])
                metrics = metrics_of(record)
                # ({}, 0) is the production contract, but it is not evidence that the rule was conformed to.
                self.assertEqual(record["status"], "comparison_output_unusable")
                for name in ("offered_options_match_rule", "skipped_count_match_rule"):
                    self.assertFalse(metrics[name]["available"], name)
                    self.assertIsNone(metrics[name]["numerator"], name)
                    self.assertIsNone(metrics[name]["rate"], name)
                    self.assertEqual(metrics[name]["unscorable_reasons"], {"comparison_output_unusable": 1}, name)
                self.assertEqual(metrics["label_eligibility_agreement"]["unscorable_reasons"],
                                 {"comparison_output_unusable": 5})
                self.assertFalse(metrics["label_eligibility_agreement"]["available"])

    def test_a_dataset_whose_e2_always_fails_pools_to_no_course_conformance(self):
        fallback = normalized(ok=False, diagnostics=["Comparison output is not valid JSON."])
        outputs = {sid: {"e1": PERFECT_E1, "e2_normalized": fallback} for sid in ("a", "b")}
        report = score.score_dataset(make_dataset(make_label("a"), make_label("b")), outputs)
        pooled = {m["metric"]: m for m in report["aggregate"]["course"]}
        for name in ("offered_options_match_rule", "skipped_count_match_rule"):
            self.assertFalse(pooled[name]["available"], name)
            self.assertIsNone(pooled[name]["rate"], name)
        self.assertEqual({s["stages"]["course"]["status"] for s in report["scenarios"]},
                         {"comparison_output_unusable"})

    def test_ambiguous_labels_are_excluded_from_label_agreement(self):
        metrics = metrics_of(score.score_course(make_label(), perfect_normalized()))
        self.assertEqual(metrics["label_eligibility_agreement"]["unscorable_reasons"], {"ambiguous_label_excluded": 1})

    def test_the_scorer_calls_no_recommender_and_scores_no_relevance(self):
        names = [m["metric"] for m in score.score_course(make_label(), perfect_normalized())["metrics"]]
        self.assertEqual(names, ["offered_options_match_rule", "skipped_count_match_rule", "label_eligibility_agreement"])
        self.assertIn("relevant", " ".join(score.LIMITATIONS_COURSE))


# ---------------------------------------------------------------------------------------------------------------
# 11-12. Determinism and JSON
# ---------------------------------------------------------------------------------------------------------------

class DeterminismAndJson(unittest.TestCase):
    def setUp(self):
        self.dataset = make_dataset(make_label("b_second"), make_label("a_first"))
        self.outputs = {"a_first": {"e1": PERFECT_E1, "e2_normalized": perfect_normalized(),
                                    "e2_raw": raw_text(crucial=[entry("Docker", "Docker is required")])},
                        "b_second": {"e1": {"technical_skills": ["Python"]},
                                     "e2_normalized": normalized(crucial=[entry("Docker", "Docker is required")])}}

    def test_the_report_is_identical_across_runs_and_independent_of_output_key_order(self):
        first = score.score_dataset(self.dataset, self.outputs)
        second = score.score_dataset(self.dataset, copy.deepcopy(dict(reversed(list(self.outputs.items())))))
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_scenarios_follow_the_dataset_id_order_and_metrics_a_fixed_order(self):
        report = score.score_dataset(self.dataset, self.outputs)
        self.assertEqual([s["scenario_id"] for s in report["scenarios"]], ["b_second", "a_first"])
        self.assertEqual([m["metric"] for m in report["scenarios"][0]["stages"]["e2"]["metrics"]], [
            "status_agreement", "status_agreement_strict", "gap_detection_recall", "false_flag_rate", "status_exact_match_given_flagged",
            "tier_agreement_determinate", "tier_agreement_ambiguous_set", "jd_evidence_found",
            "jd_quote_matches_authored_anchor", "resume_evidence_quote_found"])
        self.assertEqual(list(report["aggregate"]), ["e1", "e2", "course"])

    def test_the_report_round_trips_through_json_without_loss(self):
        report = score.score_dataset(self.dataset, self.outputs)
        self.assertEqual(json.loads(json.dumps(report)), report)
        self.assertEqual(report["report_version"], score.REPORT_VERSION)

    def test_the_input_data_are_not_modified(self):
        before = copy.deepcopy((self.dataset, self.outputs))
        score.score_dataset(self.dataset, self.outputs)
        self.assertEqual((self.dataset, self.outputs), before)

    def test_every_metric_record_carries_the_required_fields(self):
        report = score.score_dataset(self.dataset, self.outputs)
        required = {"stage", "metric", "numerator", "denominator", "rate", "available", "direction",
                    "unscorable_count", "unscorable_reasons", "note"}
        for scenario in report["scenarios"]:
            for stage in scenario["stages"].values():
                self.assertTrue(stage["matching_assumptions"] or stage["stage"] == "course")
                self.assertTrue(stage["limitations"])
                for metric in stage["metrics"]:
                    self.assertTrue(required <= set(metric), metric["metric"])
                    if not metric["available"]:
                        self.assertIn("unavailable_reason", metric)
        self.assertTrue(report["limitations"])


# ---------------------------------------------------------------------------------------------------------------
# 13. No network, no writes, real dataset untouched
# ---------------------------------------------------------------------------------------------------------------

def _hash_tree(directory):
    result = {}
    for folder in ("scenarios", "expected"):
        base = os.path.join(directory, folder)
        for name in sorted(os.listdir(base)):
            with open(os.path.join(base, name), "rb") as handle:
                result[f"{folder}/{name}"] = hashlib.sha256(handle.read()).hexdigest()
    return result


class IsolationAndRealLabels(unittest.TestCase):
    def test_scoring_makes_no_network_call(self):
        def refuse(*args, **kwargs):
            raise AssertionError("network access attempted")
        with mock.patch.object(socket.socket, "connect", refuse), mock.patch.object(socket, "create_connection", refuse), \
                mock.patch.object(socket, "getaddrinfo", refuse):
            score.score_dataset(make_dataset(make_label()), {"t1": {"e1": PERFECT_E1,
                                                                   "e2_normalized": perfect_normalized()}})

    def test_score_module_imports_only_the_standard_library_it_needs(self):
        with open(os.path.join(PIPELINE, "score.py"), "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module)
        self.assertEqual(found, {"importlib.util", "json", "os", "re", "sys"})

    def test_score_module_opens_no_file_and_names_no_protected_artifact(self):
        with open(os.path.join(PIPELINE, "score.py"), "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("open", calls)
        for forbidden in (".env", "resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                          "missing_skills.txt", "ats_optimized_bullets.txt"):
            self.assertNotIn(forbidden, source)

    def test_scoring_the_real_labels_with_no_outputs_changes_no_dataset_file(self):
        before = _hash_tree(PIPELINE)
        dataset = dataset_module.load_dataset()
        report = score.score_dataset(dataset, {})
        self.assertEqual(before, _hash_tree(PIPELINE))
        self.assertEqual(len(report["scenarios"]), 10)
        for scenario in report["scenarios"]:
            self.assertFalse(scenario["output_supplied"])
            self.assertEqual({s["status"] for s in scenario["stages"].values()}, {"output_missing"})
        for stage in report["aggregate"].values():
            for metric in stage:
                self.assertFalse(metric["available"], metric["metric"])  # nothing was supplied, nothing is "correct"

    def test_label_derived_outputs_exercise_every_real_label_shape(self):
        """FIXTURE, NOT A RESULT: outputs are generated FROM the labels, so full marks only show that the scorer can
        read all 68 real labels. They say nothing about the pipeline or a model."""
        dataset = dataset_module.load_dataset()
        outputs = {}
        for sid in dataset["ids"]:
            label = dataset["labels"][sid]
            e1 = {cat: [] for cat in score.E1_CATEGORIES}
            tiers = {"crucial": [], "good_to_have": [], "optional": []}
            for skill in label["skills"]:
                e1[skill["category"]].append(skill["extraction"]["accepted_forms"][0])
                if skill["outcome_class"] == "determinate" and skill["authoring_truth"] in ("missing", "partial"):
                    tier = (skill["jd"]["expected_tier_set"] or ["crucial"])[0]
                    tiers[tier].append(entry(skill["canonical"], skill["jd"]["mentions"][0]["anchor"],
                                             category=skill["category"], status=skill["authoring_truth"]))
            outputs[sid] = {"e1": e1, "e2_normalized": normalized(crucial=tiers["crucial"], good=tiers["good_to_have"],
                                                                  optional=tiers["optional"])}
        report = score.score_dataset(dataset, outputs)
        pooled = {stage: {m["metric"]: m for m in metrics} for stage, metrics in report["aggregate"].items()}
        self.assertEqual(pooled["e1"]["skill_recall"]["denominator"], 68)
        self.assertEqual(pooled["e1"]["skill_recall"]["rate"], 1.0)
        for name in ("status_agreement", "gap_detection_recall", "status_exact_match_given_flagged",
                     "tier_agreement_determinate", "tier_agreement_ambiguous_set"):
            with self.subTest(metric=name):
                self.assertEqual(pooled["e2"][name]["rate"], 1.0, pooled["e2"][name])
        self.assertEqual(pooled["e2"]["false_flag_rate"]["rate"], 0.0)
        self.assertEqual(pooled["e2"]["status_agreement"]["denominator"], 65)
        self.assertEqual(pooled["e2"]["status_agreement"]["unscorable_reasons"], {"ambiguous_label_excluded": 3})
        # no raw string was supplied, so the 28 demonstrated (ABSENT) labels have unknown provenance
        strict = pooled["e2"]["status_agreement_strict"]
        self.assertEqual((strict["numerator"], strict["denominator"]), (37, 37))
        self.assertEqual(strict["unscorable_reasons"], {"ambiguous_label_excluded": 3,
                                                        "absence_provenance_unknown": 28})
        self.assertEqual(pooled["course"]["offered_options_match_rule"]["rate"], 1.0)
        self.assertEqual(pooled["course"]["label_eligibility_agreement"]["rate"], 1.0,
                         pooled["course"]["label_eligibility_agreement"])
        self.assertEqual(json.loads(json.dumps(report)), report)


if __name__ == "__main__":
    unittest.main()
