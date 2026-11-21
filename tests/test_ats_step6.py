"""Offline tests for eval/ats/analyze_step6.py.

Run from the project root:  python -m unittest tests.test_ats_step6 -v
Every test builds a small SYNTHETIC baseline folder in a temporary directory. The only touch of the real baseline is
one read-only integrity check that is skipped when the folder is absent. No network, no model, no raw response, no
.env, no protected data file, no case file.
"""
import ast
import builtins
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATS_DIR = os.path.join(ROOT, "eval", "ats")
REAL_RUN = os.path.join(ATS_DIR, "results", "ats_dev_20261029T052205Z")
FIXED_NOW = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
PROTECTED = ("resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt", "missing_skills.txt",
             "ats_optimized_bullets.txt")


def _load(name, filename):
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(ATS_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


step6 = _load("ats_step6", "analyze_step6.py")
runner = _load("ats_runner", "run_ats_eval.py")
checks = runner.checks


def read_text(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def folder_hashes(folder):
    return {n: sha(os.path.join(folder, n)) for n in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, n))}


def write_csv(path, header, rows, bom=False, crlf=False, quote_all=False):
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n" if crlf else "\n",
                            quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in header])


def make_row(case, repeat, index, section, heading, wording, d3="", d4="", d6="",
             labels=("yes", "none", "high", "high", "improved")):
    support, claims, relevance, clarity, overall = labels
    return {"case_id": case, "split": "dev", "repeat": str(repeat), "call_id": f"{case}__r{repeat:02d}",
            "bullet_index": str(index), "output_section": section, "matched_source_heading": heading,
            "source_bullets": "src one || src two", "rewritten_bullet": wording,
            "checker_findings": "; ".join(x for x in (d3 and "D3", d4 and "D4", d6 and "D6") if x),
            "d2_invented_numbers": "", "d3_absent_skill_leaks": d3, "d4_unsupported_technologies": d4,
            "d6_unchanged_from_source": d6, "d6_near_duplicate_of": "",
            "supported_by_source_evidence": support, "unsupported_or_invented_claims": claims,
            "relevance_to_job_description": relevance, "clarity_and_action_orientation": clarity,
            "overall_rewrite_improvement": overall, "reviewer_notes": "note"}


class Fixture(object):
    """A synthetic baseline run: 2 cases x 2 repeats, 6 labelled bullets."""
    RUN_ID = "ats_dev_TEST"

    def __init__(self, test):
        self.temp = tempfile.TemporaryDirectory()
        test.addCleanup(self.temp.cleanup)
        self.results = os.path.join(self.temp.name, "results")
        self.run_dir = os.path.join(self.results, self.RUN_ID)
        os.makedirs(self.run_dir)
        A0 = ("Experience - Dev (2020)", "Experience")
        B0 = ("Projects - X", "Projects - X")
        self.rows = [
            make_row("caseA", 1, 0, *A0, "Built an API with Flask.", d6="yes", labels=("yes", "none", "high", "high", "unchanged")),
            make_row("caseA", 1, 1, *A0, "Wrote tests for the API.", d3="kafka", d4="kafka",
                     labels=("partially", "Invented tool X", "medium", "high", "worse")),
            make_row("caseA", 2, 0, *A0, "Built an API with Flask.", d6="yes", labels=("yes", "none", "high", "high", "unchanged")),
            make_row("caseA", 2, 1, *A0, "wrote tests for the api", d3="kafka", d4="kafka",
                     labels=("partially", "Invented tool X", "medium", "high", "worse")),
            make_row("caseB", 1, 0, *B0, "Designed a data pipeline.", labels=("yes", "Minor caveat", "high", "high", "improved")),
            make_row("caseB", 2, 0, *B0, "Created ETL jobs for data loading", labels=("yes", "none", "high", "medium", "improved"))]
        self.reviewed = None  # None = identical to rows
        self.decisions = None
        self.plan = {"run_id": self.RUN_ID, "split": "dev", "repeats": 2, "case_ids": ["caseA", "caseB"],
                     "model_expected": "m/x", "model_settings": {"max_tokens": 8192},
                     "evaluation_overrides": {"max_tokens": 4096}, "effective_settings": {"max_tokens": 4096}}
        self.calls = None
        self.write()

    def write(self):
        cols = step6.BULLETS_COLUMNS
        blank = {h: "" for h in step6.HUMAN}
        write_csv(os.path.join(self.run_dir, "bullets.csv"), cols, [dict(r, **blank) for r in self.rows])
        write_csv(os.path.join(self.run_dir, "bullets_reviewed.csv"), cols, self.reviewed or self.rows,
                  bom=True, crlf=True, quote_all=True)
        dec = self.decisions or [{c: r[c] for c in step6.DECISIONS_COLUMNS} for r in self.rows]
        write_csv(os.path.join(self.run_dir, "human_review_decisions.csv"), step6.DECISIONS_COLUMNS, dec, bom=True)
        calls = self.calls or [{"call_id": f"{c}__r{r:02d}", "case_id": c, "repeat": r, "status": "ok",
                                "finish_reason": "stop", "truncated": False, "attempts": [{"attempt": 1}]}
                               for c in ("caseA", "caseB") for r in (1, 2)]
        with open(os.path.join(self.run_dir, "calls.jsonl"), "w", encoding="utf-8", newline="") as handle:
            handle.write("".join(json.dumps(c) + "\n" for c in calls))
        for name, text in (("evaluations.jsonl", "{}\n"), ("summary.json", "{}"), ("metadata.json", "{}"),
                           ("report.md", "# baseline\n")):
            with open(os.path.join(self.run_dir, name), "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        with open(os.path.join(self.run_dir, "plan.json"), "w", encoding="utf-8", newline="") as handle:
            json.dump(self.plan, handle)

    def analyze(self):
        return step6.analyze(self.RUN_ID, self.results, FIXED_NOW)

    def metrics(self):
        target, metrics = self.analyze()
        return metrics


class HappyPath(unittest.TestCase):
    def test_three_new_files_and_the_baseline_is_untouched(self):
        fx = Fixture(self)
        before = folder_hashes(fx.run_dir)
        target, metrics = fx.analyze()
        self.assertEqual(sorted(os.listdir(target)), sorted(step6.OUTPUT_FILES))
        self.assertEqual(folder_hashes(fx.run_dir), before)
        self.assertEqual(sorted(os.listdir(fx.run_dir)), sorted(list(before) + ["step6"]))
        self.assertTrue(metrics["baseline_hashes"]["unchanged"])
        self.assertEqual(set(metrics["baseline_hashes"]["before"]), set(step6.HASHED_FILES))

    def test_bom_crlf_and_quoted_header_are_handled(self):
        fx = Fixture(self)
        with open(os.path.join(fx.run_dir, "bullets_reviewed.csv"), "rb") as handle:
            raw = handle.read()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf\""))
        self.assertIn(b"\r\n", raw)
        self.assertEqual(fx.metrics()["integrity"]["unique_keys"], 6)

    def test_main_exit_code_and_messages(self):
        fx = Fixture(self)
        lines = []
        code = step6.main(["--run-id", fx.RUN_ID], results_root=fx.results, out=lines.append, now=FIXED_NOW)
        self.assertEqual(code, 0)
        self.assertTrue(any("verified unchanged" in x for x in lines))

    def test_outputs_are_deterministic(self):
        first, second = Fixture(self), Fixture(self)
        t1, _ = first.analyze()
        t2, _ = second.analyze()
        for name in step6.OUTPUT_FILES:
            if name != "step6_metrics.json":
                self.assertEqual(read_text(os.path.join(t1, name)), read_text(os.path.join(t2, name)), name)
        m1 = json.loads(read_text(os.path.join(t1, "step6_metrics.json")))
        m2 = json.loads(read_text(os.path.join(t2, "step6_metrics.json")))
        self.assertEqual(m1["human_review"], m2["human_review"])
        self.assertEqual(m1["repeat_comparison"], m2["repeat_comparison"])


class HumanMetrics(unittest.TestCase):
    def test_every_dimension_has_counts_and_denominators(self):
        h = Fixture(self).metrics()["human_review"]
        dims = h["dimensions"]
        self.assertEqual(h["bullets"], 6)
        self.assertEqual({v: f["text"] for v, f in dims["supported_by_source_evidence"].items()},
                         {"yes": "4/6 (66.7%)", "partially": "2/6 (33.3%)", "no": "0/6 (0.0%)", "other_or_blank": "0/6 (0.0%)"})
        self.assertEqual(dims["relevance_to_job_description"]["high"]["text"], "4/6 (66.7%)")
        self.assertEqual(dims["relevance_to_job_description"]["medium"]["text"], "2/6 (33.3%)")
        self.assertEqual(dims["clarity_and_action_orientation"]["high"]["text"], "5/6 (83.3%)")
        self.assertEqual({v: f["count"] for v, f in dims["overall_rewrite_improvement"].items()},
                         {"improved": 2, "unchanged": 2, "worse": 2, "other_or_blank": 0})
        for field, values in dims.items():
            self.assertEqual(sum(f["count"] for f in values.values()), 6, field)
        claims = h["unsupported_or_invented_claims_field"]
        self.assertEqual((claims["none"]["count"], claims["explanatory_text"]["count"], claims["blank"]["count"]), (3, 3, 0))

    def test_breakdowns_by_case_and_repeat_reconcile_with_the_totals(self):
        h = Fixture(self).metrics()["human_review"]
        self.assertEqual([c["bullets"] for c in h["by_case"]], [4, 2])
        self.assertEqual(sum(c["bullets"] for c in h["by_case_repeat"]), 6)
        self.assertEqual([(c["case_id"], c["repeat"], c["bullets"]) for c in h["by_case_repeat"]],
                         [("caseA", 1, 2), ("caseA", 2, 2), ("caseB", 1, 1), ("caseB", 2, 1)])
        first = h["by_case_repeat"][0]["counts"]
        self.assertEqual((first["supported_by_source_evidence"]["partially"], first["overall_rewrite_improvement"]["worse"],
                          first["overall_rewrite_improvement"]["unchanged"]), (1, 1, 1))
        self.assertEqual(h["overall_improvement_by_support_counts"]["partially"],
                         {"bullets": 2, "improved": 0, "unchanged": 0, "worse": 2})

    def test_yes_support_with_claim_text_is_flagged_not_corrected(self):
        fx = Fixture(self)
        before = read_text(os.path.join(fx.run_dir, "human_review_decisions.csv"))
        target, metrics = fx.analyze()
        flagged = metrics["human_review"]["data_quality"]["support_yes_with_explanatory_claim_text"]
        self.assertEqual((flagged["count"], flagged["of_support_yes"]), (1, 4))
        self.assertEqual((flagged["rows"][0]["case_id"], flagged["rows"][0]["repeat"]), ("caseB", 1))
        self.assertEqual(read_text(os.path.join(fx.run_dir, "human_review_decisions.csv")), before)
        self.assertEqual(metrics["human_review"]["dimensions"]["supported_by_source_evidence"]["yes"]["count"], 4)  # still yes
        self.assertIn("1 of 4 bullets rated support `yes`", read_text(os.path.join(target, "step6_report.md")))

    def test_invalid_labels_are_counted_as_supplied_and_reported(self):
        fx = Fixture(self)
        fx.rows[0]["supported_by_source_evidence"] = "maybe"
        fx.write()
        metrics = fx.metrics()
        self.assertEqual(metrics["integrity"]["details"]["labels_outside_agreed_sets"]["count"], 2)  # reviewed + decisions
        self.assertEqual(metrics["human_review"]["dimensions"]["supported_by_source_evidence"]["other_or_blank"]["count"], 1)
        self.assertEqual(metrics["human_review"]["dimensions"]["supported_by_source_evidence"]["yes"]["count"], 3)

    def test_disagreement_between_reviewed_and_decisions_is_reported(self):
        fx = Fixture(self)
        fx.reviewed = [dict(r) for r in fx.rows]
        fx.reviewed[1]["relevance_to_job_description"] = "high"
        fx.write()
        details = fx.metrics()["integrity"]["details"]["reviewed_vs_decisions_disagreements_by_field"]
        self.assertEqual(details["relevance_to_job_description"], 1)
        self.assertEqual(sum(v for k, v in details.items() if k != "relevance_to_job_description"), 0)

    def test_no_accuracy_precision_or_recall_is_computed(self):
        metrics = Fixture(self).metrics()
        keys = set()

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    keys.add(str(key).lower())
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(metrics)
        self.assertFalse(keys & {"accuracy", "precision", "recall", "f1", "f1_score"})  # no such metric exists
        self.assertIn("accuracy", metrics["not_calculated"])
        self.assertIn("precision", metrics["not_calculated"])


class AutomatedChecks(unittest.TestCase):
    def test_d3_and_d4_overlap_is_reported_and_never_summed(self):
        auto = Fixture(self).metrics()["automated_vs_human"]
        self.assertEqual(auto["flag_totals"]["d3"]["text"], "2/6 (33.3%)")
        self.assertEqual(auto["flag_totals"]["d4"]["text"], "2/6 (33.3%)")
        o = auto["d3_d4_overlap"]
        self.assertEqual((o["both_d3_and_d4"], o["d3_only"], o["d4_only"], o["neither"]), (2, 0, 0, 4))
        self.assertNotIn("d3_plus_d4", json.dumps(auto))
        self.assertEqual(auto["union_d2_d3_d4_counts"]["by_support"]["flagged"]["bullets"], 2)  # union, not 4

    def test_crosstabs_are_plain_counts(self):
        cross = Fixture(self).metrics()["automated_vs_human"]["crosstabs_counts"]
        self.assertEqual(cross["d3_vs_support"]["flagged"], {"bullets": 2, "yes": 0, "partially": 2, "no": 0})
        self.assertEqual(cross["d6_unchanged_vs_overall"]["flagged"], {"bullets": 2, "improved": 0, "unchanged": 2, "worse": 0})
        self.assertEqual(cross["d2_vs_support"]["flagged"]["bullets"], 0)
        union = Fixture(self).metrics()["automated_vs_human"]["union_d2_d3_d4_counts"]
        self.assertEqual((union["support_partially_or_no_total"], union["support_partially_or_no_without_any_d2_d3_d4_flag"]), (2, 0))

    def test_d2_with_no_flags_is_stated_as_not_comparable(self):
        auto = Fixture(self).metrics()["automated_vs_human"]
        self.assertTrue(any("D2 flagged no bullets" in x for x in auto["not_comparable"]))


class RepeatComparison(unittest.TestCase):
    def test_exact_normalized_and_changed_wording_are_separated(self):
        rep = Fixture(self).metrics()["repeat_comparison"]
        self.assertEqual(rep["pair_status_counts"], {"paired": 3})
        w = rep["wording"]
        self.assertEqual((w["exact_equal"]["count"], w["normalized_equal_but_not_exact"]["count"],
                          w["normalized_equal_total_including_exact"]["count"], w["changed_after_normalisation"]["count"]),
                         (1, 1, 2, 1))
        self.assertEqual(w["changed_after_normalisation"]["of"], 3)
        self.assertEqual(w["word_overlap_jaccard_of_changed_pairs"]["median"], 0.111)

    def test_judgment_differences_are_reported_by_wording_class(self):
        rep = Fixture(self).metrics()["repeat_comparison"]
        self.assertEqual(rep["human_judgments_differing_between_repeats"]["clarity_and_action_orientation"]["count"], 1)
        self.assertEqual(rep["human_judgments_differing_between_repeats"]["overall_rewrite_improvement"]["count"], 0)
        self.assertEqual(rep["unsupported_claims_field_none_vs_text_differs"]["count"], 1)
        by = rep["judgments_by_wording_class"]
        self.assertEqual((by["exact_wording"]["positions"], by["exact_wording"]["any_of_four_dimensions_differs"]), (1, 0))
        self.assertEqual((by["wording_changed"]["positions"], by["wording_changed"]["any_of_four_dimensions_differs"]), (1, 1))

    def test_response_status_consistency_is_reported_honestly(self):
        status = Fixture(self).metrics()["repeat_comparison"]["response_status"]
        self.assertEqual(status["status_counts_all_calls"], {"ok": 4})
        self.assertEqual(status["cases_with_same_status_in_both_repeats"]["text"], "2/2 (100.0%)")
        self.assertIn("no information", status["note"])

    def test_unmatched_and_mismatched_pairs_are_reported_not_forced(self):
        fx = Fixture(self)
        fx.rows.append(make_row("caseA", 1, 2, "Experience - Dev (2020)", "Experience", "Extra bullet only in repeat 1"))
        fx.rows[3]["output_section"] = "Other section"          # caseA r2 #1: section differs
        fx.rows[5]["matched_source_heading"] = ""               # caseB r2 #0: heading blank
        fx.plan["case_ids"] = ["caseA", "caseB"]
        fx.write()
        rep = fx.metrics()["repeat_comparison"]
        self.assertEqual(rep["pair_status_counts"], {"ambiguous_no_matched_heading": 1, "paired": 1,
                                                     "section_mismatch": 1, "unmatched_missing_repeat": 1})
        self.assertEqual(rep["cases_with_different_bullet_counts_between_repeats"], ["caseA"])
        self.assertEqual(rep["paired_positions"], 1)
        self.assertEqual(rep["wording"]["exact_equal"]["of"], 1)  # only the paired position is compared

    def test_heading_mismatch_is_reported(self):
        fx = Fixture(self)
        fx.rows[3]["matched_source_heading"] = "Education"
        fx.write()
        self.assertEqual(fx.metrics()["repeat_comparison"]["pair_status_counts"], {"heading_mismatch": 1, "paired": 2})

    def test_pairs_csv_has_one_row_per_position_and_neutralises_formulas(self):
        fx = Fixture(self)
        fx.rows[4]["rewritten_bullet"] = "=SUM(1,2)"
        fx.write()
        target, _ = fx.analyze()
        with open(os.path.join(target, "step6_repeat_pairs.csv"), encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames), step6.PAIR_COLUMNS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["wording_r1"], "'=SUM(1,2)")
        self.assertEqual((rows[0]["exact_wording_equal"], rows[1]["exact_wording_equal"], rows[1]["normalized_wording_equal"]),
                         ("yes", "no", "yes"))

    def test_only_two_repeats_are_supported(self):
        fx = Fixture(self)
        fx.plan["repeats"] = 3
        fx.write()
        with self.assertRaises(step6.StepError) as caught:
            fx.analyze()
        self.assertEqual(caught.exception.code, 3)


class IntegrityFailures(unittest.TestCase):
    def assert_stops(self, fx, code=3, fragment=None):
        with self.assertRaises(step6.StepError) as caught:
            fx.analyze()
        self.assertEqual(caught.exception.code, code)
        if fragment:
            self.assertIn(fragment, str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(fx.run_dir, "step6")))  # nothing written

    def test_duplicate_key(self):
        fx = Fixture(self)
        fx.decisions = [{c: r[c] for c in step6.DECISIONS_COLUMNS} for r in fx.rows]
        fx.decisions.append(dict(fx.decisions[0]))
        fx.write()
        self.assert_stops(fx, fragment="duplicated key")

    def test_missing_row_in_decisions(self):
        fx = Fixture(self)
        fx.decisions = [{c: r[c] for c in step6.DECISIONS_COLUMNS} for r in fx.rows[:-1]]
        fx.write()
        self.assert_stops(fx, fragment="key sets differ")

    def test_unmatched_extra_row_in_reviewed(self):
        fx = Fixture(self)
        fx.reviewed = [dict(r) for r in fx.rows] + [make_row("caseZ", 1, 0, "S", "S", "x")]
        fx.write()
        self.assert_stops(fx, fragment="key sets differ")

    def test_reviewed_sheet_that_changes_an_original_field(self):
        fx = Fixture(self)
        fx.reviewed = [dict(r) for r in fx.rows]
        fx.reviewed[0]["rewritten_bullet"] = "Edited by hand"
        fx.write()
        self.assert_stops(fx, fragment="changed original non-review field")

    def test_wrong_columns(self):
        fx = Fixture(self)
        path = os.path.join(fx.run_dir, "human_review_decisions.csv")
        write_csv(path, step6.DECISIONS_COLUMNS[:-1], [{c: r[c] for c in step6.DECISIONS_COLUMNS[:-1]} for r in fx.rows], bom=True)
        self.assert_stops(fx, fragment="columns differ")

    def test_non_dev_split_is_refused(self):
        fx = Fixture(self)
        for row in fx.rows:
            row["split"] = "holdout"
        fx.write()
        self.assert_stops(fx, fragment="not a development-only run")

    def test_case_ids_must_match_the_plan(self):
        fx = Fixture(self)
        fx.plan["case_ids"] = ["caseA"]
        fx.write()
        self.assert_stops(fx, fragment="case ids")

    def test_calls_must_exist_and_be_ok(self):
        fx = Fixture(self)
        fx.calls = [{"call_id": f"{c}__r{r:02d}", "case_id": c, "repeat": r, "status": "ok"}
                    for c in ("caseA", "caseB") for r in (1, 2)]
        fx.calls[0]["status"] = "parse_failed"
        fx.write()
        self.assert_stops(fx, fragment="status-ok record")

    def test_missing_baseline_file(self):
        fx = Fixture(self)
        os.remove(os.path.join(fx.run_dir, "metadata.json"))
        self.assert_stops(fx, code=2, fragment="metadata.json")

    def test_invalid_run_ids(self):
        fx = Fixture(self)
        for bad in ("../x", "a/b", "", ".hidden", "nonexistent"):
            with self.assertRaises(step6.StepError):
                step6.analyze(bad, fx.results, FIXED_NOW)


class NeverOverwrites(unittest.TestCase):
    def test_existing_step6_directory_stops_the_analysis(self):
        fx = Fixture(self)
        os.mkdir(os.path.join(fx.run_dir, "step6"))
        marker = os.path.join(fx.run_dir, "step6", "step6_report.md")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("keep me")
        before = folder_hashes(fx.run_dir)
        with self.assertRaises(step6.StepError) as caught:
            fx.analyze()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("refusing to overwrite", str(caught.exception))
        self.assertEqual(read_text(marker), "keep me")
        self.assertEqual(os.listdir(os.path.join(fx.run_dir, "step6")), ["step6_report.md"])
        self.assertEqual(folder_hashes(fx.run_dir), before)

    def test_a_second_run_is_refused(self):
        fx = Fixture(self)
        target, _ = fx.analyze()
        first = {n: sha(os.path.join(target, n)) for n in os.listdir(target)}
        code = step6.main(["--run-id", fx.RUN_ID], results_root=fx.results, out=lambda *_: None)
        self.assertEqual(code, 2)
        self.assertEqual({n: sha(os.path.join(target, n)) for n in os.listdir(target)}, first)

    def test_exclusive_creation_refuses_an_existing_file(self):
        fx = Fixture(self)
        os.mkdir(os.path.join(fx.run_dir, "step6"))
        with open(os.path.join(fx.run_dir, "step6", "step6_report.md"), "w", encoding="utf-8") as handle:
            handle.write("old")
        with mock.patch.object(step6.os, "mkdir"):  # pretend the directory was just created by the script
            with self.assertRaises(FileExistsError):
                step6.write_new_outputs(fx.run_dir, {n: "new" for n in step6.OUTPUT_FILES})
        self.assertEqual(read_text(os.path.join(fx.run_dir, "step6", "step6_report.md")), "old")


class HashGuard(unittest.TestCase):
    def test_a_baseline_change_during_the_analysis_stops_everything(self):
        fx = Fixture(self)
        real = step6.repeat_comparison

        def tamper(data):
            with open(os.path.join(fx.run_dir, "report.md"), "a", encoding="utf-8") as handle:
                handle.write("tampered")
            return real(data)

        with mock.patch.object(step6, "repeat_comparison", tamper):
            with self.assertRaises(step6.StepError) as caught:
                fx.analyze()
        self.assertEqual(caught.exception.code, 4)
        self.assertFalse(os.path.exists(os.path.join(fx.run_dir, "step6")))

    def test_a_baseline_change_after_writing_is_detected(self):
        fx = Fixture(self)
        real = step6.write_new_outputs

        def write_then_tamper(run_dir, contents):
            target = real(run_dir, contents)
            with open(os.path.join(run_dir, "plan.json"), "a", encoding="utf-8") as handle:
                handle.write(" ")
            return target

        with mock.patch.object(step6, "write_new_outputs", write_then_tamper):
            with self.assertRaises(step6.StepError) as caught:
                fx.analyze()
        self.assertEqual(caught.exception.code, 4)
        self.assertIn("after the outputs were written", str(caught.exception))

    def test_all_nine_baseline_files_are_hashed(self):
        self.assertEqual(set(step6.HASHED_FILES), {"bullets.csv", "bullets_reviewed.csv", "human_review_decisions.csv",
                                                  "calls.jsonl", "evaluations.jsonl", "plan.json", "summary.json",
                                                  "metadata.json", "report.md"})


class Disclosures(unittest.TestCase):
    def test_report_states_the_required_limitations(self):
        fx = Fixture(self)
        target, metrics = fx.analyze()
        text = read_text(os.path.join(target, "step6_report.md"))
        for fragment in ("synthetic development cases", "provisional", "holdout cases were excluded", "evaluation-only",
                         "4096", "production uses 8192", "never summed", "Not ground truth", "No accuracy, precision or recall",
                         "Equal only after normalisation", "Exactly equal", "4/6 (66.7%)"):
            self.assertIn(fragment, text, fragment)
        d = metrics["disclosures"]
        self.assertEqual((d["effective_max_tokens"], d["production_max_tokens"]), (4096, 8192))


class ReportSentence(unittest.TestCase):
    NOT_CALCULATED = "Accuracy, precision and recall are not calculated because no defensible ground truth exists."
    D2_CLAUSE = "D2 flagged no bullets, so it cannot be compared with the human labels. "

    def report_text(self, fx):
        target, metrics = fx.analyze()
        return read_text(os.path.join(target, "step6_report.md")), metrics

    def test_d2_clause_appears_when_d2_flagged_nothing(self):
        text, metrics = self.report_text(Fixture(self))
        self.assertEqual(metrics["automated_vs_human"]["flag_totals"]["d2"]["count"], 0)
        self.assertIn("These are counts, not accuracy. " + self.D2_CLAUSE + self.NOT_CALCULATED, text)
        self.assertNotIn("cannot be compared. accuracy", text)  # the old run-on fragment

    def test_d2_clause_is_absent_when_d2_flagged_a_bullet(self):
        fx = Fixture(self)
        fx.rows[0]["d2_invented_numbers"] = "99%"
        fx.write()
        text, metrics = self.report_text(fx)
        self.assertEqual(metrics["automated_vs_human"]["flag_totals"]["d2"]["count"], 1)
        self.assertIn("These are counts, not accuracy. " + self.NOT_CALCULATED, text)
        self.assertNotIn("D2 flagged no bullets", text)


class Safety(unittest.TestCase):
    def test_only_expected_files_are_opened_at_run_time(self):
        fx = Fixture(self)
        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                opened.append(os.path.abspath(os.fsdecode(file)))
            return real_open(file, *args, **kwargs)

        with mock.patch.object(builtins, "open", spy):
            fx.analyze()
        allowed = {os.path.join(os.path.realpath(fx.run_dir), n) for n in step6.HASHED_FILES}
        allowed |= {os.path.join(os.path.realpath(fx.run_dir), "step6", n) for n in step6.OUTPUT_FILES}
        allowed.add(os.path.abspath(step6.SCRIPT_PATH))
        opened = {os.path.realpath(p) if os.path.exists(p) else p for p in opened}
        allowed = {os.path.realpath(p) if os.path.exists(p) else p for p in allowed}
        self.assertEqual(opened - allowed, set())
        self.assertFalse([p for p in opened if os.sep + "raw" + os.sep in p or os.path.basename(p) == ".env"])
        self.assertFalse([p for p in opened if os.path.basename(p) in PROTECTED])

    def test_script_has_no_forbidden_imports_or_references(self):
        source = read_text(os.path.join(ATS_DIR, "analyze_step6.py"))
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported, {"argparse", "csv", "hashlib", "io", "json", "os", "re", "statistics", "sys",
                                    "collections", "datetime"})
        doc_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body \
                    and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                doc_nodes.add(id(node.body[0].value))  # docstrings may name what the script avoids
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
                self.assertNotIn(".env", node.value)
                self.assertFalse(any(name in node.value for name in PROTECTED), node.value)
                self.assertNotIn("raw/", node.value)
                self.assertNotRegex(node.value, r"^(git|docker)( |$)")
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("system", "popen", "load_cases", "build_label_rows", "getenv", "environ", "CASES_DIR"):
            self.assertNotIn(banned, names)

    def test_only_the_expected_open_modes_are_used(self):
        tree = ast.parse(read_text(os.path.join(ATS_DIR, "analyze_step6.py")))
        modes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                modes.append(mode)
        self.assertEqual(sorted(set(modes)), ["r", "rb", "x"])  # no 'w' or 'a': nothing can be overwritten


class SharedRules(unittest.TestCase):
    def test_helpers_match_the_evaluation_runner_and_checker(self):
        for k, n in ((0, 0), (3, 18), (1, 3)):
            self.assertEqual(step6.rate(k, n), runner.rate(k, n))
        for value in ("=SUM(1)", "+1", "-2", "@x", "\tx", "plain", None, 5):
            self.assertEqual(step6.csv_safe(value), runner.csv_safe(value))
        for text in ("Built an API, with Flask!", "  wrote   TESTS  ", "", "A-B_c"):
            self.assertEqual(step6.normalise(text), checks.normalise_for_comparison(text))


@unittest.skipUnless(os.path.isdir(REAL_RUN), "the real baseline folder is not present in this checkout")
class RealBaselineReadOnly(unittest.TestCase):
    def test_integrity_checks_pass_on_the_real_baseline_without_writing(self):
        before = {n: sha(os.path.join(REAL_RUN, n)) for n in step6.HASHED_FILES}
        data = step6.load_and_check(REAL_RUN)
        self.assertEqual(len(data["keys"]), 102)
        self.assertEqual(data["plan"]["split"], "dev")
        self.assertEqual({k[0] for k in data["keys"]}, set(data["plan"]["case_ids"]))
        self.assertEqual({n: sha(os.path.join(REAL_RUN, n)) for n in step6.HASHED_FILES}, before)


if __name__ == "__main__":
    unittest.main()
