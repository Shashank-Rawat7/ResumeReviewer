"""Offline tests for the production E2 normalisation and course-eligibility logic (whole-pipeline evaluation, Step 2).

Run from the project root:  python -B -m unittest tests.test_pipeline_stages -v

WHAT THESE TESTS PROVE
  They prove what the production code does with a raw model-output STRING or a prioritised dict that the test supplies.
  They never prove what a live model emits, how often it emits an invalid status, or whether course recommendations
  are relevant. Every raw output below is a hand-written fixture.

HOW THE PRODUCTION CODE IS REACHED
  eval/pipeline/stages.py reads src/parser.py and frontend/display.py as TEXT and executes only the pure definitions
  the two functions need. No project module is imported, so load_dotenv() never runs, no .env is read, and no
  streamlit, requests, langchain, database or YouTube code is loaded. No protected file, no network, no model.
"""
import ast
import importlib.util
import json
import logging
import os
import sys
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGES_PATH = os.path.join(ROOT, "eval", "pipeline", "stages.py")

_spec = importlib.util.spec_from_file_location("pipeline_stages", STAGES_PATH)
stages = sys.modules.get("pipeline_stages") or importlib.util.module_from_spec(_spec)
if "pipeline_stages" not in sys.modules:
    sys.modules["pipeline_stages"] = stages
    _spec.loader.exec_module(stages)

JD = (
    "Requirements:\n"
    "Hands-on experience with Docker is required.\n"
    "Python for data pipelines is required.\n"
    "Experience building REST APIs is required.\n"
    "Experience with Kubernetes is a plus.\n"
    "Exposure to Terraform for infrastructure automation.\n"
    "Strong SQL skills are needed.\n"
    "Clear written communication with business stakeholders.\n"
)
DOCKER_Q = "Hands-on experience with Docker is required"
PYTHON_Q = "Python for data pipelines is required"
REST_Q = "Experience building REST APIs is required"
K8S_Q = "Experience with Kubernetes is a plus"
TERRAFORM_Q = "Exposure to Terraform for infrastructure automation"
COMM_Q = "Clear written communication with business stakeholders"
NOT_IN_JD = "Mandatory Rust experience for kernel work"
HELPER = json.dumps({
    "technical_skills": ["Docker", "Python", "Kubernetes", "Terraform", "SQL"],
    "soft_skills": ["Communication"],
    "academic_qualifications": [],
})
RESUME = ("Built Python ETL jobs that load daily reports.\n"
          "Wrote basic SQL queries against a PostgreSQL database.\n"
          "Ran Docker containers on a laptop for local testing.\n")
EMPTY_LEGACY = {"technical_skills": [], "soft_skills": [], "academic_qualifications": []}


def entry(skill, jd_evidence, category="technical_skills", status="missing", reason="Reason.", **extra):
    """A raw model entry, as the comparison prompt asks the model to write it."""
    data = {"skill": skill, "category": category, "status": status, "reason": reason, "jd_evidence": jd_evidence}
    data.update(extra)
    return data


def raw(crucial=(), good=(), optional=()):
    return json.dumps({"crucial": list(crucial), "good_to_have": list(good), "optional": list(optional)})


def names(result, tier):
    return [item["skill"] for item in result[tier]]


class QuietLogs(unittest.TestCase):
    """The production normaliser logs its diagnostics; keep the test output clean."""

    def setUp(self):
        logger = logging.getLogger(stages.PARSER_MODULE_NAME)
        handler = logging.NullHandler()
        old_propagate = logger.propagate
        logger.addHandler(handler)
        logger.propagate = False
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(setattr, logger, "propagate", old_propagate)
        self.parser = stages.load_parser()

    def build(self, raw_text, jd=JD, helper=HELPER, resume=None):
        return self.parser.build_prioritized_missing_skills(raw_text, jd, helper, resume)


class ValidAndInvalidEntries(QuietLogs):
    def test_valid_missing_and_partial_entries_are_placed_in_their_tiers(self):
        result, legacy = self.build(raw(
            crucial=[entry("Docker", DOCKER_Q), entry("Python", PYTHON_Q, status="partial")],
            good=[entry("Kubernetes", K8S_Q)], optional=[entry("Terraform", TERRAFORM_Q)]))
        self.assertTrue(result["normalization_ok"])
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(names(result, "crucial"), ["Docker", "Python"])
        self.assertEqual(names(result, "good_to_have"), ["Kubernetes"])
        self.assertEqual(names(result, "optional"), ["Terraform"])
        for tier in ("crucial", "good_to_have", "optional"):
            for item in result[tier]:
                self.assertIs(item["evidence_found"], True)
                self.assertIs(item["in_extracted_list"], True)
        self.assertIs(result["crucial"][0]["evidence_related"], True)

    def test_legacy_list_keeps_only_missing_entries_with_found_evidence_in_the_extracted_list(self):
        _result, legacy = self.build(raw(
            crucial=[entry("Docker", DOCKER_Q), entry("Python", PYTHON_Q, status="partial")],
            good=[entry("Kubernetes", K8S_Q)], optional=[entry("Terraform", TERRAFORM_Q)]))
        self.assertEqual(legacy["technical_skills"], ["Docker", "Kubernetes", "Terraform"])  # partial Python excluded
        self.assertEqual(legacy["soft_skills"], [])

    def test_status_and_category_are_lower_cased(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q, category="TECHNICAL_SKILLS", status="MISSING")]))
        self.assertEqual((result["crucial"][0]["status"], result["crucial"][0]["category"]),
                         ("missing", "technical_skills"))

    def test_raw_no_gap_and_other_invalid_statuses_are_dropped_not_normalised(self):
        for bad in ("no_gap", "demonstrated", "covered"):
            with self.subTest(status=bad):
                result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q),
                                                    entry("Python", PYTHON_Q, status=bad)]))
                self.assertTrue(result["normalization_ok"])  # the whole result is NOT discarded here
                self.assertEqual(names(result, "crucial"), ["Docker"])  # the invalid entry is simply absent
                self.assertEqual(result["diagnostics"], [f"Dropped malformed entry in 'crucial': invalid status '{bad}'."])

    def test_dropped_entry_diagnostic_does_not_name_the_skill(self):
        result, _ = self.build(raw(crucial=[entry("Kubernetes", K8S_Q, status="no_gap")]))
        self.assertEqual(len(result["diagnostics"]), 1)
        self.assertNotIn("Kubernetes", result["diagnostics"][0])

    def test_empty_or_non_string_status_is_reported_as_missing_or_empty(self):
        for bad in ("", None, 3):
            with self.subTest(status=bad):
                result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q, status=bad)]))
                self.assertEqual(result["crucial"], [])
                self.assertEqual(result["diagnostics"], ["Dropped malformed entry in 'crucial': missing or empty 'status'."])

    def test_each_structural_problem_drops_only_that_entry(self):
        cases = [
            ("not an object", "Docker", "entry is not an object"),
            ("missing jd_evidence", {k: v for k, v in entry("Docker", DOCKER_Q).items() if k != "jd_evidence"},
             "missing or empty 'jd_evidence'"),
            ("invalid category", entry("Docker", DOCKER_Q, category="hard_skills"), "invalid category 'hard_skills'"),
            ("unusable name", entry("!!!", DOCKER_Q), "skill name has no usable characters"),
            ("name too long", entry("a" * 101, DOCKER_Q), "skill name too long"),
            ("reason too long", entry("Docker", DOCKER_Q, reason="r" * 601), "reason too long for 'Docker'"),
            ("evidence too long", entry("Docker", "e" * 401), "jd_evidence too long for 'Docker'"),
        ]
        for label, bad, problem in cases:
            with self.subTest(label=label):
                result, _ = self.build(raw(crucial=[bad, entry("Python", PYTHON_Q)]))
                self.assertEqual(names(result, "crucial"), ["Python"])
                self.assertIn(f"Dropped malformed entry in 'crucial': {problem}.", result["diagnostics"])


class FailureFallback(QuietLogs):
    def assertFallback(self, outcome, message):
        result, legacy = outcome
        self.assertFalse(result["normalization_ok"])
        self.assertEqual([result[t] for t in ("crucial", "good_to_have", "optional")], [[], [], []])
        self.assertEqual(result["diagnostics"], [message])
        self.assertEqual(legacy, EMPTY_LEGACY)

    def test_invalid_empty_or_missing_json_falls_back(self):
        for bad in ("not json", "", "   ", None, '{"crucial": [', "```json\nnot json\n```"):
            with self.subTest(raw=bad):
                self.assertFallback(self.build(bad), "Comparison output is not valid JSON.")

    def test_json_without_any_tier_falls_back(self):
        for bad in ("[]", "{}", '{"other": []}', '"text"', "42"):
            with self.subTest(raw=bad):
                self.assertFallback(self.build(bad), "Comparison output has none of the expected tiers.")

    def test_unexpected_exception_is_contained(self):
        self.assertFallback(self.build(123), "Normalisation raised an unexpected error; see the server log.")

    def test_fenced_json_is_accepted(self):
        result, _ = self.build("```json\n" + raw(crucial=[entry("Docker", DOCKER_Q)]) + "\n```")
        self.assertTrue(result["normalization_ok"])
        self.assertEqual(names(result, "crucial"), ["Docker"])

    def test_missing_or_non_list_tier_is_noted_but_does_not_fail(self):
        result, _ = self.build(json.dumps({"crucial": [entry("Docker", DOCKER_Q)], "good_to_have": "oops"}))
        self.assertTrue(result["normalization_ok"])
        self.assertEqual(names(result, "crucial"), ["Docker"])
        self.assertEqual(result["diagnostics"], ["Tier 'good_to_have' is not a list; ignored.",
                                                 "Tier 'optional' missing from output; treated as empty."])

    def test_all_tiers_empty_is_a_valid_result(self):
        result, legacy = self.build(raw())
        self.assertTrue(result["normalization_ok"])
        self.assertEqual(legacy, EMPTY_LEGACY)

    def test_missing_jd_text_does_not_raise_and_leaves_evidence_unfound(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q)]), jd=None)
        self.assertTrue(result["normalization_ok"])
        self.assertIs(result["crucial"][0]["evidence_found"], False)


class DuplicateResolution(QuietLogs):
    def test_found_evidence_beats_a_higher_tier(self):
        result, _ = self.build(raw(crucial=[entry("Docker", NOT_IN_JD)], good=[entry("Docker", DOCKER_Q)]))
        self.assertEqual(names(result, "crucial"), [])
        self.assertEqual(names(result, "good_to_have"), ["Docker"])
        self.assertIs(result["good_to_have"][0]["evidence_found"], True)
        self.assertIn("Duplicate 'Docker' in 'crucial' removed; kept the occurrence in 'good_to_have'.",
                      result["diagnostics"])

    def test_highest_tier_wins_when_both_quotes_are_found(self):
        for tiers, kept in ((("good", "optional"), "good_to_have"), (("crucial", "optional"), "crucial"),
                            (("crucial", "good"), "crucial")):
            with self.subTest(tiers=tiers):
                kwargs = {tier: [entry("Docker", DOCKER_Q)] for tier in tiers}
                result, _ = self.build(raw(**kwargs))
                self.assertEqual([names(result, t) for t in ("crucial", "good_to_have", "optional")].count(["Docker"]), 1)
                self.assertEqual(names(result, kept), ["Docker"])

    def test_first_seen_wins_within_one_tier(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q, reason="first"),
                                            entry("docker", DOCKER_Q, reason="second")]))
        self.assertEqual([(i["skill"], i["reason"]) for i in result["crucial"]], [("Docker", "first")])
        self.assertIn("Duplicate 'docker' in 'crucial' removed; kept the occurrence in 'crucial'.",
                      result["diagnostics"])

    def test_skill_key_ignores_case_punctuation_and_a_trailing_plural_s(self):
        result, _ = self.build(raw(crucial=[entry("REST APIs", REST_Q)], good=[entry("rest api", REST_Q)]))
        self.assertEqual(names(result, "crucial"), ["REST APIs"])
        self.assertEqual(names(result, "good_to_have"), [])

    def test_duplicates_collapse_across_categories(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q)],
                                   good=[entry("Docker", DOCKER_Q, category="soft_skills")]))
        self.assertEqual(sum(len(result[t]) for t in ("crucial", "good_to_have", "optional")), 1)
        self.assertEqual(result["crucial"][0]["category"], "technical_skills")

    def test_a_dropped_invalid_entry_does_not_count_as_a_duplicate(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q, status="no_gap")],
                                   good=[entry("Docker", DOCKER_Q)]))
        self.assertEqual(names(result, "good_to_have"), ["Docker"])
        self.assertFalse(any(d.startswith("Duplicate") for d in result["diagnostics"]))

    def test_distinct_names_for_one_idea_are_not_merged(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q), entry("Containerization", DOCKER_Q)]))
        self.assertEqual(names(result, "crucial"), ["Docker", "Containerization"])  # no alias or semantic matching

    def test_output_order_inside_a_tier_follows_the_input(self):
        result, _ = self.build(raw(crucial=[entry("SQL", "Strong SQL skills are needed"), entry("Docker", DOCKER_Q)]))
        self.assertEqual(names(result, "crucial"), ["SQL", "Docker"])


class EvidenceAndAdvisoryFlags(QuietLogs):
    def test_quote_not_in_the_jd_is_kept_but_marked_and_left_out_of_legacy(self):
        result, legacy = self.build(raw(crucial=[entry("Rust", NOT_IN_JD)]))
        item = result["crucial"][0]
        self.assertIs(item["evidence_found"], False)
        self.assertIsNone(item["evidence_related"])
        self.assertEqual(legacy, EMPTY_LEGACY)
        self.assertIn("'Rust': jd_evidence was not found in the job description (evidence_found=False); "
                      "excluded from legacy missing_skills.", result["diagnostics"])

    def test_skill_outside_the_extracted_list_is_kept_but_left_out_of_legacy(self):
        result, legacy = self.build(raw(crucial=[entry("Rust", DOCKER_Q)]))
        self.assertIs(result["crucial"][0]["in_extracted_list"], False)
        self.assertEqual(legacy, EMPTY_LEGACY)
        self.assertIn("'Rust' has a quote found in the JD but is absent from the extracted skill list; kept in the "
                      "prioritised output, excluded from legacy missing_skills.", result["diagnostics"])

    def test_unusable_extracted_list_leaves_membership_unknown_and_legacy_empty(self):
        result, legacy = self.build(raw(crucial=[entry("Docker", DOCKER_Q)]), helper="Error generating job skills.")
        self.assertIsNone(result["crucial"][0]["in_extracted_list"])
        self.assertEqual(legacy, EMPTY_LEGACY)
        self.assertIn("Extracted JD skill list is unavailable; helper-list membership is unknown and the legacy "
                      "missing_skills is left empty.", result["diagnostics"])

    def test_category_that_differs_from_the_extracted_list_is_left_out_of_legacy(self):
        result, legacy = self.build(raw(crucial=[entry("Docker", DOCKER_Q, category="soft_skills")]))
        self.assertEqual(legacy, EMPTY_LEGACY)
        self.assertEqual(names(result, "crucial"), ["Docker"])
        self.assertIn("'Docker': category 'soft_skills' differs from the extracted list ('technical_skills'); "
                      "excluded from legacy missing_skills.", result["diagnostics"])

    def test_unrelated_quote_is_advisory_only(self):
        result, legacy = self.build(raw(optional=[entry("Terraform", K8S_Q)]))
        item = result["optional"][0]
        self.assertIs(item["evidence_found"], True)
        self.assertIs(item["evidence_related"], False)
        self.assertEqual(legacy["technical_skills"], ["Terraform"])  # relatedness does not change eligibility

    def test_resume_side_flags_are_advisory_and_never_change_status(self):
        result, _ = self.build(raw(
            crucial=[entry("Docker", DOCKER_Q),
                     entry("Python", PYTHON_Q, status="partial",
                           resume_evidence="Built Python ETL jobs that load daily reports"),
                     entry("SQL", "Strong SQL skills are needed", status="partial",
                           resume_evidence="Designed and tuned warehouse schemas daily")],
            good=[entry("Kubernetes", K8S_Q)]), resume=RESUME)
        by_name = {i["skill"]: i for tier in ("crucial", "good_to_have") for i in result[tier]}
        self.assertIs(by_name["Docker"]["resume_mention_found"], True)       # named in the resume
        self.assertIs(by_name["Kubernetes"]["resume_mention_found"], False)  # not named
        self.assertNotIn("resume_mention_found", by_name["Python"])          # only 'missing' entries are checked
        self.assertIs(by_name["Python"]["resume_evidence_found"], True)
        self.assertIs(by_name["SQL"]["resume_evidence_found"], False)
        self.assertEqual({k: v["status"] for k, v in by_name.items()},
                         {"Docker": "missing", "Python": "partial", "SQL": "partial", "Kubernetes": "missing"})

    def test_without_resume_text_no_resume_side_flags_are_added(self):
        result, _ = self.build(raw(crucial=[entry("Docker", DOCKER_Q)]), resume=None)
        self.assertNotIn("resume_mention_found", result["crucial"][0])


# ---------------------------------------------------------------------------------------------------------------
# Course eligibility
# ---------------------------------------------------------------------------------------------------------------

def full(skill, category="technical_skills", status="missing", found=True, related=True, listed=True):
    """A prioritised entry exactly as the parser emits it (the shape the frontend validator requires)."""
    return {"skill": skill, "category": category, "status": status, "reason": "Reason.", "jd_evidence": "Quote.",
            "evidence_found": found, "evidence_related": related, "in_extracted_list": listed}


def prioritized(crucial=(), good=(), optional=(), ok=True):
    return {"crucial": list(crucial), "good_to_have": list(good), "optional": list(optional),
            "normalization_ok": ok, "diagnostics": []}


DASH = "—"


class CourseEligibility(unittest.TestCase):
    def setUp(self):
        self.display = stages.load_display()

    def options(self, data):
        return self.display.get_recommendation_options(data)

    def test_technical_entries_are_offered_with_tier_and_state_labels(self):
        options, skipped = self.options(prioritized(
            crucial=[full("Docker"), full("Python", status="partial")], good=[full("Kubernetes")],
            optional=[full("Terraform")]))
        self.assertEqual(skipped, 0)
        self.assertEqual(options, {
            "Docker": f"Docker {DASH} Crucial, Not Demonstrated",
            "Python": f"Python {DASH} Crucial, Partially Demonstrated",
            "Kubernetes": f"Kubernetes {DASH} Good to Have, Not Demonstrated",
            "Terraform": f"Terraform {DASH} Optional, Not Demonstrated",
        })
        self.assertEqual(list(options), ["Docker", "Python", "Kubernetes", "Terraform"])  # priority order

    def test_both_missing_and_partial_qualify(self):
        options, _ = self.options(prioritized(crucial=[full("A1x", status="missing"), full("B1x", status="partial")]))
        self.assertEqual(set(options), {"A1x", "B1x"})

    def test_unverified_technical_entries_are_skipped_and_counted_whatever_their_status(self):
        options, skipped = self.options(prioritized(
            crucial=[full("Docker", found=False), full("Python", status="partial", found=False, related=None)],
            good=[full("Kubernetes")]))
        self.assertEqual(list(options), ["Kubernetes"])
        self.assertEqual(skipped, 2)

    def test_non_technical_entries_are_neither_offered_nor_counted(self):
        options, skipped = self.options(prioritized(
            crucial=[full("Communication", category="soft_skills"),
                     full("Leadership", category="soft_skills", found=False, related=None)],
            good=[full("Master's degree", category="academic_qualifications"),
                  full("PhD", category="academic_qualifications", status="partial", found=False, related=None)]))
        self.assertEqual((options, skipped), ({}, 0))

    def test_options_plus_skipped_equal_the_technical_count_of_the_priority_summary(self):
        data = prioritized(
            crucial=[full("Docker"), full("Python", found=False), full("Communication", category="soft_skills")],
            good=[full("Kubernetes", status="partial")], optional=[full("Terraform", found=False, related=None)])
        options, skipped = self.options(data)
        self.assertEqual(len(options) + skipped, self.display.summarize_priority_gaps(data)["technical"])
        self.assertEqual((len(options), skipped), (2, 2))

    def test_unconfirmed_quote_relevance_is_flagged_in_the_label(self):
        options, _ = self.options(prioritized(crucial=[full("Docker", related=False)]))
        self.assertEqual(options["Docker"], f"Docker {DASH} Crucial, Not Demonstrated · quote relevance unconfirmed")

    def test_unusable_comparison_output_gives_no_options(self):
        good = prioritized(crucial=[full("Docker")])
        broken = {
            "normalization_ok false": prioritized(crucial=[full("Docker")], ok=False),
            "not an object": None,
            "a string": "text",
            "no normalization_ok": {k: v for k, v in good.items() if k != "normalization_ok"},
            "tier not a list": dict(good, optional="x"),
            "missing tier": {k: v for k, v in good.items() if k != "optional"},
            "diagnostics not a list": dict(good, diagnostics="x"),
            "invalid status no_gap": prioritized(crucial=[full("Docker"), full("Python", status="no_gap")]),
            "evidence_found not boolean": prioritized(crucial=[full("Docker", found="yes")]),
            "invalid category": prioritized(crucial=[full("Docker", category="hard_skills")]),
        }
        for label, data in broken.items():
            with self.subTest(label=label):
                self.assertEqual(self.options(data), ({}, 0))

    def test_one_invalid_entry_makes_the_whole_frontend_result_unusable(self):
        self.assertEqual(self.options(prioritized(crucial=[full("Docker")], good=[full("Rust", status="no_gap")])),
                         ({}, 0))

    def test_frontend_keeps_the_first_duplicate_in_tier_order_without_preferring_found_evidence(self):
        # A hand-built shape the parser never emits (the parser already keeps one entry per skill key). It documents
        # that the frontend rule differs from the parser rule: tier order only, evidence is not considered.
        options, skipped = self.options(prioritized(crucial=[full("Docker", found=False)], good=[full("Docker")]))
        self.assertEqual((options, skipped), ({}, 1))

    def test_duplicate_keys_are_counted_once(self):
        options, skipped = self.options(prioritized(crucial=[full("Docker")], good=[full("docker", status="partial")],
                                                    optional=[full("DOCKERS", found=False)]))
        self.assertEqual((list(options), skipped), (["Docker"], 0))


class DerivedNoGapState(unittest.TestCase):
    """The frontend 'no_gap' is a derived display state, not a status the model can emit."""

    def setUp(self):
        self.display = stages.load_display()
        self.job_skills = {"technical_skills": ["Docker", "Python", "SQL"], "soft_skills": ["Communication"],
                           "academic_qualifications": []}

    def test_unflagged_extracted_skill_is_no_gap_with_no_tier_status_or_entry(self):
        view = self.display.build_skill_gap_records(prioritized(crucial=[full("Docker")]), self.job_skills)
        by_key = {record["skill"]: record for record in view["records"]}
        self.assertEqual(by_key["Docker"]["state"], self.display.STATE_NOT_DEMONSTRATED)
        for name in ("Python", "SQL", "Communication"):
            record = by_key[name]
            self.assertEqual(record["state"], self.display.STATE_NO_GAP)
            self.assertEqual((record["tier"], record["status"], record["entry"]), (None, None, None))
            self.assertIs(record["in_extracted_list"], True)
        self.assertEqual(view["extracted_total"], 4)

    def test_every_extracted_skill_is_in_exactly_one_state(self):
        data = prioritized(crucial=[full("Docker"), full("Python", status="partial")],
                           good=[full("SQL", found=False, related=None)])
        counts = self.display.summarize_skill_gaps(data, self.job_skills)
        self.assertEqual(counts["total"], 4)
        self.assertEqual(counts["no_gap"] + counts["partially_demonstrated"] + counts["not_demonstrated"]
                         + counts["evidence_not_verified"], counts["total"])
        self.assertEqual((counts["no_gap"], counts["partially_demonstrated"], counts["not_demonstrated"],
                          counts["evidence_not_verified"]), (1, 1, 1, 1))

    def test_a_raw_no_gap_entry_makes_the_frontend_view_unusable(self):
        data = prioritized(crucial=[full("Docker", status="no_gap")])
        self.assertIsNone(self.display.build_skill_gap_records(data, self.job_skills))
        self.assertIsNone(self.display.summarize_skill_gaps(data, self.job_skills))


class ParserToEligibility(QuietLogs):
    """The parser output feeds the eligibility rule; the raw string is a fixture, not model output."""

    def test_end_to_end_from_a_raw_string(self):
        display = stages.load_display()
        result, _ = self.build(raw(
            crucial=[entry("Docker", DOCKER_Q),
                     entry("Python", PYTHON_Q, status="no_gap"),             # invalid -> dropped by the parser
                     entry("Communication", COMM_Q, category="soft_skills"),
                     entry("Kubernetes", NOT_IN_JD)],                         # quote not in the JD -> unverified
            optional=[entry("Terraform", TERRAFORM_Q, status="partial")]))
        options, skipped = display.get_recommendation_options(result)
        self.assertEqual(list(options), ["Docker", "Terraform"])
        self.assertEqual(skipped, 1)

    def test_a_fallback_result_gives_no_options(self):
        display = stages.load_display()
        result, _ = self.build("not json")
        self.assertEqual(display.get_recommendation_options(result), ({}, 0))


class Extractor(unittest.TestCase):
    """The isolation mechanism itself, on synthetic source text."""

    SOURCE = textwrap.dedent('''
        import os
        from dotenv import load_dotenv
        load_dotenv()
        raise RuntimeError("top-level code must never run")
        K = 3
        PATTERN = re.compile("a+")
        def helper(x):
            return x + K
        def entry(x):
            return helper(x) * 2
        def unrelated():
            return os.getcwd()
    ''')

    def test_only_the_needed_definitions_run_and_other_top_level_code_is_never_executed(self):
        module = stages.extract_pure(self.SOURCE, ("entry",), "synthetic")
        self.assertEqual(module.entry(1), 8)
        self.assertEqual(module.pure_info["selected"], ["K", "entry", "helper"])
        self.assertFalse(hasattr(module, "unrelated"))
        self.assertFalse(hasattr(module, "load_dotenv"))

    def test_allowed_module_call_in_a_constant_is_accepted(self):
        module = stages.extract_pure(self.SOURCE, ("PATTERN",), "synthetic")
        self.assertTrue(module.PATTERN.match("aaa"))

    def test_a_name_that_is_not_defined_builtin_or_allowed_is_refused(self):
        with self.assertRaisesRegex(stages.PureExtractionError, "unrelated uses os"):
            stages.extract_pure(self.SOURCE, ("unrelated",), "synthetic")
        with self.assertRaisesRegex(stages.PureExtractionError, "uses st"):
            stages.extract_pure("def f():\n    return st.write('x')\n", ("f",), "synthetic")

    def test_an_assignment_that_calls_something_unsafe_is_refused(self):
        for source in ('X = open("file")', 'X = os.getenv("KEY")', "X = len(y)\ny = [1]"):
            with self.subTest(source=source):
                with self.assertRaises(stages.PureExtractionError):
                    stages.extract_pure(source, ("X",), "synthetic")

    def test_unknown_entry_name_is_refused(self):
        with self.assertRaisesRegex(stages.PureExtractionError, "no top-level definition named nothing"):
            stages.extract_pure(self.SOURCE, ("nothing",), "synthetic")

    def test_source_digest_ignores_line_endings(self):
        lf = stages.extract_pure("def f():\n    return 1\n", ("f",), "synthetic")
        crlf = stages.extract_pure("def f():\r\n    return 1\r\n", ("f",), "synthetic")
        self.assertEqual(lf.pure_info["source_sha256"], crlf.pure_info["source_sha256"])
        self.assertRegex(lf.pure_info["source_sha256"], r"^[0-9a-f]{64}$")


class RealSourceSelection(unittest.TestCase):
    def test_parser_selection_excludes_environment_pdf_model_and_class_code(self):
        selected = set(stages.load_parser().pure_info["selected"])
        self.assertIn("build_prioritized_missing_skills", selected)
        for forbidden in ("load_prompts", "ResumeParserConfig", "ResumeParser", "ResumeParserError",
                          "verify_project_bullets", "job_description"):
            self.assertNotIn(forbidden, selected)

    def test_display_selection_contains_no_rendering_code(self):
        selected = stages.load_display().pure_info["selected"]
        self.assertIn("get_recommendation_options", selected)
        self.assertEqual([name for name in selected if name.startswith("display_")], [])

    def test_provenance_is_recorded(self):
        for module, source in ((stages.load_parser(), "src/parser.py"), (stages.load_display(), "frontend/display.py")):
            self.assertEqual(module.pure_info["source"], source)
            self.assertRegex(module.pure_info["source_sha256"], r"^[0-9a-f]{64}$")

    def test_stages_module_imports_only_the_standard_library_subset(self):
        with open(STAGES_PATH, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.add((node.module or "").split(".")[0])
        self.assertEqual(found, {"ast", "builtins", "hashlib", "html", "json", "logging", "os", "re", "types",
                                 "unicodedata"})

    def test_only_two_source_files_are_read(self):
        self.assertEqual(stages.PARSER_RELATIVE, ("src", "parser.py"))
        self.assertEqual(stages.DISPLAY_RELATIVE, ("frontend", "display.py"))


if __name__ == "__main__":
    unittest.main()
