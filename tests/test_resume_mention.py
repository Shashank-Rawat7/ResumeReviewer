"""Tests for the advisory resume-mention check in src/parser.py.

Run from the project root:  python -m unittest tests.test_resume_mention -v
Synthetic data only. No network, API key, resume file or live model is used.
"""
import copy
import json
import unittest

from tests.test_bullet_verification import load_parser

parser = load_parser()
found = parser.resume_mention_found

RESUME = (
    "Built a resume parser using Python and FastAPI. Deployed with Docker on PostgreSQL-backed storage. "
    "Technologies: Python, FastAPI, Docker, PostgreSQL, scikit-learn, Node.js, C++"
)


def entry(skill, status="missing", **extra):
    base = {"skill": skill, "category": "technical_skills", "status": status,
            "reason": "synthetic reason", "jd_evidence": "synthetic quote about the skill"}
    base.update(extra)
    return base


def raw_output(*skills, status="missing"):
    return json.dumps({"crucial": [{"skill": s, "category": "technical_skills", "status": status,
                                    "reason": "r", "jd_evidence": "requires " + s} for s in skills],
                       "good_to_have": [], "optional": []})


class MentionMatching(unittest.TestCase):
    def test_four_named_skills_present(self):
        for skill in ("Python", "FastAPI", "Docker", "PostgreSQL"):
            self.assertIs(found(skill, RESUME), True, skill)

    def test_absent_skill(self):
        for skill in ("Kubernetes", "Terraform", "Quillstream"):
            self.assertIs(found(skill, RESUME), False, skill)

    def test_case_and_whitespace(self):
        self.assertIs(found("PYTHON", "we use   python\n\tdaily"), True)
        self.assertIs(found("  fastapi ", "Stack: FASTAPI."), True)
        self.assertIs(found("Machine Learning", "applied machine\n   learning methods"), True)
        self.assertIs(found("machine learning", "applied MachineLearning methods"), True)

    def test_short_or_ambiguous_names_are_false(self):
        for skill in ("Go", "R", "C", "AI", "ML"):
            self.assertIs(found(skill, "Skills: Go, R, C, AI, ML. Go to market. R&D team."), False, skill)

    def test_symbol_names_are_allowed_and_exact(self):
        self.assertIs(found("C++", "Languages: C++, Python"), True)
        self.assertIs(found("C++", "Languages: C, Python"), False)
        self.assertIs(found("C#", "Languages: C#."), True)
        self.assertIs(found("C#", "Languages: C, Java"), False)

    def test_java_versus_javascript(self):
        self.assertIs(found("Java", "Skills: JavaScript, TypeScript"), False)
        self.assertIs(found("JavaScript", "Skills: Java, Spring"), False)
        self.assertIs(found("Java", "Skills: Java, Spring"), True)
        self.assertIs(found("Java", "Skills: JavaScript and Java"), True)
        self.assertIs(found("Java", "Skills: Java/Python"), True)

    def test_edges_with_letters_digits_symbols(self):
        self.assertIs(found("Python", "Python3 scripting"), False)
        self.assertIs(found("Docker", "Dockerfile authoring"), False)
        self.assertIs(found("SQL", "NoSQL and MySQL"), False)
        self.assertIs(found("SQL", "SQL, Excel"), True)

    def test_multiword_and_punctuation_variants(self):
        self.assertIs(found("scikit-learn", RESUME), True)
        self.assertIs(found("scikit learn", RESUME), True)
        self.assertIs(found("Node.js", "Backend: NodeJS"), True)
        self.assertIs(found("Node.js", "Backend: node.js"), True)
        self.assertIs(found("CI/CD", "Set up CI CD pipelines"), True)
        self.assertIs(found("REST API", "Designed REST-API services"), True)
        self.assertIs(found("Machine Learning", "machine, learning"), False)
        self.assertIs(found("Machine Learning", "machine vision learning"), False)

    def test_documented_non_matches(self):
        self.assertIs(found("PostgreSQL", "Used Postgres"), False)             # no synonyms
        self.assertIs(found("NLP", "natural language processing"), False)      # no acronym expansion
        self.assertIs(found("REST APIs", "Built a REST API"), False)           # no plural handling
        self.assertIs(found("Python", "PythonFastAPI"), False)                 # glued text is not matched

    def test_curly_quotes_and_unicode_fold(self):
        self.assertIs(found("Python", "Ｐｙｔｈｏｎ developer"), True)         # NFKC full-width letters

    def test_malformed_inputs_give_false_and_never_raise(self):
        for skill, text in [(None, RESUME), (5, RESUME), ("", RESUME), ("   ", RESUME), ("---", RESUME),
                            ("Python", None), ("Python", 7), ("Python", ""), (["Python"], RESUME),
                            ({"a": 1}, {"b": 2})]:
            self.assertIs(found(skill, text), False, (skill, text))

    def test_regex_metacharacters_in_skill_name(self):
        for skill in ("C++ (STL)", "a.*b", "[x", "(?P<n>", "\\"):
            self.assertIsInstance(found(skill, RESUME), bool)
        self.assertIs(found("a.*b", "axxb"), False)

    def test_very_long_input_is_fast_enough(self):
        text = ("python " * 50000) + "kubernetes"
        self.assertIs(found("Kubernetes", text), True)


class Annotation(unittest.TestCase):
    def build(self, *skills, status="missing", resume=RESUME):
        jd = " ".join("requires " + s for s in skills)
        return parser.build_prioritized_missing_skills(
            raw_output(*skills, status=status), jd, json.dumps({"technical_skills": list(skills)}), resume)

    def test_field_added_only_to_missing_entries(self):
        result, _ = self.build("Python", "Kubernetes")
        flags = {e["skill"]: e["resume_mention_found"] for e in result["crucial"]}
        self.assertEqual(flags, {"Python": True, "Kubernetes": False})
        for e in result["crucial"]:
            self.assertIsInstance(e["resume_mention_found"], bool)

    def test_partial_entries_are_not_annotated(self):
        result, _ = self.build("Python", status="partial")
        self.assertNotIn("resume_mention_found", result["crucial"][0])

    def test_nothing_else_changes(self):
        with_resume, legacy_with = self.build("Python", "Kubernetes", "Docker")
        without, legacy_without = parser.build_prioritized_missing_skills(
            raw_output("Python", "Kubernetes", "Docker"), "requires Python requires Kubernetes requires Docker",
            json.dumps({"technical_skills": ["Python", "Kubernetes", "Docker"]}))
        stripped = copy.deepcopy(with_resume)
        for tier in parser.PRIORITY_TIERS:
            for e in stripped[tier]:
                e.pop("resume_mention_found")
                e.pop("resume_evidence_found")          # added by the resume-evidence check (always None here)
                e.pop("resume_evidence_names_skill")
        self.assertEqual(stripped, without)           # status, reason, tier, quote, flags, diagnostics identical
        self.assertEqual(legacy_with, legacy_without)  # legacy dict (feeds the ATS score) identical
        for e in without["crucial"]:
            self.assertNotIn("resume_mention_found", e)

    def test_status_reason_tier_and_quote_preserved_for_a_mentioned_skill(self):
        result, _ = self.build("Docker")
        e = result["crucial"][0]
        self.assertTrue(e["resume_mention_found"])
        self.assertEqual((e["status"], e["reason"], e["jd_evidence"]), ("missing", "r", "requires Docker"))
        self.assertEqual(result["good_to_have"], [])

    def test_resume_not_a_string_means_no_annotation(self):
        for resume in (None, 5, ["Python"]):
            result, _ = self.build("Python", resume=resume)
            self.assertNotIn("resume_mention_found", result["crucial"][0])

    def test_empty_resume_string_gives_false(self):
        result, _ = self.build("Python", resume="")
        self.assertIs(result["crucial"][0]["resume_mention_found"], False)

    def test_resume_text_is_not_modified(self):
        text = "Python   and\nDocker"
        before = str(text)
        self.build("Python", resume=text)
        self.assertEqual(text, before)

    def test_old_three_argument_call_still_works(self):
        result, _ = parser.build_prioritized_missing_skills(raw_output("Python"), "requires Python", "{}")
        self.assertTrue(result["normalization_ok"])

    def test_unusable_model_output_is_unchanged(self):
        result, legacy = parser.build_prioritized_missing_skills("not json", "jd", "{}", RESUME)
        self.assertFalse(result["normalization_ok"])
        self.assertEqual(result["crucial"], [])
        self.assertEqual(legacy, {c: [] for c in parser.SKILL_CATEGORIES})

    def test_annotate_tolerates_odd_structures(self):
        for odd in (None, [], "x", {"crucial": "nope"}, {"crucial": [None, 3, "s", {"status": "missing"}]}):
            parser.annotate_resume_mentions(odd, RESUME)   # must not raise
        odd = {"crucial": [{"status": "missing"}]}
        parser.annotate_resume_mentions(odd, RESUME)
        self.assertIs(odd["crucial"][0]["resume_mention_found"], False)

    def test_json_round_trip_keeps_boolean(self):
        result, _ = self.build("Python")
        self.assertIs(json.loads(json.dumps(result))["crucial"][0]["resume_mention_found"], True)

    def test_logs_counts_only(self):
        with self.assertLogs(parser.logger, level="INFO") as logs:
            self.build("Python", "Kubernetes")
        joined = " ".join(logs.output)
        self.assertIn("1 of 2", joined)
        self.assertNotIn(RESUME[:30], joined)


if __name__ == "__main__":
    unittest.main()
