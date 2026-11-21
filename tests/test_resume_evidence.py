"""Tests for the advisory resume-side quote check (resume_evidence) in src/parser.py.

Run from the project root:  python -m unittest tests.test_resume_evidence -v
Synthetic data only. No network, API key, resume file or live model is used.
A verified quote proves only that the words exist in the extracted text; it is never proof of proficiency.
"""
import copy
import json
import os
import unittest

from tests.test_bullet_verification import load_parser

parser = load_parser()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESUME = (
    "Built a resume parser service using Python and FastAPI endpoints. "
    "Containerised the service with Docker Compose and stored results in PostgreSQL tables.\n"
    "Technologies: Python, FastAPI, Docker, PostgreSQL"
)


def make_entry(skill, status="partial", **extra):
    entry = {"skill": skill, "category": "technical_skills", "status": status,
             "reason": "synthetic reason", "jd_evidence": "requires " + skill}
    entry.update(extra)
    return entry


def raw(*entries):
    return json.dumps({"crucial": list(entries), "good_to_have": [], "optional": []})


def build(*entries, resume=RESUME):
    skills = [e["skill"] for e in entries if isinstance(e, dict) and isinstance(e.get("skill"), str)]
    jd = " ".join("requires " + s for s in skills)
    return parser.build_prioritized_missing_skills(
        raw(*entries), jd, json.dumps({"technical_skills": skills}), resume)


class QuoteMatching(unittest.TestCase):
    def test_four_named_skills_found_and_name_the_skill(self):
        quotes = {"Python": "Built a resume parser service using Python",
                  "FastAPI": "using Python and FastAPI endpoints",
                  "Docker": "Containerised the service with Docker Compose",
                  "PostgreSQL": "stored results in PostgreSQL tables"}
        result, _ = build(*[make_entry(s, resume_evidence=q) for s, q in quotes.items()])
        for e in result["crucial"]:
            self.assertIs(e["resume_evidence_found"], True, e["skill"])
            self.assertIs(e["resume_evidence_names_skill"], True, e["skill"])

    def test_absent_skill_quote_not_found(self):
        result, _ = build(make_entry("Kubernetes", resume_evidence="Managed Kubernetes clusters in production"))
        e = result["crucial"][0]
        self.assertIs(e["resume_evidence_found"], False)
        self.assertIsNone(e["resume_evidence_names_skill"])

    def test_paraphrase_is_not_accepted(self):
        for quote in ("Used Python for building a resume parsing service",     # reworded
                      "Built a resume parser service using Java",              # one word changed
                      "Built resume parser service using Python",              # a word dropped
                      "Built a resume parser service ... Python and FastAPI"): # ellipsis join
            self.assertIs(parser.resume_quote_found(quote, RESUME), False, quote)

    def test_case_and_whitespace_and_glued_pdf_text(self):
        self.assertIs(parser.resume_quote_found("BUILT A RESUME PARSER SERVICE USING PYTHON", RESUME), True)
        self.assertIs(parser.resume_quote_found("Built  a resume\n parser\tservice using Python", RESUME), True)
        self.assertIs(parser.resume_quote_found("Built a resume parser service using Python", RESUME.replace(" ", "")), True)
        # A quote that is glued where the resume has spaces is no longer accepted (see WordBoundary).
        self.assertIs(parser.resume_quote_found("Builtaresumeparserserviceusing Python", RESUME), False)

    def test_punctuation_must_match_exactly(self):
        self.assertIs(parser.resume_quote_found("Technologies: Python, FastAPI, Docker, PostgreSQL", RESUME), True)
        self.assertIs(parser.resume_quote_found("Technologies Python FastAPI Docker PostgreSQL", RESUME), False)

    def test_curly_quotes_and_dashes_are_folded(self):
        text = "Led the team’s ETL work – daily batch loads of sales data"
        self.assertIs(parser.resume_quote_found("Led the team's ETL work - daily batch loads", text), True)

    def test_too_short_quote_is_not_evidence(self):
        for quote in ("Docker", "Docker Compose", "Python and", ""):
            self.assertIs(parser.resume_quote_found(quote, RESUME), False, quote)

    def test_too_long_quote_is_not_evidence(self):
        long_text = "built a data pipeline " * 20
        self.assertGreater(len(long_text), parser._MAX_RESUME_EVIDENCE_LEN)
        self.assertIs(parser.resume_quote_found(long_text, long_text + RESUME), False)

    def test_bad_types_give_false_and_never_raise(self):
        for quote, text in [(None, RESUME), (5, RESUME), (["x"], RESUME), ({"a": 1}, RESUME),
                            ("Built a resume parser service", None), ("Built a resume parser service", 5)]:
            self.assertIs(parser.resume_quote_found(quote, text), False, (quote, text))


class MentionVersusUsage(unittest.TestCase):
    def test_quote_exists_but_does_not_name_the_skill(self):
        result, _ = build(make_entry("Docker", resume_evidence="Containerised the service with"))
        e = result["crucial"][0]
        self.assertIs(e["resume_evidence_found"], True)
        self.assertIs(e["resume_evidence_names_skill"], False)

    def test_list_mention_is_verified_but_status_and_reason_are_untouched(self):
        result, _ = build(make_entry("Docker", resume_evidence="Technologies: Python, FastAPI, Docker, PostgreSQL"))
        e = result["crucial"][0]
        self.assertIs(e["resume_evidence_found"], True)
        self.assertIs(e["resume_evidence_names_skill"], True)
        self.assertEqual((e["status"], e["reason"]), ("partial", "synthetic reason"))
        # No field claims usage or proficiency.
        self.assertFalse([k for k in e if "usage" in k or "proficien" in k or "demonstrat" in k])

    def test_evidence_fields_are_separate_from_resume_mention_found(self):
        result, _ = build(make_entry("Docker", status="missing", resume_evidence="Containerised the service with"))
        e = result["crucial"][0]
        self.assertIs(e["resume_mention_found"], True)           # unchanged Step 1b behaviour
        self.assertIs(e["resume_evidence_names_skill"], False)   # this quote does not name it


class ShortSkillNames(unittest.TestCase):
    def test_names_that_cannot_be_matched_reliably_give_none_not_false(self):
        for skill, quote in (("Go", "Wrote microservices in Go for billing"), ("AI", "Applied AI methods to forecasting")):
            result, _ = build(make_entry(skill, status="missing", resume_evidence=quote), resume="x " + quote + " y")
            e = result["crucial"][0]
            self.assertIs(e["resume_evidence_found"], True)
            self.assertIsNone(e["resume_evidence_names_skill"], skill)
            note = [d for d in result["diagnostics"] if "marked missing" in d][0]
            self.assertIn("could not be checked", note)
            self.assertNotIn("does not contain", note)

    def test_checkable_name_still_reports_true_and_false(self):
        result, _ = build(make_entry("NLP", status="missing", resume_evidence="Built an NLP classifier for tickets"),
                          resume="Built an NLP classifier for tickets")
        self.assertIs(result["crucial"][0]["resume_evidence_names_skill"], True)
        result, _ = build(make_entry("Docker", status="missing", resume_evidence="Containerised the service with"))
        self.assertIs(result["crucial"][0]["resume_evidence_names_skill"], False)


class MissingEntryContradiction(unittest.TestCase):
    def test_verified_quote_on_missing_entry_is_recorded_not_reclassified(self):
        result, legacy = build(make_entry("Docker", status="missing", resume_evidence="Containerised the service with Docker Compose"))
        e = result["crucial"][0]
        self.assertEqual(e["status"], "missing")
        self.assertIs(e["resume_evidence_found"], True)
        hits = [d for d in result["diagnostics"] if "marked missing" in d and "Docker" in d]
        self.assertEqual(len(hits), 1)
        self.assertIn("status unchanged", hits[0])
        self.assertNotIn("Compose", hits[0])                     # resume text is not copied into diagnostics

    def test_unverified_quote_on_missing_entry_is_flagged_as_unverified(self):
        result, _ = build(make_entry("Docker", status="missing", resume_evidence="Deployed everything with Docker Swarm clusters"))
        e = result["crucial"][0]
        self.assertIs(e["resume_evidence_found"], False)
        self.assertTrue([d for d in result["diagnostics"] if "not found in the resume text" in d])
        self.assertFalse([d for d in result["diagnostics"] if "marked missing" in d])

    def test_legacy_dict_is_identical_with_and_without_quotes(self):
        with_q, legacy_q = build(make_entry("Docker", status="missing", resume_evidence="Containerised the service with Docker Compose"),
                                 make_entry("Kubernetes", status="missing"))
        without, legacy_n = build(make_entry("Docker", status="missing"), make_entry("Kubernetes", status="missing"))
        self.assertEqual(legacy_q, legacy_n)
        self.assertEqual([e["status"] for e in with_q["crucial"]], [e["status"] for e in without["crucial"]])


class MalformedAndCompatibility(unittest.TestCase):
    def test_malformed_quotes_never_break_the_entry(self):
        for bad in (5, 0, True, ["x"], {"a": 1}):
            result, _ = build(make_entry("Python", resume_evidence=bad))
            self.assertTrue(result["normalization_ok"], bad)
            e = result["crucial"][0]
            self.assertNotIn("resume_evidence", e, bad)
            self.assertIsNone(e["resume_evidence_found"])
            self.assertTrue([d for d in result["diagnostics"] if "not a non-empty string" in d], bad)

    def test_empty_or_null_quotes_are_simply_not_provided(self):
        for empty in (None, "", "   "):
            result, _ = build(make_entry("Python", resume_evidence=empty))
            e = result["crucial"][0]
            self.assertIsNone(e["resume_evidence_found"])
            self.assertIsNone(e["resume_evidence_names_skill"])
            self.assertFalse([d for d in result["diagnostics"] if "resume_evidence" in d], repr(empty))
        result, _ = build(make_entry("Python", resume_evidence=None))
        self.assertFalse([d for d in result["diagnostics"] if "resume_evidence" in d])

    def test_entry_without_quote_is_unchanged_apart_from_null_flags(self):
        plain, _ = build(make_entry("Python"))
        e = plain["crucial"][0]
        expected = {"skill", "category", "status", "reason", "jd_evidence", "evidence_found", "evidence_related",
                    "in_extracted_list", "resume_evidence_found", "resume_evidence_names_skill"}
        self.assertEqual(set(e), expected)
        self.assertIsNone(e["resume_evidence_found"])

    def test_nothing_else_changes_when_a_quote_is_added(self):
        quote = "Built a resume parser service using Python"
        with_q, legacy_q = build(make_entry("Python", resume_evidence=quote), make_entry("Kubernetes", status="missing"))
        without, legacy_n = build(make_entry("Python"), make_entry("Kubernetes", status="missing"))
        stripped = copy.deepcopy(with_q)
        for e in stripped["crucial"]:
            for key in ("resume_evidence", "resume_evidence_found", "resume_evidence_names_skill"):
                e.pop(key, None)
        base = copy.deepcopy(without)
        for e in base["crucial"]:
            for key in ("resume_evidence_found", "resume_evidence_names_skill"):
                e.pop(key, None)
        self.assertEqual(stripped, base)
        self.assertEqual(legacy_q, legacy_n)

    def test_resume_text_not_a_string_adds_no_flags(self):
        result, _ = build(make_entry("Python", resume_evidence="Built a resume parser service using Python"), resume=None)
        e = result["crucial"][0]
        self.assertNotIn("resume_evidence_found", e)
        self.assertNotIn("resume_mention_found", e)

    def test_old_three_argument_call_still_works(self):
        result, _ = parser.build_prioritized_missing_skills(raw(make_entry("Python")), "requires Python", "{}")
        self.assertTrue(result["normalization_ok"])
        self.assertNotIn("resume_evidence_found", result["crucial"][0])

    def test_unusable_model_output_is_unchanged(self):
        result, legacy = parser.build_prioritized_missing_skills("not json", "jd", "{}", RESUME)
        self.assertFalse(result["normalization_ok"])
        self.assertEqual(legacy, {c: [] for c in parser.SKILL_CATEGORIES})

    def test_annotate_tolerates_odd_structures(self):
        for odd in (None, [], "x", {"crucial": "nope"}, {"crucial": [None, 3, "s"]}, {"crucial": [{}], "diagnostics": "no"}):
            parser.annotate_resume_evidence(odd, RESUME)       # must not raise

    def test_diagnostics_stay_capped(self):
        entries = [make_entry(f"Skill{i}", resume_evidence=f"Quote number {i} that is not in the resume") for i in range(80)]
        result, _ = build(*entries)
        self.assertLessEqual(len(result["diagnostics"]),
                             parser._MAX_DIAGNOSTICS + 1 + parser._MAX_EVIDENCE_DIAGNOSTICS + 1)

    def test_json_round_trip_and_resume_text_untouched(self):
        text = "Built a resume parser service using Python"
        before = str(text)
        result, _ = build(make_entry("Python", resume_evidence=text), resume=text)
        self.assertEqual(text, before)
        again = json.loads(json.dumps(result))
        self.assertIs(again["crucial"][0]["resume_evidence_found"], True)

    def test_logs_counts_only(self):
        with self.assertLogs(parser.logger, level="INFO") as logs:
            build(make_entry("Python", resume_evidence="Built a resume parser service using Python"),
                  make_entry("Docker", resume_evidence="Deployed on a made up platform nobody has"))
        joined = " ".join(logs.output)
        self.assertIn("1 of 2 quotes", joined)
        self.assertNotIn("resume parser service", joined)
        self.assertNotIn("made up platform", joined)


class WordBoundary(unittest.TestCase):
    """A quote must not match by splitting or gluing real words ("the rapist" versus "therapist")."""

    def test_audit_case_split_word_is_rejected(self):
        self.assertIs(parser.resume_quote_found("the rapist organised files", "therapist organisedfiles"), False)
        self.assertIs(parser.resume_quote_found("the rapist organised files",
                                                "She is a therapist organisedfiles for clients"), False)

    def test_same_words_with_real_boundaries_still_match(self):
        self.assertIs(parser.resume_quote_found("the therapist organised files", "the therapist organisedfiles"), True)
        self.assertIs(parser.resume_quote_found("the therapist organised files", "the therapist organised files"), True)

    def test_sparse_pdf_glue_between_long_words_is_tolerated(self):
        self.assertIs(parser.resume_quote_found("Developed resume parsing pipelines",
                                                "Developed resumeparsing pipelines daily"), True)
        self.assertIs(parser.resume_quote_found("Developed resume parsing pipelines",
                                                "Developedresumeparsingpipelines daily"), True)

    def test_dense_pdf_glue_is_tolerated_including_short_words_and_mid_run_starts(self):
        glued = RESUME.replace(" ", "")
        self.assertIs(parser.resume_quote_found("Built a resume parser service using Python", glued), True)
        self.assertIs(parser.resume_quote_found("the service with Docker Compose", glued), True)

    def test_short_word_glue_in_sparse_text_is_a_documented_loss(self):
        self.assertIs(parser.resume_quote_found("Built a resume parser service", "Builtaresume parser service today"), False)

    def test_quote_cannot_start_or_end_inside_a_resume_word_in_normal_text(self):
        resume = "a therapist organised files today"
        self.assertIs(parser.resume_quote_found("rapist organised files", resume), False)
        self.assertIs(parser.resume_quote_found("a therapist organised fil", resume), False)
        self.assertIs(parser.resume_quote_found("therapist organised files", resume), True)

    def test_punctuation_next_to_the_quote_edges_is_fine(self):
        self.assertIs(parser.resume_quote_found("stored results in PostgreSQL tables", "we stored results in PostgreSQL tables."), True)

    def test_quote_glued_where_resume_has_spaces_is_not_accepted(self):
        self.assertIs(parser.resume_quote_found("resumeparserservice", "built a resume parser service now"), False)

    def test_boundary_matching_is_fast_on_large_input(self):
        import time
        text = "organised files " * 20000
        begin = time.perf_counter()
        self.assertIs(parser.resume_quote_found("the rapist organised files", text), False)
        self.assertLess(time.perf_counter() - begin, 5)


class DiagnosticBudget(unittest.TestCase):
    """Evidence diagnostics have their own fixed budget: bounded, and important ones survive a full list."""

    BOUND = parser._MAX_DIAGNOSTICS + 1 + parser._MAX_EVIDENCE_DIAGNOSTICS + 1

    def test_contradiction_survives_a_full_general_list(self):
        base = [make_entry(f"Skill{i}", status="missing", jd_evidence="quote that is not in the jd") for i in range(60)]
        docker = make_entry("Docker", status="missing", resume_evidence="Containerised the service with Docker Compose")
        result, _ = build(*base, docker)
        diagnostics = result["diagnostics"]
        self.assertIn("further diagnostics omitted", diagnostics)        # the general list really was full
        self.assertTrue([d for d in diagnostics if "'Docker' is marked missing" in d])
        self.assertLessEqual(len(diagnostics), self.BOUND)
        self.assertTrue(all(isinstance(d, str) for d in diagnostics))

    def test_growth_is_bounded(self):
        entries = [make_entry(f"Skill{i}", resume_evidence=f"Made up quote number {i} here") for i in range(200)]
        result, _ = build(*entries)
        self.assertLessEqual(len(result["diagnostics"]), self.BOUND)

    def test_contradictions_come_first_and_overflow_is_counted(self):
        entries = [make_entry(f"Skill{i}", resume_evidence=f"Made up quote number {i} here") for i in range(15)]
        entries.append(make_entry("Docker", status="missing", resume_evidence="Containerised the service with Docker Compose"))
        result, _ = build(*entries)
        evidence = [d for d in result["diagnostics"] if "resume_evidence" in d]
        self.assertEqual(len(evidence), parser._MAX_EVIDENCE_DIAGNOSTICS + 1)
        self.assertIn("'Docker' is marked missing", evidence[0])
        self.assertIn("6 further resume_evidence diagnostics omitted", evidence[-1])

    def test_entry_flags_stay_complete_when_diagnostics_are_omitted(self):
        entries = [make_entry(f"Skill{i}", resume_evidence=f"Made up quote number {i} here") for i in range(30)]
        result, _ = build(*entries)
        self.assertTrue(all(e["resume_evidence_found"] is False for e in result["crucial"]))


class StorageCap(unittest.TestCase):
    def test_quote_at_the_limit_is_stored_and_over_it_is_not(self):
        at_limit = "a" * parser._MAX_RESUME_EVIDENCE_LEN
        result, _ = build(make_entry("Python", resume_evidence=at_limit))
        self.assertEqual(result["crucial"][0]["resume_evidence"], at_limit)
        over = "a" * (parser._MAX_RESUME_EVIDENCE_LEN + 1)
        result, _ = build(make_entry("Python", resume_evidence=over))
        e = result["crucial"][0]
        self.assertNotIn("resume_evidence", e)
        self.assertIsNone(e["resume_evidence_found"])
        self.assertTrue([d for d in result["diagnostics"] if "longer than" in d])

    def test_huge_quote_is_not_kept(self):
        result, _ = build(make_entry("Python", resume_evidence="A" * 200000))
        self.assertNotIn("resume_evidence", result["crucial"][0])


class PromptContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except Exception:
            raise unittest.SkipTest("PyYAML not installed")
        with open(os.path.join(ROOT, "src", "prompts.yaml"), encoding="utf-8") as handle:
            cls.template = yaml.safe_load(handle)["job_resume_comparison"]

    def test_formats_with_only_its_three_inputs(self):
        text = self.template.format(job_description="JD", job_skills="[]", resume_content="RESUME")
        self.assertNotIn("{{", text)
        self.assertIn("resume_evidence", text)

    def test_example_json_is_valid_and_shows_the_quote_on_the_partial_entry_only(self):
        text = self.template.format(job_description="JD", job_skills="[]", resume_content="RESUME")
        block = text.split("```json", 1)[1].split("```", 1)[0]
        example = json.loads(block)
        missing = example["crucial"][0]
        partial = example["good_to_have"][0]
        self.assertEqual((missing["status"], partial["status"]), ("missing", "partial"))
        self.assertNotIn("resume_evidence", missing)
        self.assertTrue(partial["resume_evidence"])
        self.assertEqual(set(missing), {"skill", "category", "status", "reason", "jd_evidence"})

    def test_prompt_separates_mention_from_usage(self):
        self.assertIn("mention, not usage", self.template)

    def step2_bullets(self):
        """The bullets of 'Step 2 - Prioritise ...' (up to the next bold heading)."""
        block = self.template.split("**Step 2", 1)[1].split("**Rules**", 1)[0]
        return [line.strip() for line in block.splitlines() if line.strip().startswith("- ")]

    def test_priority_rules_follow_section_headings(self):
        bullets = self.step2_bullets()
        nice = [b for b in bullets if '"Nice to Have"' in b]
        required = [b for b in bullets if '"Must Have"' in b]
        self.assertEqual((len(nice), len(required)), (1, 1))
        self.assertIn('"good_to_have"', nice[0])       # nice-to-have headings -> Good to Have
        self.assertIn('only if the job description separately states', nice[0])
        self.assertIn('"crucial"', required[0])        # required headings -> Crucial

    def test_existing_conservative_rules_and_schema_are_kept(self):
        text = self.template
        self.assertIn('Never default to "crucial"', text)
        self.assertIn("When the wording is ambiguous, choose the lower tier.", text)
        self.assertIn("Do not use outside knowledge", text)
        self.assertIn("explicitly presented as optional", text)

    def test_priority_rules_name_no_individual_skill(self):
        step2 = " ".join(self.step2_bullets())
        for skill in ("Docker", "Pytest", "RAG", "Vector", "Embedding", "LLM", "Python", "FastAPI"):
            self.assertNotIn(skill, step2)


class TierIsPassedThrough(unittest.TestCase):
    """The normaliser keeps the model's tier; it never re-tiers a validated entry."""

    def test_nice_to_have_stays_good_to_have_and_required_stays_crucial(self):
        jd = "Required: Python. Nice to Have: Docker."
        entries = {
            "crucial": [make_entry("Python", status="missing", jd_evidence="Required: Python")],
            "good_to_have": [make_entry("Docker", status="missing", jd_evidence="Nice to Have: Docker")],
            "optional": [],
        }
        helper = json.dumps({"technical_skills": ["Python", "Docker"]})
        result, _ = parser.build_prioritized_missing_skills(json.dumps(entries), jd, helper, RESUME)
        self.assertEqual([e["skill"] for e in result["crucial"]], ["Python"])
        self.assertEqual([e["skill"] for e in result["good_to_have"]], ["Docker"])
        self.assertEqual(result["optional"], [])
        for e in result["crucial"] + result["good_to_have"]:
            self.assertIs(e["evidence_found"], True)


if __name__ == "__main__":
    unittest.main()
