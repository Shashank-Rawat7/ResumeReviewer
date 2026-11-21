"""Tests for the canonical skill-gap representation and the sections built from it (frontend/display.py).

Covers: canonical records, de-duplication, evidence precedence, reconciliation of the Job Skill Coverage
Overview, the Skill Gaps by Priority summary, recommendation consistency, labels and rendering, and the
legacy / unavailable paths. (The file keeps its original name, test_priority_breakdown.)

Run from the project root:  python -m unittest tests.test_priority_breakdown -v
Synthetic data only and a fake Streamlit: no network, API key, resume file or live model is used.
"""
import copy
import os
import re
import unittest

from tests.test_project_bullets import load_display

TIERS = ("crucial", "good_to_have", "optional")
TECH, SOFT, ACAD = "technical_skills", "soft_skills", "academic_qualifications"


def entry(skill, category=TECH, status="missing", evidence_found=True, evidence_related=True):
    return {
        "skill": skill, "category": category, "status": status,
        "reason": "Synthetic reason.", "jd_evidence": "synthetic job description quote",
        "evidence_found": evidence_found, "evidence_related": evidence_related, "in_extracted_list": True,
    }


def result(crucial=(), good_to_have=(), optional=(), ok=True):
    return {"normalization_ok": ok, "crucial": list(crucial), "good_to_have": list(good_to_have),
            "optional": list(optional), "diagnostics": []}


def jobs(technical=(), soft=(), academic=()):
    return {TECH: list(technical), SOFT: list(soft), ACAD: list(academic)}


JOB = jobs(["SkillA", "SkillB", "SkillC", "SkillD", "SkillE", "SkillF", "Flask", "FastAPI"], ["Communication"])


def full():
    """9 flagged entries after de-duplication (8 technical, 1 soft); 9 extracted skills; 1 outside the list."""
    return result(
        crucial=[entry("SkillA"), entry("SkillB", status="partial"), entry("SkillC", evidence_found=False),
                 entry("Flask"), entry("FastAPI")],
        good_to_have=[entry("SkillD", status="partial"), entry("SkillA", status="partial"),
                      entry("Communication", category=SOFT)],
        optional=[entry("SkillE"), entry("skillb"), entry("OutsideTool")],
    )


class FakeColumn:
    def __init__(self, fake):
        self.fake = fake

    def metric(self, label, value, *args, **kwargs):
        self.fake.metrics.append((label, value))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.display, cls.fake = load_display()
        fake = cls.fake
        fake.captions, fake.infos, fake.metrics, fake.subheaders = [], [], [], []
        fake.caption = lambda text, *a, **k: fake.captions.append(text)
        fake.info = lambda text, *a, **k: fake.infos.append(text)
        fake.progress = lambda *a, **k: None
        fake.subheader = lambda text, *a, **k: fake.subheaders.append(text)
        fake.columns = lambda n, *a, **k: [FakeColumn(fake) for _ in range(n)]

    def setUp(self):
        for items in (self.fake.markdown_calls, self.fake.captions, self.fake.infos, self.fake.metrics,
                      self.fake.subheaders):
            items.clear()

    def records(self, prioritized, job_skills=JOB):
        view = self.display.build_skill_gap_records(prioritized, job_skills)
        return None if view is None else view["records"]

    def by_skill(self, prioritized, job_skills=JOB):
        return {r["skill"]: r for r in self.records(prioritized, job_skills)}

    @property
    def rendered_text(self):
        return "\n".join(self.fake.markdown_calls + self.fake.captions + self.fake.infos
                         + [f"{label} {value}" for label, value in self.fake.metrics])


class ClassificationTests(Base):
    def test_state_rule_is_evidence_first_then_status(self):
        d = self.display
        self.assertEqual(d.classify_entry(entry("A")), d.STATE_NOT_DEMONSTRATED)
        self.assertEqual(d.classify_entry(entry("A", status="partial")), d.STATE_PARTIAL)
        self.assertEqual(d.classify_entry(entry("A", evidence_found=False)), d.STATE_UNVERIFIED)
        self.assertEqual(d.classify_entry(entry("A", status="partial", evidence_found=False)), d.STATE_UNVERIFIED)

    def test_evidence_related_is_advisory_only(self):
        d = self.display
        for related in (True, False, None):
            self.assertEqual(d.classify_entry(entry("A", evidence_related=related)), d.STATE_NOT_DEMONSTRATED)

    def test_approved_labels(self):
        d = self.display
        self.assertEqual(
            [d.STATE_LABELS[s] for s in (d.STATE_NO_GAP, d.STATE_PARTIAL, d.STATE_NOT_DEMONSTRATED, d.STATE_UNVERIFIED)],
            ["No Gap Identified", "Partially Demonstrated", "Not Demonstrated", "Evidence Not Verified"])


class CanonicalRecordTests(Base):
    def test_records_hold_flagged_and_unflagged_extracted_skills(self):
        recs = self.by_skill(full())
        self.assertEqual(set(recs), {"SkillA", "SkillB", "SkillC", "SkillD", "SkillE", "SkillF", "Flask", "FastAPI",
                                     "Communication", "OutsideTool"})
        self.assertEqual(recs["SkillF"]["state"], self.display.STATE_NO_GAP)
        self.assertIsNone(recs["SkillF"]["tier"])
        self.assertIsNone(recs["SkillF"]["entry"])

    def test_flagged_record_preserves_identity_and_fields(self):
        rec = self.by_skill(result(crucial=[entry("  SkillA  ", status="partial", evidence_related=False)]))["SkillA"]
        self.assertEqual((rec["key"], rec["skill"], rec["tier"], rec["category"], rec["status"]),
                         ("skilla", "SkillA", "crucial", TECH, "partial"))
        self.assertEqual((rec["evidence_found"], rec["evidence_related"], rec["in_extracted_list"]), (True, False, True))
        self.assertEqual(rec["state"], self.display.STATE_PARTIAL)

    def test_unflagged_record_takes_name_and_category_from_the_extracted_list(self):
        rec = self.by_skill(result(), jobs(["  Kubernetes "], ["Teamwork"]))
        self.assertEqual((rec["Kubernetes"]["category"], rec["Teamwork"]["category"]), (TECH, SOFT))
        self.assertEqual(rec["Kubernetes"]["skill"], "Kubernetes")

    def test_skill_in_several_tiers_is_one_record_at_the_highest_priority(self):
        recs = self.records(result(crucial=[entry("Docker")], good_to_have=[entry("Docker", status="partial")],
                                   optional=[entry("Docker")]), jobs(["Docker"]))
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["tier"], recs[0]["status"]), ("crucial", "missing"))

    def test_spelling_and_punctuation_variants_are_one_skill(self):
        recs = self.records(result(crucial=[entry("REST APIs")], good_to_have=[entry("rest api")],
                                   optional=[entry("Rest-API")]), jobs(["REST APIs"]))
        self.assertEqual(len(recs), 1)

    def test_duplicate_across_categories_keeps_the_first(self):
        recs = self.records(result(crucial=[entry("Communication", category=SOFT)],
                                   good_to_have=[entry("communication")]), jobs(soft=["Communication"]))
        self.assertEqual([(r["category"], r["tier"]) for r in recs], [(SOFT, "crucial")])

    def test_extracted_duplicates_and_cross_category_duplicates_count_once(self):
        view = self.display.build_skill_gap_records(result(), jobs(["Python", "python"], ["Python"], ["PYTHON"]))
        self.assertEqual(view["extracted_total"], 1)
        self.assertEqual(len(view["records"]), 1)

    def test_outside_list_flag_uses_the_frontend_name_match(self):
        recs = self.by_skill(full())
        self.assertFalse(recs["OutsideTool"]["in_extracted_list"])
        self.assertTrue(recs["Flask"]["in_extracted_list"])
        self.assertEqual(self.display.build_skill_gap_records(full(), JOB)["extracted_total"], 9)

    def test_outside_entry_is_not_added_to_the_extracted_total(self):
        view = self.display.build_skill_gap_records(result(crucial=[entry("Zed")]), jobs(["Alpha"]))
        self.assertEqual(view["extracted_total"], 1)
        self.assertEqual(sum(1 for r in view["records"] if r["in_extracted_list"]), 1)

    def test_odd_extracted_lists_are_tolerated(self):
        for odd in (None, [], "text", {TECH: "not a list"}, {TECH: [None, 5, "", "  ", "***"]}, {}):
            with self.subTest(odd=repr(odd)):
                view = self.display.build_skill_gap_records(result(), odd)
                self.assertEqual((view["extracted_total"], view["records"]), (0, []))

    def test_input_is_not_modified(self):
        data, job = full(), copy.deepcopy(JOB)
        before = copy.deepcopy(data)
        self.display.build_skill_gap_records(data, job)
        self.display.summarize_skill_gaps(data, job)
        self.display.summarize_priority_gaps(data)
        self.assertEqual((data, job), (before, JOB))


class OverviewReconciliationTests(Base):
    def counts(self, prioritized, job_skills=JOB):
        return self.display.summarize_skill_gaps(prioritized, job_skills)

    def test_full_example(self):
        c = self.counts(full())
        self.assertEqual((c["no_gap"], c["partially_demonstrated"], c["not_demonstrated"],
                          c["evidence_not_verified"], c["total"], c["outside"]), (1, 2, 5, 1, 9, 1))

    def test_four_counts_always_sum_to_the_extracted_total(self):
        cases = [(result(), JOB), (full(), JOB), (full(), jobs()), (full(), jobs(["SkillA"])),
                 (result(crucial=[entry("Communication", category=SOFT, evidence_found=False)]), JOB),
                 (result(crucial=[entry("X")], optional=[entry("x", evidence_found=False)]), jobs(["X", "x", "Y"]))]
        for prioritized, job in cases:
            with self.subTest(job=repr(job)[:40]):
                c = self.counts(prioritized, job)
                self.assertEqual(c["no_gap"] + c["partially_demonstrated"] + c["not_demonstrated"]
                                 + c["evidence_not_verified"], c["total"])

    def test_unverified_is_exclusive_of_the_two_statuses(self):
        c = self.counts(result(crucial=[entry("A", evidence_found=False), entry("B", status="partial", evidence_found=False)]),
                        jobs(["A", "B"]))
        self.assertEqual((c["evidence_not_verified"], c["not_demonstrated"], c["partially_demonstrated"]), (2, 0, 0))

    def test_all_categories_count_in_the_overview(self):
        c = self.counts(result(crucial=[entry("Teamwork", category=SOFT), entry("MBA", category=ACAD, status="partial")]),
                        jobs(["Py"], ["Teamwork"], ["MBA"]))
        self.assertEqual((c["not_demonstrated"], c["partially_demonstrated"], c["no_gap"], c["total"]), (1, 1, 1, 3))

    def test_outside_entries_are_reported_separately_and_never_counted(self):
        c = self.counts(result(crucial=[entry("Zed"), entry("Yak", evidence_found=False)]), jobs(["Alpha"]))
        self.assertEqual((c["outside"], c["total"], c["no_gap"], c["not_demonstrated"], c["evidence_not_verified"]),
                         (2, 1, 1, 0, 0))

    def test_no_flagged_entry_means_every_extracted_skill_is_no_gap(self):
        c = self.counts(result(), JOB)
        self.assertEqual((c["no_gap"], c["total"]), (9, 9))

    def test_flagged_in_list_plus_outside_equals_all_flagged(self):
        data = full()
        c, p = self.counts(data), self.display.summarize_priority_gaps(data)
        self.assertEqual(c["not_demonstrated"] + c["partially_demonstrated"] + c["evidence_not_verified"] + c["outside"],
                         p["total"])


class PriorityGapsTests(Base):
    def test_full_example(self):
        p = self.display.summarize_priority_gaps(full())
        self.assertEqual(p["tiers"]["crucial"], {"not_demonstrated": 3, "partially_demonstrated": 1,
                                                 "evidence_not_verified": 1, "total": 5})
        self.assertEqual(p["tiers"]["good_to_have"], {"not_demonstrated": 1, "partially_demonstrated": 1,
                                                      "evidence_not_verified": 0, "total": 2})
        self.assertEqual(p["tiers"]["optional"], {"not_demonstrated": 2, "partially_demonstrated": 0,
                                                  "evidence_not_verified": 0, "total": 2})
        self.assertEqual((p["total"], p["technical"]), (9, 8))
        self.assertEqual(p["categories"], {TECH: 8, SOFT: 1, ACAD: 0})

    def test_tier_totals_add_up_and_every_state_is_counted_once(self):
        p = self.display.summarize_priority_gaps(full())
        for counts in p["tiers"].values():
            self.assertEqual(counts["not_demonstrated"] + counts["partially_demonstrated"]
                             + counts["evidence_not_verified"], counts["total"])
        self.assertEqual(sum(c["total"] for c in p["tiers"].values()), p["total"])
        self.assertEqual(sum(p["categories"].values()), p["total"])
        self.assertEqual(p["technical"], p["categories"][TECH])

    def test_total_equals_the_number_of_flagged_records(self):
        data = full()
        self.assertEqual(self.display.summarize_priority_gaps(data)["total"],
                         sum(1 for r in self.records(data) if r["tier"] is not None))

    def test_duplicates_count_once_across_tiers(self):
        p = self.display.summarize_priority_gaps(result(crucial=[entry("Docker")], good_to_have=[entry("docker")],
                                                        optional=[entry("DOCKER ")]))
        self.assertEqual((p["total"], p["tiers"]["crucial"]["total"]), (1, 1))

    def test_alternatives_stay_separate(self):
        p = self.display.summarize_priority_gaps(result(crucial=[entry("Flask"), entry("FastAPI")]))
        self.assertEqual(p["tiers"]["crucial"]["not_demonstrated"], 2)

    def test_valid_empty_result_is_all_zero_not_none(self):
        p = self.display.summarize_priority_gaps(result())
        self.assertEqual((p["total"], p["technical"]), (0, 0))
        self.assertIsNotNone(p)


class RecommendationConsistencyTests(Base):
    def test_options_and_skipped_follow_the_canonical_records(self):
        data = full()
        options, skipped = self.display.get_recommendation_options(data)
        self.assertEqual(set(options), {"SkillA", "SkillB", "SkillD", "SkillE", "Flask", "FastAPI", "OutsideTool"})
        self.assertEqual(skipped, 1)
        eligible = {r["skill"] for r in self.records(data) if r["tier"] is not None and r["category"] == TECH
                    and r["state"] != self.display.STATE_UNVERIFIED}
        self.assertEqual(set(options), eligible)

    def test_options_plus_skipped_equal_the_technical_count(self):
        for data in (full(), result(), result(crucial=[entry("Teamwork", category=SOFT), entry("A", evidence_found=False)]),
                     result(crucial=[entry("A"), entry("B", status="partial", evidence_found=False)])):
            options, skipped = self.display.get_recommendation_options(data)
            self.assertEqual(len(options) + skipped, self.display.summarize_priority_gaps(data)["technical"])

    def test_unverified_and_non_technical_entries_are_never_offered(self):
        options, skipped = self.display.get_recommendation_options(result(crucial=[
            entry("A", evidence_found=False), entry("Teamwork", category=SOFT), entry("MBA", category=ACAD),
            entry("B")]))
        self.assertEqual((list(options), skipped), (["B"], 1))

    def test_labels_use_the_approved_terms(self):
        options, _ = self.display.get_recommendation_options(
            result(crucial=[entry("A")], good_to_have=[entry("B", status="partial", evidence_related=False)]))
        self.assertEqual(options["A"], "A — Crucial, Not Demonstrated")
        self.assertEqual(options["B"], "B — Good to Have, Partially Demonstrated · quote relevance unconfirmed")
        for label in options.values():
            self.assertNotRegex(label, r"\b(missing|partial)\b")

    def test_outside_list_technical_skill_with_a_verified_quote_is_offered(self):
        options, _ = self.display.get_recommendation_options(full())
        self.assertIn("OutsideTool", options)


class UnavailableAndLegacyTests(Base):
    def bad_cases(self):
        broken = entry("A")
        del broken["reason"]
        return [None, "text", 5, [], {}, result(ok=False), result(crucial=[entry("A")], ok=False),
                result(crucial=[broken]), result(crucial=["not an entry"]), result(crucial=[entry("A", status="matched")]),
                result(crucial=[entry("A", evidence_found="yes")]), result(crucial=[entry("***")]),
                dict(result(), diagnostics=[1]), dict(result(), crucial="x")]

    def test_every_function_returns_nothing_usable_for_bad_input(self):
        d = self.display
        for case in self.bad_cases():
            with self.subTest(case=repr(case)[:50]):
                self.assertIsNone(d.build_skill_gap_records(case, JOB))
                self.assertIsNone(d.summarize_skill_gaps(case, JOB))
                self.assertIsNone(d.summarize_priority_gaps(case))
                self.assertEqual(d.get_recommendation_options(case), ({}, 0))

    def test_nothing_is_rendered_for_bad_input(self):
        d = self.display
        for case in self.bad_cases():
            d.display_skill_gap_summary(case, JOB)
            d.display_prioritized_missing_skills(case, JOB)
        self.assertEqual(self.rendered_text.strip(), "")
        self.assertEqual(self.fake.subheaders, [])

    def test_valid_empty_result_is_distinguishable_from_unavailable(self):
        d = self.display
        d.display_prioritized_missing_skills(result(), JOB)
        self.assertEqual(len(self.fake.infos), 1)
        self.assertIsNotNone(d.summarize_priority_gaps(result()))
        self.assertIsNone(d.summarize_priority_gaps(result(ok=False)))

    def test_legacy_and_unavailable_state_detection_is_unchanged(self):
        d = self.display
        self.assertEqual(d.get_prioritized_view({"missing_skills": {}}), ("legacy", None))
        self.assertEqual(d.get_prioritized_view({}), ("legacy", None))
        self.assertEqual(d.get_prioritized_view({"prioritized_missing_skills": None})[0], "unavailable")
        self.assertEqual(d.get_prioritized_view({"prioritized_missing_skills": result(ok=False)})[0], "unavailable")
        self.assertEqual(d.get_prioritized_view({"prioritized_missing_skills": result()})[0], "ok")

    def test_legacy_widgets_keep_their_legacy_wording(self):
        self.display.display_skill_match(1, 4)
        self.display.display_missing_skills({"technical_skills": ["SkillB"], "soft_skills": [], "academic_qualifications": []})
        text = self.rendered_text
        self.assertIn("Matched 3 out of 4", text)
        self.assertIn("SkillB", text)

    def test_unavailable_message_names_the_current_sections(self):
        message = self.display.SKILL_ANALYSIS_UNAVAILABLE_MESSAGE
        self.assertNotIn("ATS", message)
        for phrase in ("coverage overview", "skill gaps by priority", "skill recommendations"):
            self.assertIn(phrase, message)


class RenderingTests(Base):
    OLD_WORDS = re.compile(r"\b(Missing|Partial|Unverified)\b|Not flagged|Not Flagged|ATS Score")

    def test_overview_shows_four_labelled_metrics_and_the_caveat(self):
        self.display.display_skill_gap_summary(full(), JOB)
        self.assertEqual(self.fake.subheaders, ["🎯 Job Skill Coverage Overview"])
        self.assertEqual(self.fake.metrics, [("No Gap Identified", 1), ("Partially Demonstrated", 2),
                                             ("Not Demonstrated", 5), ("Evidence Not Verified", 1)])
        caption = self.fake.captions[0]
        self.assertIn("Out of 9 distinct skills", caption)
        self.assertIn("1 further flagged item(s) are not in the extracted list", caption)
        self.assertIn("included in Skill Gaps by Priority", caption)
        self.assertIn("does not confirm the candidate's proficiency", caption)
        self.assertIn("not proof that the candidate lacks it", caption)

    def test_overview_without_outside_entries_has_no_outside_note(self):
        self.display.display_skill_gap_summary(result(crucial=[entry("SkillA")]), JOB)
        self.assertNotIn("further flagged", self.fake.captions[0])

    def test_priority_section_shows_summary_table_counts_and_lists(self):
        self.display.display_prioritized_missing_skills(full(), JOB)
        text = self.rendered_text
        table = [m for m in self.fake.markdown_calls if "| Priority |" in m][0]
        self.assertIn("| Priority | Not Demonstrated | Partially Demonstrated | Evidence Not Verified | Total |", table)
        self.assertIn("| Crucial | 3 | 1 | 1 | 5 |", table)
        self.assertIn("| Good to Have | 1 | 1 | 0 | 2 |", table)
        self.assertIn("| Optional | 2 | 0 | 0 | 2 |", table)
        self.assertIn("| **All priorities** | 6 | 2 | 1 | 9 |", table)
        self.assertIn("**Flagged skills: 9** · Technical skills: 8 · Soft skills: 1 · Academic qualifications: 0", text)
        for header in ("**Crucial** (5)", "**Good to Have** (2)", "**Optional** (2)"):
            self.assertIn(header, text)

    def test_displayed_technical_entries_agree_with_the_technical_count(self):
        self.display.display_prioritized_missing_skills(full(), JOB)
        entries = [m for m in self.fake.markdown_calls if "<span style=" in m]
        self.assertEqual(len(entries), 9)
        self.assertEqual(sum(1 for m in entries if "· Technical" in m), 8)
        self.assertEqual(sum(1 for m in entries if "· Soft skills" in m or "· Soft" in m), 1)

    def test_empty_state_groups_are_omitted_and_empty_tiers_say_so(self):
        self.display.display_prioritized_missing_skills(result(crucial=[entry("A")]), JOB)
        text = self.rendered_text
        self.assertIn("*Not Demonstrated* (1)", text)
        self.assertNotIn("*Partially Demonstrated* (", text)
        self.assertNotIn("*Evidence Not Verified* (", text)
        self.assertIn("**Good to Have** (0)", text)
        self.assertEqual(text.count("_None identified._"), 2)

    def test_unverified_entry_is_grouped_separately_and_shows_its_reported_status(self):
        self.display.display_prioritized_missing_skills(result(crucial=[entry("A", status="partial", evidence_found=False)]), JOB)
        text = self.rendered_text
        self.assertIn("*Evidence Not Verified* (1)", text)
        self.assertNotIn("*Partially Demonstrated* (", text)
        self.assertIn("reported status (not counted): Partially Demonstrated", text)
        self.assertIn("quote not found in the job description (unverified)", text)

    def test_outside_and_relevance_notes_are_kept(self):
        self.display.display_prioritized_missing_skills(
            result(crucial=[entry("OutsideTool"), entry("SkillA", evidence_related=False)]), JOB)
        text = self.rendered_text
        self.assertIn("outside the extracted skill list", text)
        self.assertIn("relevance of the quote is unconfirmed", text)

    def test_model_text_is_html_escaped(self):
        bad = entry("<b>Tool</b>")
        bad["reason"] = "<script>x</script>"
        self.display.display_prioritized_missing_skills(result(crucial=[bad]), JOB)
        text = self.rendered_text
        self.assertNotIn("<script>", text)
        self.assertNotIn("<b>Tool</b>", text)
        self.assertIn("&lt;b&gt;Tool&lt;/b&gt;", text)

    def test_valid_empty_result_shows_an_info_message_and_no_table(self):
        self.display.display_prioritized_missing_skills(result(), JOB)
        self.assertEqual(self.fake.infos, ["The model did not flag any skill gaps for this job description. "
                                           "This does not guarantee the resume meets the role."])
        self.assertFalse([m for m in self.fake.markdown_calls if "| Priority |" in m])

    def test_no_old_labels_score_or_percentage_in_the_new_sections(self):
        self.display.display_skill_gap_summary(full(), JOB)
        self.display.display_prioritized_missing_skills(full(), JOB)
        self.assertIsNone(self.OLD_WORDS.search(self.rendered_text))
        self.assertIsNone(re.search(r"\d\s*%", self.rendered_text))

    def test_old_breakdown_and_gauge_are_gone(self):
        for name in ("display_priority_breakdown", "summarize_priority_breakdown", "calculate_ats_score", "display_ats_score",
                     "_STATUS_STYLE", "_extracted_skill_keys"):
            self.assertFalse(hasattr(self.display, name), name)
        self.assertNotIn("Technical Skill Gap Breakdown", self.rendered_text)


class RecommendationsTabWordingTests(unittest.TestCase):
    """Fixed wording of the Recommendations tab in frontend/main.py.

    main.py is a Streamlit script and cannot be imported in a unit test, so its wording is checked in the
    source text. The rendered behaviour is covered by the Layer A AppTest harness.
    """

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "main.py")
        with open(path, encoding="utf-8", newline="") as handle:
            cls.source = handle.read().replace("\r\n", "\n")

    def test_current_analysis_selector_label_uses_the_approved_terms(self):
        self.assertIn(
            'selector_label = "Select a Not Demonstrated or Partially Demonstrated Skill" '
            'if skills_state == "ok" else "Select a Missing Skill"',
            self.source)
        self.assertIn("st.selectbox(selector_label, list(recommendation_options)", self.source)

    def test_legacy_wording_is_preserved_exactly(self):
        self.assertEqual(self.source.count('"Select a Missing Skill"'), 1)
        self.assertIn('        elif skills_state == "legacy":\n'
                      '            st.info("No missing skills found. Analyze a resume first.")\n', self.source)

    def test_current_analysis_empty_state_explains_that_no_eligible_skill_was_found(self):
        self.assertNotIn("No missing or partial technical skills", self.source)
        self.assertIn(
            'st.info("No eligible technical skills were identified for recommendations: none were Not '
            'Demonstrated or Partially Demonstrated with a verified job-description quote. This does not '
            'guarantee the resume meets the role.")', self.source)

    def test_unverified_only_message_and_skipped_caption_are_unchanged(self):
        self.assertIn('st.info("The flagged technical skills could not be verified against the job description, '
                      'so no recommendations are offered.")', self.source)
        self.assertIn('"job-description quote could not be found."', self.source)


if __name__ == "__main__":
    unittest.main()
