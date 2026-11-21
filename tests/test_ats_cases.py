"""Offline validation of the ATS evaluation dataset (eval/ats/cases/*.json) and technology vocabulary
(eval/ats/tech_vocabulary.json).

Run from the project root:  python -m unittest tests.test_ats_cases -v
Synthetic data only: no network, no API key, no model, no resume file, no protected data file.
The vocabulary is a LIMITED SCREENING AID, not an ontology and not proof of factual support.
"""
import glob
import importlib.util
import json
import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES_DIR = os.path.join(ROOT, "eval", "ats", "cases")
VOCAB_PATH = os.path.join(ROOT, "eval", "ats", "tech_vocabulary.json")

_spec = importlib.util.spec_from_file_location("ats_dataset", os.path.join(ROOT, "eval", "ats", "dataset.py"))
dataset = sys.modules.get("ats_dataset") or importlib.util.module_from_spec(_spec)
if "ats_dataset" not in sys.modules:
    sys.modules["ats_dataset"] = dataset
    _spec.loader.exec_module(dataset)
checks = dataset.checks  # the one checker module, loaded by dataset.py
render_resume = dataset.render_resume  # the shared renderer; GoldenRendering below pins it against literals

REQUIRED_CATEGORIES = {
    "strong_evidence", "vague_evidence", "absent_skill", "metric_heavy",
    "no_metric_trap", "inflation_trap", "fresher_backend_ai", "short_mismatch",
}
TRAP_TYPES = {
    "metric_preservation", "invented_metric", "absent_skill_leak", "vague_evidence", "scale_claim",
    "overstatement", "job_mismatch", "format_robustness",
}
CASE_KEYS = {
    "id", "category", "holdout", "title", "provenance", "job_description", "resume", "resume_text",
    "absent_skills", "trap_types", "traps",
}
# Generic or ambiguous words that must never be vocabulary terms or aliases.
BANNED_TERMS = {
    "go", "r", "c", "d", "it", "ai", "ml", "bi", "rest", "node", "spark", "ray", "express", "swift", "react",
    "lambda", "chroma", "transformers", "agents", "agent", "cloud", "data", "api", "apis", "web", "database",
    "framework", "model", "models", "tools", "testing", "backend", "frontend", "software", "code", "script",
}
ALLOWED_SHORT_TERMS = {"s3", "c#"}


def load_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(CASES_DIR, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            case = json.load(handle)
        cases.append((os.path.basename(path), case))
    return cases


def load_vocabulary():
    with open(VOCAB_PATH, encoding="utf-8") as handle:
        return json.load(handle)


CASES = load_cases()
VOCAB = load_vocabulary()


def source_sections(case):
    return [{"heading": s["heading"], "bullets": s["bullets"]} for s in case["resume"]["sections"]]


class DatasetShape(unittest.TestCase):
    def test_exactly_twelve_cases_with_unique_ids(self):
        self.assertEqual(len(CASES), 12)
        ids = [case["id"] for _, case in CASES]
        self.assertEqual(len(set(ids)), 12)

    def test_file_name_matches_id(self):
        for filename, case in CASES:
            self.assertEqual(filename, case["id"] + ".json")

    def test_all_required_categories_present_and_no_unknown_ones(self):
        categories = {case["category"] for _, case in CASES}
        self.assertEqual(categories, REQUIRED_CATEGORIES)

    def test_every_category_has_at_least_one_case(self):
        for category in REQUIRED_CATEGORIES:
            self.assertTrue(any(c["category"] == category for _, c in CASES), category)

    def test_exactly_three_holdouts_and_flag_is_boolean(self):
        for _, case in CASES:
            self.assertIsInstance(case["holdout"], bool)
        holdouts = [case["id"] for _, case in CASES if case["holdout"]]
        self.assertEqual(len(holdouts), 3, holdouts)

    def test_holdouts_span_different_categories(self):
        categories = [case["category"] for _, case in CASES if case["holdout"]]
        self.assertEqual(len(set(categories)), 3)

    def test_schema(self):
        for _, case in CASES:
            cid = case["id"]
            self.assertEqual(set(case), CASE_KEYS, cid)
            self.assertRegex(cid, r"^c\d{2}_[a-z0-9_]+$")
            for key in ("title", "provenance", "job_description", "resume_text"):
                self.assertIsInstance(case[key], str, (cid, key))
                self.assertTrue(case[key].strip(), (cid, key))
            self.assertTrue(case["traps"] and all(isinstance(t, str) and t.strip() for t in case["traps"]), cid)
            self.assertTrue(case["trap_types"], cid)
            self.assertTrue(set(case["trap_types"]) <= TRAP_TYPES, (cid, case["trap_types"]))
            resume = case["resume"]
            self.assertEqual(set(resume), {"name", "sections", "skills_line"}, cid)
            self.assertTrue(resume["name"].strip() and resume["skills_line"].strip(), cid)
            self.assertTrue(resume["sections"], cid)
            for section in resume["sections"]:
                self.assertTrue(set(section) <= {"heading", "dates", "bullets"} and "heading" in section, cid)
                self.assertTrue(section["heading"].strip(), cid)
                self.assertTrue(section["bullets"], cid)
                for bullet in section["bullets"]:
                    self.assertIsInstance(bullet, str)
                    self.assertTrue(bullet.strip() and "\n" not in bullet, (cid, bullet))
                    self.assertEqual(bullet, bullet.strip(), (cid, bullet))

    def test_section_headings_unique_and_not_substrings_of_each_other(self):
        for _, case in CASES:
            headings = [s["heading"].lower() for s in case["resume"]["sections"]]
            self.assertEqual(len(set(headings)), len(headings), case["id"])
            for i, a in enumerate(headings):
                for j, b in enumerate(headings):
                    if i != j:
                        self.assertNotIn(a, b, (case["id"], a, b))

    def test_no_duplicate_source_bullets_within_a_case(self):
        for _, case in CASES:
            bullets = [b.lower() for s in case["resume"]["sections"] for b in s["bullets"]]
            self.assertEqual(len(set(bullets)), len(bullets), case["id"])

    def test_case_and_trap_text_is_ascii(self):
        for filename, _ in CASES:
            with open(os.path.join(CASES_DIR, filename), encoding="utf-8") as handle:
                raw = handle.read()
            raw.encode("ascii")  # raises on non-ASCII


class Rendering(unittest.TestCase):
    def test_stored_resume_text_equals_renderer_output(self):
        for _, case in CASES:
            self.assertEqual(case["resume_text"], render_resume(case["resume"]), case["id"])

    def test_rendering_is_deterministic(self):
        for _, case in CASES:
            first = render_resume(case["resume"])
            self.assertEqual(first, render_resume(json.loads(json.dumps(case["resume"]))))
            self.assertEqual(first, render_resume(case["resume"]))

    def test_rendered_text_contains_every_heading_bullet_and_skills_line(self):
        for _, case in CASES:
            text = case["resume_text"]
            for section in case["resume"]["sections"]:
                self.assertIn(section["heading"], text)
                for bullet in section["bullets"]:
                    self.assertIn("- " + bullet + "\n", text)
            self.assertIn("Skills: " + case["resume"]["skills_line"], text)

    def test_rendering_uses_lf_newlines(self):
        for _, case in CASES:
            self.assertNotIn("\r", case["resume_text"])
            self.assertTrue(case["resume_text"].endswith("\n"))

    def test_render_small_example_exactly(self):
        resume = {"name": "X", "sections": [{"heading": "H", "dates": "2024", "bullets": ["a", "b"]}], "skills_line": "S"}
        self.assertEqual(render_resume(resume), "X\n\nH\n2024\n- a\n- b\n\nSkills: S\n")


class GoldenRendering(unittest.TestCase):
    """Independent of dataset.render_resume: the expected text is a literal, so the renderer is not tested only
    against itself."""

    GOLDEN_RESUME = {
        "name": "Candidate Zulu",
        "sections": [
            {"heading": "Projects - One", "dates": "Jan 2024 - Feb 2024", "bullets": ["Built A.", "Tested B."]},
            {"heading": "Work Experience - Two", "bullets": ["Fixed C."]},
        ],
        "skills_line": "Python, SQL",
    }
    GOLDEN_TEXT = (
        "Candidate Zulu\n"
        "\n"
        "Projects - One\n"
        "Jan 2024 - Feb 2024\n"
        "- Built A.\n"
        "- Tested B.\n"
        "\n"
        "Work Experience - Two\n"
        "- Fixed C.\n"
        "\n"
        "Skills: Python, SQL\n"
    )

    def test_literal_golden_output(self):
        self.assertEqual(dataset.render_resume(self.GOLDEN_RESUME), self.GOLDEN_TEXT)

    def test_dates_line_is_omitted_when_absent(self):
        self.assertNotIn("\n\nFixed", self.GOLDEN_TEXT)
        self.assertIn("Work Experience - Two\n- Fixed C.", dataset.render_resume(self.GOLDEN_RESUME))


class SharedDatasetModule(unittest.TestCase):
    def test_splits_have_expected_sizes_and_membership(self):
        dev, holdout, every = (dataset.load_cases(s) for s in ("dev", "holdout", "all"))
        self.assertEqual((len(dev), len(holdout), len(every)), (9, 3, 12))
        self.assertFalse(any(c["holdout"] for c in dev))
        self.assertTrue(all(c["holdout"] for c in holdout))
        self.assertEqual(sorted(c["id"] for c in dev + holdout), sorted(c["id"] for c in every))

    def test_default_split_is_dev(self):
        self.assertEqual([c["id"] for c in dataset.load_cases()], [c["id"] for c in dataset.load_cases("dev")])

    def test_order_is_file_name_order_and_deterministic(self):
        ids = [c["id"] for c in dataset.load_cases("all")]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, [c["id"] for c in dataset.load_cases("all")])

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            dataset.load_cases("everything")

    def test_loaded_cases_are_copies(self):
        first = dataset.load_cases("dev")
        first[0]["job_description"] = "changed"
        self.assertNotEqual(dataset.load_cases("dev")[0]["job_description"], "changed")

    def test_every_real_case_validates(self):
        for _, case in CASES:
            self.assertEqual(dataset.validate_case(case), [], case["id"])
        self.assertEqual(dataset.validate_dataset([c for _, c in CASES]), [])

    def test_validate_case_reports_each_kind_of_damage(self):
        base = json.loads(json.dumps(CASES[0][1]))

        def damaged(mutate):
            case = json.loads(json.dumps(base))
            mutate(case)
            return dataset.validate_case(case)

        self.assertTrue(damaged(lambda c: c.pop("traps")))
        self.assertTrue(damaged(lambda c: c.update(holdout="yes")))
        self.assertTrue(damaged(lambda c: c.update(category="invented")))
        self.assertTrue(damaged(lambda c: c.update(id="bad id")))
        self.assertTrue(damaged(lambda c: c.update(resume_text=c["resume_text"] + "x")))
        self.assertTrue(damaged(lambda c: c["resume"]["sections"][0]["bullets"].append(" padded ")))
        self.assertTrue(damaged(lambda c: c.update(absent_skills=[{"name": "Python", "aliases": []}])))
        self.assertTrue(damaged(lambda c: c.update(absent_skills=[{"name": "Nope", "aliases": "x"}])))
        self.assertTrue(damaged(lambda c: c["resume"]["sections"].append(
            {"heading": "Projects", "bullets": ["x"]})))
        self.assertEqual(dataset.validate_case("not a dict"), ["case is not a JSON object"])

    def test_filename_must_match_id(self):
        self.assertTrue(dataset.validate_case(CASES[0][1], "c99_other.json"))

    def test_load_cases_raises_with_every_problem_and_never_repairs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            good = json.loads(json.dumps(CASES[0][1]))
            bad = json.loads(json.dumps(CASES[1][1]))
            bad["resume_text"] += "tampered"
            for case in (good, bad):
                with open(os.path.join(folder, case["id"] + ".json"), "w", encoding="utf-8") as handle:
                    json.dump(case, handle)
            with open(os.path.join(folder, "c77_broken.json"), "w", encoding="utf-8") as handle:
                handle.write("{not json")
            with self.assertRaises(dataset.DatasetError) as caught:
                dataset.load_cases("dev", cases_dir=folder, strict=False)
            message = str(caught.exception)
            self.assertIn("differs from the deterministic rendering", message)
            self.assertIn("c77_broken.json", message)

    def test_strict_mode_requires_the_full_dataset(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            case = json.loads(json.dumps(CASES[0][1]))
            with open(os.path.join(folder, case["id"] + ".json"), "w", encoding="utf-8") as handle:
                json.dump(case, handle)
            with self.assertRaises(dataset.DatasetError):
                dataset.load_cases("all", cases_dir=folder)
            self.assertEqual(len(dataset.load_cases("all", cases_dir=folder, strict=False)), 1)

    def test_load_vocabulary_returns_only_the_terms_mapping(self):
        terms = dataset.load_vocabulary()
        self.assertEqual(terms, VOCAB["terms"])
        self.assertNotIn("meta", terms)
        self.assertNotIn("terms", terms)
        self.assertTrue(all(isinstance(v, list) for v in terms.values()))

    def test_load_vocabulary_rejects_malformed_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            def write(obj):
                path = os.path.join(folder, "v.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(obj, handle)
                return path
            for bad in ({"terms": {"A": []}}, {"meta": {}, "terms": []},
                        {"meta": {}, "terms": {"Python": ["py"], "Pyth": ["PY"]}},
                        {"meta": {}, "terms": {"Python": "py"}}, {"meta": {}, "terms": {}}):
                with self.assertRaises(dataset.DatasetError, msg=bad):
                    dataset.load_vocabulary(write(bad))

    def test_vocabulary_terms_work_with_check_technologies(self):
        result = checks.check_technologies(["Used Redis"], "Python", dataset.load_vocabulary())
        self.assertEqual(result["results"][0]["unsupported"], ["Redis"])

    def test_source_views(self):
        case = CASES[0][1]
        self.assertEqual([s["heading"] for s in dataset.source_sections(case)],
                         [s["heading"] for s in case["resume"]["sections"]])
        self.assertEqual(len(dataset.source_bullets(case)), sum(len(s["bullets"]) for s in case["resume"]["sections"]))


class Fingerprints(unittest.TestCase):
    def test_shape_and_stability(self):
        first, second = dataset.dataset_fingerprint("dev"), dataset.dataset_fingerprint("dev")
        self.assertEqual(first, second)
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(first["covers"]), 9)
        self.assertEqual(len(dataset.dataset_fingerprint("holdout")["covers"]), 3)
        self.assertEqual(len(dataset.dataset_fingerprint("all")["covers"]), 12)
        self.assertEqual([c["path"] for c in first["covers"]], sorted(c["path"] for c in first["covers"]))

    def test_splits_have_different_fingerprints(self):
        prints = {s: dataset.dataset_fingerprint(s)["sha256"] for s in ("dev", "holdout", "all")}
        self.assertEqual(len(set(prints.values())), 3)

    def test_covered_content_change_changes_the_fingerprint_and_other_split_does_not(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            copy_dir = os.path.join(folder, "cases")
            shutil.copytree(dataset.CASES_DIR, copy_dir)
            before_dev = dataset.dataset_fingerprint("dev", copy_dir)["sha256"]
            before_holdout = dataset.dataset_fingerprint("holdout", copy_dir)["sha256"]
            target = os.path.join(copy_dir, "c02_strong_ml_pipeline.json")  # a holdout file
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(" ")
            self.assertEqual(dataset.dataset_fingerprint("dev", copy_dir)["sha256"], before_dev)
            self.assertNotEqual(dataset.dataset_fingerprint("holdout", copy_dir)["sha256"], before_holdout)

    def test_line_endings_are_normalised(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            lf, crlf = os.path.join(folder, "a"), os.path.join(folder, "b")
            with open(lf, "wb") as handle:
                handle.write(b"x\ny\n")
            with open(crlf, "wb") as handle:
                handle.write(b"x\r\ny\r\n")
            self.assertEqual(dataset.sha256_file(lf), dataset.sha256_file(crlf))

    def test_vocabulary_and_checker_fingerprints(self):
        import shutil
        import tempfile
        self.assertEqual(dataset.vocabulary_fingerprint(), dataset.vocabulary_fingerprint())
        self.assertEqual(dataset.checker_fingerprint()["covers"][0]["path"], "checks.py")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "v.json")
            shutil.copy(dataset.VOCAB_PATH, path)
            before = dataset.vocabulary_fingerprint(path)["sha256"]
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertNotEqual(dataset.vocabulary_fingerprint(path)["sha256"], before)

    def test_combined_hash_depends_on_labels(self):
        self.assertNotEqual(dataset.combine_fingerprint([("a", "1" * 64)])["sha256"],
                            dataset.combine_fingerprint([("b", "1" * 64)])["sha256"])


class AbsentSkills(unittest.TestCase):
    def test_declared_absent_skills_are_valid_for_every_case(self):
        for _, case in CASES:
            problems = checks.validate_absent_skills(case["absent_skills"], case["resume_text"], case["job_description"])
            self.assertEqual(problems, [], case["id"])

    def test_every_case_declares_at_least_one_absent_skill(self):
        for _, case in CASES:
            self.assertTrue(case["absent_skills"], case["id"])

    def test_absent_skill_entries_have_name_and_alias_list(self):
        for _, case in CASES:
            for skill in case["absent_skills"]:
                self.assertEqual(set(skill), {"name", "aliases"}, case["id"])
                self.assertIsInstance(skill["aliases"], list)

    def test_absent_names_and_aliases_not_in_resume_but_in_job_description(self):
        for _, case in CASES:
            for skill in case["absent_skills"]:
                terms = [skill["name"]] + skill["aliases"]
                self.assertFalse(any(checks._term_in(case["resume_text"], t) for t in terms), (case["id"], skill))
                self.assertTrue(any(checks._term_in(case["job_description"], t) for t in terms), (case["id"], skill))

    def test_absent_skills_unique_within_a_case(self):
        for _, case in CASES:
            names = [s["name"].lower() for s in case["absent_skills"]]
            self.assertEqual(len(set(names)), len(names), case["id"])

    def test_validator_would_catch_a_present_skill(self):
        case = CASES[0][1]
        problems = checks.validate_absent_skills([{"name": "Python", "aliases": []}], case["resume_text"], case["job_description"])
        self.assertTrue(problems)


class PrivacyAndProvenance(unittest.TestCase):
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
    URL = re.compile(r"(https?://|www\.|linkedin\.com|github\.com/|\.com\b|\.in\b|\.io\b)", re.IGNORECASE)

    def _blobs(self, case):
        yield case["resume_text"]
        yield case["job_description"]
        yield "\n".join(case["traps"])
        yield case["title"]

    def test_no_emails_phones_or_urls(self):
        for _, case in CASES:
            for blob in self._blobs(case):
                self.assertIsNone(self.EMAIL.search(blob), case["id"])
                self.assertIsNone(self.URL.search(blob), case["id"])
                for match in self.PHONE.finditer(blob):
                    digits = re.sub(r"\D", "", match.group())
                    self.assertLess(len(digits), 10, (case["id"], match.group()))

    def test_candidate_names_are_placeholders(self):
        for _, case in CASES:
            self.assertRegex(case["resume"]["name"], r"^Candidate [A-Z][a-z]+$", case["id"])

    def test_provenance_states_synthetic(self):
        for _, case in CASES:
            self.assertIn("Not copied from any real", case["provenance"])

    def test_no_street_address_or_date_of_birth_markers(self):
        for _, case in CASES:
            low = case["resume_text"].lower()
            for marker in ("date of birth", "dob:", "address:", "phone:", "email:", "mobile:"):
                self.assertNotIn(marker, low, case["id"])


class VocabularyShape(unittest.TestCase):
    def test_top_level_keys_and_meta_documents_limits(self):
        self.assertEqual(set(VOCAB), {"meta", "terms"})
        meta = VOCAB["meta"]
        self.assertIn("LIMITED SCREENING AID", meta["status"])
        self.assertIn("not proof", meta["status"])
        for key in ("purpose", "matching", "alias_policy", "known_limitations", "groups"):
            self.assertIn(key, meta)

    def test_term_count_is_about_one_hundred_twenty(self):
        terms = VOCAB["terms"]
        total = len(terms) + sum(len(v) for v in terms.values())
        self.assertGreaterEqual(len(terms), 100)
        self.assertLessEqual(len(terms), 140)
        self.assertGreaterEqual(total, 150)

    def test_entries_are_well_formed(self):
        for name, aliases in VOCAB["terms"].items():
            self.assertIsInstance(name, str)
            self.assertEqual(name, name.strip())
            self.assertTrue(name)
            self.assertIsInstance(aliases, list, name)
            for alias in aliases:
                self.assertIsInstance(alias, str, name)
                self.assertEqual(alias, alias.strip(), (name, alias))
                self.assertTrue(alias, name)
                self.assertNotEqual(alias.lower(), name.lower(), (name, alias))

    def test_terms_are_ascii_without_newlines(self):
        for name, aliases in VOCAB["terms"].items():
            for term in [name] + aliases:
                term.encode("ascii")
                self.assertNotIn("\n", term)

    def test_every_term_and_alias_is_unique_case_insensitively(self):
        seen = {}
        for name, aliases in VOCAB["terms"].items():
            for term in [name] + aliases:
                key = term.lower()
                self.assertNotIn(key, seen, f"{term!r} in {name!r} duplicates {seen.get(key)!r}")
                seen[key] = name

    def test_no_alias_is_only_a_plural_or_case_of_another_entrys_name(self):
        names = {n.lower() for n in VOCAB["terms"]}
        for name, aliases in VOCAB["terms"].items():
            for alias in aliases:
                self.assertNotIn(alias.lower(), names - {name.lower()}, (name, alias))

    def test_no_banned_generic_or_ambiguous_terms(self):
        for name, aliases in VOCAB["terms"].items():
            for term in [name] + aliases:
                self.assertNotIn(term.lower(), BANNED_TERMS, (name, term))

    def test_short_terms_are_limited_to_the_allowlist(self):
        for name, aliases in VOCAB["terms"].items():
            for term in [name] + aliases:
                if len(term) < 3:
                    self.assertIn(term.lower(), ALLOWED_SHORT_TERMS, term)

    def test_groups_cover_exactly_the_terms(self):
        grouped = [t for members in VOCAB["meta"]["groups"].values() for t in members]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(VOCAB["terms"]))

    def test_backend_ai_core_terms_present(self):
        names = {n.lower() for n in VOCAB["terms"]}
        for required in ("python", "fastapi", "sql", "postgresql", "docker", "llm api", "rag", "embeddings",
                         "langchain", "faiss", "pinecone", "pgvector", "kubernetes", "kafka", "redis", "pytest"):
            self.assertIn(required, names)

    def test_every_case_absent_skill_resolves_to_a_vocabulary_term_when_it_is_a_known_technology(self):
        terms = {t.lower() for n, a in VOCAB["terms"].items() for t in [n] + a}
        missing = [(c["id"], s["name"]) for _, c in CASES for s in c["absent_skills"] if s["name"].lower() not in terms]
        self.assertEqual(missing, [], "absent skills should be vocabulary terms so D4 also sees them")


class VocabularyMatching(unittest.TestCase):
    vocab = VOCAB["terms"]

    def flagged(self, bullet, source=""):
        result = checks.check_technologies([bullet], source, self.vocab)
        return set(result["results"][0]["unsupported"])

    def test_case_insensitive(self):
        self.assertEqual(self.flagged("built it with FASTAPI"), {"FastAPI"})
        self.assertEqual(self.flagged("built it with fastapi"), {"FastAPI"})

    def test_alias_counts_as_support_for_canonical(self):
        self.assertEqual(self.flagged("stored data in PostgreSQL", "Skills: Postgres"), set())
        self.assertEqual(self.flagged("ran on Kubernetes", "used k8s daily"), set())
        self.assertEqual(self.flagged("Kafka consumers", "Apache Kafka"), set())

    def test_alias_in_bullet_is_flagged_as_canonical(self):
        self.assertEqual(self.flagged("deployed on k8s"), {"Kubernetes"})
        self.assertEqual(self.flagged("used retrieval-augmented generation"), {"RAG"})
        self.assertEqual(self.flagged("used sklearn"), {"scikit-learn"})

    def test_java_does_not_match_javascript(self):
        self.assertEqual(self.flagged("wrote JavaScript widgets"), {"JavaScript"})
        self.assertEqual(self.flagged("wrote Java services"), {"Java"})
        self.assertEqual(self.flagged("wrote JavaScript widgets", "Java"), {"JavaScript"})

    def test_sql_does_not_match_nosql_or_postgresql_support_sql(self):
        self.assertEqual(self.flagged("queried NoSQL stores"), set())
        self.assertEqual(self.flagged("tuned SQL queries"), {"SQL"})
        self.assertEqual(self.flagged("tuned SQL queries", "PostgreSQL"), {"SQL"})

    def test_git_does_not_match_github_or_gitlab(self):
        self.assertEqual(self.flagged("used Git daily", "GitHub"), {"Git"})
        self.assertEqual(self.flagged("used GitHub daily", "Git"), {"GitHub"})
        self.assertEqual(self.flagged("used GitLab runners", "GitHub"), {"GitLab"})

    def test_docker_and_docker_compose_are_separate_entries(self):
        self.assertEqual(self.flagged("ran Docker Compose", "Docker"), {"Docker Compose"})
        self.assertEqual(self.flagged("ran Docker", "Docker Compose"), set())  # 'Docker' is word-bounded inside 'Docker Compose'

    def test_redis_does_not_match_redistribute(self):
        self.assertEqual(self.flagged("redistribute the load"), set())
        self.assertEqual(self.flagged("cached results in Redis"), {"Redis"})

    def test_ambiguous_words_are_not_technologies(self):
        for sentence in ("Go to market with data", "Used R and C for coursework", "Express the results clearly",
                         "Reacted to feedback", "Spark interest in the team", "A cloud of agents", "the REST of the work"):
            self.assertEqual(self.flagged(sentence), set(), sentence)

    def test_punctuation_boundaries(self):
        self.assertEqual(self.flagged("(Python), Docker; and Redis."), {"Python", "Docker", "Redis"})
        self.assertEqual(self.flagged("CI/CD with Git/GitHub"), {"CI/CD", "Git", "GitHub"})

    def test_special_character_terms(self):
        self.assertEqual(self.flagged("Wrote C++ and C# code"), {"C++", "C#"})
        self.assertEqual(self.flagged("Built a Node.js service"), {"Node.js"})
        self.assertEqual(self.flagged("Weights & Biases dashboards"), {"Weights & Biases"})

    def test_hyphen_and_spacing_variants_need_aliases(self):
        self.assertEqual(self.flagged("fine-tuning a model"), {"fine-tuning"})
        self.assertEqual(self.flagged("fine tuning a model"), {"fine-tuning"})
        self.assertEqual(self.flagged("REST APIs and RESTful endpoints"), {"REST API"})

    def test_plural_forms_match_only_when_listed(self):
        self.assertEqual(self.flagged("stored embedding vectors"), {"embeddings"})
        self.assertEqual(self.flagged("used LLMs"), {"LLM"})
        self.assertEqual(self.flagged("wrote Dockerfiles"), {"Docker"})

    def test_digit_neighbours_are_not_boundaries(self):
        self.assertEqual(self.flagged("used Python3 daily"), set())
        self.assertEqual(self.flagged("used Python 3 daily"), {"Python"})

    def test_s3_is_matched_exactly(self):
        self.assertEqual(self.flagged("stored files in S3"), {"S3"})
        self.assertEqual(self.flagged("the S3x format"), set())

    def test_normalization_of_whitespace_and_case_in_source_support(self):
        self.assertEqual(self.flagged("Used PYTORCH", "pytorch"), set())

    def test_scratch_vocabulary_checks_flag_only_unsupported(self):
        result = checks.check_technologies(["Used Python and Redis", "Used Python"], "Skills: Python", self.vocab)
        self.assertEqual(result["bullets_with_unsupported"], 1)
        self.assertEqual(result["results"][0]["unsupported"], ["Redis"])
        self.assertEqual(result["results"][1]["unsupported"], [])

    def test_source_resumes_have_no_unsupported_vocabulary_terms_by_construction(self):
        for _, case in CASES:
            bullets = [b for s in case["resume"]["sections"] for b in s["bullets"]]
            result = checks.check_technologies(bullets, case["resume_text"], self.vocab)
            self.assertEqual(result["bullets_with_unsupported"], 0, case["id"])


if __name__ == "__main__":
    unittest.main()
