"""Offline tests for the deterministic ATS bullet checks (eval/ats/checks.py).

Run from the project root:  python -m unittest tests.test_ats_checks -v
Synthetic strings only: no network, no API key, no model, no resume file, no protected data file.
The checks are screening signals; these tests pin their documented behaviour, including known blind spots
(a test named *_blind_spot or *_limitation pins behaviour that is deliberately NOT a detection).
"""
import ast
import importlib.util
import json
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKS_PATH = os.path.join(ROOT, "eval", "ats", "checks.py")

_spec = importlib.util.spec_from_file_location("ats_checks_under_test", CHECKS_PATH)
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)


def output(*sections):
    """A well-formed model response: sections given as (heading, [bullets])."""
    return json.dumps({"optimized_bullets": [{"section": h, "bullets": b} for h, b in sections]})


class ModuleIsolation(unittest.TestCase):
    def setUp(self):
        with open(CHECKS_PATH, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_only_standard_library_modules_are_imported(self):
        names = set()
        for node in ast.walk(ast.parse(self.source)):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertEqual(names, {"json", "re", "decimal"})

    def test_no_file_network_or_protected_file_access(self):
        tree = ast.parse(self.source)
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for forbidden in ("open", "socket", "urlopen", "requests", "parse_resume", "save_text_to_file", "environ"):
            self.assertNotIn(forbidden, used, forbidden)
        strings = " ".join(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                           and isinstance(node.value, str) and node is not tree.body[0].value)
        for protected in ("resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                          "missing_skills.txt", "ats_optimized_bullets.txt", ".env"):
            self.assertNotIn(protected, strings, protected)

    def test_syntax_is_valid_for_python_3_9(self):
        ast.parse(self.source, feature_version=(3, 9))


class SchemaCheck(unittest.TestCase):
    def test_valid_output_is_ok_and_keeps_sections(self):
        parsed = checks.parse_ats_output(output(("Projects - Task API", ["Built an API."])))
        self.assertEqual(parsed["status"], "ok")
        self.assertFalse(parsed["production_would_500"])
        self.assertEqual(parsed["sections"], [{"section": "Projects - Task API", "bullets": ["Built an API."]}])

    def test_code_fences_are_stripped_like_production(self):
        fenced = "```json\n" + output(("S", ["Did a thing."])) + "\n```"
        self.assertEqual(checks.parse_ats_output(fenced)["status"], "ok")

    def test_invalid_json_is_flagged_and_would_be_a_production_500(self):
        for bad in ("not json at all", "", "   ", '{"optimized_bullets": [', None, 5):
            with self.subTest(bad=bad):
                parsed = checks.parse_ats_output(bad)
                self.assertEqual(parsed["status"], "invalid_json")
                self.assertTrue(parsed["production_would_500"])
                self.assertEqual(parsed["sections"], [])

    def test_wrong_shape_is_flagged_and_is_not_a_500(self):
        for value in ('[]', '"text"', '{}', '{"optimized_bullets": "x"}', '{"other": []}', '3'):
            with self.subTest(value=value):
                parsed = checks.parse_ats_output(value)
                self.assertEqual(parsed["status"], "wrong_shape")
                self.assertFalse(parsed["production_would_500"])

    def test_empty_optimized_bullets_is_its_own_status(self):
        self.assertEqual(checks.parse_ats_output('{"optimized_bullets": []}')["status"], "empty")

    def test_bad_items_are_wrong_shape(self):
        cases = {
            "item not an object": '{"optimized_bullets": ["x"]}',
            "blank section": '{"optimized_bullets": [{"section": "  ", "bullets": ["a b"]}]}',
            "missing section": '{"optimized_bullets": [{"bullets": ["a b"]}]}',
            "empty bullets": '{"optimized_bullets": [{"section": "S", "bullets": []}]}',
            "bullets not a list": '{"optimized_bullets": [{"section": "S", "bullets": "a"}]}',
            "non-string bullet": '{"optimized_bullets": [{"section": "S", "bullets": ["ok", 5]}]}',
            "blank bullet": '{"optimized_bullets": [{"section": "S", "bullets": ["ok", " "]}]}',
        }
        for name, value in cases.items():
            with self.subTest(name):
                parsed = checks.parse_ats_output(value)
                self.assertEqual(parsed["status"], "wrong_shape")
                self.assertTrue(parsed["problems"])

    def test_problem_messages_never_contain_the_model_text(self):
        parsed = checks.parse_ats_output('{"optimized_bullets": [{"section": "SECRET-HEADING", "bullets": [5]}]}')
        self.assertNotIn("SECRET-HEADING", " ".join(parsed["problems"]))


class NumberExtraction(unittest.TestCase):
    def canon(self, text):
        return [canonical for _raw, canonical in checks.extract_numbers(text)]

    def test_forms_are_normalised(self):
        self.assertEqual(self.canon("1,000 users"), ["1000"])
        self.assertEqual(self.canon("10k requests"), ["10000"])
        self.assertEqual(self.canon("2M rows"), ["2000000"])
        self.assertEqual(self.canon("35% and 35 %"), ["35", "35"])
        self.assertEqual(self.canon("3.50 and 3.5"), ["3.5", "3.5"])
        self.assertEqual(self.canon("a 3.5x speedup"), ["3.5"])

    def test_versions_and_years_are_numbers(self):
        self.assertEqual(self.canon("Python 3.11 and 3.11.2"), ["3.11", "3.11.2"])
        self.assertEqual(self.canon("Jan 2020 - Present"), ["2020"])
        self.assertEqual(self.canon("5-10 services in 2024-2025"), ["5", "10", "2024", "2025"])

    def test_technology_names_with_digits_are_not_numbers(self):
        for text in ("S3 bucket", "OAuth2 login", "GPT-4o model", "GPT-4 model", "Llama-3 model", "Neo4j graph",
                     "v2 API", "Top-3 accuracy", "COVID-19 data", "ISO-27001 audit", "p99 latency", "Python3 script"):
            with self.subTest(text=text):
                self.assertEqual(checks.extract_numbers(text), [])

    def test_spaced_technology_versions_are_numbers_blind_spot(self):
        # documented: only glued / hyphenated names are protected, so "GPT 4" and "Python 3" are read as numbers
        self.assertEqual(self.canon("GPT 4 and Llama 3"), ["4", "3"])
        self.assertEqual(self.canon("S3 plus 3 buckets"), ["3"])

    def test_leading_list_marker_is_ignored(self):
        self.assertEqual(self.canon("1. Built it"), [])
        self.assertEqual(self.canon("2) Built 3 things"), ["3"])
        self.assertEqual(self.canon("- Built 3 things"), ["3"])

    def test_sentence_final_period_is_not_part_of_the_number(self):
        self.assertEqual(self.canon("Reduced load by 35."), ["35"])

    def test_non_string_input_gives_nothing(self):
        self.assertEqual(checks.extract_numbers(None), [])
        self.assertEqual(checks.extract_numbers(12), [])
        self.assertEqual(checks.extract_quantities(None), [])


class InventedNumbers(unittest.TestCase):
    SOURCE = ("Built a REST API serving 1,000 requests per day. Cut report time from 10 minutes to 2 minutes "
              "(2023). Used Python 3.11 and three microservices.")

    def check(self, bullets, source=None, example=()):
        return checks.check_invented_numbers(bullets, source or self.SOURCE, example)

    def test_numbers_present_in_the_source_pass(self):
        result = self.check(["Cut report time from 10 minutes to 2 minutes.", "Served 1000 requests daily in 2023."])
        self.assertEqual(result["bullets_with_invented"], 0)

    def test_equivalent_formats_pass(self):
        result = self.check(["Served 1k requests per day."])  # 1k == 1,000
        self.assertEqual(result["bullets_with_invented"], 0)

    def test_an_invented_percentage_fails(self):
        result = self.check(["Cut report time by 80%."])
        self.assertEqual(result["bullets_with_invented"], 1)
        self.assertEqual(result["results"][0]["invented"], ["80%"])

    def test_derived_numbers_count_as_invented(self):
        # "from 10 to 2 minutes" does not state "80%" or "5x"; derived figures are invented unless in the source
        result = self.check(["Made reports 5x faster."])
        self.assertEqual(result["results"][0]["invented"], ["5x"])

    def test_unstated_year_and_version_fail_but_stated_ones_pass(self):
        result = self.check(["Shipped in 2024 on Python 3.12.", "Shipped in 2023 on Python 3.11."])
        self.assertEqual(result["results"][0]["invented"], ["2024", "3.12"])
        self.assertEqual(result["results"][1]["invented"], [])

    def test_number_words_in_the_source_allow_their_digits(self):
        self.assertEqual(self.check(["Built 3 microservices."])["bullets_with_invented"], 0)
        self.assertEqual(self.check(["Built 4 microservices."])["bullets_with_invented"], 1)

    def test_glued_tokens_are_a_documented_blind_spot(self):
        # S3 / OAuth2 are never extracted, so an invented technology name with a digit is not caught by D2
        self.assertEqual(self.check(["Stored files in S3 with OAuth2."])["bullets_with_invented"], 0)

    def test_prompt_example_echo_is_reported_separately(self):
        result = self.check(["Reduced latency by 35%.", "Reduced latency by 41%."], example={"35"})
        self.assertEqual(result["bullets_with_invented"], 2)
        self.assertEqual(result["bullets_with_example_echo"], 1)
        self.assertEqual(result["results"][0]["echo"], ["35%"])

    def test_example_numbers_come_from_the_prompts_example_block_only(self):
        prompt = ("Rules: keep 99 items.\n**JSON Output Format Example**\n"
                  '{"bullets": ["cut by 35%", "team of 5"]}\nInputs\nJob Description: {job_description}')
        self.assertEqual(checks.example_numbers_from_prompt(prompt), {"35", "5"})
        self.assertEqual(checks.example_numbers_from_prompt("no example here 35"), set())

    def test_quantity_words_are_advisory_and_not_counted(self):
        result = self.check(["Handled dozens of requests with five workers and two caches."])
        entry = result["results"][0]
        self.assertEqual(result["bullets_with_invented"], 0)   # never part of the invented-number count
        self.assertIn("dozens", entry["quantity_words"])        # not stated in the source
        self.assertIn("five", entry["quantity_words"])          # 5 is not in the source
        self.assertNotIn("two", entry["quantity_words"])        # 2 is stated in the source

    def test_non_string_bullets_are_counted_as_invalid_not_checked(self):
        result = self.check(["Valid bullet.", None, 5])
        self.assertEqual(result["bullets_checked"], 1)
        self.assertEqual(result["bullets_invalid"], 2)


class AbsentSkills(unittest.TestCase):
    ABSENT = [{"name": "Kubernetes", "aliases": ["k8s"]}, {"name": "Terraform", "aliases": []}]

    def test_name_and_alias_mentions_are_leaks_case_insensitively(self):
        result = checks.check_absent_skills(["Deployed on KUBERNETES.", "Ran workloads on k8s.", "Used Docker."],
                                            self.ABSENT)
        self.assertEqual([r["leaked"] for r in result["results"]], [["Kubernetes"], ["Kubernetes"], []])
        self.assertEqual(result["bullets_with_leak"], 2)
        self.assertEqual(result["bullets_checked"], 3)

    def test_matching_respects_word_boundaries(self):
        absent = [{"name": "Java", "aliases": []}]
        result = checks.check_absent_skills(["Wrote JavaScript services.", "Used Java 17."], absent)
        self.assertEqual([r["leaked"] for r in result["results"]], [[], ["Java"]])

    def test_no_declared_skills_means_no_leaks(self):
        self.assertEqual(checks.check_absent_skills(["Used Kubernetes."], [])["bullets_with_leak"], 0)

    def test_dataset_validation_accepts_a_good_declaration(self):
        problems = checks.validate_absent_skills(
            self.ABSENT, "Built APIs with Docker.", "We need Kubernetes (k8s) and Terraform experience.")
        self.assertEqual(problems, [])

    def test_dataset_validation_rejects_bad_declarations(self):
        jd = "Requires Kubernetes and Go."
        self.assertTrue(checks.validate_absent_skills(
            [{"name": "Kubernetes", "aliases": []}], "Ran Kubernetes locally.", jd))        # present in resume
        self.assertTrue(checks.validate_absent_skills(
            [{"name": "Terraform", "aliases": []}], "Built APIs.", jd))                       # not requested
        self.assertTrue(checks.validate_absent_skills(
            [{"name": "Go", "aliases": []}], "Built APIs.", jd))                              # ambiguous short name
        self.assertTrue(checks.validate_absent_skills([{"name": " "}], "x", jd))              # blank name
        self.assertTrue(checks.validate_absent_skills(
            [{"name": "Kubernetes", "aliases": []}, {"name": "kubernetes", "aliases": []}], "x", jd))  # duplicate


class Technologies(unittest.TestCase):
    VOCAB = {"PostgreSQL": ["Postgres"], "Docker": [], "Redis": [], "FastAPI": []}

    def test_a_technology_missing_from_the_source_is_flagged(self):
        result = checks.check_technologies(["Cached results in Redis."], "Built APIs with FastAPI and Docker.",
                                           self.VOCAB)
        self.assertEqual(result["results"][0]["unsupported"], ["Redis"])
        self.assertEqual(result["bullets_with_unsupported"], 1)

    def test_supported_technologies_and_aliases_pass(self):
        source = "Stored data in Postgres, shipped with Docker."
        result = checks.check_technologies(["Modelled data in PostgreSQL.", "Containerised with Docker."], source,
                                           self.VOCAB)
        self.assertEqual(result["bullets_with_unsupported"], 0)

    def test_terms_outside_the_vocabulary_are_not_seen(self):
        result = checks.check_technologies(["Used Cassandra."], "Built APIs.", self.VOCAB)
        self.assertEqual(result["bullets_with_unsupported"], 0)  # documented false negative


class Sections(unittest.TestCase):
    SOURCE = [
        {"heading": "Projects - Task API", "bullets": ["a", "b", "c"]},
        {"heading": "Projects - Log Analyzer", "bullets": ["d", "e"]},
    ]

    def run_check(self, *output_sections):
        return checks.check_sections([{"section": h, "bullets": b} for h, b in output_sections], self.SOURCE)

    def test_matching_ignores_case_and_whitespace_and_extra_text(self):
        result = self.run_check(("projects  -  task api (2024)", ["x", "y", "z"]), ("Projects - Log Analyzer", ["p", "q"]))
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["unmatched_output"], [])
        self.assertEqual(result["missing_from_output"], [])
        self.assertEqual(result["bullet_count_changes"], [])

    def test_unknown_missing_and_duplicate_sections_are_reported(self):
        result = self.run_check(("Projects - Task API", ["x", "y"]), ("Projects - Task API", ["z"]), ("Skills", ["w"]))
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["unmatched_output"], ["Skills"])
        self.assertEqual(result["duplicate_groups"], [{"section": "Projects - Task API", "heading": "Projects - Task API"}])
        self.assertEqual(result["missing_from_output"], ["Projects - Log Analyzer"])
        self.assertEqual(result["bullet_count_changes"], [])  # 2 + 1 bullets across both groups = 3 in the source

    def test_changed_bullet_counts_are_reported(self):
        result = self.run_check(("Projects - Task API", ["x", "y"]), ("Projects - Log Analyzer", ["p", "q", "r"]))
        self.assertEqual(result["bullet_count_changes"], [
            {"heading": "Projects - Task API", "source_count": 3, "output_count": 2},
            {"heading": "Projects - Log Analyzer", "source_count": 2, "output_count": 3},
        ])


class DuplicatesAndUnchanged(unittest.TestCase):
    def test_identical_and_near_identical_bullets_are_duplicates(self):
        bullets = ["Built a REST API with FastAPI.", "Built a REST API with FastAPI!", "Wrote unit tests for the parser."]
        pairs = checks.find_duplicates(bullets)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [(0, 1)])

    def test_clearly_different_bullets_are_not_duplicates(self):
        self.assertEqual(checks.find_duplicates(["Built an API.", "Trained a classifier on tabular data."]), [])

    def test_threshold_is_respected(self):
        a, b = "one two three four", "one two three five"  # Jaccard 3/5 = 0.6
        self.assertEqual(checks.find_duplicates([a, b], threshold=0.8), [])
        self.assertEqual(len(checks.find_duplicates([a, b], threshold=0.5)), 1)

    def test_unchanged_copies_are_found_after_normalisation(self):
        source = ["Built a REST API with FastAPI."]
        rewritten = ["  built a REST API with fastapi ", "Designed a REST API with FastAPI and Docker."]
        self.assertEqual(checks.find_unchanged(rewritten, source), [0])


class EvaluateOutput(unittest.TestCase):
    SOURCE_TEXT = "Built a REST API with FastAPI serving 1,000 requests per day."
    SOURCE_SECTIONS = [{"heading": "Projects - Task API", "bullets": ["Built a REST API with FastAPI serving 1,000 requests per day."]}]
    ABSENT = [{"name": "Kubernetes", "aliases": []}]
    VOCAB = {"FastAPI": [], "Redis": []}

    def evaluate(self, raw):
        return checks.evaluate_output(raw, self.SOURCE_TEXT, self.SOURCE_SECTIONS, self.ABSENT, self.VOCAB, {"35"})

    def test_good_output_has_no_flags(self):
        raw = output(("Projects - Task API", ["Developed a REST API in FastAPI handling 1000 requests daily."]))
        result = self.evaluate(raw)
        self.assertEqual(result["schema"]["status"], "ok")
        self.assertEqual(result["bullets_total"], 1)
        self.assertEqual(result["numbers"]["bullets_with_invented"], 0)
        self.assertEqual(result["absent_skills"]["bullets_with_leak"], 0)
        self.assertEqual(result["technologies"]["bullets_with_unsupported"], 0)
        self.assertEqual(result["sections"]["matched"], 1)
        self.assertEqual(result["duplicates"], [])

    def test_problem_output_raises_every_relevant_flag_and_is_json_serialisable(self):
        raw = output(("Projects - Task API", ["Cut latency by 35% using Redis on Kubernetes."]))
        result = self.evaluate(raw)
        self.assertEqual(result["numbers"]["bullets_with_invented"], 1)
        self.assertEqual(result["numbers"]["bullets_with_example_echo"], 1)
        self.assertEqual(result["absent_skills"]["bullets_with_leak"], 1)
        self.assertEqual(result["technologies"]["bullets_with_unsupported"], 1)
        self.assertEqual(result["numbers"]["results"][0]["section"], "Projects - Task API")
        json.dumps(result)

    def test_schema_failure_skips_bullet_checks(self):
        for raw in ("garbage", '{"optimized_bullets": []}', "[]"):
            with self.subTest(raw=raw):
                result = self.evaluate(raw)
                self.assertNotEqual(result["schema"]["status"], "ok")
                self.assertEqual(result["bullets_total"], 0)
                self.assertIsNone(result["numbers"])
                self.assertIsNone(result["sections"])

    def test_optional_checks_are_skipped_when_their_inputs_are_missing(self):
        raw = output(("Projects - Task API", ["Built an API."]))
        result = checks.evaluate_output(raw, self.SOURCE_TEXT)
        self.assertIsNotNone(result["numbers"])
        for key in ("absent_skills", "technologies", "sections", "unchanged"):
            self.assertIsNone(result[key])


def values(text):
    return [q["value"] for q in checks.extract_quantities(text)]


def units(text):
    return [q["unit"] for q in checks.extract_quantities(text)]


def invented(source, bullet):
    return checks.check_invented_numbers([bullet], source)["results"][0]


class AttachedUnits(unittest.TestCase):
    """I1: whitelisted units are numbers; attached and spaced forms give the same value."""

    def test_attached_and_spaced_forms_give_the_same_value_and_unit(self):
        pairs = [("50ms", "50 ms"), ("5GB", "5 GB"), ("15min", "15 min"), ("2hrs", "2 hrs"), ("3 mins", "3mins"),
                 ("10secs", "10 secs"), ("7days", "7 days"), ("512MB", "512 MB"), ("2TB", "2 tb")]
        for attached, spaced in pairs:
            with self.subTest(attached=attached):
                self.assertEqual(values(attached), values(spaced))
                self.assertEqual(units(attached), units(spaced))
                self.assertNotEqual(units(attached), [""])

    def test_unit_labels_are_normalised_per_family(self):
        self.assertEqual(units("1 hour, 2 hrs, 3h, 4 hours"), ["h", "h", "h", "h"])
        self.assertEqual(units("1 second, 2 secs, 3s"), ["s", "s", "s"])
        self.assertEqual(units("1 minute 2 min"), ["min", "min"])
        self.assertEqual(units("50 percent and 50%"), ["%", "%"])
        self.assertEqual(units("5 GB, 5gb, 5Gb"), ["gb", "gb", "gb"])

    def test_attached_only_units_and_ordinals(self):
        self.assertEqual(units("10s 2h"), ["s", "h"])
        self.assertEqual(units("1st 2nd 3rd 95th"), ["ordinal"] * 4)
        self.assertEqual(values("1st 2nd 3rd 95th"), ["1", "2", "3", "95"])
        # a spaced single letter is not a unit: "10 s" is the number 10 with no unit
        self.assertEqual((values("10 s"), units("10 s")), (["10"], [""]))

    def test_attached_decimals(self):
        self.assertEqual(values("3.5ms 2.5GB 1.5s 0.25h"), ["3.5", "2.5", "1.5", "0.25"])

    def test_decimal_with_an_unsupported_unit_is_ignored_not_truncated(self):
        # 3.5kg must not degrade to "3"; units outside the whitelist are a documented blind spot
        for text in ("3.5kg", "60mph", "5000rps", "20Mbps", "5G", "7.5km"):
            with self.subTest(text=text):
                self.assertEqual(checks.extract_quantities(text), [])

    def test_unlisted_units_are_not_recognised_attached_or_spaced_limitation(self):
        # Attached: the whole token is ignored. Spaced: the digits are a plain number with NO unit. The two
        # spellings are therefore not consistent; this pins the documented limitation, it is not desired behaviour.
        for attached, spaced, number in (("3.5kg", "3.5 kg", "3.5"), ("60mph", "60 mph", "60")):
            with self.subTest(attached=attached):
                self.assertEqual(checks.extract_quantities(attached), [])
                self.assertEqual(values(spaced), [number])
                self.assertEqual(units(spaced), [""])

    def test_unlisted_unit_spelling_changes_the_verdict_limitation(self):
        # false positive: same quantity, different spelling
        self.assertEqual(invented("Weighed 3.5kg.", "Weighed 3.5 kg.")["invented"], ["3.5"])
        self.assertEqual(invented("Ran at 60mph.", "Ran at 60 mph.")["invented"], ["60"])
        # the reverse direction happens to pass
        self.assertEqual(invented("Weighed 3.5 kg.", "Weighed 3.5kg.")["invented"], [])
        # false negative: an invented attached figure is never seen, the spaced one is
        self.assertEqual(invented("Built an API.", "Reached 60mph.")["invented"], [])
        self.assertEqual(invented("Built an API.", "Reached 60 mph.")["invented"], ["60"])
        # a listed unit has no such asymmetry
        self.assertEqual(invented("Built an API.", "Reached 60ms.")["invented"], ["60ms"])
        self.assertEqual(invented("Built an API.", "Reached 60 ms.")["invented"], ["60 ms"])

    def test_lower_case_m_and_spaced_single_letters_are_not_read_as_million(self):
        self.assertEqual(checks.extract_quantities("5m delay"), [])
        self.assertEqual(values("5 M users, 5 k users"), ["5", "5"])  # blind spot: kept as plain 5

    def test_an_invented_unit_figure_is_now_detected(self):
        self.assertEqual(invented("Built an API.", "Reduced latency to 50ms.")["invented"], ["50ms"])
        self.assertEqual(invented("Built an API.", "Stored 5GB of logs.")["invented"], ["5GB"])
        self.assertEqual(invented("Built an API.", "Cut build time to 15min.")["invented"], ["15min"])
        self.assertEqual(invented("Built an API.", "Ranked 1st of 40.")["invented"], ["1st", "40"])

    def test_spacing_variants_no_longer_change_the_verdict(self):
        for source, bullet in (("Latency was 50ms.", "Latency fell to 50 ms."),
                               ("Latency was 50 ms.", "Latency fell to 50ms."),
                               ("Stored 5GB.", "Stored 5 GB."), ("Stored 5 GB.", "Stored 5GB.")):
            with self.subTest(source=source, bullet=bullet):
                self.assertEqual(invented(source, bullet)["invented"], [])

    def test_uppercase_multiplier_and_multiplier_decimal(self):
        self.assertEqual((values("2X faster"), units("2X faster")), (["2"], ["x"]))
        self.assertEqual(values("3.5x"), ["3.5"])


class GroupingAndMagnitudes(unittest.TestCase):
    """I2: Indian grouping and magnitude words; ambiguous formats stay conservative."""

    def test_indian_and_international_grouping_are_equal(self):
        for text, expected in (("1,00,000", "100000"), ("100,000", "100000"), ("12,34,567", "1234567"),
                               ("1,234,567", "1234567"), ("1,00,00,000", "10000000"), ("12,34,567.89", "1234567.89"),
                               ("5,000.50", "5000.5")):
            with self.subTest(text=text):
                self.assertEqual(values(text), [expected])

    def test_magnitude_words_and_suffixes_agree(self):
        groups = [("1.5 million", "1.5M", "1500000", "1,500,000"), ("2 lakh", "200k", "200000", "2,00,000"),
                  ("3 crore", "30 million", "30000000", "3,00,00,000"), ("5 thousand", "5k", "5000", "5,000"),
                  ("2 billion", "2000 million", "2000000000", "2,000,000,000")]
        for group in groups:
            with self.subTest(group=group):
                self.assertEqual(len({tuple(values(text)) for text in group}), 1, group)

    def test_magnitude_words_ignore_case_and_accept_a_plural(self):
        self.assertEqual(values("2 Lakhs, 3 CRORE, 4 Millions"), ["200000", "30000000", "4000000"])

    def test_a_magnitude_is_not_a_unit(self):
        self.assertEqual(units("1.5 million and 2 lakh and 3k"), ["", "", ""])

    def test_equivalent_forms_do_not_count_as_invented(self):
        self.assertEqual(invented("Served 1.5 million users.", "Served 1.5M users.")["invented"], [])
        self.assertEqual(invented("Served 1.5M users.", "Served 1.5 million users.")["invented"], [])
        self.assertEqual(invented("Processed 2,00,000 rows.", "Processed 200,000 rows.")["invented"], [])
        self.assertEqual(invented("Processed 2 lakh rows.", "Processed 200k rows.")["invented"], [])

    def test_a_different_magnitude_is_invented(self):
        self.assertEqual(invented("Served 1.5 million users.", "Served 15 million users.")["invented"], ["15 million"])
        self.assertEqual(invented("Served 1.5 million users.", "Served 1.5 thousand users.")["invented"], ["1.5 thousand"])

    def test_a_magnitude_word_after_a_number_is_not_also_a_vague_quantity_word(self):
        result = invented("Built an API.", "Served 2 million users and millions more.")
        self.assertEqual(result["invented"], ["2 million"])
        self.assertEqual(result["quantity_words"].count("million"), 0)
        self.assertEqual(result["quantity_words"], ["millions"])

    def test_the_word_one_is_not_a_quantity_word_but_larger_number_words_are(self):
        result = invented("Built an API.", "Led one project with seven reviewers.")
        self.assertEqual(result["quantity_words"], ["seven"])

    def test_echo_requires_the_number_to_be_invented(self):
        result = checks.check_invented_numbers(["Cut load by 35%.", "Cut load by 41%."], "Saved 35 hours and 41 minutes.",
                                               example_numbers={"35"})
        self.assertEqual(result["bullets_with_example_echo"], 0)  # 35 is in the source, so it is not an echo
        result = checks.check_invented_numbers(["Cut load by 35%."], "Built an API.", example_numbers={"35"})
        self.assertEqual(result["bullets_with_example_echo"], 1)

    def test_a_bare_magnitude_word_is_only_advisory(self):
        result = invented("Built an API.", "Served a million users.")
        self.assertEqual(result["invented"], [])
        self.assertEqual(result["quantity_words"], ["million"])

    def test_space_grouping_is_not_merged_limitation(self):
        self.assertEqual(values("10 000 users"), ["10", "0"])
        # so the same quantity written two ways is flagged for review rather than silently guessed
        self.assertEqual(invented("Handled 10 000 requests.", "Handled 10,000 requests.")["invented"], ["10,000"])

    def test_pathological_comma_runs_stay_fast(self):
        import time
        for text in ("12,34," * 7000 + "x", "123," * 10000 + "x", "1," * 20000, "9" * 40000, "1.5 " * 10000):
            with self.subTest(length=len(text)):
                started = time.perf_counter()
                checks.extract_quantities(text)
                self.assertLess(time.perf_counter() - started, 2.0)

    def test_numbers_with_more_than_six_comma_groups_are_split_limitation(self):
        self.assertEqual(values("1,234,567,890,123,456,789"), ["1234567890123456789"])  # 6 groups: still one number
        self.assertEqual(len(values("1,234,567,890,123,456,789,012")), 2)               # 7 groups: split in two

    def test_malformed_comma_groups_are_not_merged_limitation(self):
        self.assertEqual(values("1,00 and 12,34 and 1,2,3"), ["1", "0", "12", "34", "1", "2", "3"])

    def test_ambiguous_european_style_is_compared_as_written(self):
        quantity = checks.extract_quantities("1.000 users")[0]
        self.assertEqual((quantity["value"], quantity["ambiguous"]), ("1.000", True))
        self.assertEqual(invented("Spent 1.000 euros.", "Spent 1.000 euros.")["invented"], [])
        self.assertEqual(invented("Spent 1.000 euros.", "Spent 1,000 euros.")["invented"], ["1,000"])
        self.assertEqual(invented("Spent 1,000 euros.", "Spent 1.000 euros.")["invented"], ["1.000"])
        self.assertEqual(invented("Spent 1.5 euros.", "Spent 1.500 euros.")["invented"], ["1.500"])

    def test_decimals_that_are_not_ambiguous_stay_numeric(self):
        for text in ("0.001", "3.50", "12.25", "1234.567"):
            with self.subTest(text=text):
                self.assertFalse(checks.extract_quantities(text)[0]["ambiguous"])
        self.assertEqual(values("3.50"), ["3.5"])
        self.assertEqual(values("0.001"), ["0.001"])


class UnitsAreSeparateFromMagnitude(unittest.TestCase):
    """Equal numbers with different units are NOT equivalent; the value check alone cannot tell (advisory only)."""

    def test_units_are_recorded_even_though_the_value_is_equal(self):
        quantities = checks.extract_quantities("35 hours and 35%")
        self.assertEqual([q["value"] for q in quantities], ["35", "35"])
        self.assertEqual([q["unit"] for q in quantities], ["h", "%"])

    def test_equal_value_with_a_different_unit_passes_the_invented_check_limitation(self):
        result = invented("Saved 35 hours.", "Cut costs by 35%.")
        self.assertEqual(result["invented"], [])            # value-only comparison: NOT a claim of equivalence
        self.assertEqual(result["unit_mismatch"], ["35%"])  # the advisory that exposes it

    def test_percent_against_an_unlisted_unit_is_also_a_mismatch(self):
        self.assertEqual(invented("Supported 35 users.", "Improved speed by 35%.")["unit_mismatch"], ["35%"])

    def test_same_unit_family_is_not_a_mismatch(self):
        for source, bullet in (("Saved 35 hours.", "Saved 35 hrs."), ("Took 10 minutes.", "Took 10 min."),
                               ("Used 2GB.", "Used 2 GB."), ("Up 50%.", "Up 50 percent."), ("Took 3 seconds.", "Took 3s.")):
            with self.subTest(source=source, bullet=bullet):
                self.assertEqual(invented(source, bullet)["unit_mismatch"], [])

    def test_a_bare_number_never_produces_a_mismatch(self):
        self.assertEqual(invented("Saved 35 hours.", "Reviewed 35 pull requests.")["unit_mismatch"], [])

    def test_summary_counts_unit_mismatches_separately_from_invented(self):
        result = checks.check_invented_numbers(["Cut costs by 35%.", "Cut costs by 90%."], "Saved 35 hours.")
        self.assertEqual((result["bullets_with_invented"], result["bullets_with_unit_mismatch"]), (1, 1))


class SectionMatchingRegressions(unittest.TestCase):
    """I3: longest match, duplicate groups, generic prefix headings."""

    TASK, LOG, GENERIC = ("Projects - Task API", ["a", "b", "c"]), ("Projects - Log Analyzer", ["d", "e"]), ("Projects", ["z"])

    def sections(self, *source):
        return [{"heading": h, "bullets": b} for h, b in source]

    def match(self, source, *output):
        return checks.check_sections([{"section": h, "bullets": b} for h, b in output], self.sections(*source))

    def test_the_specific_heading_wins_in_either_order(self):
        for source in ((self.GENERIC, self.TASK, self.LOG), (self.TASK, self.LOG, self.GENERIC)):
            with self.subTest(first=source[0][0]):
                result = self.match(source, ("Projects - Task API", ["x", "y", "z"]), ("Projects - Log Analyzer", ["p", "q"]))
                self.assertEqual(result["matched"], 2)
                self.assertEqual(result["ambiguous_output"], [])
                self.assertEqual(result["duplicate_groups"], [])
                self.assertEqual(result["missing_from_output"], ["Projects"])
                self.assertEqual(result["bullet_count_changes"], [])

    def test_a_decorated_output_section_matches_the_longest_heading_in_either_order(self):
        for source in ((self.GENERIC, self.TASK), (self.TASK, self.GENERIC)):
            with self.subTest(first=source[0][0]):
                result = self.match(source, ("Projects - Task API (Jan 2024 - Mar 2024)", ["a", "b", "c"]))
                self.assertEqual((result["matched"], result["ambiguous_output"], result["unmatched_output"]), (1, [], []))
                self.assertEqual(result["missing_from_output"], ["Projects"])

    def test_an_exact_match_of_a_generic_heading_is_allowed(self):
        result = self.match((self.GENERIC, self.TASK), ("Projects", ["x"]), ("Projects - Task API", ["a", "b", "c"]))
        self.assertEqual((result["matched"], result["ambiguous_output"], result["missing_from_output"]), (2, [], []))

    def test_a_specific_output_is_not_silently_assigned_to_a_generic_prefix_heading(self):
        # "Projects - Log Analyzer" is not a source heading here; only the generic "Projects" is contained in it
        result = self.match((self.TASK, self.GENERIC), ("Projects - Log Analyzer", ["x"]))
        self.assertEqual(result["ambiguous_output"], [{"section": "Projects - Log Analyzer", "heading": "Projects"}])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["missing_from_output"], ["Projects - Task API", "Projects"])

    def test_a_generic_heading_without_longer_siblings_matches_normally(self):
        result = self.match((self.GENERIC, ("Work Experience", ["w"])), ("Projects - Anything", ["x"]))
        self.assertEqual((result["matched"], result["ambiguous_output"]), (1, []))

    def test_a_repeated_group_is_reported_and_counted_with_the_first(self):
        result = self.match((self.TASK, self.LOG), ("Projects - Task API", ["x", "y"]), ("projects - task api", ["z"]),
                            ("Projects - Log Analyzer", ["p", "q"]))
        self.assertEqual(result["duplicate_groups"], [{"section": "projects - task api", "heading": "Projects - Task API"}])
        self.assertEqual(result["bullet_count_changes"], [])
        self.assertEqual(result["matched"], 2)

    def test_a_repeated_group_does_not_leak_into_a_generic_heading(self):
        result = self.match((self.GENERIC, self.TASK), ("Projects - Task API", ["x"]), ("Projects - Task API", ["y"]))
        self.assertEqual(len(result["duplicate_groups"]), 1)
        self.assertEqual(result["missing_from_output"], ["Projects"])

    def test_missing_unmatched_and_ambiguous_are_reported_separately(self):
        result = self.match((self.TASK, self.LOG), ("Education", ["x"]))
        self.assertEqual(result["unmatched_output"], ["Education"])
        self.assertEqual(result["missing_from_output"], ["Projects - Task API", "Projects - Log Analyzer"])
        self.assertEqual((result["ambiguous_output"], result["duplicate_groups"]), ([], []))

    def test_legitimate_bullet_count_changes_are_signals_not_failures(self):
        result = self.match((self.TASK, self.LOG), ("Projects - Task API", ["merged one", "merged two"]),
                            ("Projects - Log Analyzer", ["d", "e", "split extra"]))
        self.assertEqual(result["bullet_count_changes"], [
            {"heading": "Projects - Task API", "source_count": 3, "output_count": 2},
            {"heading": "Projects - Log Analyzer", "source_count": 2, "output_count": 3},
        ])
        self.assertEqual((result["matched"], result["unmatched_output"], result["missing_from_output"]), (2, [], []))

    def test_a_blank_source_heading_never_matches(self):
        result = self.match((("", ["a"]), self.TASK), ("Projects - Task API", ["x"]))
        self.assertEqual((result["matched"], result["missing_from_output"]), (1, [""]))
        result = self.match((("", ["a"]), self.TASK), ("Education", ["x"]))  # only the blank heading "contains" it
        self.assertEqual((result["matched"], result["unmatched_output"]), (0, ["Education"]))

    def test_results_flow_through_evaluate_output(self):
        raw = output(("Projects - Task API", ["x"]), ("Projects - Task API", ["y"]))
        result = checks.evaluate_output(raw, "src", self.sections(self.TASK, self.GENERIC))
        self.assertEqual(len(result["sections"]["duplicate_groups"]), 1)
        self.assertEqual(result["sections"]["missing_from_output"], ["Projects"])


class SkillBoundaryCases(unittest.TestCase):
    """D3/D4 intended behaviour that can be stated unambiguously."""

    def leaked(self, bullet, *names):
        absent = [{"name": name, "aliases": []} for name in names]
        return checks.check_absent_skills([bullet], absent)["results"][0]["leaked"]

    def test_punctuated_names_match_as_written(self):
        self.assertEqual(self.leaked("Wrote C++ services.", "C++"), ["C++"])
        self.assertEqual(self.leaked("Built Node.js apps.", "Node.js"), ["Node.js"])
        self.assertEqual(self.leaked("Built CI/CD pipelines.", "CI/CD"), ["CI/CD"])

    def test_a_name_followed_by_a_digit_is_not_matched_limitation(self):
        # digits count as part of a word: "Java8" and "C++17" are different tokens from "Java" and "C++"
        self.assertEqual(self.leaked("Used Java8 streams.", "Java"), [])
        self.assertEqual(self.leaked("Wrote C++17 code.", "C++"), [])
        self.assertEqual(self.leaked("Used Java 8 streams.", "Java"), ["Java"])

    def test_sql_is_not_matched_inside_longer_words(self):
        for bullet in ("Used NoSQL stores.", "Used SQLAlchemy.", "Used PostgreSQL.", "Used MySQL."):
            with self.subTest(bullet=bullet):
                self.assertEqual(self.leaked(bullet, "SQL"), [])
        for bullet in ("Wrote SQL queries.", "Wrote sql, then Python.", "Used SQL-based tools."):
            with self.subTest(bullet=bullet):
                self.assertEqual(self.leaked(bullet, "SQL"), ["SQL"])

    def test_a_negated_mention_is_still_a_leak_by_design(self):
        self.assertEqual(self.leaked("No Kubernetes experience yet.", "Kubernetes"), ["Kubernetes"])

    def test_spelling_variants_match_only_when_listed_as_aliases(self):
        self.assertEqual(self.leaked("Built CI-CD pipelines.", "CI/CD"), [])
        self.assertEqual(self.leaked("Used Amazon Web Services.", "AWS"), [])
        absent = [{"name": "AWS", "aliases": ["Amazon Web Services"]}]
        self.assertEqual(checks.check_absent_skills(["Used Amazon Web Services."], absent)["bullets_with_leak"], 1)

    def test_a_generic_alias_collides_with_longer_names_so_the_validator_cannot_save_you(self):
        # "Node" would also match "node-fetch" and "Node 18"; datasets should not use it as an alias
        absent = [{"name": "Node.js", "aliases": ["Node"]}]
        self.assertEqual(checks.check_absent_skills(["Used node-fetch."], absent)["bullets_with_leak"], 1)

    def test_validator_rejects_two_character_names_even_when_unambiguous_limitation(self):
        self.assertTrue(checks.validate_absent_skills([{"name": "C#", "aliases": []}], "x", "We use C#."))

    def test_redis_is_not_matched_inside_redistribute(self):
        vocab = {"Redis": []}
        result = checks.check_technologies(["Redistribute work across Redisson nodes.", "Cached in redis."], "Built APIs.", vocab)
        self.assertEqual([r["unsupported"] for r in result["results"]], [[], ["Redis"]])

    def test_a_technology_supported_through_an_alias_is_not_flagged(self):
        vocab = {"PostgreSQL": ["Postgres", "psql"]}
        result = checks.check_technologies(["Tuned PostgreSQL queries."], "Used psql daily.", vocab)
        self.assertEqual(result["bullets_with_unsupported"], 0)


class SchemaEdgeCases(unittest.TestCase):
    def status(self, raw):
        return checks.parse_ats_output(raw)["status"]

    def test_non_string_section_values_are_wrong_shape(self):
        for section in (5, None, ["S"], {"a": 1}, True, 1.5):
            with self.subTest(section=section):
                raw = json.dumps({"optimized_bullets": [{"section": section, "bullets": ["valid bullet text"]}]})
                parsed = checks.parse_ats_output(raw)
                self.assertEqual(parsed["status"], "wrong_shape")
                self.assertEqual(parsed["sections"], [])

    def test_one_bad_item_makes_the_whole_response_wrong_shape_limitation(self):
        raw = json.dumps({"optimized_bullets": [{"section": "S", "bullets": ["good bullet"]},
                                                {"section": "T", "bullets": []}]})
        parsed = checks.parse_ats_output(raw)
        self.assertEqual(parsed["status"], "wrong_shape")  # stricter than production, which shows a placeholder
        self.assertEqual(parsed["sections"], [])
        self.assertFalse(parsed["production_would_500"])

    def test_output_that_production_cannot_parse_but_looks_json_like(self):
        fenced_without_language = "```\n" + output(("S", ["a b"])) + "\n```"
        self.assertEqual(self.status(fenced_without_language), "invalid_json")  # same as production's cleaning
        self.assertEqual(self.status("﻿" + output(("S", ["a b"]))), "invalid_json")
        self.assertEqual(self.status("Here you go: " + output(("S", ["a b"]))), "invalid_json")

    def test_extreme_json_does_not_raise(self):
        for raw in ("[" * 100000, '{"optimized_bullets": [], "n": ' + "9" * 5000 + "}", b"{}", "\x00", "NaN"):
            with self.subTest(raw=repr(raw)[:30]):
                self.assertIn(self.status(raw), {"invalid_json", "wrong_shape", "empty"})


class MalformedConfiguration(unittest.TestCase):
    RAW = output(("Projects - A", ["Built 5 APIs"]))

    def test_malformed_model_output_never_raises(self):
        for raw in ("garbage", "", None, "[]", '{"optimized_bullets": [null]}', output(("S", ["x y"])).upper()):
            with self.subTest(raw=raw):
                checks.evaluate_output(raw, "src", [{"heading": "S", "bullets": ["a"]}], [{"name": "X"}], {"A": []})

    def test_malformed_source_sections_raise_documented_exceptions(self):
        for bad in ([{"bullets": []}], [{"heading": "S"}], [{"heading": None, "bullets": []}], ["not a dict"]):
            with self.subTest(bad=bad):
                with self.assertRaises((KeyError, TypeError, AttributeError)):
                    checks.evaluate_output(self.RAW, "src", bad)

    def test_malformed_absent_skills_and_vocabulary_entries_are_tolerated(self):
        result = checks.evaluate_output(self.RAW, "src", None, [None, "x", {"name": None}, {"name": "X", "aliases": None}],
                                        {"A": None, "B": ["b", None]})
        self.assertEqual(result["absent_skills"]["bullets_with_leak"], 0)
        self.assertEqual(result["technologies"]["bullets_with_unsupported"], 0)

    def test_results_are_json_serialisable_with_the_new_fields(self):
        result = checks.evaluate_output(output(("Projects - A", ["Cut latency to 50ms by 35%."])), "Saved 35 hours.",
                                        [{"heading": "Projects - A", "bullets": ["x"]}], [], {})
        json.dumps(result)
        self.assertEqual(result["numbers"]["results"][0]["invented"], ["50ms"])
        self.assertEqual(result["numbers"]["results"][0]["unit_mismatch"], ["35%"])


if __name__ == "__main__":
    unittest.main()
