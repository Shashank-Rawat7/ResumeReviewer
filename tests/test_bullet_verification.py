"""Tests for project-bullet verification (src/parser.py) and its warning in the UI (frontend/display.py).

Run from the project root:  python -m unittest tests.test_bullet_verification -v
Synthetic data only. No network, API key, resume file or live model is used. If LangChain & co. are
not installed, light stand-ins are used (the function under test does not use them), and python-dotenv
is always stubbed so the project's .env is never loaded by the tests.
"""
import contextlib
import importlib
import importlib.util
import json
import logging
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OPTIONAL = ["PyPDF2", "langchain", "langchain.chains", "langchain.chains.summarize", "langchain.text_splitter",
             "langchain.docstore", "langchain.docstore.document", "langchain.prompts", "langchain.output_parsers",
             "langchain_groq"]


def load_parser():
    patched = {"dotenv": mock.MagicMock()}          # never read the project's .env during tests
    for name in _OPTIONAL:
        try:
            importlib.import_module(name)
        except Exception:
            patched[name] = mock.MagicMock()
    with mock.patch.dict(sys.modules, patched):
        spec = importlib.util.spec_from_file_location("parser_under_test", os.path.join(ROOT, "src", "parser.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.markdown_calls = []

    def subheader(self, *args, **kwargs):
        pass

    def markdown(self, body, *args, **kwargs):
        self.markdown_calls.append(body)

    @contextlib.contextmanager
    def expander(self, title, *args, **kwargs):
        yield


def load_display():
    fake = FakeStreamlit()
    saved = {name: sys.modules.get(name) for name in ("streamlit", "requests")}
    sys.modules["streamlit"] = fake
    if "requests" not in sys.modules:
        try:
            import requests  # noqa: F401
        except ImportError:
            sys.modules["requests"] = types.ModuleType("requests")
    try:
        spec = importlib.util.spec_from_file_location("display_under_test", os.path.join(ROOT, "frontend", "display.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, fake


PARSER = load_parser()
WARNING = "Could not verify against extracted resume text"

SOURCE = ("Alpha Project\n"
          "• Built a planner pipeline using an   orchestration\nframework.\n"
          "• Added 12 automated tests for file handling.\n"
          "Beta Project\n"
          "• Created a small command line tool in Python.\n")

B_EXACT = "Added 12 automated tests for file handling."
B_SPACED = "Built a planner pipeline using an orchestration framework."     # source has extra spaces and a newline
B_PARAPHRASE = "Wrote twelve automated tests covering file handling."
B_CASE = "added 12 automated tests for file handling."
B_BETA = "Created a small command line tool in Python."


def project(bullets, name="P", **extra):
    return {"project_name": name, "objective": "an objective", "outcomes": "a summary", "technologies_used": ["Python"],
            "bullets": bullets, **extra}


def summary_json(*projects):
    return json.dumps({"skills": ["x"], "key_projects": list(projects)})


def verify(text, source=SOURCE):
    return json.loads(PARSER.verify_project_bullets(text, source))


class VerifyProjectBullets(unittest.TestCase):
    def test_exact_match_is_verified(self):
        self.assertEqual(verify(summary_json(project([B_EXACT])))["key_projects"][0]["bullets_unverified"], [])

    def test_match_after_whitespace_normalisation_is_verified(self):
        self.assertEqual(verify(summary_json(project([B_SPACED, "  " + B_EXACT + "\n"])))["key_projects"][0]["bullets_unverified"], [])

    def test_paraphrase_stays_unverified_and_is_kept_unchanged(self):
        out = verify(summary_json(project([B_EXACT, B_PARAPHRASE])))["key_projects"][0]
        self.assertEqual(out["bullets_unverified"], [1])
        self.assertEqual(out["bullets"], [B_EXACT, B_PARAPHRASE])

    def test_case_differences_do_not_match(self):
        self.assertEqual(verify(summary_json(project([B_CASE, B_EXACT])))["key_projects"][0]["bullets_unverified"], [0])

    def test_small_wording_changes_do_not_match(self):
        altered = ["Added 12 automated test for file handling.", B_EXACT + " Extra words."]
        self.assertEqual(verify(summary_json(project(altered)))["key_projects"][0]["bullets_unverified"], [0, 1])

    def test_known_limitation_a_shortened_excerpt_of_a_source_bullet_counts_as_found(self):
        # The approved rule is an exact substring match, so a bullet that is only a shortened piece of a
        # real source bullet is found. Documented here so the behaviour is explicit, not accidental.
        excerpt = "Added 12 automated tests"
        self.assertEqual(verify(summary_json(project([excerpt])))["key_projects"][0]["bullets_unverified"], [])

    def test_multiple_projects_mixed_results_with_per_project_indices(self):
        out = verify(summary_json(project([B_PARAPHRASE, B_EXACT, B_CASE], "A"), project([B_BETA, B_PARAPHRASE], "B"), project([B_SPACED], "C")))
        self.assertEqual([p["bullets_unverified"] for p in out["key_projects"]], [[0, 2], [1], []])

    def test_metadata_is_sorted_zero_based_ints_and_text_is_preserved(self):
        bullets = [B_PARAPHRASE, B_EXACT, B_CASE]
        entry = verify(summary_json(project(bullets)))["key_projects"][0]
        self.assertEqual(entry["bullets_unverified"], [0, 2])
        self.assertTrue(all(type(i) is int for i in entry["bullets_unverified"]))
        self.assertEqual(entry["bullets"], bullets)
        self.assertEqual((entry["objective"], entry["outcomes"], entry["technologies_used"]), ("an objective", "a summary", ["Python"]))

    def test_model_supplied_bullets_unverified_is_overwritten(self):
        entry = verify(summary_json(project([B_EXACT, B_PARAPHRASE], bullets_unverified=[0])))["key_projects"][0]
        self.assertEqual(entry["bullets_unverified"], [1])

    def test_empty_bullets_list_gets_an_empty_flag_list(self):
        self.assertEqual(verify(summary_json(project([])))["key_projects"][0]["bullets_unverified"], [])

    def test_blank_and_non_string_items_are_reported_by_position(self):
        entry = verify(summary_json(project([None, B_EXACT, "", 7, "   ", {"a": 1}])))["key_projects"][0]
        self.assertEqual(entry["bullets_unverified"], [0, 2, 3, 4, 5])

    def test_missing_or_malformed_bullets_return_the_text_unchanged(self):
        cases = {
            "no bullets key (older summary)": summary_json({"project_name": "Old", "outcomes": "s"}),
            "bullets is a string": summary_json(project("not a list")),
            "bullets is a dict": summary_json(project({"a": 1})),
            "bullets is null": summary_json(project(None)),
            "bullets is a number": summary_json(project(5)),
            "no key_projects": json.dumps({"skills": []}),
            "key_projects is a string": json.dumps({"key_projects": "x"}),
            "project is not an object": json.dumps({"key_projects": ["x", 5, None]}),
        }
        for label, text in cases.items():
            with self.subTest(label):
                self.assertEqual(PARSER.verify_project_bullets(text, SOURCE), text)

    def test_mixed_old_and_new_projects(self):
        out = verify(summary_json(project([B_EXACT]), {"project_name": "Old", "outcomes": "s"}))
        self.assertEqual(out["key_projects"][0]["bullets_unverified"], [])
        self.assertNotIn("bullets_unverified", out["key_projects"][1])

    def test_invalid_json_and_unexpected_inputs_do_not_raise(self):
        for odd in ("not json", "", "[]", "null", "5", None, 5, object(), b"\xff\xfe", ["x"], {"key_projects": []}):
            with self.subTest(repr(odd)[:20]):
                result = PARSER.verify_project_bullets(odd, SOURCE)
                if isinstance(odd, str):
                    self.assertEqual(result, odd)

    def test_unusable_resume_text_flags_every_bullet(self):
        for source in (None, "", 5, b"bytes"):
            with self.subTest(repr(source)):
                self.assertEqual(verify(summary_json(project([B_EXACT, B_BETA])), source)["key_projects"][0]["bullets_unverified"], [0, 1])

    def test_non_ascii_text_round_trips(self):
        text = "Résumé • façade ✓ naïve café déjà vu"
        entry = verify(summary_json(project([text])), "x " + text + " y")["key_projects"][0]
        self.assertEqual(entry["bullets_unverified"], [])
        self.assertEqual(entry["bullets"], [text])


GLUED = ("Gamma Project\n"
         "\u2022 Builta plannerpipeline usingan orchestration framework.\n"
         "\u2022 Added12 automated\ttests for\n\n  file handling; all passed.\n"
         "\u2022 Created a small, command-line tool in Python (v3).\n")
G1 = "Built a planner pipeline using an orchestration framework."
G2 = "Added 12 automated tests for file handling; all passed."
G3 = "Created a small, command-line tool in Python (v3)."


def flagged(bullets, source=GLUED):
    return verify(summary_json(project(bullets)), source)["key_projects"][0]["bullets_unverified"]


class WhitespaceInsensitiveMatching(unittest.TestCase):
    def test_missing_spaces_between_words_in_the_source_still_verify(self):
        self.assertEqual(flagged([G1, G2, G3]), [])

    def test_newlines_tabs_repeated_and_unicode_whitespace_do_not_cause_mismatches(self):
        messy = "Built\ta   planner\n\npipeline\u00a0using an\r\norchestration     framework."
        self.assertEqual(flagged([messy]), [])
        self.assertEqual(flagged([G1], "x " + "Built a\tplanner \n pipeline using\n\n\nan orchestration framework. y"), [])

    def test_case_differences_remain_unverified(self):
        self.assertEqual(flagged([G1.lower(), G2.upper(), "Built a Planner pipeline using an orchestration framework."]), [0, 1, 2])

    def test_punctuation_differences_remain_unverified(self):
        altered = [G2.replace(";", ","), G2.replace("handling;", "handling."), G3.replace("-", "\u2013"),
                   G3.replace("(v3)", "[v3]"), G1.replace("an orchestration", "an, orchestration")]
        self.assertEqual(flagged(altered), [0, 1, 2, 3, 4])

    def test_word_substitution_deletion_and_reordering_remain_unverified(self):
        substituted = G1.replace("planner", "scheduler")
        deleted = G2.replace("automated ", "")
        reordered = "Built a pipeline planner using an orchestration framework."
        self.assertEqual(flagged([substituted, deleted, reordered]), [0, 1, 2])

    def test_minimum_length_guard(self):
        source = "Alpha " + "abcdefghij0123456789" + " omega abcdefghij012345678 end"
        twenty = "abcdefghij 0123456789"          # 20 non-whitespace characters, found in the source
        nineteen = "abcdefghij012345678"          # 19 non-whitespace characters, also present in the source
        self.assertEqual(len(PARSER._remove_whitespace(twenty)), 20)
        self.assertEqual(len(PARSER._remove_whitespace(nineteen)), 19)
        self.assertEqual(flagged([twenty], source), [])
        self.assertEqual(flagged([nineteen], source), [0])

    def test_whitespace_does_not_count_towards_the_minimum_length(self):
        source = "abcdefghij abcdefghij"
        self.assertEqual(flagged(["abcdefghi      j   abcdefghi        "], source), [0])   # 19 non-space characters
        self.assertEqual(flagged(["a b c d e f g h i j a b c d e f g h i j"], source), [])  # 20 non-space characters

    def test_short_real_bullets_are_unverified_even_though_they_are_in_the_text(self):
        source = "Python  \u2022 Docker \u2022 SQL and Git"
        self.assertEqual(flagged(["Python", "Docker", "SQL and Git"], source), [0, 1, 2])

    def test_original_bullet_text_is_not_modified(self):
        odd = "Built   a planner\tpipeline using an\n orchestration framework."
        entry = verify(summary_json(project([odd])), GLUED)["key_projects"][0]
        self.assertEqual(entry["bullets"], [odd])
        self.assertEqual(entry["bullets_unverified"], [])

    def test_indexes_stay_zero_based_in_original_order_with_the_guard(self):
        self.assertEqual(flagged(["short", G1, "x" * 19, G2, None, G3.lower()]), [0, 2, 4, 5])

    def test_two_source_bullets_merged_into_one_stay_unverified_while_the_bullet_symbol_separates_them(self):
        self.assertEqual(flagged([G1 + " " + G2]), [0])

    def test_known_limitation_neighbouring_source_lines_without_a_bullet_symbol_can_read_as_one_bullet(self):
        # Whitespace is ignored everywhere, so if the extracted text has no symbol between two lines,
        # a bullet made of both lines is found. Documented consequence of the approved rule.
        no_symbols = "Delta\nBuilta plannerpipeline usingan orchestration framework.\nAdded12 automated tests for file handling; all passed.\n"
        self.assertEqual(flagged([G1 + " " + G2], no_symbols), [])


class Logging(unittest.TestCase):
    def test_logs_contain_counts_only(self):
        secret_a, secret_b = "Zorblax synthetic confidential sentence one.", "Quuxington synthetic confidential sentence two."
        with self.assertLogs(PARSER.logger, level=logging.WARNING) as captured:
            PARSER.verify_project_bullets(summary_json(project([secret_a, B_EXACT, secret_b])), SOURCE)
        joined = " ".join(captured.output)
        self.assertIn("2 of 3", joined)
        for leaked in ("Zorblax", "Quuxington", "confidential", "automated", "planner", "an objective", "Alpha"):
            self.assertNotIn(leaked, joined)

    def test_nothing_is_logged_when_everything_is_verified(self):
        with self.assertNoLogs(PARSER.logger, level=logging.WARNING) if hasattr(self, "assertNoLogs") else contextlib.nullcontext():
            PARSER.verify_project_bullets(summary_json(project([B_EXACT])), SOURCE)

    def test_skipped_verification_logs_only_the_exception_type(self):
        with mock.patch.object(PARSER, "_remove_whitespace", side_effect=RuntimeError("Zorblax private text in message")):
            with self.assertLogs(PARSER.logger, level=logging.WARNING) as captured:
                result = PARSER.verify_project_bullets(summary_json(project([B_EXACT])), SOURCE)
        joined = " ".join(captured.output)
        self.assertIn("RuntimeError", joined)
        self.assertNotIn("Zorblax", joined)
        self.assertEqual(result, summary_json(project([B_EXACT])))


def render(projects):
    module, fake = load_display()
    module.display_projects(projects)
    return "\n".join(fake.markdown_calls)


class WarningInTheUI(unittest.TestCase):
    def test_warning_appears_only_for_flagged_bullets(self):
        html_out = render([project(["Verified one.", "Flagged one.", "Verified two."], bullets_unverified=[1])])
        self.assertIn("<li>Verified one.</li>", html_out)
        self.assertIn("<li>Verified two.</li>", html_out)
        self.assertIn(f"<li>Flagged one. <small>⚠ {WARNING}</small></li>", html_out)
        self.assertEqual(html_out.count(WARNING), 1)

    def test_explanatory_note_only_when_something_is_flagged(self):
        flagged = render([project(["A.", "B."], bullets_unverified=[0])])
        self.assertIn("does not mean the bullet is wrong", flagged)
        for clean in (project(["A."], bullets_unverified=[]), project(["A."]), project(["A."], bullets_unverified=None)):
            self.assertNotIn("does not mean the bullet is wrong", render([clean]))
            self.assertNotIn(WARNING, render([clean]))

    def test_wording_never_calls_a_bullet_false_or_fabricated(self):
        html_out = render([project(["A.", "B."], bullets_unverified=[0, 1])]).lower()
        for word in ("fabricat", "hallucinat", "false", "fake", "wrong bullet", "invented"):
            self.assertNotIn(word, html_out.replace("does not mean the bullet is wrong", ""))

    def test_indices_refer_to_the_original_list_even_when_blank_items_are_skipped(self):
        html_out = render([project(["", "First real.", None, "Second real."], bullets_unverified=[3])])
        self.assertIn("<li>First real.</li>", html_out)
        self.assertIn(f"<li>Second real. <small>⚠ {WARNING}</small></li>", html_out)

    def test_malformed_flag_values_are_ignored_without_crashing(self):
        for flag in (None, "1", 1, {"a": 1}, [True], [False], ["0"], [None], [1.0], [-1], [99], [[0]]):
            with self.subTest(repr(flag)):
                html_out = render([project(["Only bullet."], bullets_unverified=flag)])
                self.assertIn("<li>Only bullet.</li>", html_out)
                self.assertNotIn(WARNING, html_out)

    def test_html_like_bullet_stays_escaped_when_flagged(self):
        evil = '<script>alert(1)</script> & <img src=x onerror=alert(2)> "q"'
        html_out = render([project([evil], bullets_unverified=[0])])
        self.assertNotIn("<script>", html_out)
        self.assertNotIn("<img", html_out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;img", html_out)
        self.assertIn(f"<small>⚠ {WARNING}</small>", html_out)

    def test_legacy_projects_still_render(self):
        legacy = {"project_name": "Old", "objective": "obj", "outcomes": "Old summary.", "technologies_used": ["Python"]}
        html_out = render([legacy, project([], name="Empty"), project("not a list", name="Bad")])
        self.assertEqual(html_out.count("<strong>Outcome:</strong>"), 3)
        self.assertIn("<li>Old summary.</li>", html_out)
        self.assertNotIn(WARNING, html_out)

    def test_helper_functions_never_raise(self):
        module, _ = load_display()
        for value in (None, [], "x", 5, {}, object(), [None], [[1]]):
            self.assertEqual(module._indexed_bullets(value), [])
            self.assertEqual(module._unverified_indices(value) if not isinstance(value, list) else set(), set())
        self.assertEqual(module._indexed_bullets(["", " a ", None, "b"]), [(1, "a"), (3, "b")])
        self.assertEqual(module._usable_bullets(["", " a ", None, "b"]), ["a", "b"])
        self.assertEqual(module._unverified_indices([0, 2, True, "1", 2]), {0, 2})


class ParserAndUITogether(unittest.TestCase):
    """The parser's indices must point at the same bullets the UI shows (including skipped blank items)."""

    def test_indices_line_up_end_to_end(self):
        bullets = [B_EXACT, "", B_PARAPHRASE, None, B_BETA, B_CASE]
        verified = verify(summary_json(project(bullets)))
        self.assertEqual(verified["key_projects"][0]["bullets_unverified"], [1, 2, 3, 5])
        html_out = render(verified["key_projects"])
        self.assertIn(f"<li>{B_EXACT}</li>", html_out)
        self.assertIn(f"<li>{B_BETA}</li>", html_out)
        self.assertIn(f"<li>{B_PARAPHRASE} <small>⚠ {WARNING}</small></li>", html_out)
        self.assertIn(f"<li>{B_CASE} <small>⚠ {WARNING}</small></li>", html_out)
        self.assertEqual(html_out.count(WARNING), 2)


class WiringInParseResume(unittest.TestCase):
    def test_parse_resume_calls_the_verifier_before_saving_the_summary(self):
        with open(os.path.join(ROOT, "src", "parser.py"), encoding="utf-8") as handle:
            source = handle.read()
        call = source.index("summary_text = verify_project_bullets(summary_text, resume_text)")
        save = source.index("self.save_text_to_file(ResumeParserConfig.RESUME_SUMMARY_FILE_PATH, summary_text)")
        self.assertLess(call, save)
        self.assertLess(source.index('resume_text = " ".join('), call)
        self.assertEqual(source.count("verify_project_bullets("), 2)       # the definition and the single call


if __name__ == "__main__":
    unittest.main()
