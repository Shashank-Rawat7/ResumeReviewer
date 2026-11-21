"""Tests for blank/invalid bullet handling, HTML escaping and unique-ID session selection.

Run from the project root:  python -m unittest tests.test_ui_robustness -v
Uses only synthetic data and a fake Streamlit, so no network, API key, database or resume file is needed.
"""
import os
import re
import unittest

from tests.test_project_bullets import ROOT, load_display


def render_work(experiences):
    module, fake = load_display()
    module.display_work_experience(experiences)
    return "\n".join(fake.markdown_calls)


def render_projects(projects):
    module, fake = load_display()
    module.display_projects(projects)
    return "\n".join(fake.markdown_calls)


def experience(**overrides):
    base = {
        "job_title": "Analyst",
        "company": "Synthetic Co",
        "responsibilities": ["Built dashboards."],
        "technologies_used": ["Python", "SQL"],
    }
    base.update(overrides)
    return base


def legacy_project(**overrides):
    base = {
        "project_name": "Legacy Project",
        "objective": "Build a demo.",
        "technologies_used": ["Python"],
        "outcomes": ["Shipped the demo."],
    }
    base.update(overrides)
    return base


class WorkExperienceTests(unittest.TestCase):
    def test_blank_whitespace_and_non_string_responsibilities_are_dropped(self):
        out = render_work([experience(responsibilities=["First.", "", "   ", None, 5, ["x"], "Second."])])
        self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["First.", "Second."])
        self.assertNotIn("No responsibilities specified.", out)

    def test_order_is_preserved(self):
        out = render_work([experience(responsibilities=["C.", "A.", "B."])])
        self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["C.", "A.", "B."])

    def test_all_unusable_responsibilities_show_fallback(self):
        for value in ([], [""], ["  ", "\n"], [None, 3], "", "   ", None, 7, {"a": 1}):
            with self.subTest(value=value):
                out = render_work([experience(responsibilities=value)])
                self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["No responsibilities specified."])

    def test_string_responsibility_is_one_item(self):
        out = render_work([experience(responsibilities="Only one.")])
        self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["Only one."])

    def test_html_is_escaped(self):
        out = render_work([experience(
            responsibilities=["<script>alert(1)</script> & more"],
            technologies_used=["<b>Py</b>", "C&C"],
        )])
        self.assertNotIn("<script>", out)
        self.assertNotIn("<b>Py</b>", out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; more", out)
        self.assertIn("&lt;b&gt;Py&lt;/b&gt;, C&amp;C", out)

    def test_non_string_technologies_do_not_crash(self):
        out = render_work([experience(technologies_used=["Python", None, 3, "", "SQL"])])
        self.assertIn("<strong>Technologies:</strong> Python, SQL", out)

    def test_empty_technologies_fallback_kept(self):
        out = render_work([experience(technologies_used=[None, ""])])
        self.assertIn("No technologies used", out)


class LegacyProjectTests(unittest.TestCase):
    def test_unusable_outcomes_show_fallback(self):
        for value in ("", "   ", [""], [None, 5], [], None):
            with self.subTest(value=value):
                out = render_projects([legacy_project(outcomes=value)])
                self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["No outcomes specified."])

    def test_real_outcomes_shown_and_blanks_dropped(self):
        out = render_projects([legacy_project(outcomes=["Real one.", " ", None, "Another."])])
        self.assertEqual(re.findall(r"<li>(.*?)</li>", out), ["Real one.", "Another."])

    def test_html_is_escaped(self):
        out = render_projects([legacy_project(
            objective="<img src=x onerror=1>",
            outcomes=["<script>1</script>"],
            technologies_used=["<i>T</i>"],
        )])
        for raw in ("<img src=x", "<script>", "<i>T</i>"):
            self.assertNotIn(raw, out)
        self.assertIn("&lt;img src=x onerror=1&gt;", out)
        self.assertIn("&lt;script&gt;1&lt;/script&gt;", out)
        self.assertIn("&lt;i&gt;T&lt;/i&gt;", out)

    def test_non_string_technologies_do_not_crash(self):
        out = render_projects([legacy_project(technologies_used=[None, 2, "Go"])])
        self.assertIn("<strong>Technologies:</strong> Go", out)


class SessionOptionTests(unittest.TestCase):
    def setUp(self):
        self.module, _ = load_display()

    def test_duplicate_names_get_distinct_labels(self):
        ids, labels = self.module.build_session_options([
            {"id": 6, "name": "Analysis", "created_at": "x"},
            {"id": 4, "name": "Analysis", "created_at": "y"},
        ])
        self.assertEqual(ids, [4, 6])
        self.assertEqual(labels[4], "Analysis (#4)")
        self.assertEqual(labels[6], "Analysis (#6)")

    def test_selection_returns_the_id(self):
        ids, labels = self.module.build_session_options([{"id": 9, "name": "A"}, {"id": 2, "name": "A"}])
        chosen = next(i for i in ids if labels[i] == "A (#9)")
        self.assertEqual(chosen, 9)

    def test_malformed_entries_are_skipped(self):
        ids, labels = self.module.build_session_options([
            None, "x", {"name": "no id"}, {"id": "3", "name": "str id"}, {"id": True, "name": "bool"},
            {"id": 5, "name": "Good"},
        ])
        self.assertEqual(ids, [5])
        self.assertEqual(labels, {5: "Good (#5)"})

    def test_blank_name_is_untitled(self):
        _, labels = self.module.build_session_options([{"id": 1, "name": "  "}, {"id": 2, "name": None}])
        self.assertEqual(labels, {1: "Untitled (#1)", 2: "Untitled (#2)"})

    def test_non_list_gives_empty(self):
        for value in (None, {}, "abc", 5):
            self.assertEqual(self.module.build_session_options(value), ([], {}))

    def test_main_uses_ids_not_names(self):
        with open(os.path.join(ROOT, "frontend", "main.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn('s["name"] == choice', source)
        self.assertNotIn('[s["name"] for s in sessions]', source)
        self.assertIn("build_session_options(", source)
        self.assertIn("st.session_state.session_id = choice", source)
        self.assertIn("format_func", source)


if __name__ == "__main__":
    unittest.main()
