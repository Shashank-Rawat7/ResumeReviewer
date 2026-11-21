"""Tests for preserving project bullets (prompt schema + frontend rendering).

Run from the project root:  python -m unittest tests.test_project_bullets -v
Uses only synthetic data and a fake Streamlit, so no network, API key or resume file is needed.
"""
import contextlib
import importlib.util
import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FakeStreamlit(types.ModuleType):
    """Records markdown/subheader calls and the expander titles of display_projects."""

    def __init__(self):
        super().__init__("streamlit")
        self.markdown_calls = []
        self.expander_titles = []

    def subheader(self, *args, **kwargs):
        pass

    def markdown(self, body, *args, **kwargs):
        self.markdown_calls.append(body)

    @contextlib.contextmanager
    def expander(self, title, *args, **kwargs):
        self.expander_titles.append(title)
        yield


def load_display():
    """Import frontend/display.py against a fake Streamlit; return (module, fake)."""
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


def project(**overrides):
    base = {
        "project_name": "Synthetic Project",
        "objective": "Build a demo service.",
        "technologies_used": ["Python", "FastAPI"],
        "outcomes": "A short summary of results.",
    }
    base.update(overrides)
    return base


def render(projects):
    module, fake = load_display()
    module.display_projects(projects)
    return "\n".join(fake.markdown_calls), fake


class NewFormatProject(unittest.TestCase):
    def test_each_bullet_is_a_separate_list_item_in_order(self):
        html_out, _ = render([project(bullets=["First original bullet.", "Second original bullet.", "Third one."])])
        first = html_out.index("<li>First original bullet.</li>")
        second = html_out.index("<li>Second original bullet.</li>")
        third = html_out.index("<li>Third one.</li>")
        self.assertTrue(first < second < third)
        self.assertEqual(html_out.count("<li>"), 3)

    def test_objective_and_technologies_still_shown_and_summary_not_duplicated(self):
        html_out, _ = render([project(bullets=["Only bullet."])])
        self.assertIn("Build a demo service.", html_out)
        self.assertIn("Python, FastAPI", html_out)
        self.assertIn("Bullets:", html_out)
        self.assertNotIn("A short summary of results.", html_out)
        self.assertNotIn("Outcome:", html_out)

    def test_bullets_are_stripped(self):
        html_out, _ = render([project(bullets=["   padded bullet   "])])
        self.assertIn("<li>padded bullet</li>", html_out)


class LegacyProject(unittest.TestCase):
    def test_objective_and_outcomes_rendering_is_unchanged(self):
        html_out, fake = render([project()])
        self.assertIn("<strong>Outcome:</strong>", html_out)
        self.assertIn("<li>A short summary of results.</li>", html_out)
        self.assertIn("Build a demo service.", html_out)
        self.assertNotIn("Bullets:", html_out)
        self.assertEqual(fake.expander_titles, ["**Synthetic Project**"])

    def test_legacy_outcomes_list_and_missing_outcomes(self):
        html_out, _ = render([project(outcomes=["One.", "Two."])])
        self.assertIn("<li>One.</li><li>Two.</li>", html_out)
        html_out, _ = render([{k: v for k, v in project().items() if k != "outcomes"}])
        self.assertIn("No outcomes specified.", html_out)


class UnusableBullets(unittest.TestCase):
    CASES = {
        "missing key": None,
        "empty list": [],
        "None value": "NONE",
        "plain string": "A single string, not a list",
        "dict": {"a": 1},
        "number": 5,
        "list of non-strings": [None, 5, {"x": 1}],
        "list of blanks": ["", "   ", "\n"],
    }

    def test_fall_back_to_legacy_view(self):
        for label, value in self.CASES.items():
            with self.subTest(label):
                if value is None:
                    item = project()
                elif value == "NONE":
                    item = project(bullets=None)
                else:
                    item = project(bullets=value)
                html_out, _ = render([item])
                self.assertIn("<strong>Outcome:</strong>", html_out)
                self.assertIn("<li>A short summary of results.</li>", html_out)
                self.assertNotIn("Bullets:", html_out)

    def test_mixed_list_keeps_only_the_usable_strings(self):
        html_out, _ = render([project(bullets=["Real bullet.", None, 7, "", "  Another real one. "])])
        self.assertEqual(html_out.count("<li>"), 2)
        self.assertIn("<li>Real bullet.</li>", html_out)
        self.assertIn("<li>Another real one.</li>", html_out)

    def test_usable_bullets_helper_never_raises(self):
        module, _ = load_display()
        for value in (None, [], "x", 5, {}, [None], [["nested"]], object()):
            self.assertEqual(module._usable_bullets(value), [])
        self.assertEqual(module._usable_bullets([" a ", "b"]), ["a", "b"])

    def test_mixed_formats_in_one_list(self):
        html_out, _ = render([project(project_name="New", bullets=["New-format bullet."]), project(project_name="Old")])
        self.assertIn("<li>New-format bullet.</li>", html_out)
        self.assertIn("<li>A short summary of results.</li>", html_out)


class Escaping(unittest.TestCase):
    def test_html_like_bullet_text_is_escaped(self):
        evil = '<script>alert(1)</script> & <img src=x onerror=alert(2)> "quoted"'
        html_out, _ = render([project(bullets=[evil], objective="<b>obj</b>", technologies_used=["<i>tech</i>"])])
        for raw in ("<script>", "<img", "<b>obj</b>", "<i>tech</i>"):
            self.assertNotIn(raw, html_out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;img", html_out)
        self.assertIn("&quot;quoted&quot;", html_out)

    def test_newlines_and_blank_lines_cannot_break_the_html_block(self):
        html_out, _ = render([project(bullets=["line one\n\nline two"])])
        self.assertIn("<li>line one line two</li>", html_out)

    def test_text_that_looks_like_markdown_stays_literal_text_inside_html(self):
        html_out, _ = render([project(bullets=["Built **bold** claims with $x$ maths"])])
        self.assertIn("<li>Built **bold** claims with $x$ maths</li>", html_out)


class PromptSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except ImportError:
            raise unittest.SkipTest("PyYAML is not installed")
        with open(os.path.join(ROOT, "src", "prompts.yaml"), encoding="utf-8") as handle:
            cls.prompt = yaml.safe_load(handle)["resume_summary_generator"]

    def test_bullets_field_and_verbatim_rules_present(self):
        self.assertIn("`bullets`", self.prompt)
        for phrase in ("copied verbatim", "Do NOT merge, split, shorten, reorder, paraphrase, correct or add to any bullet",
                       "Never write bullets yourself"):
            self.assertIn(phrase, self.prompt)

    def test_existing_project_fields_are_kept(self):
        for field in ("project_name", "objective", "technologies_used", "outcomes"):
            self.assertIn(f"`{field}`", self.prompt)
            self.assertIn(f'"{field}"', self.prompt)

    def test_example_json_contains_bullets_list(self):
        self.assertIn('"bullets": [', self.prompt)

    def test_prompt_still_formats_with_only_the_text_variable(self):
        rendered = self.prompt.format(text="SYNTHETIC RESUME TEXT")
        self.assertIn("SYNTHETIC RESUME TEXT", rendered)
        self.assertIn('"bullets": [', rendered)

    def test_only_the_projects_section_gained_bullets(self):
        work_section = self.prompt[self.prompt.index("3. **Work Experience**"):self.prompt.index("4. **Key Projects**")]
        self.assertNotIn("bullets", work_section)


if __name__ == "__main__":
    unittest.main()
