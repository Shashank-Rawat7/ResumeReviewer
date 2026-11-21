"""Regression tests: a blank line inside an st.markdown HTML block turns the rest into a code block.

Streamlit cleans markdown with textwrap.dedent(...).strip(), which makes whitespace-only lines truly blank.
A blank line ends a Markdown HTML block, and the indented line after it is then shown as literal text
(e.g. "<strong>Technologies:</strong>"). An empty placeholder on its own line caused exactly that.

Run from the project root:  python -m unittest tests.test_markdown_blocks -v
Uses synthetic data and a fake Streamlit. If markdown-it-py happens to be installed, a second check renders
the markdown; it is skipped otherwise (no extra dependency is required).
"""
import textwrap
import unittest

from tests.test_project_bullets import load_display

ID = "dQw4w9WgXcQ"


def cleaned(fake):
    """What Streamlit actually hands to the Markdown renderer, one string per st.markdown call."""
    return [textwrap.dedent(body).strip() for body in fake.markdown_calls]


def blank_lines(text):
    return [i for i, line in enumerate(text.split("\n")) if not line.strip()]


def render_html(text):
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None
    return MarkdownIt("commonmark").render(text)


def project(**overrides):
    base = {"project_name": "P", "objective": "Obj.", "technologies_used": ["Python", "SQL"]}
    base.update(overrides)
    return base


class ProjectBlocks(unittest.TestCase):
    def check(self, projects):
        module, fake = load_display()
        module.display_projects(projects)
        blocks = cleaned(fake)
        self.assertEqual(len(blocks), len(projects))
        for text in blocks:
            self.assertEqual(blank_lines(text), [], text)
            self.assertIn("<strong>Technologies:</strong> Python, SQL", text)
            rendered = render_html(text)
            if rendered is not None:
                self.assertNotIn("<pre>", rendered)
                self.assertNotIn("&lt;strong&gt;", rendered)
                self.assertIn("<strong>Technologies:</strong> Python, SQL", rendered)
        return blocks

    def test_bullets_view_with_empty_note(self):
        blocks = self.check([project(bullets=["Built it.", "Shipped it."])])
        self.assertNotIn("<small>", blocks[0])

    def test_bullets_view_with_note(self):
        blocks = self.check([project(bullets=["Built it.", "Shipped it."], bullets_unverified=[1])])
        self.assertIn("<small>", blocks[0])
        self.assertIn("<li>Built it.</li>", blocks[0])

    def test_legacy_view(self):
        self.check([project(outcomes=["Did a thing.", "Did another."])])

    def test_legacy_view_without_outcomes(self):
        self.check([project(outcomes=[])])


class CourseCardBlocks(unittest.TestCase):
    def check(self, channel):
        module, fake = load_display()
        fake.caption = lambda *a, **k: None
        module.display_youtube_courses(
            [{"title": "Docker intro", "channel": channel, "link": "https://www.youtube.com/watch?v=" + ID}],
            "Docker",
        )
        (text,) = cleaned(fake)
        self.assertEqual(blank_lines(text), [], text)
        self.assertIn("Watch on YouTube", text)
        self.assertIn("https://www.youtube.com/watch?v=" + ID, text)
        rendered = render_html(text)
        if rendered is not None:
            self.assertNotIn("<pre>", rendered)
            self.assertIn("<button", rendered)
        return text

    def test_empty_channel(self):
        self.assertNotIn("Channel:", self.check(""))

    def test_non_empty_channel(self):
        self.assertIn("Channel: Example Channel", self.check("Example Channel"))


if __name__ == "__main__":
    unittest.main()
