"""Tests for the course-recommendation failure signal, result shape and safe HTML rendering.

Run from the project root:  python -m unittest tests.test_course_recommendations -v
Synthetic data and a fake YouTube client only. No network, API key or .env file is used: the Google client
library and dotenv are replaced by stubs while src/recommender.py is imported.
"""
import importlib.util
import os
import re
import sys
import types
import unittest

from tests.test_project_bullets import load_display

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_recommender():
    """Import src/recommender.py with stubbed googleapiclient and dotenv; return the module."""
    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *args, **kwargs: None
    package = types.ModuleType("googleapiclient")
    package.discovery = discovery
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: False
    names = ("googleapiclient", "googleapiclient.discovery", "dotenv")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update({"googleapiclient": package, "googleapiclient.discovery": discovery, "dotenv": dotenv})
    try:
        spec = importlib.util.spec_from_file_location(
            "recommender_under_test", os.path.join(ROOT, "src", "recommender.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


recommender = load_recommender()


class FakeRequest:
    def __init__(self, outcome):
        self.outcome = outcome

    def execute(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeYouTube:
    """search: one response (or Exception) per search call, the last one repeating.
    details: {video_id: response or Exception} for the video-details calls."""

    def __init__(self, search, details=None):
        self.search_outcomes = search if isinstance(search, list) else [search]
        self.details = details or {}
        self.search_calls = 0

    def search(self):
        return self

    def videos(self):
        return self

    def list(self, **kwargs):
        if "q" in kwargs:
            index = min(self.search_calls, len(self.search_outcomes) - 1)
            self.search_calls += 1
            return FakeRequest(self.search_outcomes[index])
        return FakeRequest(self.details[kwargs["id"]])


def found(*video_ids):
    return {"items": [{"id": {"videoId": video_id}} for video_id in video_ids]}


def detail(title, channel="Example Channel", views="1000"):
    return {"items": [{
        "snippet": {"title": title, "channelTitle": channel, "publishedAt": "2024-01-01T00:00:00Z",
                    "description": "synthetic"},
        "statistics": {"viewCount": views, "likeCount": str(int(views) // 50), "commentCount": "1"},
        "contentDetails": {"duration": "PT15M"},
    }]}


def make(search, details=None):
    instance = recommender.SkillRecommender.__new__(recommender.SkillRecommender)
    instance.youtube = FakeYouTube(search, details)
    return instance


ID_A, ID_B, ID_C = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"


class FailureSignal(unittest.TestCase):
    def test_all_searches_failing_is_a_provider_error_not_an_empty_list(self):
        rec = make(RuntimeError("quota exceeded key=SECRET-KEY"))
        with self.assertRaises(recommender.RecommenderProviderError) as caught:
            rec.get_missing_skills_resources("Docker")
        self.assertNotIn("SECRET-KEY", str(caught.exception))
        self.assertEqual(str(caught.exception), recommender.PROVIDER_FAILURE_MESSAGE)

    def test_successful_search_with_no_videos_is_a_normal_empty_result(self):
        rec = make(found())
        self.assertEqual(rec.fetch_youtube_resources("Docker"), [])
        self.assertEqual(rec.get_missing_skills_resources("Docker"), [])

    def test_videos_found_but_every_details_call_failing_is_a_provider_error(self):
        rec = make(found(ID_A, ID_B), {ID_A: RuntimeError("boom"), ID_B: RuntimeError("boom")})
        with self.assertRaises(recommender.RecommenderProviderError):
            rec.get_missing_skills_resources("Docker")

    def test_a_video_without_details_is_skipped_and_the_result_stays_a_normal_empty_list(self):
        rec = make(found(ID_A), {ID_A: {"items": []}})
        self.assertEqual(rec.get_missing_skills_resources("Docker"), [])

    def test_some_failed_searches_still_return_the_results_of_the_others(self):
        failing = RuntimeError("temporary")
        rec = make([failing, failing, failing, failing, found(ID_A)], {ID_A: detail("Docker basics")})
        self.assertEqual([r["title"] for r in rec.get_missing_skills_resources("Docker")], ["Docker basics"])

    def test_one_details_failure_does_not_discard_the_other_results(self):
        rec = make(found(ID_A, ID_B), {ID_A: RuntimeError("boom"), ID_B: detail("Docker deep dive", "Chan B")})
        result = rec.get_missing_skills_resources("Docker")
        self.assertEqual([(r["title"], r["channel"]) for r in result], [("Docker deep dive", "Chan B")])


class ResultShape(unittest.TestCase):
    def test_titles_and_channels_survive_with_link_only_extra(self):
        rec = make(found(ID_A, ID_B), {ID_A: detail("First video", "Chan A", "5000"),
                                       ID_B: detail("Second video", "Chan B", "50")})
        result = rec.get_missing_skills_resources("Pytest")
        self.assertEqual({tuple(sorted(item)) for item in result}, {("channel", "link", "title")})
        by_title = {item["title"]: item for item in result}
        self.assertEqual(by_title["First video"]["channel"], "Chan A")
        self.assertEqual(by_title["Second video"]["channel"], "Chan B")
        self.assertIn("v=" + ID_A, by_title["First video"]["link"])

    def test_existing_ordering_duplicates_and_limit_are_unchanged(self):
        rec = make(found(ID_A, ID_B, ID_C, ID_A),
                   {ID_A: detail("low", views="10"), ID_B: detail("high", views="2000000"),
                    ID_C: detail("mid", views="20000")})
        result = rec.get_missing_skills_resources("Pytest", max_results=2)
        self.assertEqual([r["title"] for r in result], ["high", "mid"])


class EndpointAndWiringSource(unittest.TestCase):
    """backend/main.py and frontend/main.py cannot be imported in a test, so their source text is checked."""

    @staticmethod
    def read(*parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_endpoint_reports_provider_failure_with_a_non_success_status_and_no_raw_error(self):
        source = self.read("backend", "main.py")
        function = source.split("def recommend_courses", 1)[1].split("\n\n", 1)[0]
        self.assertIn("except RecommenderProviderError", function)
        self.assertIn("status_code=502", function)
        self.assertIn("status_code=500", function)
        self.assertIn("return {'courses': courses}", function)
        self.assertNotIn("str(e)", function)
        self.assertNotIn("'error'", function)

    def test_frontend_uses_the_shared_result_handler(self):
        source = self.read("frontend", "main.py")
        self.assertIn("display_course_recommendations(resp.status_code, body, skill)", source)


class CourseRendering(unittest.TestCase):
    def setUp(self):
        self.display, self.fake = load_display()
        self.fake.infos, self.fake.errors, self.fake.captions, self.fake.subheaders = [], [], [], []
        self.fake.info = lambda message, *a, **k: self.fake.infos.append(message)
        self.fake.error = lambda message, *a, **k: self.fake.errors.append(message)
        self.fake.caption = lambda message, *a, **k: self.fake.captions.append(message)
        self.fake.subheader = lambda message, *a, **k: self.fake.subheaders.append(message)

    def html(self):
        return "\n".join(self.fake.markdown_calls)

    def course(self, **overrides):
        item = {"title": "Docker in 20 minutes", "channel": "Example Channel",
                "link": "https://www.youtube.com/watch?v=" + ID_A + "&search=Docker"}
        item.update(overrides)
        return item

    def test_title_channel_and_a_working_link_are_shown(self):
        self.display.display_youtube_courses([self.course()], "Docker")
        out = self.html()
        self.assertIn("Docker in 20 minutes", out)
        self.assertIn("Channel: Example Channel", out)
        self.assertIn("For skill: Docker", out)
        self.assertIn('href="https://www.youtube.com/watch?v=' + ID_A + '"', out)
        self.assertIn("https://img.youtube.com/vi/" + ID_A + "/0.jpg", out)

    @staticmethod
    def luminance(hex_color):
        digits = hex_color.lstrip("#")
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        channels = [int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(self, foreground, background):
        a, b = sorted((self.luminance(foreground), self.luminance(background)), reverse=True)
        return (a + 0.05) / (b + 0.05)

    def test_title_is_dark_bold_readable_on_the_light_card(self):
        self.display.display_youtube_courses([self.course(title="Unique Title Text")], "Docker")
        out = self.html()
        title_style = re.search(r'<div style="([^"]*)">Unique Title Text</div>', out).group(1)
        card_style = re.search(r'<div style="(display: flex[^"]*)"', out).group(1)
        text_color = re.search(r"(?<![-a-z])color:\s*(#[0-9a-fA-F]{3,6})", title_style).group(1)
        card_color = re.search(r"background-color:\s*(#[0-9a-fA-F]{3,6})", card_style).group(1)
        self.assertEqual(card_color.lower(), "#f9f9f9")                  # card background unchanged
        self.assertGreaterEqual(self.contrast(text_color, card_color), 7)  # WCAG AAA for normal text
        self.assertIn("font-weight:600", title_style)
        self.assertNotIn("<h4", out)             # a Streamlit-styled heading would add its own padding and colour
        self.assertIn("color: #111", card_style)  # every text inside the card inherits a dark colour

    def test_every_text_line_on_the_card_is_readable(self):
        self.display.display_youtube_courses([self.course()], "Docker")
        out = self.html()
        for label in ("For skill: Docker", "Channel: Example Channel"):
            style = re.search(r'<div style="([^"]*)">' + re.escape(label), out).group(1)
            color = re.search(r"(?<![-a-z])color:\s*(#[0-9a-fA-F]{3,6})", style).group(1)
            self.assertGreaterEqual(self.contrast(color, "#f9f9f9"), 7, label)

    def test_long_titles_wrap_instead_of_overflowing(self):
        self.display.display_youtube_courses([self.course(title="W" * 300)], "Docker")
        out = self.html()
        self.assertIn("overflow-wrap:anywhere", out)
        self.assertIn("min-width: 0", out)        # lets the text column shrink beside the fixed-width thumbnail
        self.assertIn("flex-shrink: 0", out)      # the thumbnail keeps its width

    def test_hostile_skill_title_and_channel_are_escaped(self):
        hostile = self.course(title="<script>alert(1)</script>", channel='"><b>x</b>')
        self.display.display_youtube_courses([hostile], "<img src=x onerror=alert(1)>")
        out = self.html()
        for raw in ("<img src=x", "<script>", "<b>x</b>"):
            self.assertNotIn(raw, out)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", out)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", out)

    def test_links_that_are_not_valid_youtube_watch_links_are_skipped(self):
        bad = ["javascript:alert(1)", "http://www.youtube.com/watch?v=" + ID_A,
               "https://evil.example/watch?v=" + ID_A, "https://www.youtube.com/watch?v=short",
               'https://www.youtube.com/watch?v=' + ID_A + '"onmouseover="x', "", None]
        self.display.display_youtube_courses([self.course(link=link) for link in bad], "Docker")
        self.assertEqual(self.html(), "")
        self.assertEqual(len(self.fake.captions), 1)
        self.assertIn(str(len(bad)), self.fake.captions[0])

    def test_plain_link_strings_from_older_callers_still_render(self):
        self.display.display_youtube_courses(["https://www.youtube.com/watch?v=" + ID_B], "Docker")
        out = self.html()
        self.assertIn("Docker Course", out)
        self.assertIn('href="https://www.youtube.com/watch?v=' + ID_B + '"', out)

    def test_success_with_courses_shows_the_cards(self):
        self.display.display_course_recommendations(200, {"courses": [self.course()]}, "Docker")
        self.assertEqual(self.fake.subheaders, ["🎓 Recommended YouTube Courses for Docker"])
        self.assertIn("Docker in 20 minutes", self.html())
        self.assertEqual((self.fake.infos, self.fake.errors), ([], []))

    def test_a_genuinely_empty_result_is_an_info_message_not_an_error(self):
        self.display.display_course_recommendations(200, {"courses": []}, "Docker")
        self.assertEqual(len(self.fake.infos), 1)
        self.assertIn("found no matching", self.fake.infos[0])
        self.assertEqual(self.fake.errors, [])

    def test_a_provider_failure_is_an_error_and_never_looks_like_no_results(self):
        self.display.display_course_recommendations(502, {"detail": "provider"}, "Docker")
        self.assertEqual(self.fake.errors, [self.display.COURSE_PROVIDER_ERROR_MESSAGE])
        self.assertEqual(self.fake.infos, [])
        self.assertEqual(self.fake.subheaders, [])

    def test_other_failures_and_malformed_bodies_show_the_generic_error(self):
        for status, body in ((500, {"detail": "x"}), (200, None), (200, {"error": "x"}), (404, None)):
            self.fake.errors.clear()
            self.display.display_course_recommendations(status, body, "Docker")
            self.assertEqual(self.fake.errors, ["Failed to fetch recommendations."], (status, body))


if __name__ == "__main__":
    unittest.main()
