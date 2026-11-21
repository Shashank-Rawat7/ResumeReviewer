"""End-to-end test of /parse_resume with the REAL parser, LangChain and Groq SDK against a local MOCK Groq server.

Run from the project root:  python -m unittest tests.test_mocked_pipeline -v

No live Groq or YouTube call is made and no API key is used: tests/docker/mock_groq.py answers on 127.0.0.1 and the
backend is pointed at it with GROQ_API_BASE. This proves the plumbing (upload file -> parse -> five model calls ->
clean -> save -> metrics line); it does NOT prove anything about the real model.

The scenario runs in a subprocess (importing backend.main has side effects) with a throw-away SQLite database, a
stubbed python-dotenv (the project's .env is never read) and the parser's file-saving hook replaced, so the generated
src/*.txt files are never written. Skipped when FastAPI, LangChain, Groq or PyPDF2 are not installed.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCENARIO = r'''
import contextlib, importlib.util, io, json, logging, os, sys, threading, types

root = os.environ["RR_ROOT"]
sys.path.insert(0, root)
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mock = load("mock_groq", "tests/docker/mock_groq.py")
minipdf = load("minipdf", "tests/docker/minipdf.py")
server = mock.serve(0, "127.0.0.1")
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["GROQ_API_BASE"] = "http://127.0.0.1:%d" % server.server_address[1]

try:
    import googleapiclient.discovery  # noqa: F401
except Exception:
    gac = types.ModuleType("googleapiclient"); disc = types.ModuleType("googleapiclient.discovery")
    disc.build = lambda *a, **k: object(); gac.discovery = disc
    sys.modules["googleapiclient"] = gac; sys.modules["googleapiclient.discovery"] = disc

from fastapi.testclient import TestClient
import backend.main as main
from src import metrics

out = {}
try:
    main.auth.verify_password("x", main.auth.hash_password("x")); out["real_hashing"] = True
except Exception:
    out["real_hashing"] = False
    main.auth.hash_password = lambda p: "plain$" + p
    main.auth.verify_password = lambda p, h: h == "plain$" + p

main.parser.save_text_to_file = lambda *a, **k: None          # never write the generated src/*.txt files
buf = io.StringIO()
metrics.configure_metrics_logger(stream=buf)
# everything else the backend prints or logs is captured too
cap_out, cap_err, cap_log = io.StringIO(), io.StringIO(), io.StringIO()
handler = logging.StreamHandler(cap_log); handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(handler); logging.getLogger().setLevel(logging.DEBUG)
stack = contextlib.ExitStack()
stack.enter_context(contextlib.redirect_stdout(cap_out)); stack.enter_context(contextlib.redirect_stderr(cap_err))
sys.excepthook = lambda *a: (stack.close(), sys.__excepthook__(*a))   # a crash must still be visible
out["retry_tracking_available"] = metrics._STATE.available

client = TestClient(main.app, raise_server_exceptions=False)
client.post("/register", json={"username": "carol", "email": "carol@example.test", "password": "pw"})
token = client.post("/login", data={"username": "carol", "password": "pw"}).json()["access_token"]
H = {"Authorization": "Bearer " + token}
sid = client.post("/start_session", params={"session_name": "mock"}, headers=H).json()["session_id"]

upload_dir = os.environ["UPLOAD_DIR"]
up = client.post("/upload_resume", files={"file": ("mock_resume.pdf", minipdf.make_pdf(minipdf.RESUME_LINES), "application/pdf")}, headers=H)
stored = up.json()["resume_filename"]
JD = "We need a Python engineer. " + mock.JD_SENTENCE
r = client.post("/parse_resume", data={"resume_filename": stored, "job_description": JD, "session_id": sid}, headers=H)
out["status"] = r.status_code
body = r.json() if r.status_code == 200 else {}
out["body_keys"] = sorted(body) if isinstance(body, dict) else None
out["body_text"] = json.dumps(body)
leftovers = [f for d, _, fs in os.walk(upload_dir) for f in fs]
out["upload_removed"] = leftovers == []
stack.close()
out["captured"] = cap_out.getvalue() + cap_err.getvalue() + cap_log.getvalue()
out["mock_requests"] = mock.Handler and __import__("urllib.request").request.urlopen(os.environ["GROQ_API_BASE"] + "/stats").read().decode()
saved = client.get("/load_session", params={"session_id": sid}, headers=H).json()
out["saved_keys"] = sorted(saved.get("resume_analysis", {}))
lines = [l for l in buf.getvalue().splitlines() if l.strip()]
out["metrics_lines"] = len(lines)
out["record"] = json.loads(lines[-1]) if lines else None
out["log_text"] = buf.getvalue()
print("RESULT:" + json.dumps(out))
'''


def _have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


@unittest.skipUnless(_have("fastapi", "httpx", "sqlalchemy", "jose", "passlib", "langchain_groq", "langchain", "PyPDF2", "yaml"),
                     "the full application dependencies are not installed")
class MockedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = os.path.join(tmp, "uploads")
            os.makedirs(uploads)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("POSTGRES_", "GROQ_", "YOUTUBE_", "DATABASE_", "SECRET_", "UPLOAD_", "API_BASE"))}
            env.update({"RR_ROOT": ROOT, "UPLOAD_DIR": uploads, "SECRET_KEY": "test-only-0f4b7c2e9a1d4c68b3e5a7f2c9d01e84-not-a-real-secret",
                        "GROQ_API_KEY": "mock-key-not-real", "YOUTUBE_API_KEY": "mock-key-not-real",
                        "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "t.db").replace("\\", "/"),
                        "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
            done = subprocess.run([sys.executable, "-B", "-c", SCENARIO], env=env, cwd=tmp, capture_output=True, text=True,
                                  timeout=170)
        marker = [l for l in done.stdout.splitlines() if l.startswith("RESULT:")]
        if not marker:
            raise AssertionError("scenario failed:\n" + done.stdout[-1500:] + "\n" + done.stderr[-3000:])
        cls.out = json.loads(marker[-1][7:])

    def test_the_request_succeeds_and_the_upload_is_consumed(self):
        self.assertEqual(self.out["status"], 200)
        self.assertTrue(self.out["upload_removed"])

    def test_the_response_shape_is_unchanged_and_is_saved(self):
        expected = ["ats_content", "credentials", "job_skills", "missing_skills", "summary"]
        for key in expected:
            self.assertIn(key, self.out["body_keys"])
        self.assertEqual(self.out["saved_keys"], self.out["body_keys"])
        self.assertNotIn("request_id", self.out["body_text"])
        self.assertNotIn("latency_ms", self.out["body_text"])

    def test_exactly_five_model_calls_reached_the_mock(self):
        self.assertEqual(json.loads(self.out["mock_requests"])["requests"], 5)

    def test_one_metrics_record_with_all_stages_and_tokens(self):
        self.assertEqual(self.out["metrics_lines"], 1)
        rec = self.out["record"]
        self.assertEqual(rec["outcome"], "ok")
        self.assertEqual(sorted(rec["stages"]), ["credentials", "e1", "e2", "e3", "summary"])
        for name, stage in rec["stages"].items():
            self.assertEqual(stage["status"], "ok", name)
            self.assertGreater(stage["input_tokens"], 0, name)
            self.assertEqual(stage["llm_calls"], 1, name)
        self.assertTrue(rec["totals"]["tokens_complete"])
        self.assertIs(rec["normalization_ok"], True)
        self.assertGreater(rec["pipeline_ms"], 0)

    def test_retries_are_counted_when_the_sdk_supports_it(self):
        rec = self.out["record"]
        if self.out["retry_tracking_available"]:
            self.assertEqual(rec["totals"]["retries"], 0)
            self.assertIsNone(rec["retries_unavailable_reason"])
        else:
            self.assertIsNone(rec["totals"]["retries"])
            self.assertEqual(rec["retries_unavailable_reason"], "tracking_unavailable")

    def test_nothing_resume_derived_reaches_stdout_stderr_or_logs(self):
        for text in ("Mock Candidate", "mock@example.test", "Kubernetes", "Mock Project", "mock-key-not-real", "pw"):
            self.assertNotIn(text, self.out["captured"], text)

    def test_the_metrics_line_holds_no_resume_job_or_secret_text(self):
        for text in ("Mock Candidate", "mock@example.test", "Kubernetes is required", "mock-key-not-real", "carol"):
            self.assertNotIn(text, self.out["log_text"])


if __name__ == "__main__":
    unittest.main()
