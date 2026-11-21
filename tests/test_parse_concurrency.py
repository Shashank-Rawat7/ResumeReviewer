"""Tests for how POST /parse_resume (backend/main.py) runs the analysis: in a worker thread, one analysis at a time.

Run from the project root:  python -m unittest tests.test_parse_concurrency -v

Why a real server: FastAPI's TestClient gives every request its own event loop, so it can never show that one request
blocks another. The scenario therefore starts a real Uvicorn server on 127.0.0.1 (in a thread of a SUBPROCESS) and
talks to it over HTTP. Everything else is offline and mocked: a throw-away SQLite file, a stubbed python-dotenv (the
real .env is never read), fake keys and a fake parser that sleeps instead of calling a model. No Docker, no
PostgreSQL, no Groq, no YouTube. The tests are skipped when the FastAPI test dependencies are not installed.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCENARIO = r'''
import asyncio, json, logging, os, socket, sys, threading, time, types
import urllib.error, urllib.parse, urllib.request

root = os.environ["RR_ROOT"]
sys.path.insert(0, root)
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)   # never read a real .env
try:
    import googleapiclient.discovery  # noqa: F401
except Exception:
    gac = types.ModuleType("googleapiclient")
    disc = types.ModuleType("googleapiclient.discovery")
    disc.build = lambda *a, **k: object()
    gac.discovery = disc
    sys.modules["googleapiclient"] = gac
    sys.modules["googleapiclient.discovery"] = disc

from fastapi.testclient import TestClient
import uvicorn
import src.metrics as metrics_module
import backend.main as main

out = {}
try:
    main.auth.verify_password("x", main.auth.hash_password("x"))
except Exception:
    main.auth.hash_password = lambda p: "plain$" + p
    main.auth.verify_password = lambda p, h: h == "plain$" + p

# ---- the fake parser: records where and how it ran, sleeps instead of calling a model ----
state = {"sleep": 0.0, "fail": False}
calls = []
active = {"now": 0, "max": 0}
guard = threading.Lock()
started = threading.Event()

def fake_parse(path, jd, metrics=None):
    try:
        asyncio.get_running_loop()
        on_loop = True
    except RuntimeError:
        on_loop = False
    with guard:
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
    call = {"thread": threading.get_ident(), "on_loop": on_loop, "start": time.monotonic(),
            "stage_visible": None, "file_there": os.path.exists(path)}
    calls.append(call)
    started.set()
    try:
        with metrics.stage("e1"):
            call["stage_visible"] = metrics_module._CURRENT_STAGE.get() is not None
            # what the Groq SDK logs when it retries a request; credited to the running stage via a ContextVar
            logging.getLogger("groq._base_client").info("Retrying request to %s in %f seconds", "https://api.example/x", 0.5)
            time.sleep(state["sleep"])
            if state["fail"]:
                raise RuntimeError("provider said: SECRET-RESUME-TEXT")
        return (json.dumps({"note": jd}), "{}", "{}", "{}", json.dumps({"bullets": []}),
                json.dumps({"normalization_ok": False, "skills": []}))
    finally:
        call["end"] = time.monotonic()
        with guard:
            active["now"] -= 1
main.parser.parse_resume = fake_parse

metric_lines = []
class Collect(logging.Handler):
    def emit(self, record):
        metric_lines.append(record.getMessage())
logging.getLogger("resume_reviewer.metrics").addHandler(Collect())

# ---- setup through the in-process client (registration, login, sessions, uploads) ----
client = TestClient(main.app, raise_server_exceptions=False)
upload_dir = os.environ["UPLOAD_DIR"]

def login_headers(name):
    client.post("/register", json={"username": name, "email": name + "@example.test", "password": "pw-" + name})
    token = client.post("/login", data={"username": name, "password": "pw-" + name}).json()["access_token"]
    return {"Authorization": "Bearer " + token}

def new_session(h, label):
    return client.post("/start_session", params={"session_name": label}, headers=h).json()["session_id"]

def new_upload(h):
    r = client.post("/upload_resume", files={"file": ("r.pdf", b"%PDF-1.4 fake resume", "application/pdf")}, headers=h)
    return r.json()["resume_filename"]

A = login_headers("alice")
sessions = [new_session(A, "s%d" % i) for i in range(6)]

# ---- a real server over HTTP ----
sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()
for _ in range(100):
    if server.started:
        break
    time.sleep(0.1)
base = "http://127.0.0.1:%d" % port
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def http(method, path, headers=None, data=None, timeout=60):
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    t0 = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    return {"status": status, "body": body.decode("utf-8", "replace"), "seconds": time.monotonic() - t0}

def parse_request(name, session_id):
    form = urllib.parse.urlencode({"resume_filename": name, "job_description": "JD", "session_id": session_id}).encode()
    return http("POST", "/parse_resume", headers=A, data=form)

def on_thread(results, key, fn, *args):
    t = threading.Thread(target=lambda: results.__setitem__(key, fn(*args)))
    t.start()
    return t

out["lock_is_threading_lock"] = type(main._ANALYSIS_LOCK) is type(threading.Lock())
out["server_thread"] = server_thread.ident

# ---- 1. the server stays responsive while one slow analysis runs ----
state["sleep"] = 1.5
started.clear()
results = {}
runner = on_thread(results, "slow", parse_request, new_upload(A), sessions[0])
started.wait(20)
health = [http("GET", "/health") for _ in range(5)]
out["analysis_still_running_after_probes"] = runner.is_alive()
runner.join()
slow_first = results["slow"]
# the authenticated endpoint gets its own slow analysis, so a blocked loop cannot hide behind the /health probes above
started.clear()
results = {}
runner = on_thread(results, "slow", parse_request, new_upload(A), sessions[5])
started.wait(20)
listing = [http("GET", "/my_sessions", headers=A) for _ in range(5)]
out["sessions_probes_during_analysis"] = runner.is_alive()
runner.join()
out["responsive"] = {
    "health_status": [h["status"] for h in health], "health_max_s": max(h["seconds"] for h in health),
    "sessions_status": [h["status"] for h in listing], "sessions_max_s": max(h["seconds"] for h in listing),
    "slow_status": slow_first["status"], "slow_seconds": slow_first["seconds"],
}

# ---- 2. two analyses at once: serialised, both succeed, metrics tell queue time from pipeline time ----
state["sleep"] = 0.6
active["max"] = 0
before_calls = len(calls)
del metric_lines[:]
results = {}
names = [new_upload(A), new_upload(A)]
t1 = on_thread(results, "a", parse_request, names[0], sessions[1])
t2 = on_thread(results, "b", parse_request, names[1], sessions[2])
t1.join(); t2.join()
pair = sorted(calls[before_calls:], key=lambda c: c["start"])
records = sorted((json.loads(line) for line in metric_lines), key=lambda r: r["request_ms"])
out["pair"] = {
    "statuses": sorted([results["a"]["status"], results["b"]["status"]]),
    "max_active": active["max"], "calls": len(pair),
    "second_started_after_first_ended": len(pair) == 2 and pair[1]["start"] >= pair[0]["end"] - 0.02,
    "on_loop": [c["on_loop"] for c in calls], "worker_is_server_thread": [c["thread"] == server_thread.ident for c in calls],
    "stage_visible": [c["stage_visible"] for c in calls],
    "records": [{"request_ms": r["request_ms"], "pipeline_ms": r["pipeline_ms"], "outcome": r["outcome"],
                 "e1_retries": r["stages"]["e1"]["retries"], "retries": r["totals"]["retries"],
                 "reason": r["retries_unavailable_reason"], "e1_status": r["stages"]["e1"]["status"]} for r in records],
}

# ---- 3. a parser failure: generic 500, upload removed, metrics say error, the lock is released ----
state["sleep"] = 0.05
state["fail"] = True
del metric_lines[:]
bad_name = new_upload(A)
bad = parse_request(bad_name, sessions[3])
state["fail"] = False
leftovers = [os.path.join(d, f) for d, _, fs in os.walk(upload_dir) for f in fs if bad_name in f]
after = parse_request(new_upload(A), sessions[4])
out["failure"] = {
    "status": bad["status"], "body": bad["body"], "leftovers": leftovers,
    "outcome": [json.loads(l)["outcome"] for l in metric_lines][:1],
    "next_status": after["status"], "next_seconds": after["seconds"],
}

server.should_exit = True
server_thread.join(timeout=15)
print("RESULT:" + json.dumps(out))
'''


def _have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


@unittest.skipUnless(_have("fastapi", "httpx", "sqlalchemy", "jose", "passlib", "langchain_groq", "yaml", "uvicorn", "multipart"),
                     "FastAPI test dependencies are not installed")
class ParseRunsOffTheEventLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = os.path.join(tmp, "uploads")
            os.makedirs(uploads)
            env = dict(os.environ)
            for name in list(env):
                if name.startswith(("POSTGRES_", "GROQ_", "YOUTUBE_", "DATABASE_", "SECRET_", "UPLOAD_", "API_BASE")):
                    env.pop(name)
            env.update({
                "RR_ROOT": ROOT, "UPLOAD_DIR": uploads,
                "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "t.db").replace("\\", "/"),
                "SECRET_KEY": "test-only-0f4b7c2e9a1d4c68b3e5a7f2c9d01e84-not-a-real-secret",
                "GROQ_API_KEY": "gsk_test_not_real", "YOUTUBE_API_KEY": "yt-test-not-real",
                "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            })
            done = subprocess.run([sys.executable, "-B", "-c", SCENARIO], env=env, cwd=tmp, capture_output=True,
                                  text=True, timeout=170)
        marker = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
        if not marker:
            raise AssertionError("scenario failed:\n" + done.stdout[-1500:] + "\n" + done.stderr[-3000:])
        cls.out = json.loads(marker[-1][len("RESULT:"):])

    # --- the server stays responsive during a slow analysis (the 1.5 s analysis would block a loop-bound parser) ---
    def test_health_stays_responsive_during_a_slow_analysis(self):
        r = self.out["responsive"]
        self.assertTrue(self.out["analysis_still_running_after_probes"])      # the probes really ran DURING the analysis
        self.assertEqual(r["health_status"], [200] * 5)
        self.assertLess(r["health_max_s"], 0.8)                              # blocked behaviour would be about 1.5 s
        self.assertEqual(r["slow_status"], 200)                              # and the analysis itself still succeeds
        self.assertGreaterEqual(r["slow_seconds"], 1.4)

    def test_an_authenticated_session_endpoint_stays_responsive(self):
        r = self.out["responsive"]
        self.assertTrue(self.out["sessions_probes_during_analysis"])         # probes ran while an analysis was running
        self.assertEqual(r["sessions_status"], [200] * 5)
        self.assertLess(r["sessions_max_s"], 0.8)

    # --- where the parser runs ---
    def test_the_parser_runs_outside_the_event_loop_thread(self):
        pair = self.out["pair"]
        self.assertTrue(pair["on_loop"] and not any(pair["on_loop"]))        # no running loop inside the parser
        self.assertFalse(any(pair["worker_is_server_thread"]))               # not the thread that runs the event loop
        self.assertTrue(all(pair["stage_visible"]))                          # the request's context reached the worker

    # --- one analysis at a time ---
    def test_two_concurrent_analyses_are_serialised_and_both_complete(self):
        pair = self.out["pair"]
        self.assertTrue(self.out["lock_is_threading_lock"])
        self.assertEqual(pair["statuses"], [200, 200])
        self.assertEqual(pair["calls"], 2)
        self.assertEqual(pair["max_active"], 1)
        self.assertTrue(pair["second_started_after_first_ended"])

    # --- metrics ---
    def test_metrics_separate_queue_time_from_pipeline_time(self):
        first, second = self.out["pair"]["records"]                          # sorted by request_ms
        for rec in (first, second):
            self.assertEqual(rec["outcome"], "ok")
            self.assertGreaterEqual(rec["pipeline_ms"], 450)                 # the analysis itself took about 600 ms
            self.assertLess(rec["pipeline_ms"], 1000)
        self.assertLess(first["request_ms"] - first["pipeline_ms"], 450)     # the first request did not queue
        self.assertGreaterEqual(second["request_ms"] - second["pipeline_ms"], 450)   # the second waited for the first
        self.assertGreaterEqual(second["request_ms"], 1050)                  # request_ms includes the wait

    def test_retry_attribution_still_works_in_the_worker_thread(self):
        for rec in self.out["pair"]["records"]:
            self.assertIsNone(rec["reason"])                                 # retries are trusted, not "unavailable"
            self.assertEqual(rec["e1_status"], "ok")
            self.assertEqual(rec["e1_retries"], 1)                           # each request is credited its own retry
            self.assertEqual(rec["retries"], 1)

    # --- failure handling ---
    def test_a_parser_failure_keeps_the_generic_500_and_cleans_up(self):
        f = self.out["failure"]
        self.assertEqual(f["status"], 500)
        self.assertEqual(json.loads(f["body"]), {"detail": "Resume analysis failed due to an internal error."})
        self.assertNotIn("SECRET-RESUME-TEXT", f["body"])
        self.assertEqual(f["leftovers"], [])                                 # no stored file and no .processing file
        self.assertEqual(f["outcome"], ["error"])

    def test_the_lock_is_released_after_a_failure(self):
        f = self.out["failure"]
        self.assertEqual(f["next_status"], 200)
        self.assertLess(f["next_seconds"], 5)                                # not stuck behind a lock that was never freed


class ParseCodeShape(unittest.TestCase):
    def source(self):
        with open(os.path.join(ROOT, "backend", "main.py"), "rb") as handle:
            return handle.read().decode("utf-8").replace("\r\n", "\n")

    def test_the_lock_is_a_module_level_threading_lock(self):
        source = self.source()
        self.assertEqual(len(re.findall(r"(?m)^_ANALYSIS_LOCK = threading\.Lock\(\)$", source)), 1)
        self.assertNotIn("asyncio.Lock", source)

    def test_the_parser_is_called_once_inside_the_helper_with_the_pipeline_timer_after_the_lock(self):
        source = self.source()
        self.assertEqual(source.count("parser.parse_resume("), 1)
        helper = source[source.index("def _run_analysis("):source.index("@app.post('/parse_resume')")]
        self.assertLess(helper.index("with _ANALYSIS_LOCK:"), helper.index("with request_metrics.pipeline():"))
        self.assertLess(helper.index("with request_metrics.pipeline():"), helper.index("parser.parse_resume("))
        self.assertIn("metrics=request_metrics", helper)

    def test_the_endpoint_awaits_the_helper_in_a_thread_and_keeps_the_metrics_guards(self):
        source = self.source()
        endpoint = source[source.index("async def parse_resume("):source.index("@app.get('/recommend_courses')")]
        self.assertIn("await asyncio.to_thread(_run_analysis, claimed_path, job_description, request_metrics)", endpoint)
        self.assertNotIn("parser.parse_resume(", endpoint)                   # never called directly on the event loop
        self.assertNotIn("request_metrics.pipeline()", endpoint)
        # the guards of tests/test_metrics.py
        self.assertEqual(source.count("request_metrics.emit(outcome)"), 1)
        self.assertRegex(source, r"finally:\s*\n\s*request_metrics\.emit\(outcome\)")
        self.assertEqual(len(re.findall(r"clean_resume_analysis\(", source)), 2)


if __name__ == "__main__":
    unittest.main()
