"""Security tests for the backend API (backend/main.py).

Covers: the removed unauthenticated /db_check and /get_all_data endpoints, authentication on the
session/analysis endpoints, cross-user data isolation, the upload lifecycle (server-generated names, per-user
ownership, strict file-name validation, ordering of the ownership checks, cleanup after success and failure,
concurrent requests), the authenticated and length-limited /recommend_courses, credentials in a JSON body (never
in a URL or an access log) and the absence of resume content in stdout, stderr and logs.

Run from the project root:  python -m unittest tests.test_security_endpoints -v

How: the scenario runs in a SUBPROCESS so importing backend.main (which connects to its database and builds the
parser at import time) cannot leak modules into the rest of the test run. It uses a throw-away SQLite file in a
temporary directory, fake API keys, a stubbed python-dotenv (the project's .env is never read), a mocked
parser and a mocked course recommender. No network, no PostgreSQL, no live Groq or YouTube call. The tests are
skipped when FastAPI / its test client / SQLAlchemy are not installed.
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
import contextlib, io, json, logging, os, sys, threading, types
from unittest import mock

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
import backend.main as main

SECRET = "RESUME-CONTENT-XYZ"
out = {}
try:                                    # real hashing when the bcrypt backend exists, else a plain stand-in
    main.auth.verify_password("x", main.auth.hash_password("x"))
    out["real_password_hashing"] = True
except Exception:
    out["real_password_hashing"] = False
    main.auth.hash_password = lambda p: "plain$" + p
    main.auth.verify_password = lambda p, h: h == "plain$" + p

calls = []
mode = {"fail": False, "sleep": 0.0}
def fake_parse(path, jd, metrics=None):
    calls.append(os.path.basename(path))
    if mode["sleep"]:
        import time; time.sleep(mode["sleep"])
    if mode["fail"]:
        try:
            raise RuntimeError("provider said: " + SECRET)
        except RuntimeError:
            logging.exception("Resume summarization failed")     # what src/parser.py does
            raise
    return (json.dumps({"note": jd, "text": SECRET}), "{}", "{}", "{}", json.dumps({"bullets": [SECRET]}),
            json.dumps({"normalization_ok": False, "skills": []}))
main.parser.parse_resume = fake_parse
rec_calls = []
main.recommender.get_missing_skills_resources = lambda skill: rec_calls.append(skill) or [{"title": "x"}]

client = TestClient(main.app, raise_server_exceptions=False)
upload_dir = os.environ["UPLOAD_DIR"]
out["upload_dir_used"] = os.path.abspath(main.UPLOAD_DIR) == os.path.abspath(upload_dir)
out["routes"] = sorted({r.path for r in main.app.routes})

# everything the app prints or logs from here on is captured
cap_out, cap_err, cap_log = io.StringIO(), io.StringIO(), io.StringIO()
handler = logging.StreamHandler(cap_log); handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(handler); logging.getLogger().setLevel(logging.DEBUG)
stack = contextlib.ExitStack()
stack.enter_context(contextlib.redirect_stdout(cap_out)); stack.enter_context(contextlib.redirect_stderr(cap_err))

def status(resp):
    return resp.status_code

def register(name, **over):
    body = {"username": name, "email": name + "@example.test", "password": "pw-" + name}
    body.update(over)
    return client.post("/register", json=body)

def register_login(name):
    r = register(name)
    l = client.post("/login", data={"username": name, "password": "pw-" + name})
    token = l.json().get("access_token")
    return status(r), status(l), {"Authorization": "Bearer " + str(token)}

def tree():
    found = {}
    for d, _, files in os.walk(upload_dir):
        for f in files:
            found[os.path.relpath(os.path.join(d, f), upload_dir).replace(os.sep, "/")] = open(os.path.join(d, f), "rb").read()
    return found

def upload(headers, content, original="resume.pdf"):
    return client.post("/upload_resume", files={"file": (original, content, "application/pdf")}, headers=headers)

def parse(headers, name, sid, jd="JD"):
    return client.post("/parse_resume", data={"resume_filename": name, "job_description": jd, "session_id": sid}, headers=headers)

# ---- 1. users, removed endpoints, authentication ----
sc_a, sl_a, A = register_login("alice")
sc_b, sl_b, B = register_login("bob")
out["register_login"] = [sc_a, sl_a, sc_b, sl_b]
out["removed_unauth"] = [status(client.get("/db_check")), status(client.get("/get_all_data"))]
out["removed_authed"] = [status(client.get("/db_check", headers=A)), status(client.get("/get_all_data", headers=A))]
bad = {"Authorization": "Bearer not-a-token"}
out["unauth"] = {
    "my_sessions": status(client.get("/my_sessions")),
    "start_session": status(client.post("/start_session", params={"session_name": "s"})),
    "load_session": status(client.get("/load_session", params={"session_id": 1})),
    "parse_resume": status(client.post("/parse_resume", data={"resume_filename": "a" * 32 + ".pdf", "job_description": "x", "session_id": 1})),
    "upload_resume": status(client.post("/upload_resume", files={"file": ("a.pdf", b"x", "application/pdf")})),
    "recommend_courses": status(client.get("/recommend_courses", params={"skill": "Python"})),
    "bad_token_my_sessions": status(client.get("/my_sessions", headers=bad)),
    "bad_token_load": status(client.get("/load_session", params={"session_id": 1}, headers=bad)),
    "bad_token_upload": status(client.post("/upload_resume", files={"file": ("a.pdf", b"x", "application/pdf")}, headers=bad)),
    "bad_token_recommend": status(client.get("/recommend_courses", params={"skill": "Python"}, headers=bad)),
}
out["upload_dir_untouched_by_unauth"] = tree() == {}

# ---- 2. sessions are private ----
sa = client.post("/start_session", params={"session_name": "alice-session"}, headers=A).json()["session_id"]
sb = client.post("/start_session", params={"session_name": "bob-session"}, headers=B).json()["session_id"]
out["my_sessions_a"] = [s["id"] for s in client.get("/my_sessions", headers=A).json()]
out["my_sessions_b"] = [s["id"] for s in client.get("/my_sessions", headers=B).json()]
out["ids"] = [sa, sb]
out["load_own"] = status(client.get("/load_session", params={"session_id": sa}, headers=A))
out["load_other"] = status(client.get("/load_session", params={"session_id": sa}, headers=B))
out["load_missing"] = status(client.get("/load_session", params={"session_id": 99999}, headers=A))

# ---- 3. two users upload files with the SAME original name ----
ra = upload(A, b"%PDF-1.4 ALICE-FILE")
rb = upload(B, b"%PDF-1.4 BOB-FILE")
name_a, name_b = ra.json()["resume_filename"], rb.json()["resume_filename"]
files = tree()
out["upload_status"] = [ra.status_code, rb.status_code]
out["names"] = [name_a, name_b]
out["same_original_distinct_names"] = name_a != name_b
out["stored_paths_depth"] = sorted(len(p.split("/")) for p in files)
out["contents_intact"] = sorted(files.values()) == sorted([b"%PDF-1.4 ALICE-FILE", b"%PDF-1.4 BOB-FILE"])
out["original_name_not_used"] = not any("resume" in p for p in files)
out["owner_dirs_differ"] = len({p.split("/")[0] for p in files}) == 2
path_b = [p for p, c in files.items() if c.endswith(b"BOB-FILE")][0]
path_a = [p for p, c in files.items() if c.endswith(b"ALICE-FILE")][0]
out["empty_upload"] = status(upload(A, b""))

# ---- 4. user A attacks user B's upload ----
before = len(calls)
r1 = parse(A, name_b, sa)                      # A's own session, B's file name
out["a_parse_b_file"] = status(r1)
out["a_parse_b_file_detail"] = r1.json().get("detail")
r2 = parse(A, name_b, sb)                      # B's session, B's file name
out["a_parse_b_file_in_b_session"] = status(r2)
r3 = parse(B, name_a, sb)                      # B tries A's file
out["b_parse_a_file"] = status(r3)
after = tree()
out["b_file_survived"] = after.get(path_b) == b"%PDF-1.4 BOB-FILE"
out["a_file_survived"] = after.get(path_a) == b"%PDF-1.4 ALICE-FILE"
out["no_model_call_for_foreign_file"] = len(calls) == before
# A uploading again cannot replace B's file: a new, different name in A's own folder
ra2 = upload(A, b"%PDF-1.4 ALICE-AGAIN")
out["reupload_new_name"] = ra2.json()["resume_filename"] not in (name_a, name_b)
out["b_file_still_intact"] = tree().get(path_b) == b"%PDF-1.4 BOB-FILE"

# ---- 5. file-name validation (400 / 422, never a filesystem error, no model call) ----
good = name_a
invalid = ["../outside.pdf", "..\\outside.pdf", "a/b.pdf", "/etc/passwd", "..", ".", "resume.pdf", good.upper(),
           good[:-4] + ".PDF", good[:-4] + ".txt", good + "\n", good[:-5] + "g.pdf", good[:-1], good + ".processing",
           "0" * 31 + ".pdf", "0" * 33 + ".pdf", os.path.join(os.environ["RR_TMP"], "x.pdf"), " " + good, good + " ", ""]
before = len(calls)
out["invalid_status"] = {repr(n): status(parse(A, n, sa)) for n in invalid}
out["invalid_made_no_model_call"] = len(calls) == before
out["invalid_left_files_alone"] = tree().get(path_a) == b"%PDF-1.4 ALICE-FILE"
out["nonexistent_valid_name"] = status(parse(A, "0" * 32 + ".pdf", sa))

# ---- 6. session ownership is checked before the file is touched or a model is called ----
before = len(calls)
r = parse(A, name_a, sb)                       # A's own file, but B's session
out["a_file_into_b_session"] = status(r)
out["foreign_session_no_model_call"] = len(calls) == before
out["foreign_session_file_not_consumed"] = tree().get(path_a) == b"%PDF-1.4 ALICE-FILE"
out["missing_session"] = status(parse(A, name_a, 99999))
out["missing_session_file_not_consumed"] = tree().get(path_a) == b"%PDF-1.4 ALICE-FILE"

# ---- 7. success: saved only into the caller's session, file removed, response has no metrics ----
r = parse(A, name_a, sa, jd="ALICE-DATA")
out["parse_own"] = status(r)
out["parse_own_body_has_metrics"] = ("request_id" in json.dumps(r.json())) if r.status_code == 200 else None
out["after_success_files"] = sorted(p for p in tree() if not p.endswith(path_b.split("/")[-1]))
saved = client.get("/load_session", params={"session_id": sa}, headers=A).json()
out["alice_data_saved"] = "ALICE-DATA" in json.dumps(saved)
out["bob_cannot_read_alice"] = "ALICE-DATA" not in json.dumps(client.get("/load_session", params={"session_id": sb}, headers=B).json())
out["second_parse_same_file"] = status(parse(A, name_a, sa))          # already consumed

# ---- 8. cleanup after a failure (and the error does not leak) ----
mode["fail"] = True
rf = upload(A, b"%PDF-1.4 FAIL")
fail_name = rf.json()["resume_filename"]
r = parse(A, fail_name, sa)
out["failure_status"] = status(r)
out["failure_body"] = json.dumps(r.json())
out["failure_file_removed"] = not any(p.endswith(fail_name) or p.endswith(fail_name + ".processing") for p in tree())
out["failure_other_users_file_kept"] = tree().get(path_b) == b"%PDF-1.4 BOB-FILE"
mode["fail"] = False

# ---- 9. concurrent requests for the same upload: exactly one wins ----
mode["sleep"] = 0.3
rc = upload(A, b"%PDF-1.4 RACE")
race_name = rc.json()["resume_filename"]
results = []
def worker():
    c = TestClient(main.app, raise_server_exceptions=False)
    results.append(c.post("/parse_resume", data={"resume_filename": race_name, "job_description": "R", "session_id": sa}, headers=A).status_code)
threads = [threading.Thread(target=worker) for _ in range(2)]
[t.start() for t in threads]; [t.join() for t in threads]
mode["sleep"] = 0.0
out["race_statuses"] = sorted(results)
out["race_no_leftovers"] = not any(race_name in p for p in tree())

# ---- 10. /recommend_courses ----
rec_before = len(rec_calls)
out["rec_unauth_no_call"] = len(rec_calls) == rec_before
out["rec_authed"] = status(client.get("/recommend_courses", params={"skill": "Python"}, headers=A))
out["rec_authed_body"] = client.get("/recommend_courses", params={"skill": "Python"}, headers=A).json()
out["rec_empty"] = status(client.get("/recommend_courses", params={"skill": ""}, headers=A))
out["rec_missing"] = status(client.get("/recommend_courses", headers=A))
out["rec_100"] = status(client.get("/recommend_courses", params={"skill": "x" * 100}, headers=A))
out["rec_101"] = status(client.get("/recommend_courses", params={"skill": "x" * 101}, headers=A))
out["rec_101_no_call"] = ("x" * 101) not in rec_calls

# ---- 11. credentials: JSON body only ----
req = register("carol")
out["register_ok"] = [status(req), req.json()]
out["register_request_url"] = str(req.request.url)
out["register_duplicate"] = status(register("carol"))
out["register_query_only"] = status(client.post("/register", params={"username": "dave", "email": "d@example.test", "password": "pw-dave-QUERY"}))
out["register_query_only_created_nothing"] = status(client.post("/login", data={"username": "dave", "password": "pw-dave-QUERY"}))
out["register_missing"] = [status(client.post("/register", json={"username": "x"})), status(client.post("/register", json={}))]
out["register_empty_fields"] = [status(register("e1", username="")), status(register("e2", email="")), status(register("e3", password=""))]
lg = client.post("/login", data={"username": "carol", "password": "pw-carol"})
out["login_ok"] = [status(lg), sorted(lg.json())]
out["login_request_url"] = str(lg.request.url)
out["login_wrong_password"] = status(client.post("/login", data={"username": "carol", "password": "nope"}))
out["login_unknown_user"] = status(client.post("/login", data={"username": "ghost", "password": "x"}))

# ---- 12. nothing sensitive in stdout / stderr / logs ----
stack.close()
out["captured_stdout"] = cap_out.getvalue()
out["captured_stderr"] = cap_err.getvalue()
out["captured_log"] = cap_log.getvalue()
out["parser_calls"] = len(calls)

# ---- 13. a real server's access log: no credentials in the request line ----
import socket, urllib.request, urllib.parse, uvicorn
access = io.StringIO()
with contextlib.redirect_stdout(access), contextlib.redirect_stderr(access):
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True); thread.start()
    import time
    for _ in range(100):
        if server.started: break
        time.sleep(0.1)
    base = "http://127.0.0.1:%d" % port
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/register", data=json.dumps({"username": "erin", "email": "erin@example.test", "password": "PW-ERIN-SECRET"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    opener.open(req, timeout=20).read()
    req = urllib.request.Request(base + "/login", data=urllib.parse.urlencode({"username": "erin", "password": "PW-ERIN-SECRET"}).encode(), method="POST")
    opener.open(req, timeout=20).read()
    server.should_exit = True; thread.join(timeout=10)
out["access_log"] = access.getvalue()
print("RESULT:" + json.dumps(out))
'''
def _have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


@unittest.skipUnless(_have("fastapi", "httpx", "sqlalchemy", "jose", "passlib", "langchain_groq", "yaml", "uvicorn", "multipart"),
                     "FastAPI test dependencies are not installed")
class SecurityEndpoints(unittest.TestCase):
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
                "RR_ROOT": ROOT, "RR_TMP": tmp, "UPLOAD_DIR": uploads,
                "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "t.db").replace("\\", "/"),
                "SECRET_KEY": "test-only-0f4b7c2e9a1d4c68b3e5a7f2c9d01e84-not-a-real-secret", "GROQ_API_KEY": "gsk_test_not_real", "YOUTUBE_API_KEY": "yt-test-not-real",
                "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            })
            done = subprocess.run([sys.executable, "-B", "-c", SCENARIO], env=env, cwd=tmp, capture_output=True,
                                  text=True, timeout=170)
        marker = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
        if not marker:
            raise AssertionError("scenario failed:\n" + done.stdout[-1500:] + "\n" + done.stderr[-3000:])
        cls.out = json.loads(marker[-1][len("RESULT:"):])
        cls.everything_logged = "\n".join(cls.out[k] for k in ("captured_stdout", "captured_stderr", "captured_log"))

    # --- setup, removed endpoints, authentication, sessions (unchanged protections) ---
    def test_setup_registers_and_logs_in_two_users(self):
        self.assertEqual(self.out["register_login"], [200, 200, 200, 200])

    def test_upload_dir_comes_from_the_environment(self):
        self.assertTrue(self.out["upload_dir_used"])

    def test_the_unauthenticated_data_dump_endpoints_are_gone(self):
        for path in ("/db_check", "/get_all_data"):
            self.assertNotIn(path, self.out["routes"])
        self.assertEqual(self.out["removed_unauth"], [404, 404])
        self.assertEqual(self.out["removed_authed"], [404, 404])

    def test_endpoints_require_authentication(self):
        for name, code in self.out["unauth"].items():
            self.assertEqual(code, 401, name)
        self.assertTrue(self.out["upload_dir_untouched_by_unauth"])

    def test_each_user_sees_only_their_own_sessions(self):
        sa, sb = self.out["ids"]
        self.assertEqual(self.out["my_sessions_a"], [sa])
        self.assertEqual(self.out["my_sessions_b"], [sb])

    def test_loading_another_users_session_is_refused(self):
        self.assertEqual(self.out["load_own"], 200)
        self.assertEqual(self.out["load_other"], 404)
        self.assertEqual(self.out["load_missing"], 404)

    # --- upload ownership ---
    def test_same_original_file_name_from_two_users_does_not_collide(self):
        self.assertEqual(self.out["upload_status"], [200, 200])
        self.assertTrue(self.out["same_original_distinct_names"])
        for name in self.out["names"]:
            self.assertRegex(name, r"\A[0-9a-f]{32}\.pdf\Z")
        self.assertTrue(self.out["contents_intact"])
        self.assertTrue(self.out["original_name_not_used"])
        self.assertTrue(self.out["owner_dirs_differ"])
        self.assertEqual(self.out["stored_paths_depth"], [2, 2])
        self.assertEqual(self.out["empty_upload"], 400)

    def test_user_a_cannot_parse_user_bs_upload(self):
        self.assertEqual(self.out["a_parse_b_file"], 404)
        self.assertEqual(self.out["a_parse_b_file_detail"], "Resume file not found")
        self.assertEqual(self.out["a_parse_b_file_in_b_session"], 404)   # not even inside B's own session id
        self.assertEqual(self.out["b_parse_a_file"], 404)
        self.assertTrue(self.out["no_model_call_for_foreign_file"])

    def test_user_a_cannot_delete_or_overwrite_user_bs_upload(self):
        self.assertTrue(self.out["b_file_survived"])
        self.assertTrue(self.out["a_file_survived"])
        self.assertTrue(self.out["reupload_new_name"])
        self.assertTrue(self.out["b_file_still_intact"])

    def test_invalid_file_names_are_rejected_without_touching_files_or_models(self):
        for name, code in self.out["invalid_status"].items():
            self.assertIn(code, (400, 422), name)
        self.assertGreaterEqual(len(self.out["invalid_status"]), 15)
        self.assertTrue(self.out["invalid_made_no_model_call"])
        self.assertTrue(self.out["invalid_left_files_alone"])
        self.assertEqual(self.out["nonexistent_valid_name"], 404)

    def test_session_ownership_is_checked_before_file_or_model(self):
        self.assertEqual(self.out["a_file_into_b_session"], 404)
        self.assertTrue(self.out["foreign_session_no_model_call"])
        self.assertTrue(self.out["foreign_session_file_not_consumed"])
        self.assertEqual(self.out["missing_session"], 404)
        self.assertTrue(self.out["missing_session_file_not_consumed"])

    def test_success_saves_into_the_callers_session_and_removes_the_upload(self):
        self.assertEqual(self.out["parse_own"], 200)
        self.assertIs(self.out["parse_own_body_has_metrics"], False)
        self.assertTrue(self.out["alice_data_saved"])
        self.assertTrue(self.out["bob_cannot_read_alice"])
        self.assertEqual(len(self.out["after_success_files"]), 1)          # only the not-yet-processed re-upload remains
        self.assertEqual(self.out["second_parse_same_file"], 404)

    def test_a_failed_analysis_cleans_up_and_does_not_leak(self):
        self.assertEqual(self.out["failure_status"], 500)
        self.assertNotIn("RESUME-CONTENT-XYZ", self.out["failure_body"])
        self.assertNotIn("provider said", self.out["failure_body"])
        self.assertTrue(self.out["failure_file_removed"])
        self.assertTrue(self.out["failure_other_users_file_kept"])

    def test_concurrent_requests_for_one_upload_give_exactly_one_winner(self):
        self.assertEqual(self.out["race_statuses"], [200, 404])
        self.assertTrue(self.out["race_no_leftovers"])

    # --- /recommend_courses ---
    def test_recommend_courses_requires_a_token_and_validates_input(self):
        self.assertEqual(self.out["unauth"]["recommend_courses"], 401)
        self.assertEqual(self.out["unauth"]["bad_token_recommend"], 401)
        self.assertTrue(self.out["rec_unauth_no_call"])
        self.assertEqual(self.out["rec_authed"], 200)
        self.assertEqual(self.out["rec_authed_body"], {"courses": [{"title": "x"}]})
        self.assertEqual(self.out["rec_100"], 200)
        for key in ("rec_empty", "rec_missing", "rec_101"):
            self.assertEqual(self.out[key], 422, key)
        self.assertTrue(self.out["rec_101_no_call"])

    # --- credentials ---
    def test_registration_uses_a_json_body_and_keeps_its_response(self):
        self.assertEqual(self.out["register_ok"], [200, {"message": "User registered successfully"}])
        self.assertEqual(self.out["register_duplicate"], 400)
        self.assertNotIn("?", self.out["register_request_url"])
        self.assertNotIn("pw-carol", self.out["register_request_url"])
        self.assertNotIn("carol", self.out["register_request_url"])

    def test_registration_rejects_credentials_in_the_query_string_and_bad_bodies(self):
        self.assertEqual(self.out["register_query_only"], 422)
        self.assertEqual(self.out["register_query_only_created_nothing"], 400)
        self.assertEqual(self.out["register_missing"], [422, 422])
        self.assertEqual(self.out["register_empty_fields"], [422, 422, 422])

    def test_login_behaviour_is_unchanged_and_not_in_the_url(self):
        self.assertEqual(self.out["login_ok"], [200, ["access_token", "token_type"]])
        self.assertNotIn("?", self.out["login_request_url"])
        self.assertEqual(self.out["login_wrong_password"], 400)
        self.assertEqual(self.out["login_unknown_user"], 400)

    def test_access_log_of_a_real_server_has_no_credentials(self):
        log = self.out["access_log"]
        self.assertIn("POST /register", log)
        self.assertIn("POST /login", log)
        for text in ("PW-ERIN-SECRET", "erin@example.test", "password", "username="):
            self.assertNotIn(text, log)

    # --- logs ---
    def test_resume_content_and_secrets_never_reach_stdout_stderr_or_logs(self):
        self.assertEqual(self.out["parser_calls"], 3)      # success, failure and the one race winner
        for text in ("RESUME-CONTENT-XYZ", "provider said", "ALICE-DATA", "pw-alice", "pw-carol", "test-only-0f4b7c2e9a1d4c68b3e5a7f2c9d01e84-not-a-real-secret",
                     "gsk_test_not_real", "yt-test-not-real", "Traceback"):
            self.assertNotIn(text, self.everything_logged, text)

    def test_the_failure_is_still_logged_with_the_exception_class(self):
        self.assertIn("Resume analysis failed (RuntimeError)", self.everything_logged)
        self.assertIn("Resume summarization failed (RuntimeError)", self.everything_logged)

    def test_frontend_dependent_endpoints_still_work(self):
        self.assertEqual(self.out["rec_authed"], 200)
        for path in ("/register", "/login", "/start_session", "/my_sessions", "/load_session", "/upload_resume",
                     "/parse_resume", "/recommend_courses", "/health"):
            self.assertIn(path, self.out["routes"])


class FrontendWiring(unittest.TestCase):
    """The Streamlit files cannot be imported in a test, so their source text is checked."""

    @staticmethod
    def read(*parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_credentials_are_sent_in_the_request_body(self):
        text = self.read("frontend", "credentials.py")
        register = text.split("def register_user", 1)[1].split("def login_user", 1)[0]
        self.assertIn("json={", register)
        self.assertNotIn("params=", register)
        self.assertNotIn("st.error(f\"Error: {e}\")", text)       # exception text may quote the request

    def test_upload_goes_through_the_backend_and_uses_the_returned_name(self):
        text = self.read("frontend", "main.py")
        self.assertIn("/upload_resume", text)
        self.assertIn('upload_resp.json()["resume_filename"]', text)
        self.assertNotIn("uploaded_file.name", text)
        self.assertNotIn("open(resume_path", text)
        self.assertNotIn("UPLOAD_DIR", text)

    def test_recommendations_send_the_auth_headers(self):
        text = self.read("frontend", "main.py")
        self.assertIn('params={"skill": skill}, headers=headers', text)

    def test_nothing_prints_resume_derived_content(self):
        for path in (("frontend", "main.py"), ("backend", "main.py")):
            text = self.read(*path)
            self.assertNotIn("print(", text, path)


class FrontendDoesNotUseRemovedEndpoints(unittest.TestCase):
    def test_no_source_or_doc_refers_to_the_removed_endpoints(self):
        hits = []
        for folder in ("frontend", "backend", "src"):
            for dirpath, dirs, files in os.walk(os.path.join(ROOT, folder)):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for name in files:
                    if name.endswith(".py"):
                        with open(os.path.join(dirpath, name), encoding="utf-8-sig") as handle:
                            text = handle.read()
                        if "db_check" in text or "get_all_data" in text:
                            hits.append(os.path.join(folder, name))
        self.assertEqual(hits, [])


class LogRedaction(unittest.TestCase):
    def test_filter_keeps_the_message_and_class_but_drops_message_and_traceback(self):
        import io
        import logging
        spec = importlib.util.spec_from_file_location("log_safety", os.path.join(ROOT, "src", "log_safety.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger = logging.getLogger("rr_test_redaction")
        logger.propagate = False
        buffer = io.StringIO()
        logger.addHandler(logging.StreamHandler(buffer))
        module._LOGGERS = ("rr_test_redaction",)
        module.install_exception_redaction()
        module.install_exception_redaction()                       # idempotent
        self.assertEqual(len([f for f in logger.filters if isinstance(f, module.ExceptionClassOnly)]), 1)
        try:
            raise ValueError("quotes the resume: SECRET-TEXT")
        except ValueError:
            logger.exception("Step %s failed", "one")
        text = buffer.getvalue()
        self.assertIn("Step one failed (ValueError)", text)
        self.assertNotIn("SECRET-TEXT", text)
        self.assertNotIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
