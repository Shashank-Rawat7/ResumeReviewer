"""Tests for the upload limits of POST /upload_resume (backend/main.py): 10 MB maximum, PDF-only, nothing left behind.

Run from the project root:  python -m unittest tests.test_upload_limits -v

How: like tests/test_security_endpoints.py, the scenario runs in a SUBPROCESS against a throw-away SQLite file with a
stubbed python-dotenv (the real .env is never read), fake keys and a mocked parser. No network, no PostgreSQL, no Groq,
no YouTube. The temporary directory used by the web framework for large request bodies (TMPDIR) is also pointed at a
folder of the test so that leftovers there can be detected. The tests are skipped when the FastAPI test dependencies
are not installed.
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
MAX_BYTES = 10 * 1024 * 1024
CHUNK = 64 * 1024

SCENARIO = r'''
import json, os, re, sys, types, importlib.util
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
import starlette.datastructures as sd
import backend.main as main

spec = importlib.util.spec_from_file_location("minipdf", os.path.join(root, "tests", "docker", "minipdf.py"))
minipdf = importlib.util.module_from_spec(spec); spec.loader.exec_module(minipdf)

out = {}
MAX, CHUNK = main.MAX_UPLOAD_BYTES, main.UPLOAD_CHUNK_BYTES
out["limits"] = [MAX, CHUNK]
try:
    main.auth.verify_password("x", main.auth.hash_password("x"))
except Exception:
    main.auth.hash_password = lambda p: "plain$" + p
    main.auth.verify_password = lambda p, h: h == "plain$" + p

parsed = []
def fake_parse(path, jd, metrics=None):
    parsed.append(os.path.basename(path))
    return (json.dumps({"note": jd}), "{}", "{}", "{}", json.dumps({"bullets": []}),
            json.dumps({"normalization_ok": False, "skills": []}))
main.parser.parse_resume = fake_parse

# Spy on every read of an uploaded file: which size was requested, how many bytes came back, optional injected failure.
reads = []
inject = {"after": None, "exc": None}
orig_read = sd.UploadFile.read
async def spy(self, size=-1):
    if inject["after"] is not None and len(reads) >= inject["after"]:
        raise inject["exc"]
    data = await orig_read(self, size)
    reads.append((size, len(data)))
    return data
sd.UploadFile.read = spy

client = TestClient(main.app, raise_server_exceptions=False)
upload_dir = os.environ["UPLOAD_DIR"]
spool_dir = os.environ["TMPDIR"]

def files_on_disk(base):
    found = {}
    for d, _, names in os.walk(base):
        for n in names:
            p = os.path.join(d, n)
            found[os.path.relpath(p, base).replace(os.sep, "/")] = os.path.getsize(p)
    return found

def register_login(name):
    client.post("/register", json={"username": name, "email": name + "@example.test", "password": "pw-" + name})
    token = client.post("/login", data={"username": name, "password": "pw-" + name}).json()["access_token"]
    return {"Authorization": "Bearer " + token}

def upload(headers, content, filename="resume.pdf", ctype="application/pdf"):
    del reads[:]
    r = client.post("/upload_resume", files={"file": (filename, content, ctype)}, headers=headers)
    return r

def brief(r):
    try:
        body = r.json()
    except Exception:
        body = {}
    return {"status": r.status_code, "detail": body.get("detail") if isinstance(body, dict) else None, "text": r.text[:500],
            "name": body.get("resume_filename") if isinstance(body, dict) else None}

A = register_login("alice")
B = register_login("bob")
pdf = minipdf.make_pdf(minipdf.RESUME_LINES)
big_ok = b"%PDF-1.4\n" + b"0" * (MAX - 9)             # exactly MAX bytes
out["sizes"] = [len(pdf), len(big_ok)]

# ---- accepted uploads ----
r = upload(A, pdf)
res = brief(r); res["reads"] = list(reads)
res["stored"] = files_on_disk(upload_dir)
res["same_bytes"] = bool(res["name"]) and any(
    open(os.path.join(d, res["name"]), "rb").read() == pdf for d, _, n in os.walk(upload_dir) if res["name"] in n)
out["valid"] = res
a_name = res["name"]
out["upper_ext"] = brief(upload(A, pdf, filename="RESUME.PDF"))
out["mime_not_trusted_ok"] = brief(upload(A, pdf, ctype="text/plain"))            # real PDF, wrong MIME: accepted
r = upload(A, big_ok)
out["exact_limit"] = brief(r); out["exact_limit"]["stored_sizes"] = sorted(files_on_disk(upload_dir).values())[-1:]
out["exact_limit"]["reads"] = [sum(n for _, n in reads), max(s for s, _ in reads)]

# ---- rejected uploads: nothing may be added to the upload folder or the framework's temp folder ----
before = files_on_disk(upload_dir)
rej = {}
rej["one_over"] = upload(A, b"%PDF-1.4\n" + b"0" * (MAX - 8)); rej["one_over_reads"] = [sum(n for _, n in reads), len(reads)]
rej["twelve_mb"] = upload(A, b"%PDF-1.4\n" + b"0" * (12 * 1024 * 1024)); twelve_reads = [sum(n for _, n in reads), len(reads)]
rej["text_as_pdf"] = upload(A, b"This is plain text, not a PDF.", ctype="application/pdf")
rej["html_as_pdf"] = upload(A, b"<html><body>hi</body></html>")
rej["png_as_pdf"] = upload(A, b"\x89PNG\r\n\x1a\n" + b"0" * 100)
rej["signature_not_first"] = upload(A, b"\n%PDF-1.4 late signature")
rej["pdf_as_txt"] = upload(A, pdf, filename="resume.txt")
rej["pdf_no_ext"] = upload(A, pdf, filename="resume")
rej["pdf_double_ext"] = upload(A, pdf, filename="resume.pdf.exe")
rej["pdf_empty_name"] = upload(A, pdf, filename="")
rej["empty_pdf"] = upload(A, b"")
inject.update(after=1, exc=OSError("disk full: " + upload_dir))
rej["io_error"] = upload(A, b"%PDF-1.4\n" + b"0" * (3 * CHUNK))
inject.update(after=1, exc=RuntimeError("client went away"))
rej["other_error"] = upload(A, b"%PDF-1.4\n" + b"0" * (3 * CHUNK))
inject.update(after=None, exc=None)
rej["unauthenticated"] = client.post("/upload_resume", files={"file": ("resume.pdf", pdf, "application/pdf")})
out["rejected"] = {k: brief(v) for k, v in rej.items() if not k.endswith("_reads")}
out["one_over_reads"] = rej["one_over_reads"]
out["twelve_reads"] = twelve_reads
after = files_on_disk(upload_dir)
out["rejected_left_new_files"] = sorted(set(after) - set(before))
out["rejected_changed_files"] = sorted(k for k in before if before[k] != after.get(k))
out["part_or_tmp_names"] = sorted(k for k in after if not re.fullmatch(r"\d+/[0-9a-f]{32}\.pdf", k))
out["spool_leftovers"] = files_on_disk(spool_dir)

# ---- cross-user isolation still holds ----
sa = client.post("/start_session", params={"session_name": "a"}, headers=A).json()["session_id"]
sb = client.post("/start_session", params={"session_name": "b"}, headers=B).json()["session_id"]
bob_up = brief(upload(B, pdf)); b_name = bob_up["name"]
bob_bad = brief(upload(B, b"not a pdf at all"))
def parse(h, name, sid):
    return client.post("/parse_resume", data={"resume_filename": name, "job_description": "JD", "session_id": sid}, headers=h)
snap = files_on_disk(upload_dir)
out["iso"] = {
    "bob_parses_alices": parse(B, a_name, sb).status_code,
    "alice_parses_bobs": parse(A, b_name, sa).status_code,
    "both_survive": all(any(k.endswith(n) for k in files_on_disk(upload_dir)) for n in (a_name, b_name)),
    "bob_reject_status": bob_bad["status"],
    "folders": sorted({k.split("/")[0] for k in snap}),
    "parser_calls": len(parsed),
}
# ---- the frontend's exact request shape still works end to end ----
fe = client.post("/upload_resume", files={"file": ("resume.pdf", pdf, "application/pdf")}, headers=A)
fe_name = fe.json().get("resume_filename", "")
pr = parse(A, fe_name, sa)
out["frontend_flow"] = {"upload": fe.status_code, "keys": sorted(fe.json()), "parse": pr.status_code,
                        "consumed": not any(k.endswith(fe_name) for k in files_on_disk(upload_dir))}
print("RESULT:" + json.dumps(out))
'''


def _have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


@unittest.skipUnless(_have("fastapi", "httpx", "sqlalchemy", "jose", "passlib", "langchain_groq", "yaml", "uvicorn", "multipart"),
                     "FastAPI test dependencies are not installed")
class UploadLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = os.path.join(tmp, "uploads")
            spool = os.path.join(tmp, "spool")
            os.makedirs(uploads)
            os.makedirs(spool)
            env = dict(os.environ)
            for name in list(env):
                if name.startswith(("POSTGRES_", "GROQ_", "YOUTUBE_", "DATABASE_", "SECRET_", "UPLOAD_", "API_BASE")):
                    env.pop(name)
            env.update({
                "RR_ROOT": ROOT, "UPLOAD_DIR": uploads, "TMPDIR": spool, "TEMP": spool, "TMP": spool,
                "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "t.db").replace("\\", "/"),
                "SECRET_KEY": "test-only-0f4b7c2e9a1d4c68b3e5a7f2c9d01e84-not-a-real-secret",
                "GROQ_API_KEY": "gsk_test_not_real", "YOUTUBE_API_KEY": "yt-test-not-real",
                "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            })
            done = subprocess.run([sys.executable, "-B", "-c", SCENARIO], env=env, cwd=tmp, capture_output=True,
                                  text=True, timeout=170)
            cls.upload_path = uploads
        marker = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
        if not marker:
            raise AssertionError("scenario failed:\n" + done.stdout[-1500:] + "\n" + done.stderr[-3000:])
        cls.out = json.loads(marker[-1][len("RESULT:"):])

    # --- accepted uploads ---
    def test_the_limits_are_ten_mib_and_small_chunks(self):
        self.assertEqual(self.out["limits"], [MAX_BYTES, CHUNK])

    def test_a_valid_pdf_under_the_limit_is_accepted_and_stored_unchanged(self):
        v = self.out["valid"]
        self.assertEqual(v["status"], 200)
        self.assertRegex(v["name"], r"^[0-9a-f]{32}\.pdf$")          # server-generated name, same contract as before
        self.assertTrue(v["same_bytes"])
        self.assertEqual(len(v["stored"]), 1)                        # one file, in the caller's own folder
        self.assertRegex(next(iter(v["stored"])), r"^\d+/[0-9a-f]{32}\.pdf$")

    def test_extension_is_case_insensitive_and_the_mime_type_is_not_trusted(self):
        self.assertEqual(self.out["upper_ext"]["status"], 200)
        self.assertEqual(self.out["mime_not_trusted_ok"]["status"], 200)     # real PDF sent as text/plain is fine
        self.assertEqual(self.out["rejected"]["text_as_pdf"]["status"], 415)  # text sent as application/pdf is not

    def test_a_file_of_exactly_ten_mib_is_accepted(self):
        e = self.out["exact_limit"]
        self.assertEqual(e["status"], 200)
        self.assertEqual(e["stored_sizes"], [MAX_BYTES])

    def test_reads_are_bounded_chunks_never_the_whole_file(self):
        self.assertEqual(self.out["valid"]["reads"][-1][1], 0)               # the loop ends on an empty read
        self.assertTrue(all(size == CHUNK for size, _ in self.out["valid"]["reads"]), self.out["valid"]["reads"])
        total, largest_request = self.out["exact_limit"]["reads"]
        self.assertEqual((total, largest_request), (MAX_BYTES, CHUNK))

    # --- rejected uploads ---
    def test_a_file_over_ten_mib_is_rejected_with_413(self):
        for key in ("one_over", "twelve_mb"):
            r = self.out["rejected"][key]
            self.assertEqual(r["status"], 413, key)
            self.assertIn("10 MB", r["detail"], key)

    def test_reading_stops_at_the_limit_instead_of_consuming_the_whole_file(self):
        for total, calls in (self.out["one_over_reads"], self.out["twelve_reads"]):
            self.assertLessEqual(total, MAX_BYTES + CHUNK)                   # at most one chunk past the limit
            self.assertLessEqual(calls, MAX_BYTES // CHUNK + 2)
        self.assertLess(self.out["twelve_reads"][0], 12 * 1024 * 1024)       # the 12 MiB body was not read to the end

    def test_non_pdf_content_with_a_pdf_name_is_rejected_with_415(self):
        for key in ("text_as_pdf", "html_as_pdf", "png_as_pdf", "signature_not_first"):
            r = self.out["rejected"][key]
            self.assertEqual(r["status"], 415, key)
            self.assertIn("not a valid PDF", r["detail"], key)

    def test_pdf_content_with_a_wrong_extension_is_rejected_with_415(self):
        for key in ("pdf_as_txt", "pdf_no_ext", "pdf_double_ext", "pdf_empty_name"):
            r = self.out["rejected"][key]
            self.assertIn(r["status"], (415, 422), key)                      # an empty file name may be refused by the form parser
        for key in ("pdf_as_txt", "pdf_no_ext", "pdf_double_ext"):
            self.assertEqual(self.out["rejected"][key]["status"], 415, key)
            self.assertIn(".pdf", self.out["rejected"][key]["detail"], key)

    def test_empty_files_and_unauthenticated_uploads_behave_as_before(self):
        self.assertEqual(self.out["rejected"]["empty_pdf"]["status"], 400)
        self.assertEqual(self.out["rejected"]["unauthenticated"]["status"], 401)

    def test_storage_failures_return_500_without_internal_details(self):
        for key in ("io_error", "other_error"):
            r = self.out["rejected"][key]
            self.assertEqual(r["status"], 500, key)
        self.assertEqual(self.out["rejected"]["io_error"]["detail"], "The upload could not be stored.")

    def test_error_responses_never_contain_paths_or_exception_text(self):
        for key, r in self.out["rejected"].items():
            for leaked in ("uploads", "disk full", "client went away", "Traceback", "/tmp", ROOT):
                self.assertNotIn(leaked, r["text"], key)

    def test_rejected_uploads_leave_no_files_behind(self):
        self.assertEqual(self.out["rejected_left_new_files"], [])            # no stored file, no partial file
        self.assertEqual(self.out["rejected_changed_files"], [])             # earlier uploads untouched
        self.assertEqual(self.out["part_or_tmp_names"], [])                  # nothing but <user id>/<32 hex>.pdf
        self.assertEqual(self.out["spool_leftovers"], {})                    # the framework's temp folder is clean too

    # --- unchanged behaviour ---
    def test_cross_user_isolation_is_intact(self):
        iso = self.out["iso"]
        self.assertEqual((iso["bob_parses_alices"], iso["alice_parses_bobs"]), (404, 404))
        self.assertTrue(iso["both_survive"])
        self.assertEqual(iso["bob_reject_status"], 415)
        self.assertEqual(iso["parser_calls"], 0)                             # no model call for another user's file
        self.assertGreaterEqual(len(iso["folders"]), 2)                      # one folder per user id

    def test_the_frontends_request_shape_still_works_end_to_end(self):
        flow = self.out["frontend_flow"]
        self.assertEqual(flow["upload"], 200)
        self.assertEqual(flow["keys"], ["resume_filename"])                  # the response contract the frontend reads
        self.assertEqual(flow["parse"], 200)
        self.assertTrue(flow["consumed"])

    def test_the_frontend_source_still_matches_the_backend_contract(self):
        with open(os.path.join(ROOT, "frontend", "main.py"), "rb") as handle:
            source = handle.read().decode("utf-8")
        self.assertIn('f"{API_BASE_URL}/upload_resume"', source)
        self.assertIn('files={"file": ("resume.pdf", uploaded_file.getvalue(), "application/pdf")}', source)
        self.assertIn('upload_resp.json()["resume_filename"]', source)
        self.assertIn('type=["pdf"]', source)


class UploadCodeShape(unittest.TestCase):
    def endpoint_source(self):
        with open(os.path.join(ROOT, "backend", "main.py"), "rb") as handle:
            text = handle.read().decode("utf-8").replace("\r\n", "\n")
        start = text.index('@app.post("/upload_resume")')
        return text[start:text.index('@app.post(\'/parse_resume\')', start)]

    def test_the_endpoint_never_reads_the_whole_upload_at_once(self):
        source = self.endpoint_source()
        self.assertNotRegex(source, r"\.read\(\s*\)")
        self.assertNotRegex(source, r"\.read\(\s*-1\s*\)")
        self.assertIn("file.read(UPLOAD_CHUNK_BYTES)", source)
        self.assertIn("MAX_UPLOAD_BYTES", source)


if __name__ == "__main__":
    unittest.main()
