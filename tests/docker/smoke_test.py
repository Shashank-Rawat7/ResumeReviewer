"""End-to-end Docker smoke test: the real three-service stack with a MOCK model. No live Groq/YouTube call.

Run from the project root (Docker Desktop / Docker Engine with Compose v2 must be running):

    python tests/docker/smoke_test.py

It starts an isolated stack (project name "resume-smoke", its own volumes and network, loopback ports
15432/18000/18501/18081 and dummy credentials from tests/docker/smoke.env), so it never touches your real stack,
database or .env. A mock Groq server (tests/docker/mock_groq.py) replaces the model.

What is asserted (each check has an ID; failures are listed by ID at the end):
  1  build + start with `compose up --build --wait`, 1b depends_on conditions in the merged configuration
  2-4  PostgreSQL / backend / frontend health; 3b-3c startup ORDER from Docker events (backend starts after PostgreSQL
       reported healthy, frontend after the backend did); 4c-4f both app containers run as uid 10001 on Python 3.12, the
       backend reaches PostgreSQL through the service name, the frontend reaches the backend over the Compose network;
       4g the uploads directory is owned by uid 10001 and writable (write/read/delete probe); 4h the frontend container
       has no mount of the uploads volume
  5-7, 12  registration (JSON body), login, sessions, authentication and cross-user isolation, removed endpoints
  8  upload through POST /upload_resume: random name, per-user folder in the uploads volume, other users cannot parse
     it, traversal refused, and 8f the frontend image's own `requests` + API_BASE_URL can reach /upload_resume
  9  mocked analysis: five model calls, one metrics line, upload removed; 9f a failed analysis returns 500, removes the
     upload and leaves the previously saved analysis unchanged; 9g-9h two concurrent requests for one upload
  10-11  database and uploads-volume persistence across `down`/`up` (no -v)
  13  explicit `compose restart` of backend and frontend
  14  database start-up retry: with PostgreSQL stopped the backend (restarted meanwhile) keeps retrying instead of
     exiting, logs the retry, and becomes healthy once PostgreSQL returns, with data intact

NOT tested (and not claimed): automatic crash recovery of the `restart: unless-stopped` policy (the application is
PID 1 and cannot be crashed deterministically and safely from outside the container), the Streamlit UI itself (check 8f
exercises the frontend image's HTTP client and API_BASE_URL, not the page code or the file_uploader widget; that would
need browser automation), and anything about the real Groq or YouTube services.

Safety:
  * Before anything starts, the script looks for existing containers, volumes or networks of the project "resume-smoke"
    (read-only `docker ... ls` calls). If any exist it STOPS with an explanation and deletes nothing.
  * Cleanup (`docker compose -p resume-smoke ... down -v`) runs only for a stack this run created, even when a check
    fails or an error occurs, unless --keep is given. No global Docker cleanup (prune) is ever run.
  * Images built by the test (resume-smoke-backend / resume-smoke-frontend) and the pulled base images
    (python:3.12.16-slim-trixie, postgres:15.20-trixie) and build cache REMAIN after the test.

Options:  --keep   leave the stack running afterwards (the command to remove it later is printed)
          --no-build   reuse existing images
Exit codes: 0 all checks passed, 1 a check failed, 2 Docker/Compose unavailable, 3 an existing resume-smoke stack found.
A passing run proves the Docker plumbing. It does NOT prove anything about the real Groq or YouTube integrations.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROJECT = "resume-smoke"
LABEL = "com.docker.compose.project=" + PROJECT
COMPOSE = ["docker", "compose", "-p", PROJECT, "--env-file", os.path.join("tests", "docker", "smoke.env"),
           "-f", "docker-compose.yaml", "-f", os.path.join("tests", "docker", "docker-compose.mock.yaml")]
BACKEND, FRONTEND, MOCK = "http://127.0.0.1:18000", "http://127.0.0.1:18501", "http://127.0.0.1:18081"
JD = "We need a Python engineer. Experience with Kubernetes is required."

# Calls to 127.0.0.1 must never go through a configured HTTP proxy.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

results = []        # (id, name, "PASS" | "FAIL")
not_verified = []   # (id, name, reason)


def say(text):
    print(text, flush=True)


def check(cid, name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((cid, name, status))
    line = "%s  [%s] %s" % (status, cid, name)
    if not ok and detail != "":
        line += " - " + str(detail)[:700].replace("\n", " ")
    say(line)
    return bool(ok)


def not_checked(cid, name, reason):
    not_verified.append((cid, name, reason))
    say("NOT VERIFIED  [%s] %s - %s" % (cid, name, reason))


class section:
    """Run a group of checks; an unexpected error is reported as a failed check instead of aborting the whole run."""

    def __init__(self, cid, name):
        self.cid, self.name = cid, name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        if issubclass(exc_type, Exception):
            check(self.cid, self.name + " (aborted by an unexpected error)", False, "%s: %s" % (exc_type.__name__, exc))
            return True
        return False            # KeyboardInterrupt etc.: propagate (cleanup still runs in main's finally block)


def run(args, **kw):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def compose(*args, **kw):
    return run(COMPOSE + list(args), **kw)


def parse_body(raw):
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except ValueError:
            return text
    return text


def http(method, url, headers=None, data=None, params=None, json_body=None, files=None, timeout=60):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = dict(headers or {})
    body = None
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    elif files is not None:
        boundary = "----smoke" + str(int(time.time() * 1000))
        name, content = files
        body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                "Content-Type: application/pdf\r\n\r\n" % (boundary, name)).encode() + content + ("\r\n--%s--\r\n" % boundary).encode()
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    elif data is not None:
        body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, parse_body(response.read())
    except urllib.error.HTTPError as err:
        return err.code, parse_body(err.read())


def unreachable(url):
    """True when nothing answers on the URL (connection refused / reset / timeout)."""
    try:
        http("GET", url, timeout=5)
    except Exception:
        return True
    return False


def wait_healthy(timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if http("GET", BACKEND + "/health", timeout=5)[0] == 200 and http("GET", FRONTEND + "/_stcore/health", timeout=5)[0] == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def service_health(service):
    out = compose("ps", "--format", "json", service).stdout.strip()
    try:
        parsed = json.loads(out)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except ValueError:
        rows = []
        for line in out.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows[0].get("Health", "") if rows else ""


def container_id(service):
    ids = compose("ps", "-q", service).stdout.split()
    return ids[0] if ids else ""


def exec_in(service, code):
    return compose("exec", "-T", service, "python", "-c", code)


def login(name):
    http("POST", BACKEND + "/register", json_body={"username": name, "email": name + "@example.test", "password": "pw-" + name})
    status, body = http("POST", BACKEND + "/login", data={"username": name, "password": "pw-" + name})
    return status, {"Authorization": "Bearer " + str(body.get("access_token") if isinstance(body, dict) else "")}


def existing_resources():
    """Containers, volumes and networks that already belong to the project name (read-only listing)."""
    queries = (
        ("containers", ["docker", "ps", "-a", "--filter", "label=" + LABEL, "--format", "{{.Names}}"]),
        ("volumes", ["docker", "volume", "ls", "--filter", "label=" + LABEL, "--format", "{{.Name}}"]),
        ("volumes (by name)", ["docker", "volume", "ls", "--filter", "name=" + PROJECT + "_", "--format", "{{.Name}}"]),
        ("networks", ["docker", "network", "ls", "--filter", "label=" + LABEL, "--format", "{{.Name}}"]),
        ("networks (by name)", ["docker", "network", "ls", "--filter", "name=" + PROJECT + "_", "--format", "{{.Name}}"]),
    )
    found = {}
    for kind, command in queries:
        done = run(command)
        if done.returncode != 0:
            raise RuntimeError("could not list %s: %s" % (kind, done.stderr.strip()[-300:]))
        names = sorted(set(done.stdout.split()))
        if names:
            found[kind] = names
    return found


class Events:
    """Docker events of this project's containers, collected while the stack starts (used for the ORDER checks)."""

    def __init__(self):
        self.items = []
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            ["docker", "events", "--filter", "label=" + LABEL, "--filter", "type=container", "--format", "{{json .}}"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._read, daemon=True).start()
        time.sleep(1.5)

    def _read(self):
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            attrs = (event.get("Actor") or {}).get("Attributes") or {}
            self.items.append((int(event.get("timeNano") or 0), attrs.get("com.docker.compose.service", ""),
                               event.get("Action") or event.get("status") or ""))

    def first(self, service, action):
        times = [t for t, s, a in self.items if s == service and a == action]
        return min(times) if times else None

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


def run_checks(args, pdf, events):
    S = {}

    # ---- 1. build and start ------------------------------------------------------------------------------------
    build = compose("up", "-d", "--build" if not args.no_build else "--no-build", "--wait", "--wait-timeout", "300")
    check("1", "fresh image build and start (compose up --build --wait)", build.returncode == 0, build.stderr[-1500:])
    if build.returncode != 0:
        return

    with section("1b", "depends_on conditions in the merged configuration"):
        cfg = json.loads(compose("config", "--format", "json").stdout)
        deps = {name: svc.get("depends_on", {}) for name, svc in cfg["services"].items()}
        check("1b", "backend waits for postgres 'service_healthy' and frontend for backend 'service_healthy'",
              deps["backend"].get("postgres", {}).get("condition") == "service_healthy"
              and deps["frontend"].get("backend", {}).get("condition") == "service_healthy", deps)

    with section("2", "PostgreSQL healthy"):
        check("2", "PostgreSQL healthy", service_health("postgres") == "healthy", service_health("postgres"))
    with section("3", "backend healthy"):
        check("3", "backend healthy (/health)", http("GET", BACKEND + "/health") == (200, {"status": "healthy"}))
    with section("4", "frontend healthy"):
        check("4", "frontend running (Streamlit health)", wait_healthy(60))

    # ---- startup ORDER, from Docker events recorded while the stack came up -------------------------------------
    with section("3b", "startup order"):
        time.sleep(2)
        if not events.items:
            not_checked("3b", "startup order", "no Docker events could be captured, so only eventual health is known")
        else:
            pg_ready, be_start = events.first("postgres", "health_status: healthy"), events.first("backend", "start")
            be_ready, fe_start = events.first("backend", "health_status: healthy"), events.first("frontend", "start")
            check("3b", "backend container started only AFTER PostgreSQL reported healthy",
                  pg_ready is not None and be_start is not None and pg_ready <= be_start, (pg_ready, be_start))
            check("3c", "frontend container started only AFTER the backend reported healthy",
                  be_ready is not None and fe_start is not None and be_ready <= fe_start, (be_ready, fe_start))

    # ---- runtime: user, Python, database by service name, network, volume permissions --------------------------
    for service in ("backend", "frontend"):
        with section("4c/4d-" + service, service + " runtime"):
            check("4c-" + service, "%s runs as uid 10001" % service,
                  exec_in(service, "import os; print(os.getuid())").stdout.strip() == "10001")
            check("4d-" + service, "%s runs Python 3.12" % service,
                  exec_in(service, "import sys; print(sys.version_info[:2])").stdout.strip() == "(3, 12)")
    with section("4e", "database by service name"):
        check("4e", "backend reaches PostgreSQL through the service name 'postgres'",
              exec_in("backend", "import os, psycopg2; psycopg2.connect(host='postgres', port=5432, user=os.environ['POSTGRES_USER'], "
                                 "password=os.environ['POSTGRES_PASSWORD'], dbname=os.environ['POSTGRES_DB']).close(); print('ok')"
                      ).stdout.strip() == "ok")
    with section("4f", "frontend to backend"):
        check("4f", "frontend reaches the backend over the Compose network",
              exec_in("frontend", "import urllib.request; print(urllib.request.urlopen('http://backend:8000/health', timeout=5).status)"
                      ).stdout.strip() == "200")
    with section("4g", "uploads permissions"):
        probe = exec_in("backend", "import os, json; p = '/app/uploads'; q = os.path.join(p, '.perm_probe'); "
                                   "open(q, 'w').write('x'); ok = open(q).read() == 'x'; os.remove(q); "
                                   "print(json.dumps({'uid': os.getuid(), 'owner': os.stat(p).st_uid, "
                                   "'writable': os.access(p, os.W_OK), 'roundtrip': ok}))").stdout.strip()
        info = json.loads(probe) if probe else {}
        check("4g", "uploads directory is owned by uid 10001, writable by the backend user (write/read/delete probe)",
              info.get("uid") == 10001 and info.get("owner") == 10001 and info.get("writable") is True
              and info.get("roundtrip") is True, probe)
    with section("4h", "frontend mounts"):
        mounts = run(["docker", "inspect", "--format", "{{json .Mounts}}", container_id("frontend")]).stdout.strip()
        check("4h", "the frontend container has no mount at /app/uploads (it does not need the uploads volume)",
              mounts != "" and "/app/uploads" not in mounts and "resume-smoke_uploads" not in mounts, mounts)

    # ---- users, authentication, session ------------------------------------------------------------------------
    with section("5", "registration and login"):
        sa, A = login("alice")
        sb, B = login("bob")
        S["A"], S["B"] = A, B
        check("5", "registration (JSON body) + 6 login (two users)", sa == 200 and sb == 200)
    A, B = S.get("A", {}), S.get("B", {})
    with section("12a", "removed endpoints"):
        check("12a", "/db_check and /get_all_data are gone",
              http("GET", BACKEND + "/db_check")[0] == 404 and http("GET", BACKEND + "/get_all_data", headers=A)[0] == 404)
    with section("12b", "authentication required"):
        check("12b", "protected endpoints refuse unauthenticated requests",
              http("GET", BACKEND + "/my_sessions")[0] == 401 and http("GET", BACKEND + "/load_session", params={"session_id": 1})[0] == 401)
    with section("7", "session creation"):
        s1 = http("POST", BACKEND + "/start_session", headers=A, params={"session_name": "smoke"})
        S["sid"] = s1[1].get("session_id") if isinstance(s1[1], dict) else None
        check("7", "session creation", s1[0] == 200 and bool(S["sid"]), s1)
    sid = S.get("sid")
    with section("12e", "recommend_courses auth"):
        check("12e", "/recommend_courses requires a token (no YouTube call is made)",
              http("GET", BACKEND + "/recommend_courses", params={"skill": "Python"})[0] == 401)
    with section("12f", "credentials in query"):
        check("12f", "credentials in a query string are refused",
              http("POST", BACKEND + "/register", params={"username": "x", "email": "x@example.test", "password": "pw"})[0] == 422)
    with section("12c", "cross-user session"):
        check("12c", "another user cannot load the session",
              http("GET", BACKEND + "/load_session", headers=B, params={"session_id": sid})[0] == 404)

    # ---- 8. uploads: stored by the BACKEND under a random name in the caller's own folder -------------------------
    find = ("import os; print([os.path.relpath(os.path.join(d, f), '/app/uploads') for d, _, fs in os.walk('/app/uploads') "
            "for f in fs])")
    with section("8", "upload"):
        status, up = http("POST", BACKEND + "/upload_resume", headers=A, files=("smoke_resume.pdf", pdf))
        stored = up.get("resume_filename") if isinstance(up, dict) else ""
        S["stored"] = stored
        check("8a", "upload accepted and renamed to <32 hex>.pdf",
              status == 200 and len(stored) == 36 and stored.endswith(".pdf") and stored != "smoke_resume.pdf", (status, up))
        listing = exec_in("backend", find).stdout.strip()
        check("8b", "the file sits in a per-user folder of the uploads volume", bool(stored) and stored in listing and "/" in listing, listing)
        check("8c", "the frontend container has no access to the uploads volume",
              exec_in("frontend", "import os; print(os.path.exists('/app/uploads'))").stdout.strip() == "False")
        check("8d", "another user cannot parse the upload (and it survives)",
              http("POST", BACKEND + "/parse_resume", headers=B,
                   data={"resume_filename": stored, "job_description": JD, "session_id": sid})[0] == 404
              and bool(stored) and stored in exec_in("backend", find).stdout)
        check("8e", "a traversal file name is refused",
              http("POST", BACKEND + "/parse_resume", headers=A,
                   data={"resume_filename": "../x.pdf", "job_description": JD, "session_id": sid})[0] == 400)
    stored = S.get("stored", "")
    with section("8f", "frontend image upload wiring"):
        token = B["Authorization"].split(" ", 1)[1]
        code = ("import os, json, requests; base = os.environ['API_BASE_URL']; "
                "r = requests.post(base + '/upload_resume', files={'file': ('probe.pdf', b'%PDF-1.4 probe', 'application/pdf')}, "
                "headers={'Authorization': 'Bearer " + token + "'}); "
                "print(json.dumps({'base': base, 'status': r.status_code, 'name': r.json().get('resume_filename', '')}))")
        out = exec_in("frontend", code).stdout.strip()
        info = json.loads(out) if out else {}
        name = info.get("name", "")
        check("8f", "the frontend image (requests + API_BASE_URL) uploads to the backend; the file lands in the caller's folder",
              info.get("base") == "http://backend:8000" and info.get("status") == 200 and len(name) == 36
              and name in exec_in("backend", find).stdout, out)

    # ---- 9. mocked analysis, failed analysis, concurrency --------------------------------------------------------
    with section("9a", "mocked analysis"):
        before = http("GET", MOCK + "/stats")[1]["requests"]
        status, body = http("POST", BACKEND + "/parse_resume", headers=A,
                            data={"resume_filename": stored, "job_description": JD, "session_id": sid})
        check("9a", "processing with the mocked model returns 200", status == 200, body if status != 200 else "")
        check("9b", "exactly five mocked model calls", http("GET", MOCK + "/stats")[1]["requests"] - before == 5)
        gone = exec_in("backend", find).stdout
        check("9c", "the processed upload was removed", bool(stored) and stored not in gone and ".processing" not in gone, gone)
    with section("9d", "metrics record"):
        logs = compose("logs", "backend", "--no-log-prefix").stdout
        records = [json.loads(l) for l in logs.splitlines() if l.startswith("{") and '"record_version"' in l]
        rec = records[-1] if records else {}
        check("9d", "one metrics record: five stages ok, tokens complete, normalization ok",
              bool(rec) and all(s["status"] == "ok" for s in rec["stages"].values()) and rec["totals"]["tokens_complete"]
              and rec["normalization_ok"] is True, rec)
        check("9e", "the log holds no resume / job text", "Mock Candidate" not in logs and "Kubernetes is required" not in logs)

    with section("9f", "failed analysis"):
        before_status, before_data = http("GET", BACKEND + "/load_session", headers=A, params={"session_id": sid})
        status, bad = http("POST", BACKEND + "/upload_resume", headers=A, files=("bad.pdf", b"%PDF-1.4 this is not a real pdf"))   # passes the upload checks, cannot be parsed
        bad_name = bad.get("resume_filename", "") if isinstance(bad, dict) else ""
        status_bad = http("POST", BACKEND + "/parse_resume", headers=A,
                          data={"resume_filename": bad_name, "job_description": JD, "session_id": sid})[0]
        left = exec_in("backend", find).stdout
        check("9f", "failed parsing: 500, the upload is removed, nothing left behind",
              status_bad == 500 and bool(bad_name) and bad_name not in left and ".processing" not in left, (status_bad, left))
        after_status, after_data = http("GET", BACKEND + "/load_session", headers=A, params={"session_id": sid})
        check("9f2", "the previously saved analysis is unchanged after the failed parse",
              before_status == 200 and after_status == 200 and isinstance(before_data, dict)
              and "summary" in before_data.get("resume_analysis", {})
              and json.dumps(before_data, sort_keys=True) == json.dumps(after_data, sort_keys=True))

    with section("9g", "concurrent parse"):
        _, again = http("POST", BACKEND + "/upload_resume", headers=A, files=("again.pdf", pdf))
        race_name = again.get("resume_filename", "") if isinstance(again, dict) else ""
        before = http("GET", MOCK + "/stats")[1]["requests"]
        codes = []

        def race():
            try:
                codes.append(http("POST", BACKEND + "/parse_resume", headers=A,
                                  data={"resume_filename": race_name, "job_description": JD, "session_id": sid})[0])
            except Exception as exc:                      # a crashed worker thread must show up as a failure
                codes.append("error: " + type(exc).__name__)
        threads = [threading.Thread(target=race) for _ in range(2)]
        [x.start() for x in threads]
        [x.join() for x in threads]
        check("9g", "concurrent parse of one upload: exactly one 200 and one 404, five model calls",
              bool(race_name) and sorted(map(str, codes)) == ["200", "404"]
              and http("GET", MOCK + "/stats")[1]["requests"] - before == 5, codes)
        check("9h", "concurrent parse left no file behind", bool(race_name) and race_name not in exec_in("backend", find).stdout)

    # ---- 10/11. persistence across a full stop/start (named volumes are kept; no -v) ----------------------------
    with section("10", "persistence"):
        exec_in("backend", "open('/app/uploads/persist.txt','w').write('kept')")
        down = compose("down")
        check("10a", "stack stops", down.returncode == 0, down.stderr[-800:])
        up = compose("up", "-d", "--no-build", "--wait", "--wait-timeout", "300")
        check("10b", "stack starts again", up.returncode == 0, up.stderr[-1500:])
        check("10c", "healthy again", wait_healthy(120))
        s, A2 = login("alice")
        check("10d", "database persisted: same user can log in", s == 200)
        saved = http("GET", BACKEND + "/load_session", headers=A2, params={"session_id": sid})
        check("10e", "database persisted: saved analysis is still there",
              saved[0] == 200 and isinstance(saved[1], dict) and "summary" in saved[1].get("resume_analysis", {}))
        kept = exec_in("backend", "print(open('/app/uploads/persist.txt').read())").stdout.strip()
        check("11", "uploaded-file volume persisted across restart", kept == "kept", kept)
        s, B2 = login("bob")
        mine = http("GET", BACKEND + "/my_sessions", headers=B2)[1]
        check("12d", "isolation still holds after restart",
              http("GET", BACKEND + "/load_session", headers=B2, params={"session_id": sid})[0] == 404
              and isinstance(mine, list) and [x["id"] for x in mine] == [])

    # ---- 13. explicit restart of the application containers -------------------------------------------------------
    with section("13", "explicit restart"):
        restarted = compose("restart", "backend", "frontend")
        check("13a", "backend and frontend restart", restarted.returncode == 0, restarted.stderr[-800:])
        check("13b", "healthy again after the restart", wait_healthy(120))
        s, A3 = login("alice")
        check("13c", "data survives a container restart",
              s == 200 and http("GET", BACKEND + "/load_session", headers=A3, params={"session_id": sid})[0] == 200)

    # ---- 14. database start-up retry: the backend boots while PostgreSQL is down -----------------------------------
    with section("14", "database start-up retry"):
        stopped = compose("stop", "postgres")
        check("14a", "PostgreSQL stopped", stopped.returncode == 0, stopped.stderr[-500:])
        restarted = compose("restart", "backend")           # `restart` ignores depends_on: the backend boots with no database
        check("14b", "backend restarted while PostgreSQL is down", restarted.returncode == 0, restarted.stderr[-500:])
        time.sleep(10)
        running = run(["docker", "inspect", "--format", "{{.State.Running}}", container_id("backend")]).stdout.strip()
        check("14c", "the backend keeps running (retrying) instead of exiting, and is not serving yet",
              running == "true" and unreachable(BACKEND + "/health"), running)
        logs = compose("logs", "backend", "--no-log-prefix", "--tail", "300").stdout
        check("14d", "the retry was exercised: 'Database not ready' with the exception class is logged",
              "Database not ready" in logs and "OperationalError" in logs)
        started = compose("start", "postgres")
        check("14e", "PostgreSQL started again", started.returncode == 0, started.stderr[-500:])
        check("14f", "the backend becomes healthy once PostgreSQL returns (no manual restart)", wait_healthy(180))
        s, A4 = login("alice")
        check("14g", "data intact after the outage",
              s == 200 and http("GET", BACKEND + "/load_session", headers=A4, params={"session_id": sid})[0] == 200)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # never depend on the console's default encoding
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    try:
        ok = run(["docker", "compose", "version"]).returncode == 0
    except OSError:
        ok = False
    if not ok:
        say("Docker Compose v2 is not available. Start Docker Desktop and try again.")
        return 2
    try:
        existing = existing_resources()
    except Exception as exc:
        say("Could not check for an existing '%s' stack (%s). Is the Docker daemon running? Nothing was started." % (PROJECT, exc))
        return 2
    if existing:
        say("STOPPED: resources of an existing '%s' stack were found. Nothing was started and nothing was deleted." % PROJECT)
        for kind, names in existing.items():
            say("  %s: %s" % (kind, ", ".join(names)))
        say("This usually means an earlier run used --keep or was interrupted. If you are sure they are only test resources, "
            "remove them yourself and run again:")
        say("  " + " ".join(COMPOSE + ["down", "-v"]))
        return 3

    spec = importlib.util.spec_from_file_location("minipdf", os.path.join(HERE, "minipdf.py"))
    minipdf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(minipdf)
    pdf = minipdf.make_pdf(minipdf.RESUME_LINES)

    events = Events()
    created = False
    try:
        created = True              # from here on, every resource carrying the project name was created by this run
        events.start()
        run_checks(args, pdf, events)
    except KeyboardInterrupt:
        say("Interrupted.")
        check("0", "run completed", False, "interrupted")
    except Exception as exc:        # anything outside a section
        check("0", "run completed", False, "%s: %s" % (type(exc).__name__, exc))
    finally:
        events.stop()
        if created and not args.keep:
            cleanup = compose("down", "-v")
            say("Cleanup (docker compose -p %s down -v): exit code %s" % (PROJECT, cleanup.returncode))
            try:
                left = existing_resources()
            except Exception as exc:
                left = {"could not verify": [str(exc)]}
            check("99", "cleanup removed every resource of the smoke project", not left, left)
        elif args.keep:
            say("--keep: the stack was left running. Remove it later with:\n  " + " ".join(COMPOSE + ["down", "-v"]))

    failed = [(cid, name) for cid, name, status in results if status == "FAIL"]
    say("\n%d checks, %d passed, %d failed" % (len(results), len(results) - len(failed), len(failed)))
    if failed:
        say("Failed check IDs: " + ", ".join(cid for cid, _ in failed))
    for cid, name, reason in not_verified:
        say("Not verified [%s]: %s (%s)" % (cid, name, reason))
    say("Not tested: automatic crash recovery (restart policy), the Streamlit UI itself, live Groq/YouTube behaviour.")
    say("Built images (resume-smoke-*), pulled base images and build cache remain; remove them yourself if you want.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
