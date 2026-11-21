"""Static checks of the Docker / deployment configuration, plus the database start-up retry.

Run from the project root:  python -m unittest tests.test_deploy_config -v

Nothing here needs Docker, a database server or the network. The compose file and Dockerfiles are parsed as text
(and as YAML when PyYAML is available), requirements.txt is checked against what the code really imports, and the
database retry runs in a subprocess against a throw-away SQLite file.
"""
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTECTED = ("resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt", "missing_skills.txt",
             "ats_optimized_bullets.txt")


def read(*parts):
    with open(os.path.join(ROOT, *parts), "rb") as handle:
        raw = handle.read()
    return raw.decode("utf-8").replace("\r\n", "\n")


def compose():
    import yaml
    return yaml.safe_load(read("docker-compose.yaml"))


HAS_YAML = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class ComposeFile(unittest.TestCase):
    def setUp(self):
        self.c = compose()
        self.s = self.c["services"]

    def test_three_services(self):
        self.assertEqual(set(self.s), {"postgres", "backend", "frontend"})

    def test_postgres_15_with_pg_isready_healthcheck_and_persistent_volume(self):
        pg = self.s["postgres"]
        self.assertEqual(pg["image"], "postgres:15.20-trixie")
        self.assertIn("pg_isready", " ".join(pg["healthcheck"]["test"]))
        self.assertIn("postgres_data:/var/lib/postgresql/data", pg["volumes"])
        self.assertIn("postgres_data", self.c["volumes"])

    def test_startup_order_waits_for_healthy_dependencies(self):
        self.assertEqual(self.s["backend"]["depends_on"]["postgres"]["condition"], "service_healthy")
        self.assertEqual(self.s["frontend"]["depends_on"]["backend"]["condition"], "service_healthy")

    def test_backend_and_frontend_have_healthchecks(self):
        self.assertIn("/health", " ".join(self.s["backend"]["healthcheck"]["test"]))
        self.assertIn("_stcore/health", " ".join(self.s["frontend"]["healthcheck"]["test"]))

    def test_every_published_port_is_bound_to_loopback(self):
        for name, service in self.s.items():
            for port in service.get("ports", []):
                self.assertTrue(str(port).startswith("127.0.0.1:"), (name, port))

    def test_internal_service_addresses(self):
        backend, frontend = self.s["backend"]["environment"], self.s["frontend"]["environment"]
        self.assertEqual(backend["POSTGRES_HOST"], "postgres")
        self.assertEqual(str(backend["POSTGRES_PORT"]), "5432")
        self.assertEqual(frontend["API_BASE_URL"], "http://backend:8000")

    def test_uploads_live_in_a_named_volume_that_only_the_backend_mounts(self):
        # Uploads are stored by the backend (POST /upload_resume); the frontend never touches the volume.
        self.assertIn("uploads:/app/uploads", self.s["backend"]["volumes"])
        self.assertEqual(self.s["backend"]["environment"]["UPLOAD_DIR"], "/app/uploads")
        self.assertNotIn("volumes", self.s["frontend"])
        self.assertNotIn("UPLOAD_DIR", self.s["frontend"]["environment"])
        self.assertIn("uploads", self.c["volumes"])

    def test_no_source_tree_or_env_file_is_mounted_or_passed_wholesale(self):
        for name in ("backend", "frontend"):
            service = self.s[name]
            self.assertNotIn("env_file", service)
            for volume in service.get("volumes", []):
                self.assertFalse(str(volume).startswith(("./", "../", "/")), (name, volume))
                self.assertNotIn("src", str(volume).split(":")[0])

    def test_empty_groq_base_url_is_never_injected(self):
        # An empty GROQ_API_BASE would override the SDK default and break every model call.
        self.assertNotIn("GROQ_API_BASE", self.s["backend"]["environment"])

    def test_no_secret_values_are_written_in_the_file(self):
        text = read("docker-compose.yaml")
        self.assertNotRegex(text, r"gsk_[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}")
        for line in text.splitlines():
            if re.match(r"\s*(POSTGRES_PASSWORD|SECRET_KEY|GROQ_API_KEY|YOUTUBE_API_KEY):", line):
                self.assertIn("${", line, line)

    def test_the_old_override_file_is_not_present(self):
        self.assertFalse(os.path.exists(os.path.join(ROOT, "docker-compose.override.yaml")))


class Dockerfiles(unittest.TestCase):
    def check(self, path):
        text = "\n".join(l for l in read(path).splitlines() if not l.lstrip().startswith("#"))
        self.assertIn("FROM python:3.12.16-slim-trixie", text)
        self.assertRegex(text, r"(?m)^USER app$")
        self.assertNotIn("--reload", text)
        self.assertNotRegex(text, r"(?mi)^(COPY|ADD)\s+.*\.env")
        self.assertIn("HEALTHCHECK", text)
        self.assertIn("pip install -r requirements.txt", text)
        return text

    def test_backend(self):
        text = self.check("backend/Dockerfile")
        self.assertIn("COPY --chown=app:app backend/", text)
        self.assertIn("COPY --chown=app:app src/", text)
        self.assertIn("UPLOAD_DIR=/app/uploads", text)
        self.assertIn('"uvicorn", "backend.main:app"', text)

    def test_frontend(self):
        text = self.check("frontend/Dockerfile")
        self.assertNotRegex(text, r"(?m)^COPY .* src/")
        self.assertIn('"streamlit", "run", "frontend/main.py"', text)

    def test_dockerignore_keeps_secrets_and_personal_data_out_of_the_build_context(self):
        rules = {line.strip() for line in read(".dockerignore").splitlines() if line.strip() and not line.startswith("#")}
        for needed in ("**/.env", "**/.env.*", ".git", "data", "**/*.pdf", "src/*.txt", "**/venv", "**/.venv",
                       "**/__pycache__", "**/logs", "eval", "tests"):
            self.assertIn(needed, rules)
        for runtime in ("requirements.txt", "backend", "frontend", "src"):
            self.assertNotIn(runtime, rules)          # required runtime files must not be excluded


class Requirements(unittest.TestCase):
    def lines(self):
        with open(os.path.join(ROOT, "requirements.txt"), "rb") as handle:
            raw = handle.read()
        self.assertFalse(raw.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")), "must be plain UTF-8")
        self.assertNotIn(b"\x00", raw)
        text = raw.decode("utf-8")
        return [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]

    @staticmethod
    def name(line):
        return re.split(r"[\[=<>!~ ]", line, maxsplit=1)[0].lower().replace("_", "-")

    def test_no_duplicates_and_everything_is_pinned(self):
        lines = self.lines()
        names = [self.name(l) for l in lines]
        self.assertEqual(len(names), len(set(names)), "duplicate requirement")
        for line in lines:
            self.assertRegex(line, r"==\d", line)

    def test_every_imported_third_party_package_is_listed(self):
        mapping = {"dotenv": "python-dotenv", "jose": "python-jose", "yaml": "pyyaml", "googleapiclient": "google-api-python-client",
                   "langchain_groq": "langchain-groq", "langchain_core": "langchain-core", "PyPDF2": "pypdf2",
                   "psycopg2": "psycopg2-binary", "multipart": "python-multipart"}
        listed = {self.name(l) for l in self.lines()}
        local = {"backend", "frontend", "src", "db", "eval", "tests", "helper"}
        imported = set()
        for folder in ("backend", "frontend", "src"):
            for dirpath, dirs, files in os.walk(os.path.join(ROOT, folder)):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for name in files:
                    if name.endswith(".py"):
                        tree = ast.parse(read(os.path.relpath(os.path.join(dirpath, name), ROOT)))
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                imported.update(a.name.split(".")[0] for a in node.names)
                            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                                imported.add(node.module.split(".")[0])
        third_party = {m for m in imported if m not in local and m not in sys.stdlib_module_names}
        missing = sorted(m for m in third_party if mapping.get(m, m.lower().replace("_", "-")) not in listed)
        self.assertEqual(missing, [])

    def test_packages_the_code_never_imports_were_removed(self):
        listed = {self.name(l) for l in self.lines()}
        for unused in ("pymupdf", "google-generativeai", "langchain-google-genai", "dotenv"):
            self.assertNotIn(unused, listed)


class UploadDirectory(unittest.TestCase):
    def test_backend_defaults_to_src_and_honours_upload_dir_and_the_frontend_does_not_use_it(self):
        text = read("backend/main.py")
        self.assertIn('os.getenv("UPLOAD_DIR") or os.path.join(os.path.dirname(__file__), "..", "src")', text)
        self.assertNotIn("UPLOAD_DIR", read("frontend/main.py"))
        self.assertNotIn("UPLOAD_DIR", read("frontend/Dockerfile"))

    def test_generated_intermediate_files_are_not_redirected_to_the_upload_volume(self):
        text = read("src", "parser.py")
        self.assertIn("CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))", text)
        self.assertNotIn("UPLOAD_DIR", text)


class DatabaseStartup(unittest.TestCase):
    SCRIPT = r'''
import os, sys, types
from unittest import mock
sys.path.insert(0, os.environ["RR_ROOT"])
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)
from sqlalchemy.exc import OperationalError
import backend.db.database as database
import backend.db.models  # noqa: F401  (registers the tables)

def boom():
    return OperationalError("stmt", {}, Exception("password=hunter2 host=db"))

# 1. database not ready for two attempts, then ready: init_db waits and succeeds
calls = []
real = database.Base.metadata.create_all
def flaky(bind=None):
    calls.append(1)
    if len(calls) < 3:
        raise boom()
    return real(bind=bind)
sleeps = []
with mock.patch.object(database.Base.metadata, "create_all", side_effect=flaky), \
     mock.patch.object(database.time, "sleep", side_effect=sleeps.append), \
     mock.patch.object(database.logging, "warning") as warn:
    database.init_db(attempts=5, delay=0.5)
ok_attempts, ok_sleeps = len(calls), list(sleeps)
logged = " ".join(str(a) for c in warn.call_args_list for a in c.args)

# 2. never ready: the last error is raised after the allowed attempts
calls2 = []
def never(bind=None):
    calls2.append(1)
    raise boom()
raised = None
with mock.patch.object(database.Base.metadata, "create_all", side_effect=never), \
     mock.patch.object(database.time, "sleep"):
    try:
        database.init_db(attempts=3, delay=0)
    except OperationalError as exc:
        raised = type(exc).__name__

# 3. other errors are not swallowed or retried
calls3 = []
def other(bind=None):
    calls3.append(1)
    raise ValueError("bug")
bug = None
with mock.patch.object(database.Base.metadata, "create_all", side_effect=other), mock.patch.object(database.time, "sleep"):
    try:
        database.init_db(attempts=3, delay=0)
    except ValueError:
        bug = "ValueError"

# 4. the real thing against SQLite creates the tables on the first attempt
database.init_db(attempts=1, delay=0)
tables = sorted(database.Base.metadata.tables)
import json
print("RESULT:" + json.dumps({"ok_attempts": ok_attempts, "ok_sleeps": ok_sleeps, "never_attempts": len(calls2), "raised": raised,
                              "bug_attempts": len(calls3), "bug": bug, "log_has_secret": "hunter2" in logged, "tables": tables}))
'''

    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("sqlalchemy",)), "SQLAlchemy not installed")
    def test_init_db_waits_for_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("POSTGRES_", "DATABASE_"))}
            env.update({"RR_ROOT": ROOT, "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "t.db").replace("\\", "/"),
                        "PYTHONDONTWRITEBYTECODE": "1"})
            done = subprocess.run([sys.executable, "-B", "-c", self.SCRIPT], env=env, cwd=tmp, capture_output=True,
                                  text=True, timeout=60)
        marker = [l for l in done.stdout.splitlines() if l.startswith("RESULT:")]
        self.assertTrue(marker, done.stdout[-800:] + done.stderr[-2000:])
        import json
        out = json.loads(marker[-1][7:])
        self.assertEqual(out["ok_attempts"], 3)
        self.assertEqual(out["ok_sleeps"], [0.5, 0.5])
        self.assertEqual((out["never_attempts"], out["raised"]), (3, "OperationalError"))
        self.assertEqual((out["bug_attempts"], out["bug"]), (1, "ValueError"))
        self.assertFalse(out["log_has_secret"])
        self.assertEqual(out["tables"], ["sessions", "users"])


class DatabaseDriverUrl(unittest.TestCase):
    """SQLAlchemy 2.1 maps a driver-less postgresql:// URL to psycopg (v3), but only psycopg2 is installed."""
    SCRIPT = r'''
import json, os, sys, types
from unittest import mock
sys.path.insert(0, os.environ["RR_ROOT"])
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)
import sqlalchemy
from sqlalchemy.engine import make_url

# create_engine is replaced, so no engine, driver import or connection can happen while the module loads
with mock.patch.object(sqlalchemy, "create_engine") as fake_engine:
    import backend.db.database as database
def parts(value):
    """Components as SQLAlchemy sees them (str() of a URL hides the password, so read the fields)."""
    url = make_url(value)
    return {"drivername": url.drivername, "username": url.username, "password": url.password, "host": url.host,
            "port": url.port, "database": url.database}

passed_to_engine = [c.args[0] for c in fake_engine.call_args_list]
url = make_url(database.DATABASE_URL)
dialect = url.get_dialect()          # resolves the dialect class only; does not import or call the DBAPI
print("RESULT:" + json.dumps({
    "module_url": parts(database.DATABASE_URL), "engine_calls": [parts(v) for v in passed_to_engine],
    "engine_arg_is_module_url": len(passed_to_engine) == 1 and passed_to_engine[0] is database.DATABASE_URL,
    "engine_arg_is_str": [isinstance(v, str) for v in passed_to_engine], "engine_str": [v for v in passed_to_engine if isinstance(v, str)],
    "backend": url.get_backend_name(), "driver": url.get_driver_name(), "dialect_module": dialect.__module__,
    "psycopg_imported": "psycopg" in sys.modules,
}))
'''

    def run_script(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("POSTGRES_", "DATABASE_"))}
        env.update({"RR_ROOT": ROOT, "PYTHONDONTWRITEBYTECODE": "1"})
        env.update(extra)
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run([sys.executable, "-B", "-c", self.SCRIPT], env=env, cwd=tmp, capture_output=True,
                                  text=True, timeout=60)
        marker = [l for l in done.stdout.splitlines() if l.startswith("RESULT:")]
        self.assertTrue(marker, done.stdout[-800:] + done.stderr[-2000:])
        import json
        return json.loads(marker[-1][7:])

    def check_fallback(self, user, password, host="postgres", port="5432", db="d"):
        out = self.run_script(POSTGRES_USER=user, POSTGRES_PASSWORD=password, POSTGRES_DB=db, POSTGRES_HOST=host,
                              POSTGRES_PORT=port)
        expected = {"drivername": "postgresql+psycopg2", "username": user, "password": password, "host": host,
                    "port": int(port), "database": db}
        self.assertEqual(out["module_url"], expected)          # the raw values come back out unchanged
        self.assertEqual(out["engine_calls"], [expected])      # and that is exactly what create_engine receives
        self.assertTrue(out["engine_arg_is_module_url"])
        self.assertEqual((out["backend"], out["driver"]), ("postgresql", "psycopg2"))
        self.assertTrue(out["dialect_module"].endswith("postgresql.psycopg2"), out["dialect_module"])
        self.assertFalse(out["psycopg_imported"])

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_fallback_url_uses_the_psycopg2_driver(self):
        self.check_fallback("u", "p")

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_password_with_at_sign(self):
        self.check_fallback("resume_user", "p@ss@word")      # a plain f-string URL read the host as "ss@word@postgres"

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_password_with_percent_sign(self):
        self.check_fallback("resume_user", "pa%41ss%")       # a plain f-string URL decoded "%41" to "A"

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_password_with_many_url_sensitive_characters(self):
        self.check_fallback("user.name+x", "a@b:c/d?e#f%g&h=i+j k[l]m\\n", host="db.internal", port="6543", db="my_db")

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_explicit_database_url_is_used_unchanged(self):
        override = "sqlite:///override.db"
        out = self.run_script(DATABASE_URL=override, POSTGRES_USER="u", POSTGRES_PASSWORD="p@ss", POSTGRES_DB="d",
                              POSTGRES_HOST="postgres", POSTGRES_PORT="5432")
        self.assertEqual(out["engine_str"], [override])       # the very same string, never rebuilt from POSTGRES_*
        self.assertEqual(out["engine_arg_is_str"], [True])
        self.assertEqual(out["module_url"]["drivername"], "sqlite")
        self.assertEqual(out["module_url"]["database"], "override.db")

    @unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "SQLAlchemy not installed")
    def test_explicit_database_url_takes_precedence_over_the_fallback(self):
        override = "postgresql+psycopg2://other:s3cret@elsewhere:6000/otherdb"
        out = self.run_script(DATABASE_URL=override, POSTGRES_USER="u", POSTGRES_PASSWORD="p", POSTGRES_DB="d",
                              POSTGRES_HOST="postgres", POSTGRES_PORT="5432")
        self.assertEqual(out["engine_str"], [override])
        self.assertEqual(out["module_url"], {"drivername": "postgresql+psycopg2", "username": "other",
                                             "password": "s3cret", "host": "elsewhere", "port": 6000,
                                             "database": "otherdb"})


DEVELOPER_SCRIPTS = ("backend/testing.py", "backend/db/reset_db.py", "backend/db/connection.py")


def _docker_pattern(pattern):
    """A .dockerignore pattern as a regular expression: '**' crosses folders, '*' and '?' stay inside one name."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and "]" in pattern[i:]:
            end = pattern.index("]", i)
            out.append(pattern[i:end + 1])
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _in_build_context(path, rules):
    """Docker's rule: a path is dropped when a pattern matches it or one of its parent folders; the last rule wins."""
    parts = path.split("/")
    dropped = False
    for rule in rules:
        negate = rule.startswith("!")
        regex = _docker_pattern(rule.lstrip("!").strip("/"))
        if any(regex.fullmatch("/".join(parts[:n])) for n in range(1, len(parts) + 1)):
            dropped = not negate
    return not dropped


class DeveloperScriptsStayOutOfTheBackendImage(unittest.TestCase):
    """Both images are built with the project root as context, so the root .dockerignore decides what `COPY backend/`
    brings in (backend/.dockerignore is only read when the context is backend/ itself)."""

    @classmethod
    def setUpClass(cls):
        cls.rules = [l.strip() for l in read(".dockerignore").splitlines() if l.strip() and not l.startswith("#")]
        cls.files = sorted(
            os.path.relpath(os.path.join(folder, name), ROOT).replace(os.sep, "/")
            for folder, folders, names in os.walk(os.path.join(ROOT, "backend")) for name in names
            if "__pycache__" not in folder
        )

    def test_the_context_check_itself_behaves_like_docker(self):
        self.assertFalse(_in_build_context("backend/db/__pycache__/x.pyc", self.rules))      # '**/' rule, nested
        self.assertFalse(_in_build_context("tests/test_deploy_config.py", self.rules))        # a folder rule
        self.assertTrue(_in_build_context("backend/main.py", self.rules))
        self.assertTrue(_in_build_context("src/parser.py", self.rules))
        self.assertTrue(_in_build_context("backend/db/reset_db.py", ["README.md"]))           # no rule, no exclusion

    def test_developer_scripts_are_not_in_the_backend_build_context(self):
        for script in DEVELOPER_SCRIPTS:
            self.assertIn(script, self.files)                                                  # still in the repository
            self.assertFalse(_in_build_context(script, self.rules), script)

    def test_everything_the_backend_needs_is_still_in_the_build_context(self):
        for needed in ("backend/__init__.py", "backend/main.py", "backend/db/__init__.py", "backend/db/auth.py",
                       "backend/db/crud.py", "backend/db/database.py", "backend/db/models.py"):
            self.assertTrue(_in_build_context(needed, self.rules), needed)

    def test_no_shipped_code_imports_a_developer_script(self):
        shipped = [f for f in self.files if f.endswith(".py") and _in_build_context(f, self.rules)]
        for folder in ("frontend", "src"):
            shipped += [folder + "/" + n for n in sorted(os.listdir(os.path.join(ROOT, folder))) if n.endswith(".py")]
        forbidden = {"backend.testing", "backend.db.reset_db", "backend.db.connection",
                     "db.reset_db", "db.connection", "testing", "reset_db", "connection"}
        for path in shipped:
            package = path.rsplit("/", 1)[0].replace("/", ".")
            for node in ast.walk(ast.parse(read(path))):
                found = set()
                if isinstance(node, ast.Import):
                    found = {alias.name for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                        base = parent + ("." + base if base else "")
                    found = {base} | {base + "." + alias.name for alias in node.names}
                self.assertFalse(found & forbidden, "%s imports %s" % (path, sorted(found & forbidden)))


class NoProtectedOrPersonalFilesInTheDeploymentConfig(unittest.TestCase):
    def test_ignore_files_cover_generated_files_and_secrets(self):
        git = read(".gitignore")
        for needed in (".env", "/src/resume_text.txt", "/src/resume_rewritten.txt", "logs/", "__pycache__/", "/data/raw_resume/*.pdf"):
            self.assertIn(needed, git)
        for name in PROTECTED:
            self.assertIn("/src/" + name, git)


if __name__ == "__main__":
    unittest.main()
