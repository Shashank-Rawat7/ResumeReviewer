"""Offline tests for the JWT signing-key validation in backend/db/auth.py.

Run from the project root:  python -m unittest tests.test_secret_key -v

Nothing here needs Docker, PostgreSQL or the network. Each case imports backend.db.auth in a subprocess with a stubbed
python-dotenv, so the real .env file is never read. The key values below are throw-away test strings, not secrets.
The checks are sanity checks: they reject obviously weak or copied keys; they do not prove a key is random.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCRIPT = r'''
import contextlib, importlib, io, json, logging, os, sys, types
sys.path.insert(0, os.environ["RR_ROOT"])
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False)
cases = json.loads(os.environ["RR_CASES"])

class Collect(logging.Handler):
    def __init__(self):
        super().__init__(0)
        self.lines = []
    def emit(self, record):
        self.lines.append(record.getMessage())
collector = Collect()
logging.getLogger().addHandler(collector)
logging.getLogger().setLevel(logging.DEBUG)

results = {}
for name, value in cases:
    if value is None:
        os.environ.pop("SECRET_KEY", None)
    else:
        os.environ["SECRET_KEY"] = value
    sys.modules.pop("backend.db.auth", None)
    collector.lines.clear()
    out, err = io.StringIO(), io.StringIO()
    entry = {"ok": False, "exc": None, "msg": ""}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            auth = importlib.import_module("backend.db.auth")
        entry["ok"] = True
    except Exception as exc:
        entry["exc"], entry["msg"] = type(exc).__name__, str(exc)
    if entry["ok"]:
        import time
        from datetime import datetime, timedelta
        from jose import jwt, JWTError
        token = auth.create_access_token({"sub": "alice"})
        claims = jwt.decode(token, value, algorithms=["HS256"])
        # a token produced independently with the same key and algorithm is still accepted
        foreign = jwt.encode({"sub": "bob", "exp": datetime.utcnow() + timedelta(minutes=5)}, value, algorithm="HS256")
        try:
            jwt.decode(token, value + "x", algorithms=["HS256"])
            wrong_key_rejected = False
        except JWTError:
            wrong_key_rejected = True
        entry["jwt"] = {
            "header_alg": jwt.get_unverified_header(token)["alg"], "sub": claims["sub"],
            "lifetime_s": round(claims["exp"] - time.time()),
            "foreign_sub": jwt.decode(foreign, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])["sub"],
            "wrong_key_rejected": wrong_key_rejected, "algorithm": auth.ALGORITHM,
            "expire_minutes": auth.ACCESS_TOKEN_EXPIRE_MINUTES, "key_unchanged": auth.SECRET_KEY == value,
        }
    entry["captured"] = out.getvalue() + err.getvalue() + "\n".join(collector.lines)
    results[name] = entry
print("RESULT:" + json.dumps(results))
'''

# Fixed, random-looking test values (a fixed value keeps the tests deterministic).
VALID_HEX_64 = "8f3c1a9e5b7d4026e1a8c3f59b0d7e42a6c9f1e3d5b8a04719c2e6b4d8f0a35c"
VALID_32 = "q7Zk2xV9mPb4Rt8LwN3cYd6HsJ5aGe1F"
VALID_URLSAFE_43 = "Tz4xK9_mQ2vLp7Rt4wBn8Hs-Jd5aGe1FcYu3Xo6Vi0E"
SHORT_16 = "Xk9$mQ2vLp7#Rt4w"
ALMOST_32 = "q7Zk2xV9mPb4Rt8LwN3cYd6HsJ5aGe1"


def env_file_value(*parts):
    """SECRET_KEY value from a KEY=value file (.env.example or the smoke-test env file). Never .env."""
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            if line.startswith("SECRET_KEY="):
                return line.split("=", 1)[1].strip()
    raise AssertionError("no SECRET_KEY line in " + "/".join(parts))


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("jose", "passlib")), "python-jose / passlib not installed")
class SecretKeyValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.placeholder = env_file_value(".env.example")
        cls.smoke_key = env_file_value("tests", "docker", "smoke.env")
        cls.cases = {
            "missing": None,
            "empty": "",
            "blank": "   ",
            "env_example_placeholder": cls.placeholder,
            "placeholder_short": "change-me",
            "placeholder_long": "Your-Secret-Key-Goes-Here-1234567890abcdef",
            "too_short": SHORT_16,
            "one_short_of_minimum": ALMOST_32,
            "repeated_single_char": "a" * 64,
            "repeated_pair": "ab" * 32,
            "repeated_digits": "1234567890" * 4,
            "repeated_block": "abcdefgh" * 8,
            "two_characters_only": "a" * 20 + "b" * 20,
            "valid_hex_64": VALID_HEX_64,
            "valid_minimum_length": VALID_32,
            "valid_urlsafe": VALID_URLSAFE_43,
            "smoke_test_key": cls.smoke_key,
        }
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SECRET_", "DATABASE_", "POSTGRES_"))}
        env.update({"RR_ROOT": ROOT, "RR_CASES": json.dumps(list(cls.cases.items())), "PYTHONDONTWRITEBYTECODE": "1"})
        done = subprocess.run([sys.executable, "-B", "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=120)
        marker = [l for l in done.stdout.splitlines() if l.startswith("RESULT:")]
        assert marker, done.stdout[-800:] + done.stderr[-2000:]
        cls.out = json.loads(marker[-1][7:])

    def rejected(self, name):
        entry = self.out[name]
        self.assertFalse(entry["ok"], name + " was accepted")
        self.assertEqual(entry["exc"], "RuntimeError", name)
        self.assertIn("SECRET_KEY", entry["msg"], name)
        self.assertIn("secrets.token_hex(32)", entry["msg"], name)     # tells the operator how to fix it
        return entry["msg"]

    def test_missing_and_empty_keys_are_rejected(self):
        for name in ("missing", "empty", "blank"):
            self.assertIn("not set", self.rejected(name), name)

    def test_the_env_example_placeholder_is_rejected(self):
        self.assertIn("placeholder", self.rejected("env_example_placeholder"))
        for name in ("placeholder_short", "placeholder_long"):
            self.assertIn("placeholder", self.rejected(name), name)

    def test_too_short_keys_are_rejected(self):
        for name in ("too_short", "one_short_of_minimum"):
            self.assertIn("too short", self.rejected(name), name)

    def test_repeated_character_keys_are_rejected(self):
        for name in ("repeated_single_char", "repeated_pair", "repeated_digits", "repeated_block", "two_characters_only"):
            self.assertIn("repetitive", self.rejected(name), name)

    def test_long_random_looking_keys_are_accepted(self):
        self.assertEqual(len(VALID_32), 32)
        for name in ("valid_hex_64", "valid_minimum_length", "valid_urlsafe", "smoke_test_key"):
            self.assertTrue(self.out[name]["ok"], name + " was rejected: " + self.out[name]["msg"][:80])

    def test_the_smoke_test_key_is_not_the_placeholder(self):
        self.assertNotEqual(self.smoke_key, self.placeholder)
        self.assertGreaterEqual(len(self.smoke_key), 32)

    def test_no_secret_value_appears_in_errors_output_or_logs(self):
        checked = 0
        for name, value in self.cases.items():
            if not value or not value.strip():
                continue                                              # nothing to leak (and "   " would match any text)
            entry = self.out[name]
            self.assertNotIn(value, entry["msg"], name)
            self.assertNotIn(value, entry["captured"], name)          # stdout, stderr and log records during import
            checked += 1
        self.assertGreaterEqual(checked, 14)

    def test_jwt_behaviour_is_unchanged_for_a_valid_key(self):
        for name in ("valid_hex_64", "valid_minimum_length", "valid_urlsafe", "smoke_test_key"):
            jwt_info = self.out[name]["jwt"]
            self.assertEqual(jwt_info["header_alg"], "HS256", name)
            self.assertEqual(jwt_info["algorithm"], "HS256", name)
            self.assertEqual(jwt_info["expire_minutes"], 60, name)
            self.assertLessEqual(abs(jwt_info["lifetime_s"] - 3600), 5, name)
            self.assertEqual(jwt_info["sub"], "alice", name)
            self.assertEqual(jwt_info["foreign_sub"], "bob", name)    # tokens signed with the same key and algorithm
            self.assertTrue(jwt_info["wrong_key_rejected"], name)
            self.assertTrue(jwt_info["key_unchanged"], name)          # the key is used exactly as configured

    def test_a_weak_key_stops_the_process_without_printing_it(self):
        weak = "Wk" * 20
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SECRET_", "DATABASE_", "POSTGRES_"))}
        env.update({"RR_ROOT": ROOT, "SECRET_KEY": weak, "PYTHONDONTWRITEBYTECODE": "1"})
        code = ("import os, sys, types; sys.path.insert(0, os.environ['RR_ROOT']); "
                "sys.modules['dotenv'] = types.SimpleNamespace(load_dotenv=lambda *a, **k: False); "
                "import backend.db.auth")
        done = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("RuntimeError", done.stderr)
        self.assertIn("SECRET_KEY", done.stderr)
        self.assertNotIn(weak, done.stdout + done.stderr)


class StartupWiring(unittest.TestCase):
    def read(self, *parts):
        with open(os.path.join(ROOT, *parts), "rb") as handle:
            return handle.read().decode("utf-8").replace("\r\n", "\n")

    def test_the_key_is_validated_before_the_app_is_created(self):
        main = self.read("backend", "main.py")
        self.assertRegex(main, r"(?m)^from backend\.db import crud, auth$")   # importing auth runs the validation
        self.assertLess(main.index("from backend.db import crud, auth"), main.index("init_db()"))
        self.assertLess(main.index("from backend.db import crud, auth"), main.index("app = FastAPI()"))
        auth = self.read("backend", "db", "auth.py")
        self.assertRegex(auth, r"(?m)^SECRET_KEY = validate_secret_key\(os\.getenv\(\"SECRET_KEY\"\)\)$")

    def test_compose_supplies_no_fallback_key(self):
        lines = [l for l in self.read("docker-compose.yaml").splitlines() if re.match(r"\s*SECRET_KEY:", l)]
        self.assertEqual(len(lines), 1)
        self.assertIn("${SECRET_KEY:?", lines[0])        # required: compose stops when it is unset or empty
        self.assertNotIn(":-", lines[0])                 # no default value
        self.assertNotRegex(lines[0], r"^\s*SECRET_KEY:\s*[^$\s]")   # no literal value


if __name__ == "__main__":
    unittest.main()
