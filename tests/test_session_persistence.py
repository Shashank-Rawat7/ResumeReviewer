"""Regression test: a saved resume_analysis must persist to the sessions table.

Background: `session.data["resume_analysis"] = ...` changes the JSON dict in place, which SQLAlchemy does
not detect on a plain JSON column, so `db.commit()` wrote nothing. /parse_resume now assigns a new dict.

Run from the project root:  python -m unittest tests.test_session_persistence -v
Uses synthetic data and an in-memory SQLite database only; it never connects to PostgreSQL.
"""
import os
import re
import unittest

# Must be set before backend.db.database is imported: that module creates its engine at import time.
os.environ["DATABASE_URL"] = "sqlite://"

from backend.db import crud  # noqa: E402
from backend.db.database import Base, SessionLocal, engine  # noqa: E402
from backend.db.models import Session as UserSession  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The statement /parse_resume uses to save the analysis (backend/main.py).
SAVE_STATEMENT = 'session.data = {**(session.data or {}), "resume_analysis": resume_analysis}'
OLD_STATEMENT = 'session.data["resume_analysis"] = resume_analysis'

ANALYSIS = {"summary": {"note": "synthetic"}, "job_skills": {"technical_skills": ["Python"]}}


def save_analysis(session, resume_analysis):
    """Apply SAVE_STATEMENT to a session row (same expression as in backend/main.py)."""
    session.data = {**(session.data or {}), "resume_analysis": resume_analysis}


def squeeze(text):
    """Remove whitespace and a trailing comma before '}' so formatting differences do not matter."""
    return re.sub(r"\s+", "", text).replace(",}", "}")


def setUpModule():
    # Safety: never run against anything but the in-memory SQLite database.
    if engine.url.drivername != "sqlite":
        raise AssertionError("test_session_persistence must only use SQLite, got " + engine.url.drivername)


class SessionPersistenceTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        db = SessionLocal()
        try:
            user = crud.create_user(db, "synthetic-user", "synthetic@example.com", "not-a-real-hash")
            self.session_id = crud.create_session(db, user.id, "synthetic-session").id
        finally:
            db.close()

    def tearDown(self):
        Base.metadata.drop_all(engine)

    def read_back(self):
        """Read the row through a brand-new database session."""
        fresh = SessionLocal()
        try:
            return fresh.get(UserSession, self.session_id).data
        finally:
            fresh.close()

    def test_resume_analysis_persists_and_reads_back(self):
        db = SessionLocal()
        try:
            save_analysis(db.get(UserSession, self.session_id), ANALYSIS)
            db.commit()
        finally:
            db.close()
        self.assertEqual(self.read_back()["resume_analysis"], ANALYSIS)

    def test_existing_keys_are_preserved(self):
        db = SessionLocal()
        try:
            row = db.get(UserSession, self.session_id)
            row.data = {"skill_recommendation": {"note": "synthetic"}, "extra_key": 1}
            db.commit()
            save_analysis(row, ANALYSIS)
            db.commit()
        finally:
            db.close()
        data = self.read_back()
        self.assertEqual(data["resume_analysis"], ANALYSIS)
        self.assertEqual(data["skill_recommendation"], {"note": "synthetic"})
        self.assertEqual(data["extra_key"], 1)


class ParseResumeSourceGuard(unittest.TestCase):
    """Fails if /parse_resume goes back to changing session.data in place."""

    def test_parse_resume_reassigns_session_data(self):
        with open(os.path.join(ROOT, "backend", "main.py"), encoding="utf-8") as handle:
            source = squeeze(handle.read())
        self.assertIn(squeeze(SAVE_STATEMENT), source)
        self.assertNotIn(squeeze(OLD_STATEMENT), source)


if __name__ == "__main__":
    unittest.main()
