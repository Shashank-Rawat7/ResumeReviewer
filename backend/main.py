import sys
import os
import re
import uuid
import asyncio
import logging
import threading

# Get the root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from fastapi import FastAPI, Form, Depends, HTTPException, UploadFile, File, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.parser import ResumeParser, ResumeParserConfig
from src.log_safety import install_exception_redaction
from src.metrics import RequestMetrics, configure_metrics_logger, install_retry_tracking
from src.recommender import SkillRecommender, RecommenderProviderError
from jose import jwt, JWTError
from backend.db.database import init_db, SessionLocal
from backend.db import crud, auth
from backend.db.models import User, Session as UserSession
from sqlalchemy.orm import Session as DBSession
import json

# Load environment variables
load_dotenv()
# Where the backend stores uploaded resume PDFs (one sub-folder per user id): a named volume in Docker
# (UPLOAD_DIR=/app/uploads), the src/ folder when run locally (the long-standing default).
UPLOAD_DIR = os.getenv("UPLOAD_DIR") or os.path.join(os.path.dirname(__file__), "..", "src")
# Uploads are created only by /upload_resume, as UPLOAD_DIR/<user id>/<32 hex characters>.pdf.
UPLOAD_NAME_RE = re.compile(r"[0-9a-f]{32}\.pdf")
# Upload limits: at most 10 MiB, read in small chunks, and only real PDFs (extension AND file signature).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
PDF_SIGNATURE = b"%PDF-"
# Initialize DB
init_db()
app = FastAPI()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


chat_sessions = {}  # {session_id: chat_history list}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Initialize Resume Parser and Skill Recommender
parser = ResumeParser()
recommender = SkillRecommender(youtube_api_key)

# Observability: one JSON metrics line per /parse_resume request (see README "Observability").
configure_metrics_logger()
install_retry_tracking()  # idempotent; counts SDK retries via a single filter, never touches handlers
install_exception_redaction()  # exception class names only in logs: messages can quote resume text


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: DBSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        user = crud.get_user_by_username(db, username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


_PRIORITY_TIERS = ("crucial", "good_to_have", "optional")
_SKILL_CATEGORIES = ("technical_skills", "soft_skills", "academic_qualifications")
_SKILL_STATUSES = ("missing", "partial")


def _unavailable_prioritized(message):
    """Prioritised result meaning "analysis unavailable" (not "nothing is missing")."""
    return {
        "crucial": [],
        "good_to_have": [],
        "optional": [],
        "normalization_ok": False,
        "diagnostics": [message],
    }


def _entry_problem(entry):
    """Describe what is wrong with a prioritised entry, or return None if it is valid."""
    if not isinstance(entry, dict):
        return "entry is not an object"
    for field in ("skill", "reason", "jd_evidence"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            return f"missing or empty '{field}'"
    if entry.get("category") not in _SKILL_CATEGORIES:
        return "invalid 'category'"
    if entry.get("status") not in _SKILL_STATUSES:
        return "invalid 'status'"
    if not isinstance(entry.get("evidence_found"), bool):
        return "'evidence_found' is not a boolean"
    for field in ("evidence_related", "in_extracted_list"):
        if field not in entry or not (entry[field] is None or isinstance(entry[field], bool)):
            return f"'{field}' is missing or not a boolean/null"
    return None


def _load_prioritized_missing_skills(value):
    """Parse and validate the parser's prioritised missing-skills result. Never raises.

    A valid result is returned exactly as the parser produced it (no fields are rebuilt
    or dropped), so normalization_ok, all three tiers, diagnostics and the per-entry
    evidence fields are preserved. Handling of anything else:
      - not parseable, or wrong outer shape (not an object, normalization_ok not a boolean,
        a tier that is not a list, diagnostics that are not a list of strings):
        replaced by an empty result with normalization_ok=False;
      - normalization_ok=True but any entry is structurally invalid: the WHOLE result is
        replaced by an empty result with normalization_ok=False (a partly-wrong analysis is
        not shown as valid); the diagnostic names the tier and position, not the content;
      - normalization_ok=False from the parser: passed through unchanged.

    Frontend contract: check normalization_ok before showing the skill-match bar, the
    missing-skills result, skill recommendations or the ATS score. When it is False the
    legacy missing_skills is an empty placeholder: "analysis unavailable", not "nothing is
    missing". A saved session with no prioritized_missing_skills key is a legacy session.
    """
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        logging.warning("Prioritised missing-skills result is not valid JSON")
        return _unavailable_prioritized("Prioritised missing-skills result was not valid JSON.")
    diagnostics = data.get("diagnostics") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("normalization_ok"), bool)
        or not all(isinstance(data.get(tier), list) for tier in _PRIORITY_TIERS)
        or not isinstance(diagnostics, list)
        or not all(isinstance(item, str) for item in diagnostics)
    ):
        logging.warning("Prioritised missing-skills result has an unexpected shape")
        return _unavailable_prioritized("Prioritised missing-skills result had an unexpected shape.")
    if data["normalization_ok"]:
        for tier in _PRIORITY_TIERS:
            for position, entry in enumerate(data[tier]):
                problem = _entry_problem(entry)
                if problem:
                    logging.warning(
                        "Prioritised missing-skills result discarded: invalid entry in '%s' at position %d (%s)",
                        tier, position, problem,
                    )
                    return _unavailable_prioritized(
                        f"Prioritised missing-skills result discarded: invalid entry in '{tier}' "
                        f"at position {position} ({problem})."
                    )
    return data


def clean_resume_analysis(summary, credentials, missing_skills, job_skills, ats_content,
                          prioritized_missing_skills=None):
    fields = {
        "summary": summary,
        "credentials": credentials,
        "missing_skills": missing_skills,
        "job_skills": job_skills,
        "ats_content": ats_content,
    }

    # NOTE: a non-JSON legacy field (e.g. the parser's "Error generating missing skills.")
    # deliberately still raises ValueError, which the endpoint turns into HTTP 500. Graceful
    # degradation (a 200 with normalization_ok=False) can be considered once the frontend
    # handles unavailable analysis.
    for field_name, value in fields.items():
        if isinstance(value, str):
            try:
                fields[field_name] = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in LLM field '{field_name}': {e}"
                ) from e

    summary = fields["summary"]
    credentials = fields["credentials"]
    missing_skills = fields["missing_skills"]
    job_skills = fields["job_skills"]
    ats_content = fields["ats_content"]

    # Fallback defaults
    summary = summary or {}
    credentials = credentials or {}
    missing_skills = missing_skills or {}
    job_skills = job_skills or {}
    ats_content = ats_content or {}

    analysis = {
        "summary": summary,
        "credentials": credentials,
        "missing_skills": missing_skills,
        "job_skills": job_skills,
        "ats_content": ats_content
    }

    # Additive key: only present when the caller supplies it, so sessions saved before this
    # field existed (and callers that omit it) keep the original five-key shape.
    if prioritized_missing_skills is not None:
        analysis["prioritized_missing_skills"] = _load_prioritized_missing_skills(
            prioritized_missing_skills
        )
    return analysis


# --- AUTH ROUTES ---
class RegisterRequest(BaseModel):
    """Sent as a JSON body so the password never appears in a URL (and so never in access logs)."""
    username: str = Field(min_length=1)
    email: str = Field(min_length=1)
    password: str = Field(min_length=1)


@app.post("/register")
def register(payload: RegisterRequest, db: DBSession = Depends(get_db)):
    if crud.get_user_by_username(db, payload.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_pw = auth.hash_password(payload.password)
    crud.create_user(db, payload.username, payload.email, hashed_pw)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: DBSession = Depends(get_db)):
    user = crud.get_user_by_username(db, form_data.username)
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- SESSION ROUTES ---
@app.post("/start_session")
def start_session(session_name: str, user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    session = crud.create_session(db, user.id, session_name)
    return {"session_id": session.id, "session_name": session.session_name}

@app.get("/my_sessions")
def my_sessions(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    sessions = crud.get_sessions_for_user(db, user.id)
    return [{"id": s.id, "name": s.session_name, "created_at": s.created_at} for s in sessions]

@app.get("/load_session")
def load_session(session_id: int, user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    session = db.query(UserSession).filter(
        UserSession.id == session_id,
        UserSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.data or {
    "resume_analysis": {},
    "skill_recommendation": {}
    }


class ResumeRequest(BaseModel):
    file_path: str
    job_description: str


def _user_upload_dir(user_id):
    """Each user's uploads live in their own directory, named after the authenticated user's id."""
    return os.path.join(UPLOAD_DIR, str(int(user_id)))


@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Store an uploaded resume under a server-generated name that belongs to the caller.

    The browser's file name is only checked for a .pdf extension, never used for storage. The returned name is what
    /parse_resume expects; because the file is stored in the caller's own directory, no other user can parse,
    overwrite or delete it.

    Checks run cheapest first and before anything is written: .pdf extension (415), non-empty (400), PDF signature
    in the first chunk (415). The body is then read in chunks and stored; past MAX_UPLOAD_BYTES the upload is refused
    with 413 without reading the rest. Whatever was written is removed on every failure.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted: the file name must end in .pdf.")
    chunk = await file.read(UPLOAD_CHUNK_BYTES)
    if not chunk:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if not chunk.startswith(PDF_SIGNATURE):
        raise HTTPException(status_code=415, detail="The file is not a valid PDF.")
    directory = _user_upload_dir(user.id)
    name = uuid.uuid4().hex + ".pdf"
    path = os.path.join(directory, name)
    created = stored = False
    size = 0
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "xb") as handle:   # "x": never overwrite an existing file
            created = True
            while chunk:
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The file is larger than the 10 MB limit.")
                handle.write(chunk)
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
        stored = True
    except OSError as exc:
        logging.error("Resume upload could not be stored (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="The upload could not be stored.")
    finally:
        if created and not stored:     # too large, disk error, cancelled request: never leave a partial file behind
            try:
                os.remove(path)
            except OSError as exc:
                logging.warning("Partial upload could not be removed (%s)", type(exc).__name__)
    return {"resume_filename": name}


# The parser is synchronous (PDF extraction and five blocking model calls), so it runs in a worker thread and the
# event loop stays free for /health and every other request. It also writes intermediate files to fixed paths, so
# only ONE analysis may run at a time per process: others wait here (in their own worker thread) until it is done.
# A threading.Lock (not an asyncio one) works across event loops and costs no event-loop time while waiting.
_ANALYSIS_LOCK = threading.Lock()


def _run_analysis(resume_path, job_description, request_metrics):
    """Run the parser once, alone. Executed in a worker thread via asyncio.to_thread, which copies the request's
    context, so retry attribution (a ContextVar) works exactly as before. The pipeline timer starts only after the
    lock is held, so pipeline_ms is parser time without queueing; request_ms still includes the wait."""
    with _ANALYSIS_LOCK:
        with request_metrics.pipeline():
            return parser.parse_resume(resume_path, job_description, metrics=request_metrics)


@app.post('/parse_resume')
async def parse_resume(
    resume_filename: str = Form(...),
    job_description: str = Form(...),
    session_id: int = Form(...),
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    request_metrics = RequestMetrics(session_id=session_id, model=ResumeParserConfig.GROQ_MODEL)
    outcome = "error"
    claimed_path = None
    try:
        # 1. Strict file-name check: exactly what /upload_resume generates, so no separators, traversal or odd extensions.
        if not UPLOAD_NAME_RE.fullmatch(resume_filename):
            raise HTTPException(status_code=400, detail="Invalid resume file name")

        # 2. Session ownership, before any file operation or model call.
        session = db.query(UserSession).filter(
            UserSession.id == session_id,
            UserSession.user_id == user.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # 3. Upload ownership: only the caller's own directory is searched, so another user's file (or a name that
        #    never existed) is simply "not found". Renaming is atomic: of two concurrent requests for the same upload
        #    exactly one gets the file, the other gets 404. The claimed copy is always removed in the finally block.
        resume_path = os.path.join(_user_upload_dir(user.id), resume_filename)
        claim = resume_path + ".processing"
        try:
            os.rename(resume_path, claim)
        except OSError:
            raise HTTPException(status_code=404, detail="Resume file not found")
        claimed_path = claim

        (
            summary, credentials, missing_skills, job_skills, ats_content,
            prioritized_missing_skills,
        ) = await asyncio.to_thread(_run_analysis, claimed_path, job_description, request_metrics)
        # Clean and validate LLM output
        resume_analysis = clean_resume_analysis(
            summary, credentials, missing_skills, job_skills, ats_content,
            prioritized_missing_skills,
        )

        # Save resume summary & results into session data
        session.data = {
            **(session.data or {}),
            "resume_analysis": resume_analysis,
        }
        db.commit()
        db.refresh(session)

        outcome = "ok"
        return resume_analysis

    except HTTPException as exc:
        outcome = f"http_{exc.status_code}"
        raise
    except Exception as e:
        # Class name only: provider and parser error messages can quote the resume or the request.
        logging.error("Resume analysis failed (%s)", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="Resume analysis failed due to an internal error."
        ) from e
    finally:
        request_metrics.emit(outcome)  # never raises; logs ids, numbers and error class names only
        if claimed_path:               # success or failure: the claimed upload never stays behind
            try:
                os.remove(claimed_path)
            except OSError as exc:
                logging.warning("Processed upload could not be removed (%s)", type(exc).__name__)


@app.get('/recommend_courses')
def recommend_courses(skill: str = Query(..., min_length=1, max_length=100), user: User = Depends(get_current_user)):
    try:
        courses = recommender.get_missing_skills_resources(skill)
    except RecommenderProviderError:
        # The provider failed: say so with a non-success status instead of returning an empty list.
        raise HTTPException(
            status_code=502,
            detail="The course provider could not be reached or rejected the request. Please try again later."
        )
    except Exception as e:
        # Log the exception type only (the message of a provider error can contain the request URL and API key).
        logging.error("Course recommendation failed (%s)", type(e).__name__)
        raise HTTPException(status_code=500, detail="Course recommendations could not be loaded.")
    return {'courses': courses}
    
