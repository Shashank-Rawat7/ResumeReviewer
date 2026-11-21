# ResumeReviewer

ResumeReviewer compares a candidate's resume (PDF) with a job description and returns a structured analysis: a resume summary, extracted credentials, the skills the job asks for, the skills missing from the resume (with priority tiers and quoted evidence), ATS-oriented bullet rewrites, and course suggestions for the gaps. It is a portfolio project: a FastAPI backend, a Streamlit frontend, PostgreSQL, and an LLM (Groq, `openai/gpt-oss-20b`) behind a five-stage prompt pipeline, plus a small offline evaluation framework used to measure that pipeline.

> **Status:** project closed out for portfolio use. It is a demonstration, not a production service, and the evaluation covers synthetic data only (see [Evaluation](#evaluation-methodology-and-measured-results) and [Known limitations](#known-limitations)).

## Architecture

```mermaid
flowchart LR
    U[Browser] --> F[Streamlit frontend :8501]
    F -- REST + JWT --> B[FastAPI backend :8000]
    B -->|"stores uploads/user-id/random.pdf, removes it after analysis"| V[(uploads volume)]
    B --> P[(PostgreSQL 15)]
    B -- 5 prompts per resume --> G[Groq API]
    B -- course search --> Y[YouTube Data API]
    B -. one JSON metrics line per request .-> L[stdout log]
```

Request flow: the frontend sends the PDF to `POST /upload_resume`; the backend stores it under a server-generated name in the caller's own folder and returns that name. `POST /parse_resume` then validates the name and the session, reads the file, makes five model calls (summary, credentials, job skills, missing skills, ATS bullets), cleans the output, saves it to the user's session in PostgreSQL and always removes the PDF afterwards. The API response and the saved session are the same JSON; metrics go only to the log.

## Technology stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, SQLAlchemy 2, psycopg2, python-jose (JWT), passlib + bcrypt |
| Frontend | Streamlit, requests |
| LLM | Groq via `langchain-groq` / `groq`, model `openai/gpt-oss-20b`, prompts in `src/prompts.yaml` |
| Resume parsing | PyPDF2 |
| Recommendations | YouTube Data API (`google-api-python-client`) |
| Database | PostgreSQL 15 (users and sessions; analysis stored as JSON) |
| Packaging | Docker, Docker Compose, Python 3.12 slim images |

Exact pinned versions are in `requirements.txt`.

## Features

- Registration and login with JWT (credentials travel in the request body, never in a URL); every session and every upload belongs to one user and other users cannot load, parse, overwrite or delete them.
- Resumable sessions: the analysis is stored in PostgreSQL and can be reloaded later.
- Job-skill extraction, missing-skill detection with priority tiers, ATS bullet rewrites and course recommendations.
- Per-request observability (latency, tokens, retries) without logging resume or job text.
- A frozen offline evaluation framework (`eval/`) and a test suite that needs no network and no API keys.

## Evaluation methodology and measured results

The evaluation lives in `eval/` and is deliberately conservative: hand-authored **synthetic** resumes and job descriptions, labels written by the author, fixed repeats, raw model calls logged, and metrics reported as numerator/denominator. The committed result artifacts are frozen. Reports: [pipeline summary](eval/pipeline/results/pipeline_918e73269cf5/summary.md). ATS bullets, in the order they were produced: (1) [baseline automated checks](eval/ats/results/ats_dev_20261029T052205Z/report.md), (2) [human review of all 102 bullets](eval/ats/results/ats_dev_20261029T052205Z/step5_human_review_report.md), (3) [Step 6 integrity check and repeat analysis](eval/ats/results/ats_dev_20261029T052205Z/step6/step6_report.md).

**Omitted from the public release.** The local job-posting inputs `eval/jobs/strong.txt`, `partial.txt` and `poor.txt` (third-party postings that cannot be redistributed) and the nine historical Stage 2 skill-gap result files under `eval/results/` that quote them are not part of this repository. Only each run's `metadata.json` (hashes and sizes) is kept, and `eval/jobs/SOURCES.md` records where the postings came from. To run `eval/run_skill_eval.py` (it makes live Groq calls) you must supply your own resume PDF (`--resume`) and your own job-posting text files named `strong.txt`, `partial.txt` and `poor.txt` in the directory given by `--jobs` (default `eval/jobs`). The omitted result files cannot be checked from the public repository. The metrics reported in this README come from the pipeline and ATS results listed above.

**Pipeline evaluation** (10 synthetic scenarios, 3 repeats, one model; ranges are over repeats):

| Stage | Metric | Result |
|---|---|---|
| Job skills (E1) | skill recall | 64-65 of 68 (0.94-0.96) |
| Missing skills (E2) | status agreement | 57-60 of 65 (0.88-0.92) |
| Missing skills (E2) | gap-detection recall | 30-34 of 37 (0.81-0.92) |
| Missing skills (E2) | false-flag rate | 0 of 28 in every repeat |
| Courses | offered options match the rule | 10 of 10 in every repeat |
| ATS bullets (E3) | deterministic screens flagged (unsupported number, absent skill, guard phrase) | 0 flagged in every repeat |
| Calls | succeeded | 90 of 90 (one retry) |

E1 precision and F1 are not reported (`n/a`): the labels do not define a complete set of non-required skills.

**ATS bullet evaluation** (9 development cases, 2 repeats, 102 bullets): no API or schema failures in 18 calls; invented numbers 0/102; absent-skill flags 6/102 (all from one case). Provisional human review of all 102 bullets: supported by the resume 70 yes / 32 partially / 0 no; overall 70 improved, 18 unchanged, 14 worse.

**Reading the three ATS reports.** They are successive stages of one frozen run (`ats_dev_20261029T052205Z`), not competing results. The baseline report was produced before any human review was recorded, so it states `Bullets reviewed: 0/102`; that was true when it was generated, and the file is kept as generated. The human review of all 102 bullets is recorded separately in `bullets_reviewed.csv`, `human_review_decisions.csv` and the Step 5 report, and the original `bullets.csv` is not overwritten. Step 6 re-reads those files, checks their integrity and compares the two repeats. The automated-check numbers above come from the baseline report; the human-review numbers come from the Step 5 and Step 6 reports.

**What these numbers do not show**
- They come from ten (pipeline) and nine (ATS) hand-authored synthetic scenarios, one model and draft labels. They are a baseline, not evidence of accuracy on real resumes.
- E3 results are screens: zero flags does not prove a bullet is factually correct.
- The ATS human review is provisional and not independent.
- Ranges across repeats show run-to-run variation; they are not confidence intervals.
- The ATS run used an evaluation-only output cap of 4096 tokens (production uses 8192).
- Evaluation was run before the observability code was added; that code does not change prompts, parameters, inputs or call order.

## Observability

Each `/parse_resume` request writes **one JSON line** to stdout on the `resume_reviewer.metrics` logger. It is not part of the API response and is not stored in the database or `session.data`. Implementation: `src/metrics.py`; tests: `tests/test_metrics.py`.

**Captured per request:** `request_id`, `session_id`, `model`, `outcome` (`ok`, `http_<status>` or `error`), endpoint latency (`request_ms`), model-pipeline latency (`pipeline_ms`), totals (input/output/total tokens, retries, failed and not-run stages) and `normalization_ok` (whether the missing-skills result was normalised successfully).

**Captured per stage** (`summary`, `credentials`, `e1` job-skill extraction, `e2` missing-skill comparison, `e3` ATS bullets): `status` (`ok`, `failed`, `not_run`), `latency_ms`, `input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens` (only if the provider reports it), `finish_reason`, `retries`, and for a failed stage the exception class name and HTTP status code. Resume text, job-description text, prompts, model output, exception messages, API keys and user names/e-mails are never logged.

Example (shortened):

```json
{"model":"openai/gpt-oss-20b","outcome":"ok","pipeline_ms":21840,"request_ms":21902,"request_id":"3f9c...","session_id":12,"normalization_ok":true,"retries_unavailable_reason":null,"stages":{"e1":{"status":"ok","latency_ms":3120,"input_tokens":812,"output_tokens":240,"retries":0,"finish_reason":"stop","error_class":null,"status_code":null},"e2":{"status":"ok","...":"..."}},"totals":{"input_tokens":9120,"output_tokens":3410,"tokens_complete":true,"retries":0,"failed_stages":0,"not_run_stages":0}}
```

**Limitations**
- A value that could not be measured is `null`, never `0`. Token totals are `null` unless every stage that ran reported usage (`tokens_complete`).
- `output_tokens` is the provider's completion count; for reasoning models it may include reasoning tokens. `reasoning_tokens` is reported separately only when the API returns it and is never added or subtracted.
- Retry counts are best effort. The Groq SDK retries internally and only logs it, so a single filter on the `groq._base_client` logger counts those log lines. If they cannot be attributed with certainty (an overlapping or context-less event) or tracking is disrupted or unsupported by the installed SDK version, `retries` is `null` and `retries_unavailable_reason` says why. The logger's visible output is unchanged.
- Latency includes retries and waiting. `pipeline_ms` covers only the time the analysis itself ran; if the request had to queue behind another analysis (see Known limitations), that wait is part of `request_ms` but not of `pipeline_ms`. Metrics are for one process; there is no aggregation or dashboard, and nothing is persisted beyond the log stream.
- The model call has **no request timeout** configured in production. Adding one is a separate operational change that needs its own review; it is intentionally not part of this feature.
- The pre-existing evaluation baseline was measured with the parser before this instrumentation was added; the added code does not change prompts, model parameters, inputs or call order.

## Running with Docker (recommended)

**Prerequisites:** Docker Desktop or Docker Engine with Compose v2, and your own Groq and YouTube Data API keys.

1. Create `.env` from the template and fill in real values (never commit it):
   ```bash
   cp .env.example .env
   ```
   Use plain `KEY=value` lines with no spaces around `=` (Compose reads `.env` for interpolation). `docker compose` fails with a clear message if a required variable is missing.
2. Build and start everything:
   ```bash
   docker compose up --build
   ```
   PostgreSQL starts first; the backend waits until it is healthy and the frontend waits for the backend.
3. Open the app at http://localhost:8501 (API at http://localhost:8000, `/health` for a liveness check). All ports are published on `127.0.0.1` only.
4. Stop: `docker compose down` (data is kept). To also delete the database and uploads: `docker compose down -v`.

**Containers and volumes**

| Service | Image | Notes |
|---|---|---|
| `postgres` | `postgres:15.20-trixie` | volume `postgres_data`, healthcheck `pg_isready` |
| `backend` | built from `backend/Dockerfile` (python:3.12.16-slim-trixie) | non-root user, no `--reload`, healthcheck on `/health`, tables created at startup (retries while the database starts) |
| `frontend` | built from `frontend/Dockerfile` (python:3.12.16-slim-trixie) | non-root user, healthcheck on Streamlit's health endpoint |

Uploaded PDFs are stored by the backend in the named volume `uploads` (mounted at `/app/uploads` in the backend container only); the frontend never touches it and the source tree is not mounted. Inside the Compose network the services use `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432` and `API_BASE_URL=http://backend:8000`.

## Configuration

See `.env.example` for names and safe placeholders.

| Variable | Purpose |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | database credentials (required) |
| `POSTGRES_HOST`, `POSTGRES_PORT` | database location for non-Docker runs (Compose overrides them) |
| `DATABASE_URL` | full SQLAlchemy connection URL for non-Docker runs (optional), for example `postgresql+psycopg2://user:password@localhost:5432/dbname`. When set, it is used exactly as written and the backend ignores the `POSTGRES_*` variables, so any `@`, `:`, `/` or `%` in the password must be percent-encoded by you. When unset, the backend builds the URL from the `POSTGRES_*` variables with the `postgresql+psycopg2` driver and the raw password (no escaping needed). `psycopg2` is the only PostgreSQL driver installed. Docker Compose does not pass `DATABASE_URL` to the backend container, so it has no effect there. |
| `GROQ_API_KEY`, `YOUTUBE_API_KEY` | external APIs (required) |
| `SECRET_KEY` | JWT signing key; use a long random value (required) |
| `API_BASE_URL` | frontend -> backend URL (optional; default `http://backend:8000`) |
| `UPLOAD_DIR` | where the backend stores uploads, one sub-folder per user id (optional; default `./src` locally, `/app/uploads` in Docker) |
| `POSTGRES_HOST_PORT`, `BACKEND_HOST_PORT`, `FRONTEND_HOST_PORT` | host ports (optional; 5432 / 8000 / 8501) |

## Running without Docker

Use Python 3.12 and a PostgreSQL 15 instance (or set `DATABASE_URL`, described under Configuration). Install `requirements.txt`, set `POSTGRES_HOST=localhost`, `API_BASE_URL=http://localhost:8000` and the variables above, then run `uvicorn backend.main:app --port 8000` and `streamlit run frontend/main.py` from the project root.

## Demo workflow

1. Open http://localhost:8501, register and log in.
2. Start a session, upload a resume PDF and paste a job description.
3. Wait for the analysis (five model calls, typically tens of seconds).
4. Review the summary, credentials, job skills, missing skills with priorities, ATS bullets and suggested courses. Reload the session later from the sidebar.

Use a resume you are comfortable sending to a third-party model provider: resume and job text are sent to Groq for processing.

## API overview

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | no | liveness |
| `POST /register` | no | create a user; JSON body `{"username", "email", "password"}` (all non-empty) |
| `POST /login` | no | OAuth2 password form, returns a JWT |
| `POST /start_session` | yes | create a session |
| `GET /my_sessions` | yes | list the caller's sessions |
| `GET /load_session` | yes | load one of the caller's sessions (404 for other users' sessions) |
| `POST /upload_resume` | yes | multipart PDF, at most 10 MB (413 above that); the name must end in `.pdf` and the content must start with `%PDF-` (415 otherwise); stored as `<user id>/<32 hex>.pdf`, returns `{"resume_filename": ...}` (the browser's file name is not used for storage) |
| `POST /parse_resume` | yes | form fields `resume_filename` (exactly what `/upload_resume` returned), `job_description`, `session_id`; analyses the caller's own upload and saves the result in the caller's own session |
| `GET /recommend_courses` | yes | course suggestions for one skill (1-100 characters); makes YouTube API calls |

The former `/db_check` and `/get_all_data` endpoints exposed database contents without authentication and have been removed.

### Security model for uploads and credentials

- **Ownership by construction.** The backend, not the browser, creates every upload name (`uuid4().hex + ".pdf"`) and stores the file in a directory named after the authenticated user's id. `/parse_resume` looks only in the caller's own directory, so another user's upload (or a guessed name) is simply "not found" and cannot be parsed, overwritten or deleted. No extra database table is needed.
- **Upload limits.** `/upload_resume` checks the `.pdf` extension and the `%PDF-` signature before writing anything, reads the body in 64 KiB chunks, and refuses more than 10 MB (10,485,760 bytes) with 413. A partly written file is deleted on every failure. The checks confirm a PDF header, not that the PDF is well-formed or safe.
- **Strict names.** `/parse_resume` accepts only 32 lowercase hex characters plus `.pdf`; separators, traversal, other extensions and malformed names get 400/422.
- **Order of checks.** File name, then session ownership, then the file, then the model. A request for someone else's session never consumes a file or spends a model call.
- **Cleanup and concurrency.** The file is claimed with an atomic rename, so of two simultaneous requests exactly one wins and the other gets 404. The claimed file is removed in a `finally` block after success or failure.
- **Credentials.** `/register` takes a JSON body and `/login` a form body, so passwords never appear in URLs, the frontend's error messages or Uvicorn's access log.
- **Logs.** Resume text and analysis are never printed. Exception logging records the exception class name only (provider messages can quote the request), through `src/log_safety.py`. Only the structured metrics line from the Observability section is emitted per request.

## Tests

Install `requirements.txt` and `requirements-dev.txt` first (the second adds only `httpx`, which FastAPI's test client needs; the tests use `unittest`, not pytest). Test modules that need the application's dependencies are skipped, not failed, when those are missing, so check a run for unexpected skips.

Run one module from the project root, for example:

```bash
python -m unittest tests.test_metrics -v
```

To run every module, give each its own process. In PowerShell:

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { python -m unittest "tests.$($_.BaseName)" }
```

In bash:

```bash
for f in tests/test_*.py; do python -m unittest "tests.$(basename "$f" .py)"; done
```

Separate processes matter: several modules assert that production libraries (for example `backend` or `langchain_groq`) are not imported, and those assertions can fail when another module in the same process has already imported them, so a single `python -m unittest discover` run is not a reliable way to run everything.

The tests need no network and no API keys. Security and pipeline tests run the app in a subprocess against a throw-away SQLite database and a local mock Groq server (`tests/docker/mock_groq.py`); `tests/test_security_endpoints.py` covers upload ownership, name validation, cleanup, concurrency, authentication, credentials and log hygiene. Run the Docker end-to-end check, which builds the stack and exercises it with the mock model on isolated ports and volumes, with:

```bash
python tests/docker/smoke_test.py
```

It makes no live Groq or YouTube call.

**Verification status.** The Docker smoke test was run once on the maintainer's Windows machine during an earlier validation run and passed all 55 checks (55 passed, 0 failed), using the mock model. That run did not use live Groq or YouTube, and it is not evidence of production readiness. Later changes (database URL construction, `SECRET_KEY` validation, upload limits, running the analysis in a worker thread, and the Docker-ignore rules for developer scripts) were checked by the offline tests only; the smoke test has not been re-run since them.

## Project structure

```
backend/    FastAPI app, auth, database models (backend/db/)
frontend/   Streamlit UI
src/        prompt pipeline (parser.py, prompts.yaml), recommender, metrics
eval/       frozen offline evaluation framework, datasets and result artifacts
tests/      unit, security, deployment-config and mocked-pipeline tests; tests/docker/ smoke kit
docker-compose.yaml, requirements.txt, .env.example
```

### Legacy developer scripts (do not import or run casually)

Three scripts under `backend/` are kept from early development. Nothing in the application imports them, and all three are excluded from the Docker images through `.dockerignore`. Use them only deliberately, and only against a throw-away local database.

| Script | What it does | Risk |
|---|---|---|
| `backend/testing.py` | Prints every user's username and email and every session's name, user ID and stored analysis data from the configured database. It has no `__main__` guard, so **importing it runs it**. | Exposes personal data in the terminal or logs. Never run it against real data. |
| `backend/db/reset_db.py` | When run as a script, **drops every table** (users, sessions, stored analyses) in the database selected by `DATABASE_URL` / `POSTGRES_*` (or a `.env` file found by python-dotenv), then recreates empty tables. There is no confirmation prompt. Importing it does not drop anything. | Irreversible data loss. Check the target first. |
| `backend/db/connection.py` | **Importing it** opens and closes a PostgreSQL connection to the hard-coded `localhost:5432`, database `resume_reviewer`, user `postgres` (password from `POSTGRES_PASSWORD`) and prints the result. | Ignores `DATABASE_URL`, so a "connected" message does not prove the app's configuration works. It changes nothing. Do not import it from application code. |

## Known limitations

- Evaluation is synthetic-only, single-model and author-labelled; it says nothing certain about real-world accuracy.
- Live Groq and YouTube behaviour was not re-verified in the Docker setup; the Docker and pipeline checks use a mock model.
- Uploads that are never analysed (for example the user closes the tab between upload and analysis) stay in the user's folder; there is no expiry job, and a crash during analysis can leave a `.processing` file. Fixed names are not reused, so this is retention, not leakage.
- The parser still writes intermediate text files (resume text, summaries) to fixed paths inside the backend container; the next request overwrites them. They are not served by any endpoint.
- Analyses run one at a time per backend process (the parser writes intermediate files to fixed paths, so it is serialised with a lock). The analysis runs in a worker thread, so `/health`, login and the other endpoints stay responsive while it runs, but further `/parse_resume` requests queue behind it and simply wait, with no timeout and no extra status code. Each waiting request keeps one database connection from SQLAlchemy's default pool (5 plus 10 overflow), so about 15 simultaneous queued analyses could exhaust the pool and make other database-backed endpoints fail after 30 seconds. This is a demonstration service, not sized for that load. Running several Uvicorn workers would serialise each process separately, not the whole service.
- Uploads are limited to 10 MB and must look like a PDF (extension and `%PDF-` header), but the framework (Starlette) still spools a whole multipart request to a temporary file before the endpoint runs, so a very large request is not stopped early; put a request-size limit in a reverse proxy for any public deployment. A file with a PDF header that cannot be parsed still gives a generic 500 at analysis time and is removed.
- Resume and job text are sent to a third-party LLM provider. No retention or PII-redaction controls are implemented.
- Passwords are hashed (bcrypt) but there is no rate limiting, password policy, email verification, password reset or token refresh, and registration reveals whether a username exists. The backend refuses to start unless `SECRET_KEY` has at least 32 characters and at least 8 distinct characters, and is not a known placeholder (including the one in `.env.example`) or a short pattern repeated to fill the key. These are sanity checks that catch obviously weak or copied values; they do not prove the key is random, so generate it with a proper random source (for example `python -c "import secrets; print(secrets.token_hex(32))"`).
- Tables are created with `create_all`; there is no migration tool, so schema changes need manual handling.
- The model call has no request timeout configured (see Observability).
- Direct dependencies in `requirements.txt` are pinned to exact versions, but their own (transitive) dependencies, for example `httpx`, `anyio` and `urllib3`, are not locked, and no lock or constraints file is provided. A fresh build can therefore resolve newer transitive versions than the ones tested.
- The frontend image installs the backend's dependencies too because there is one requirements file.
- Single-process metrics only; no dashboard or persistence of metrics.

## Acknowledgements and attribution

This repository builds on the upstream project srivabhi22/ResumeReviewer; the original authorship is preserved and credited there. The evaluation framework, observability, security fixes and Docker setup were added afterwards. No licence has been selected yet; until one is added, all rights are reserved by the respective authors.
