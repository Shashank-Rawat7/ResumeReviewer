"""Offline-first evaluation runner for the ATS bullet rewrite (Stage 5, checkpoint C).

    python eval/ats/run_ats_eval.py plan [--split dev|holdout] [--repeats N] [--pace-seconds S]
                                         [--eval-max-tokens [N]]                    (offline; the default workflow)
    python eval/ats/run_ats_eval.py run --plan RUN_ID --live --confirm-calls N         (LIVE model calls)
    python eval/ats/run_ats_eval.py summarize RUN_ID
    python eval/ats/run_ats_eval.py label-sheet RUN_ID

`plan` never touches the network, an LLM provider, an API key, `.env` or any model. `run` is the ONLY command that can
call a model, and only with --live, a --confirm-calls count equal to the plan, and a plan that still matches the
current prompt, checker, vocabulary and dataset. Holdout runs need a separate phrase and a completed, frozen dev run.

The model call mirrors production's ATS step ((ats_bullet_prompt | llm).invoke({job_description, resume_content}))
and NEVER calls ResumeParser.parse_resume, which would overwrite protected src/*.txt files. Artifacts are written
only under eval/ats/results/<run-id>/. All reports are SCREENING signals on 12 synthetic cases; they do not show
that the feature generalises to real resumes.
"""
import argparse
import ast
import csv
import io
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import textwrap
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

ATS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(ATS_DIR, "..", ".."))
RESULTS_DIR = os.path.join(ATS_DIR, "results")


def _load_sibling(module_name, filename):
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(ATS_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


dataset = _load_sibling("ats_dataset", "dataset.py")
checks = dataset.checks
RUNNER_PATH = os.path.abspath(__file__)
DATASET_MODULE_PATH = os.path.abspath(dataset.__file__)

# ---------------------------------------------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------------------------------------------
PROMPT_KEY = "resume_ats_optimizer"
SOURCE_FILES_ALLOWED = ("prompts.yaml", "parser.py")  # the only files this runner reads from src/
PROTECTED_FILES = ("resume_text.txt", "resume_summary.txt", "resume_skills.txt", "job_skills.txt",
                   "missing_skills.txt", "ats_optimized_bullets.txt")
RUN_SPLITS = ("dev", "holdout")  # "all" is deliberately not exposed (it would mix holdouts into a dev run)
HOLDOUT_PHRASE = "I-APPROVE-HOLDOUT-EVALUATION"
DEFAULT_REPEATS, MAX_REPEATS = 2, 5
DEFAULT_PACE_SECONDS = 40
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF = 60
DEFAULT_ABORT_AFTER = 3
DEFAULT_ABORT_AFTER_TRUNCATED = 2  # consecutive responses cut off by the output cap
DEFAULT_EVAL_MAX_TOKENS = 4096  # used only when `plan --eval-max-tokens` is given without a value
MIN_EVAL_MAX_TOKENS = 512
OUTPUT_TOKENS_ASSUMED = 1500  # ASSUMPTION per call (answer plus low-effort reasoning), not a measurement
CHARS_PER_TOKEN = 4  # ESTIMATE
CALL_STATUSES = ("ok", "empty_response", "api_error", "parse_failed", "schema_invalid", "evaluation_error",
                 "truncated")
HUMAN_COLUMNS = ("supported_by_source_evidence", "unsupported_or_invented_claims", "relevance_to_job_description",
                 "clarity_and_action_orientation", "overall_rewrite_improvement", "reviewer_notes")
CSV_COLUMNS = ("case_id", "split", "repeat", "call_id", "bullet_index", "output_section", "matched_source_heading",
               "source_bullets", "rewritten_bullet", "checker_findings", "d2_invented_numbers",
               "d3_absent_skill_leaks", "d4_unsupported_technologies", "d6_unchanged_from_source",
               "d6_near_duplicate_of") + HUMAN_COLUMNS
PLAN_SCHEMA_VERSION = 3
PROVIDER_PACKAGES = ("langchain", "langchain-core", "langchain-groq", "groq")
UNKNOWN = "unknown"
NOT_INSTALLED = "not installed"
NOT_SET_IN_PARSER = "not set in parser.py (library default applies)"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


class Refused(Exception):
    """A safety rule stopped the command. Nothing was written (or, for run, nothing further was called)."""


# ---------------------------------------------------------------------------------------------------------------
# Small helpers: JSON, sanitising, rates
# ---------------------------------------------------------------------------------------------------------------

def dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def jsonl_line(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"


def utc_now():
    return datetime.now(timezone.utc)


def stamp(moment):
    return moment.strftime("%Y%m%dT%H%M%SZ")


def rate(k, n):
    """'3/18 (16.7%)'; the denominator is always shown."""
    return f"{k}/{n} ({100.0 * k / n:.1f}%)" if n else f"{k}/0 (n/a)"


_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(api[_-]?key|authorization|token|secret|password)(\"?\s*[:=]\s*\"?)[^\s\",}]+"),
    re.compile(r"[A-Za-z0-9_\-]{32,}"),
)


def sanitize_message(text, limit=200):
    """Short, single-line error text with anything key-like removed. Never includes model output."""
    text = " ".join(str(text).split())
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text[:limit]


def classify_exception(exc):
    """kind: rate_limit | transport | auth | server_error | api_error; plus the HTTP status when the error has one.
    Only rate_limit and transport are retried."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    status = status if isinstance(status, int) else None
    name = type(exc).__name__
    if status == 429 or "RateLimit" in name:
        kind = "rate_limit"
    elif status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        kind = "auth"
    elif isinstance(exc, (TimeoutError, ConnectionError)) or any(w in name for w in ("Timeout", "Connection", "Network")):
        kind = "transport"
    elif status is not None and status >= 500:
        kind = "server_error"
    else:
        kind = "api_error"
    return kind, status


def csv_safe(value):
    """Neutralise spreadsheet formulas: a cell starting with = + - @ (or tab/CR) gets a leading apostrophe."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


# ---------------------------------------------------------------------------------------------------------------
# Guarded file access
# ---------------------------------------------------------------------------------------------------------------

def read_source_file(name, root=None):
    """Reads ONLY src/prompts.yaml or src/parser.py (text). Any other name, in particular the six protected
    src/*.txt files, is refused before any file is opened."""
    if name not in SOURCE_FILES_ALLOWED:
        raise Refused(f"refusing to read src/{name}: only {', '.join(SOURCE_FILES_ALLOWED)} may be read")
    with open(os.path.join(root or ROOT, "src", name), "rb") as handle:
        return handle.read().replace(b"\r\n", b"\n").decode("utf-8")


class RunDir:
    """The only writer. Every path is resolved and must stay inside <results_root>/<run_id>/."""

    def __init__(self, results_root, run_id):
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or ".." in run_id:
            raise Refused(f"invalid run id {run_id!r}")
        self.results_root = os.path.realpath(results_root)
        self.path = os.path.join(self.results_root, run_id)
        self.run_id = run_id

    def _resolve(self, relative):
        if os.path.isabs(relative) or ".." in relative.replace("\\", "/").split("/"):
            raise Refused(f"refusing to write outside the run directory: {relative!r}")
        target = os.path.realpath(os.path.join(self.path, relative))
        if os.path.commonpath([os.path.realpath(self.path), target]) != os.path.realpath(self.path):
            raise Refused(f"refusing to write outside the run directory: {relative!r}")
        return target

    def exists(self, relative=""):
        return os.path.exists(self._resolve(relative) if relative else self.path)

    def make(self):
        os.makedirs(self.results_root, exist_ok=True)
        try:
            os.mkdir(self.path)
        except FileExistsError:
            raise Refused(f"run directory {self.run_id!r} already exists; choose another run id")

    def write_bytes(self, relative, data):
        """Atomic: write a temporary file in the same folder, flush to disk, then replace."""
        target = self._resolve(relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = os.path.join(os.path.dirname(target), ".tmp-" + os.path.basename(target))
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def write_text(self, relative, text):
        self.write_bytes(relative, text.encode("utf-8", errors="surrogatepass"))

    def write_json(self, relative, obj):
        self.write_text(relative, dumps(obj))

    def append_jsonl(self, relative, obj):
        target = self._resolve(relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "ab") as handle:
            handle.write(jsonl_line(obj).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    def read_text(self, relative):
        with open(self._resolve(relative), "rb") as handle:
            return handle.read().decode("utf-8")

    def read_json(self, relative):
        return json.loads(self.read_text(relative))

    def read_jsonl(self, relative):
        if not self.exists(relative):
            return []
        return [json.loads(line) for line in self.read_text(relative).splitlines() if line.strip()]


def open_run_dir(results_root, run_id):
    run = RunDir(results_root, run_id)
    if not os.path.isdir(run.path):
        raise Refused(f"no run directory {run_id!r} under the results folder")
    return run


# ---------------------------------------------------------------------------------------------------------------
# The prompt block and plan
# ---------------------------------------------------------------------------------------------------------------

def extract_prompt_block(yaml_text, key=PROMPT_KEY):
    """The text of one top-level YAML block scalar ('key: |'), dedented. Plain text parsing: no YAML library, so the
    fingerprint covers only this prompt and `plan` imports nothing beyond the standard library."""
    lines = yaml_text.split("\n")
    for start, line in enumerate(lines):
        if line.startswith(key + ":"):
            break
    else:
        raise Refused(f"prompt {key!r} not found in src/prompts.yaml")
    block = []
    for line in lines[start + 1:]:
        if line.strip() and not line[:1].isspace():
            break
        block.append(line)
    return textwrap.dedent("\n".join(block)).strip("\n") + "\n"


def expected_model_name(parser_text):
    match = re.search(r'GROQ_MODEL\s*=\s*"([^"]+)"', parser_text)
    if not match:
        raise Refused("could not find GROQ_MODEL in src/parser.py")
    return match.group(1)


def parse_llm_settings(parser_text):
    """Settings of the production ChatGroq(...) call, read from the TEXT of src/parser.py with ast (parsed, never
    executed, nothing imported). A value that is not a plain literal, or is absent, is recorded as 'unknown' (or as
    'not set in parser.py' for the retry and timeout settings, where absence means the library default applies)."""
    settings = {"model": expected_model_name(parser_text), "temperature": UNKNOWN, "max_tokens": UNKNOWN,
                "reasoning_effort": UNKNOWN, "max_retries": NOT_SET_IN_PARSER, "request_timeout": NOT_SET_IN_PARSER,
                "source": "static parse of src/parser.py (ChatGroq call); not read from a running client"}
    try:
        tree = ast.parse(parser_text)
    except SyntaxError:
        return settings
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ChatGroq":
            for keyword in node.keywords:
                name = {"timeout": "request_timeout"}.get(keyword.arg, keyword.arg)
                if name in ("temperature", "max_tokens", "reasoning_effort", "max_retries", "request_timeout"):
                    try:
                        settings[name] = ast.literal_eval(keyword.value)
                    except (ValueError, SyntaxError):
                        settings[name] = UNKNOWN
            break
    return settings


def package_versions(version_lookup=None):
    """Installed versions from package METADATA only (importlib.metadata reads dist-info files; the packages are not
    imported). 'not installed' when absent, 'unknown' if the lookup itself fails."""
    lookup = version_lookup or importlib_metadata.version
    versions = {}
    for name in PROVIDER_PACKAGES:
        try:
            versions[name] = lookup(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = NOT_INSTALLED
        except Exception:
            versions[name] = UNKNOWN
    return versions


_LANGCHAIN_GROQ_RETRY_RE = re.compile(r"^[ \t]*max_retries[ \t]*:[^=\n]*=[ \t]*(\d+)[ \t]*(?:#.*)?$", re.MULTILINE)
_GROQ_SDK_RETRY_RE = re.compile(r"^DEFAULT_MAX_RETRIES[ \t]*=[ \t]*(\d+)", re.MULTILINE)


def _scan_default(find_spec, package, filename, pattern):
    """The integer default found by a text scan of one installed source file, 'not installed', or 'unknown'.
    find_spec on a top-level name does not import or run the package."""
    try:
        spec = find_spec(package)
    except Exception:
        return UNKNOWN
    if spec is None:
        return NOT_INSTALLED
    try:
        for location in list(spec.submodule_search_locations or []):
            path = os.path.join(location, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    found = pattern.findall(handle.read())
                return int(found[0]) if len(found) >= 1 else UNKNOWN
    except Exception:
        return UNKNOWN
    return UNKNOWN


def discover_provider_retry(find_spec=None):
    """Retry defaults of the provider stack WITHOUT importing it and without any network request. NOT a runtime
    measurement: it is a text scan of the installed source, so 'unknown' is a normal result."""
    finder = find_spec or importlib.util.find_spec
    return {
        "langchain_groq_default_max_retries": _scan_default(finder, "langchain_groq", "chat_models.py",
                                                            _LANGCHAIN_GROQ_RETRY_RE),
        "groq_sdk_default_max_retries": _scan_default(finder, "groq", "_constants.py", _GROQ_SDK_RETRY_RE),
        "method": "static text scan of the installed package source; nothing imported, no network; "
                  "not verified at runtime",
    }


def discover_environment(find_spec=None, version_lookup=None):
    return {"packages": package_versions(version_lookup), "provider_retry": discover_provider_retry(find_spec),
            "python": sys.version.split()[0]}


def _estimate_tokens(chars):
    return int(math.ceil(chars / float(CHARS_PER_TOKEN)))


def validate_eval_max_tokens(value, model_settings):
    """The evaluation-only output cap, or None when no override was requested. It may only LOWER the production
    cap, and it may not be so small that it would truncate ordinary answers."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refused("--eval-max-tokens must be a whole number")
    if value < MIN_EVAL_MAX_TOKENS:
        raise Refused(f"--eval-max-tokens must be at least {MIN_EVAL_MAX_TOKENS}; a smaller cap would truncate "
                      "ordinary answers and measure the cap, not the model")
    production = model_settings.get("max_tokens")
    if isinstance(production, int) and not isinstance(production, bool) and value > production:
        raise Refused(f"--eval-max-tokens {value} is above the production max_tokens {production}; "
                      "the override may only lower the cap")
    return value


def effective_settings_for(model_settings, override):
    """The max_tokens the evaluation client must use: the override when there is one, else the production value."""
    if override is not None:
        return {"max_tokens": override, "max_tokens_source": "evaluation override (--eval-max-tokens)"}
    return {"max_tokens": model_settings.get("max_tokens", UNKNOWN),
            "max_tokens_source": "production setting in src/parser.py (no override)"}


def build_plan(split, repeats, results_root=None, cases_dir=None, vocab_path=None, source_root=None,
               frozen_from=None, pace_seconds=DEFAULT_PACE_SECONDS, environment=None, eval_max_tokens=None):
    """The plan dict (JSON-serialisable) for the current state of the repository. Reads only the case files, the
    vocabulary, checks.py, src/prompts.yaml and src/parser.py. Raises Refused or DatasetError."""
    if split not in RUN_SPLITS:
        raise Refused(f"split must be one of {RUN_SPLITS}; 'all' is not offered because it would mix holdouts "
                      "into a development run")
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise Refused(f"repeats must be a whole number from 1 to {MAX_REPEATS}")
    cases = dataset.load_cases(split, cases_dir=cases_dir)
    prompt_block = extract_prompt_block(read_source_file("prompts.yaml", source_root))
    parser_text = read_source_file("parser.py", source_root)
    settings = parse_llm_settings(parser_text)
    model = settings["model"]
    override = validate_eval_max_tokens(eval_max_tokens, settings)
    effective = effective_settings_for(settings, override)
    environment = environment if environment is not None else discover_environment()
    schedule = [{"case_id": c["id"], "repeat": r} for c in cases for r in range(1, repeats + 1)]
    chars = [len(prompt_block) + len(c["job_description"]) + len(c["resume_text"]) for c in cases]
    input_estimate = sum(_estimate_tokens(n) for n in chars) * repeats
    output_estimate = OUTPUT_TOKENS_ASSUMED * len(schedule)
    largest_input = max([_estimate_tokens(n) for n in chars] or [0])
    cap = effective["max_tokens"]
    plan = {
        "kind": "plan",
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "split": split,
        "repeats": repeats,
        "case_ids": [c["id"] for c in cases],
        "schedule": schedule,
        "planned_calls": len(schedule),
        "model_expected": model,
        "fingerprints": {
            "dataset": dataset.dataset_fingerprint(split, cases_dir)["sha256"],
            "vocabulary": dataset.vocabulary_fingerprint(vocab_path)["sha256"],
            "checker": dataset.checker_fingerprint()["sha256"],
            "prompt": dataset.sha256_bytes(prompt_block.encode("utf-8")),
        },
        "code_fingerprints": {
            "runner": dataset.sha256_file(RUNNER_PATH),
            "dataset_module": dataset.sha256_file(DATASET_MODULE_PATH),
            "parser_py": dataset.sha256_bytes(parser_text.encode("utf-8")),
        },
        "model_settings": settings,
        "evaluation_overrides": {} if override is None else {"max_tokens": override},
        "effective_settings": effective,
        "packages": environment["packages"],
        "provider_retry": environment["provider_retry"],
        "python_version": environment.get("python", UNKNOWN),
        "call_count_meaning": "planned_calls counts application-level invocations of the ATS chain. It does NOT "
                              "guarantee that many HTTP requests: if the provider library retries internally "
                              "(see provider_retry), up to planned_calls x (1 + max_retries) requests may be sent "
                              "and the retries are invisible to this runner.",
        "prompt_key": PROMPT_KEY,
        "prompt_example_numbers": sorted(checks.example_numbers_from_prompt(prompt_block)),
        "pacing": {"pace_seconds": pace_seconds,
                   "minimum_wall_seconds": max(0, len(schedule) - 1) * pace_seconds},
        "token_estimates": {
            "ARE_ESTIMATES_NOT_MEASURED_USAGE": True,
            "method": f"input = characters of filled prompt / {CHARS_PER_TOKEN}; output = "
                      f"{OUTPUT_TOKENS_ASSUMED} tokens per call assumed",
            "input_tokens_estimate": input_estimate,
            "output_tokens_assumed": output_estimate,
            "total_tokens_estimate": input_estimate + output_estimate,
            "per_call_average_estimate": int((input_estimate + output_estimate) / max(1, len(schedule))),
            "largest_input_tokens_estimate": largest_input,
            "largest_request_tokens_estimate": (largest_input + cap if isinstance(cap, int) else UNKNOWN),
            "largest_request_meaning": "ESTIMATE of the biggest single request as a provider that reserves the "
                                       "requested output cap would see it: input estimate + effective max_tokens",
        },
        "frozen_from": frozen_from,
    }
    return plan


FROZEN_CONFIGURATION = ("model_settings", "packages", "provider_retry", "evaluation_overrides",
                        "effective_settings")


def frozen_reference(results_root, dev_run_id, current):
    """Holdout planning requires a COMPLETED dev run made with the same checker, vocabulary, prompt, parser.py,
    model settings, package versions and provider retry configuration (`current` is the freshly built plan)."""
    dev = open_run_dir(results_root, dev_run_id)
    try:
        meta = dev.read_json("metadata.json")
        dev_plan = dev.read_json("plan.json")
    except (OSError, ValueError):
        raise Refused(f"{dev_run_id!r} has no readable metadata.json and plan.json")
    if dev_plan.get("split") != "dev" or meta.get("status") != "complete":
        raise Refused(f"{dev_run_id!r} is not a completed development run (status {meta.get('status')!r})")
    changed = [k for k in ("checker", "vocabulary", "prompt")
               if dev_plan["fingerprints"].get(k) != current["fingerprints"][k]]
    changed += [k for k in FROZEN_CONFIGURATION if dev_plan.get(k) != current[k]]
    if dev_plan.get("code_fingerprints", {}).get("parser_py") != current["code_fingerprints"]["parser_py"]:
        changed.append("parser_py")
    if changed:
        raise Refused("the development run used a different " + ", ".join(changed)
                      + "; the configuration is not frozen, so the holdout split would not be a clean test")
    return {"run_id": dev_run_id,
            "fingerprints": {k: dev_plan["fingerprints"][k] for k in ("checker", "vocabulary", "prompt")},
            "configuration": {k: dev_plan[k] for k in FROZEN_CONFIGURATION}}


def new_run_id(split, now):
    return f"ats_{split}_{stamp(now)}"


def cmd_plan(split, repeats, results_root, now=None, run_id=None, frozen_run_id=None, cases_dir=None,
             vocab_path=None, source_root=None, pace_seconds=DEFAULT_PACE_SECONDS, out=print, environment=None,
             eval_max_tokens=None):
    results_root = results_root or RESULTS_DIR
    now = now or utc_now()
    plan = build_plan(split, repeats, results_root, cases_dir, vocab_path, source_root, None, pace_seconds,
                      environment, eval_max_tokens)
    if split == "holdout":
        if not frozen_run_id:
            raise Refused("a holdout plan needs --frozen-from <completed dev run id>")
        plan["frozen_from"] = frozen_reference(results_root, frozen_run_id, plan)
    elif frozen_run_id:
        raise Refused("--frozen-from applies to holdout plans only")
    run = RunDir(results_root, run_id or new_run_id(split, now))
    run.make()
    plan["run_id"] = run.run_id
    plan["created_utc"] = now.isoformat()
    run.write_json("plan.json", plan)
    metadata = dict(plan)
    metadata.update(kind="run_metadata", status="planned", live_calls_made=0)
    run.write_json("metadata.json", metadata)
    describe_plan(plan, out)
    return plan


def describe_plan(plan, out=print):
    est = plan["token_estimates"]
    out(f"PLAN {plan['run_id']}  (OFFLINE: no model was called, no key was read)")
    out(f"  split: {plan['split']}   cases: {len(plan['case_ids'])}   repeats: {plan['repeats']}")
    out(f"  planned model calls: {plan['planned_calls']}")
    for case_id in plan["case_ids"]:
        out(f"    - {case_id}")
    out(f"  pacing: >= {plan['pacing']['pace_seconds']} s between calls, minimum wall time "
        f"{plan['pacing']['minimum_wall_seconds']} s")
    out(f"  ESTIMATED tokens (not measured): input ~{est['input_tokens_estimate']}, output assumed "
        f"~{est['output_tokens_assumed']}, total ~{est['total_tokens_estimate']} "
        f"(~{est['per_call_average_estimate']} per call)")
    out(f"  model expected: {plan['model_expected']}")
    ms = plan["model_settings"]
    out(f"  model settings (static, from src/parser.py): temperature={ms['temperature']}, "
        f"max_tokens={ms['max_tokens']}, reasoning_effort={ms['reasoning_effort']}, max_retries={ms['max_retries']}")
    if plan["evaluation_overrides"]:
        out(f"  EVALUATION-ONLY OVERRIDE: max_tokens={plan['evaluation_overrides']['max_tokens']} "
            f"(production stays {ms['max_tokens']}; src/parser.py is not changed)")
    else:
        out("  evaluation override: none (the live client keeps the production max_tokens)")
    out(f"  effective max_tokens for the evaluation client: {plan['effective_settings']['max_tokens']} "
        f"({plan['effective_settings']['max_tokens_source']})")
    out(f"  ESTIMATE of the largest single request (input estimate + effective max_tokens): "
        f"~{est['largest_request_tokens_estimate']} tokens")
    if plan["pacing"]["pace_seconds"] < 60:
        out("  NOTE: pacing is under 60 s, so the previous call's tokens may still count in a provider's rolling "
            "one-minute window")
    retry = plan["provider_retry"]
    out(f"  provider retry defaults (static scan, unverified at runtime): langchain-groq="
        f"{retry['langchain_groq_default_max_retries']}, groq SDK={retry['groq_sdk_default_max_retries']}")
    out("  packages: " + ", ".join(f"{k}={v}" for k, v in sorted(plan["packages"].items())))
    out("  NOTE: planned calls are application-level invocations; hidden provider retries are not counted here "
        "and could send more HTTP requests (see call_count_meaning in plan.json)")
    for key, value in plan["code_fingerprints"].items():
        out(f"  code fingerprint {key}: {value[:16]}...")
    for key, value in plan["fingerprints"].items():
        out(f"  fingerprint {key}: {value[:16]}...")
    if plan["split"] == "holdout":
        out(f"  frozen from dev run: {plan['frozen_from']['run_id']}")
    out(f"  artifacts: eval/ats/results/{plan['run_id']}/ (plan.json, metadata.json)")
    needed = f"--live --confirm-calls {plan['planned_calls']}"
    if plan["split"] == "holdout":
        needed += f" --confirm-holdout {HOLDOUT_PHRASE}"
    out(f"  to execute (LIVE, needs your explicit approval): run --plan {plan['run_id']} {needed}")


# ---------------------------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------------------------

class ModelResponse:
    """What a client returns: the model text (`content`, verbatim), optional token usage and the provider's finish
    reason when it reports one (None means NOT reported; it is never guessed)."""

    def __init__(self, content, usage=None, finish_reason=None):
        self.content = content
        self.usage = usage
        self.finish_reason = finish_reason


def _usage_from_message(message):
    """Integers only, from LangChain's usage_metadata or response_metadata['token_usage']; else None."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        pairs = (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"), ("total_tokens", "total_tokens"))
    else:
        metadata = getattr(message, "response_metadata", None) or {}
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
        pairs = (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens"))
    if not isinstance(usage, dict):
        return None
    found = {out: usage[src] for out, src in pairs if isinstance(usage.get(src), int) and not isinstance(usage.get(src), bool)}
    return found or None


def _finish_reason_from_message(message):
    """The provider's finish reason (LangChain puts it in response_metadata), or None when it is not reported."""
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("finish_reason")
        if isinstance(value, str) and value:
            return value
    return None


def build_live_client(max_tokens_override=None):
    """LIVE ONLY. Imports the production parser lazily, builds the production ATS chain exactly as
    ResumeParser does ((ats_bullet_prompt | llm)) and returns (call, info). Importing src.parser loads the
    environment through python-dotenv (that is how the application and eval/run_skill_eval.py get the key); this
    runner never reads, prints or stores it. parse_resume is NOT used."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src.parser import ResumeParser  # noqa: E402  (lazy: never imported by plan/summarize/label-sheet)
    if not os.getenv("GROQ_API_KEY"):
        raise Refused("GROQ_API_KEY is not available to this process (it is read by the application from the "
                      "environment or its .env; this runner does not touch it)")
    parser = ResumeParser()
    llm = parser.llm
    production_max_tokens = getattr(llm, "max_tokens", None)
    # The evaluation-only cap is applied PER REQUEST with bind(): the wrapper merges bound kwargs over its own
    # defaults when it builds the request, so the production llm object and its max_tokens stay untouched.
    bound = llm if max_tokens_override is None else llm.bind(max_tokens=max_tokens_override)
    chain = parser.ats_bullet_prompt | bound

    def call(job_description, resume_content):
        message = chain.invoke({"job_description": job_description, "resume_content": resume_content})
        return ModelResponse(message.content, _usage_from_message(message), _finish_reason_from_message(message))

    info = {"model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
            "temperature": getattr(llm, "temperature", None),
            "max_tokens": production_max_tokens,  # the PRODUCTION client's value, to prove it is not mutated
            "max_tokens_override": max_tokens_override,
            "effective_max_tokens": production_max_tokens if max_tokens_override is None else max_tokens_override,
            "reasoning_effort": getattr(llm, "reasoning_effort", None),
            "max_retries": getattr(llm, "max_retries", None),  # the provider's INTERNAL retries (not ours)
            "source": "attributes of the constructed client object; no request was made"}
    return call, info


def verify_live_settings(plan, info):
    """Compare the constructed client with the plan BEFORE any call. Returns (mismatches, unverified): a value is
    'unverified' when either side is unknown/absent (it is then recorded in the run metadata, not guessed)."""
    planned, mismatches, unverified = plan["model_settings"], [], []
    for key in ("model", "temperature", "max_tokens", "reasoning_effort"):
        expected, live = planned.get(key, UNKNOWN), info.get(key)
        if expected == UNKNOWN or live in (None, UNKNOWN):
            unverified.append(key)
        elif expected != live:
            mismatches.append(f"{key} (plan {expected!r}, live client {live!r})")
    override = (plan.get("evaluation_overrides") or {}).get("max_tokens")
    reported = info.get("effective_max_tokens")
    if override is not None:
        if reported is None:
            mismatches.append(f"effective max_tokens (plan {override!r}, but the client did not report it)")
        elif reported != override:
            mismatches.append(f"effective max_tokens (plan {override!r}, live client {reported!r})")
    else:
        expected_cap = (plan.get("effective_settings") or {}).get("max_tokens", UNKNOWN)
        if reported is not None and expected_cap != UNKNOWN and reported != expected_cap:
            mismatches.append(f"effective max_tokens (plan {expected_cap!r}, live client {reported!r})")
    set_in_parser = planned.get("max_retries")
    expected_retries = (set_in_parser if isinstance(set_in_parser, int) and not isinstance(set_in_parser, bool)
                        else plan["provider_retry"].get("langchain_groq_default_max_retries"))
    live_retries = info.get("max_retries")
    if isinstance(expected_retries, int) and isinstance(live_retries, int):
        if expected_retries != live_retries:
            mismatches.append(f"provider max_retries (plan {expected_retries}, live client {live_retries})")
    else:
        unverified.append("max_retries")
    return mismatches, unverified


def _raw_name(case_id, repeat):
    return f"raw/{case_id}__r{repeat:02d}.txt"


def _call_one(call, case, repeat, split, run, vocab, example_numbers, config, sleep, clock, now):
    """One scheduled call: attempts (bounded retries for 429/transport only), raw saved BEFORE parsing, one record."""
    case_id = case["id"]
    record = {"call_id": f"{case_id}__r{repeat:02d}", "case_id": case_id, "repeat": repeat, "split": split,
              "started_utc": now().isoformat(), "attempts": [], "status": None, "response_received": False,
              "raw_path": None, "usage": None, "latency_s": None, "error_kind": None, "http_status": None,
              "error_class": None, "error_message": None, "schema_status": None, "production_would_500": None,
              "problems": [], "evaluation_error": None, "bullets_total": 0,
              "finish_reason": None, "finish_reason_available": None, "truncated": False}
    response = None
    for attempt in range(1, config["max_retries"] + 2):
        began = clock()
        try:
            response = call(case["job_description"], case["resume_text"])
        except Exception as exc:  # every provider error is recorded, never swallowed
            kind, status = classify_exception(exc)
            took = round(clock() - began, 3)
            record["attempts"].append({"attempt": attempt, "outcome": kind, "http_status": status,
                                       "error_class": type(exc).__name__,
                                       "error_message": sanitize_message(exc), "latency_s": took})
            record.update(error_kind=kind, http_status=status, error_class=type(exc).__name__,
                          error_message=sanitize_message(exc))
            if kind in ("rate_limit", "transport") and attempt <= config["max_retries"]:
                sleep(config["retry_backoff"])
                continue
            break
        took = round(clock() - began, 3)
        record["attempts"].append({"attempt": attempt, "outcome": "response", "http_status": None,
                                   "error_class": None, "error_message": None, "latency_s": took})
        record.update(latency_s=took, error_kind=None, http_status=None, error_class=None, error_message=None)
        break
    if response is None:
        record["status"] = "api_error"
        record["finished_utc"] = now().isoformat()
        return record, None
    record["response_received"] = True
    record["usage"] = response.usage if isinstance(response.usage, dict) else None
    content = response.content
    raw_text = content if isinstance(content, str) else json.dumps(content, default=str, sort_keys=True)
    raw_path = _raw_name(case_id, repeat)
    run.write_text(raw_path, raw_text)  # verbatim, before any parsing or fence removal
    record["raw_path"] = raw_path
    finish = getattr(response, "finish_reason", None)
    finish = finish if isinstance(finish, str) and finish else None  # never invented: None = not reported
    record.update(finish_reason=finish, finish_reason_available=finish is not None, truncated=finish == "length")
    evaluation = None
    try:
        evaluation = checks.evaluate_output(
            content if isinstance(content, str) else raw_text, case["resume_text"],
            source_sections=dataset.source_sections(case), absent_skills=case["absent_skills"],
            vocabulary=vocab, example_numbers=example_numbers)
    except Exception as exc:
        record["evaluation_error"] = type(exc).__name__
    if evaluation is not None:
        schema = evaluation["schema"]
        record.update(schema_status=schema["status"], production_would_500=schema["production_would_500"],
                      problems=list(schema["problems"]), bullets_total=evaluation["bullets_total"])
    if record["truncated"]:
        # the provider says the output cap was reached: a cap artifact, not an ordinary JSON/schema/empty failure
        record["status"] = "truncated"
        if not isinstance(content, str) or not content.strip():
            record["production_would_500"] = True
        record["problems"] = ["output cut off by the max_tokens cap (finish_reason == 'length')"] + record["problems"]
    elif not isinstance(content, str):
        # production calls .strip() on the text and would crash on anything that is not a string
        record["status"] = "parse_failed"
        record["production_would_500"] = True
        record["problems"] = ["response content is not text"] + record["problems"]
    elif not content.strip():
        record["status"] = "empty_response"
        record["production_would_500"] = True
    elif evaluation is None:
        record["status"] = "evaluation_error"  # the checker itself crashed; the raw response is already saved
    elif evaluation["schema"]["status"] == "ok":
        record["status"] = "ok"
    elif evaluation["schema"]["status"] == "invalid_json":
        record["status"] = "parse_failed"
    else:
        record["status"] = "schema_invalid"  # wrong_shape, or an empty optimized_bullets list
    record["finished_utc"] = now().isoformat()
    return record, evaluation if record["status"] == "ok" else None


def execute_run(run, plan, call, client_info, results_root, cases_dir=None, vocab_path=None, pace_seconds=None,
                max_retries=DEFAULT_MAX_RETRIES, retry_backoff=DEFAULT_RETRY_BACKOFF, abort_after=DEFAULT_ABORT_AFTER,
                sleep=time.sleep, clock=time.monotonic, now=utc_now, out=print, unverified_settings=(),
                abort_after_truncated=DEFAULT_ABORT_AFTER_TRUNCATED):
    """Runs the scheduled calls with the supplied `call(job_description, resume_content) -> ModelResponse`.
    max_retries/retry_backoff here are THIS RUNNER's bounded retries (one extra invocation each); they are separate
    from any retries the provider library performs inside one invocation."""
    pace = plan["pacing"]["pace_seconds"] if pace_seconds is None else pace_seconds
    config = {"max_retries": max_retries, "retry_backoff": retry_backoff, "abort_after": abort_after,
              "abort_after_truncated": abort_after_truncated, "pace_seconds": pace}
    cases = {c["id"]: c for c in dataset.load_cases(plan["split"], cases_dir=cases_dir)}
    vocab = dataset.load_vocabulary(vocab_path)
    metadata = run.read_json("metadata.json")
    metadata.update(status="running", started_utc=now().isoformat(), execution_config=config,
                    client=client_info, live_calls_made=0, unverified_settings=list(unverified_settings))
    run.write_json("metadata.json", metadata)
    consecutive, attempted, aborted_reason, stop_index = 0, 0, None, len(plan["schedule"])
    consecutive_truncated = 0
    for index, item in enumerate(plan["schedule"]):
        if index:
            sleep(pace)
        record, evaluation = _call_one(call, cases[item["case_id"]], item["repeat"], plan["split"], run, vocab,
                                       set(plan["prompt_example_numbers"]), config, sleep, clock, now)
        run.append_jsonl("calls.jsonl", record)
        if evaluation is not None:
            run.append_jsonl("evaluations.jsonl", {"call_id": record["call_id"], "case_id": record["case_id"],
                                                   "repeat": record["repeat"], "split": record["split"],
                                                   "evaluation": evaluation})
        attempted += 1
        metadata["live_calls_made"] += len(record["attempts"])
        out(f"  [{attempted}/{plan['planned_calls']}] {record['call_id']}: {record['status']}")
        consecutive = consecutive + 1 if record["status"] == "api_error" else 0
        consecutive_truncated = consecutive_truncated + 1 if record["status"] == "truncated" else 0
        if record["error_kind"] == "auth" and record["status"] == "api_error":
            aborted_reason, stop_index = "authentication error", index + 1
        elif consecutive >= abort_after:
            aborted_reason, stop_index = f"{consecutive} consecutive API failures", index + 1
        elif consecutive_truncated >= abort_after_truncated:
            aborted_reason, stop_index = (f"{consecutive_truncated} consecutive truncated responses (the output "
                                          "cap was reached before the model finished)"), index + 1
        if aborted_reason:
            out(f"  ABORTED: {aborted_reason}; remaining planned calls were not attempted")
            break
    metadata.update(status="aborted" if aborted_reason else "complete", finished_utc=now().isoformat(),
                    attempted_calls=attempted, aborted_reason=aborted_reason,
                    not_attempted=[dict(i) for i in plan["schedule"][stop_index:]])
    run.write_json("metadata.json", metadata)
    write_label_sheet(run, cases_dir=cases_dir)
    return metadata


def check_plan_is_current(plan, run, results_root, cases_dir=None, vocab_path=None, source_root=None,
                          environment=None):
    """Rebuild the plan from the repository and environment as they are NOW and compare every field that could
    change the run. Returns the list of mismatches (a plan from an older format fails on plan_schema_version)."""
    try:
        current = build_plan(plan["split"], plan["repeats"], results_root, cases_dir, vocab_path, source_root,
                             None, plan["pacing"]["pace_seconds"], environment,
                             (plan.get("evaluation_overrides") or {}).get("max_tokens"))
    except Refused as error:
        return [f"evaluation_overrides ({error})"]
    mismatches = []
    for key in ("plan_schema_version", "split", "repeats", "case_ids", "schedule", "planned_calls",
                "model_expected", "prompt_key", "model_settings", "packages", "provider_retry",
                "evaluation_overrides", "effective_settings", "pacing"):
        if plan.get(key) != current[key]:
            mismatches.append(key)
    for key in current["code_fingerprints"]:
        if (plan.get("code_fingerprints") or {}).get(key) != current["code_fingerprints"][key]:
            mismatches.append(f"code:{key}")
    for key in ("dataset", "vocabulary", "checker", "prompt"):
        if plan["fingerprints"].get(key) != current["fingerprints"][key]:
            mismatches.append(f"fingerprint:{key}")
    try:
        copy = run.read_json("metadata.json")
    except (OSError, ValueError):
        copy = {}
    for key in ("evaluation_overrides", "effective_settings", "pacing", "model_settings"):
        if copy.get(key) != plan.get(key):  # plan.json must still equal the copy written when it was planned
            mismatches.append(f"metadata:{key}")
    if plan["split"] == "holdout":
        frozen = plan.get("frozen_from") or {}
        try:
            frozen_reference(results_root, frozen.get("run_id", ""), current)
        except Refused as error:
            mismatches.append(f"frozen_from ({error})")
    return mismatches


def cmd_run(plan_id, live, confirm_calls, confirm_holdout, results_root, client_factory=None, cases_dir=None,
            vocab_path=None, source_root=None, pace_seconds=None, max_retries=DEFAULT_MAX_RETRIES,
            retry_backoff=DEFAULT_RETRY_BACKOFF, abort_after=DEFAULT_ABORT_AFTER, sleep=time.sleep,
            clock=time.monotonic, now=utc_now, out=print, environment=None,
            abort_after_truncated=DEFAULT_ABORT_AFTER_TRUNCATED):
    results_root = results_root or RESULTS_DIR
    if not live:
        raise Refused("this command makes LIVE model calls; add --live (and --confirm-calls) to proceed")
    run = open_run_dir(results_root, plan_id)
    try:
        plan = run.read_json("plan.json")
    except (OSError, ValueError):
        raise Refused(f"{plan_id!r} has no readable plan.json; create one with the plan command")
    if confirm_calls is None:
        raise Refused(f"--confirm-calls is required and must equal the planned call count ({plan['planned_calls']})")
    if confirm_calls != plan["planned_calls"]:
        raise Refused(f"--confirm-calls {confirm_calls} does not match the plan's {plan['planned_calls']} calls")
    if plan["split"] != "dev" and confirm_holdout != HOLDOUT_PHRASE:
        raise Refused(f"a holdout run also needs --confirm-holdout {HOLDOUT_PHRASE}; the generic live "
                      "confirmation is not enough")
    if pace_seconds is not None and pace_seconds != plan["pacing"]["pace_seconds"]:
        raise Refused(f"--pace-seconds {pace_seconds} differs from the pacing frozen in the plan "
                      f"({plan['pacing']['pace_seconds']} s); create a new plan with the pacing you want")
    if not isinstance(abort_after_truncated, int) or abort_after_truncated < 1:
        raise Refused("--abort-after-truncated must be a whole number of at least 1")
    if run.exists("calls.jsonl") or run.read_json("metadata.json").get("status") != "planned":
        raise Refused("this plan has already been executed (or started); create a new plan")
    mismatches = check_plan_is_current(plan, run, results_root, cases_dir, vocab_path, source_root, environment)
    if mismatches:
        raise Refused("the plan is stale: " + ", ".join(mismatches) + " changed since planning; create a new plan")
    override = (plan.get("evaluation_overrides") or {}).get("max_tokens")  # the PLAN decides, never a hardcoded value
    call, info = (client_factory or build_live_client)(max_tokens_override=override)
    if info.get("model") and info["model"] != plan["model_expected"]:
        raise Refused(f"the live client uses model {info['model']!r}, the plan expects {plan['model_expected']!r}")
    differences, unverified = verify_live_settings(plan, info)
    if differences:
        raise Refused("the live client differs from the plan: " + "; ".join(differences))
    out(f"RUN {plan_id}: {plan['planned_calls']} LIVE invocations, split {plan['split']}")
    out(f"  effective max_tokens: {plan['effective_settings']['max_tokens']} "
        f"({plan['effective_settings']['max_tokens_source']}); pacing {plan['pacing']['pace_seconds']} s")
    if unverified:
        out("  settings that could not be verified against the live client (recorded in metadata): "
            + ", ".join(unverified))
    return execute_run(run, plan, call, info, results_root, cases_dir, vocab_path, pace_seconds, max_retries,
                       retry_backoff, abort_after, sleep, clock, now, out, unverified_settings=unverified,
                       abort_after_truncated=abort_after_truncated)


# ---------------------------------------------------------------------------------------------------------------
# Label sheet
# ---------------------------------------------------------------------------------------------------------------

def _matched_heading(output_section, case):
    """The source heading checks.check_sections assigns to a single output group, or '' if none/ambiguous."""
    sections = dataset.source_sections(case)
    result = checks.check_sections([{"section": output_section, "bullets": ["x"]}], sections)
    if result["matched"] != 1:
        return ""
    unmatched = set(result["missing_from_output"])
    left = [s["heading"] for s in sections if s["heading"] not in unmatched]
    return left[0] if len(left) == 1 else ""


def _findings(evaluation, index):
    def pick(check, key):
        block = evaluation.get(check)
        return list(block["results"][index][key]) if block else []
    invented, leaked = pick("numbers", "invented"), pick("absent_skills", "leaked")
    unsupported = pick("technologies", "unsupported")
    unchanged = bool(evaluation.get("unchanged")) and index in evaluation["unchanged"]
    duplicates = [p["a"] if p["b"] == index else p["b"] for p in (evaluation.get("duplicates") or [])
                  if index in (p["a"], p["b"])]
    parts = []
    if invented:
        parts.append("D2 invented numbers: " + ", ".join(invented))
    if leaked:
        parts.append("D3 absent skills: " + ", ".join(leaked))
    if unsupported:
        parts.append("D4 technologies not in resume (advisory): " + ", ".join(unsupported))
    if unchanged:
        parts.append("D6 identical to a source bullet")
    if duplicates:
        parts.append("D6 near-duplicate of bullet(s) " + ", ".join(str(d) for d in duplicates))
    return invented, leaked, unsupported, unchanged, duplicates, "; ".join(parts)


def build_label_rows(run, cases_dir=None):
    cases = {c["id"]: c for c in dataset.load_cases("all", cases_dir=cases_dir)}
    calls = {r["call_id"]: r for r in run.read_jsonl("calls.jsonl")}
    rows = []
    for item in run.read_jsonl("evaluations.jsonl"):
        record, evaluation, case = calls[item["call_id"]], item["evaluation"], cases[item["case_id"]]
        raw = run.read_text(record["raw_path"])
        sections = checks.parse_ats_output(raw)["sections"]
        index = 0
        for section in sections:
            heading = _matched_heading(section["section"], case)
            source = ""
            if heading:
                source = " || ".join(next(s["bullets"] for s in dataset.source_sections(case) if s["heading"] == heading))
            for bullet in section["bullets"]:
                invented, leaked, unsupported, unchanged, duplicates, summary = _findings(evaluation, index)
                row = {"case_id": item["case_id"], "split": item["split"], "repeat": item["repeat"],
                       "call_id": item["call_id"], "bullet_index": index, "output_section": section["section"],
                       "matched_source_heading": heading, "source_bullets": source, "rewritten_bullet": bullet,
                       "checker_findings": summary, "d2_invented_numbers": ", ".join(invented),
                       "d3_absent_skill_leaks": ", ".join(leaked),
                       "d4_unsupported_technologies": ", ".join(unsupported),
                       "d6_unchanged_from_source": "yes" if unchanged else "",
                       "d6_near_duplicate_of": ", ".join(str(d) for d in duplicates)}
                row.update({column: "" for column in HUMAN_COLUMNS})
                rows.append(row)
                index += 1
    return rows


def rows_to_csv(rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([csv_safe(row[column]) for column in CSV_COLUMNS])
    return buffer.getvalue()


def _has_human_labels(text):
    reader = csv.DictReader(io.StringIO(text))
    return any((row.get(column) or "").strip() for row in reader for column in HUMAN_COLUMNS)


def write_label_sheet(run, cases_dir=None):
    """Writes bullets.csv. Refuses to overwrite a sheet that already contains human labels."""
    if run.exists("bullets.csv") and _has_human_labels(run.read_text("bullets.csv")):
        raise Refused("bullets.csv already contains human labels; refusing to overwrite it")
    rows = build_label_rows(run, cases_dir)
    run.write_text("bullets.csv", rows_to_csv(rows))
    return len(rows)


# ---------------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------------

def _flag_counts(evaluations, check, key):
    """(bullets checked, bullets flagged) for one checker over the evaluations that ran it."""
    checked = flagged = 0
    for item in evaluations:
        block = item["evaluation"].get(check)
        if block:
            checked += len(block["results"])
            flagged += sum(1 for r in block["results"] if r[key])
    return checked, flagged


def summarize_run(run):
    plan = run.read_json("plan.json")
    metadata = run.read_json("metadata.json")
    calls = run.read_jsonl("calls.jsonl")
    evaluations = run.read_jsonl("evaluations.jsonl")
    by_status = {s: sum(1 for c in calls if c["status"] == s) for s in CALL_STATUSES}
    received = sum(1 for c in calls if c["response_received"])
    truncated = by_status["truncated"]  # cap artifacts: excluded from the production-failure rate
    production_failures = sum(1 for c in calls if c["response_received"] and c["production_would_500"]
                              and c["status"] != "truncated")
    no_finish_reason = sum(1 for c in calls if c["response_received"] and not c.get("finish_reason_available"))
    bullets_checked = sum(item["evaluation"]["bullets_total"] for item in evaluations)
    attempts = sum(len(c["attempts"]) for c in calls)
    reviewed, rows_total = 0, 0
    if run.exists("bullets.csv"):
        for row in csv.DictReader(io.StringIO(run.read_text("bullets.csv"))):
            rows_total += 1
            if (row.get("supported_by_source_evidence") or "").strip():
                reviewed += 1
    d2c, d2f = _flag_counts(evaluations, "numbers", "invented")
    d2e = sum(1 for i in evaluations for r in (i["evaluation"]["numbers"] or {"results": []})["results"] if r["echo"])
    d2u = sum(1 for i in evaluations for r in (i["evaluation"]["numbers"] or {"results": []})["results"] if r["unit_mismatch"])
    d3c, d3f = _flag_counts(evaluations, "absent_skills", "leaked")
    d4c, d4f = _flag_counts(evaluations, "technologies", "unsupported")
    sections = {"outputs": len(evaluations), "with_unmatched_output_section": 0, "with_missing_source_section": 0,
                "with_bullet_count_change": 0, "with_ambiguous_or_duplicate_group": 0}
    unchanged = near_duplicate_outputs = 0
    for item in evaluations:
        ev = item["evaluation"]
        block = ev.get("sections") or {}
        sections["with_unmatched_output_section"] += bool(block.get("unmatched_output"))
        sections["with_missing_source_section"] += bool(block.get("missing_from_output"))
        sections["with_bullet_count_change"] += bool(block.get("bullet_count_changes"))
        sections["with_ambiguous_or_duplicate_group"] += bool(block.get("ambiguous_output") or block.get("duplicate_groups"))
        unchanged += len(ev.get("unchanged") or [])
        near_duplicate_outputs += bool(ev.get("duplicates"))
    eval_index = {i["call_id"]: i["evaluation"] for i in evaluations}
    per_case = []
    for case_id in plan["case_ids"]:
        mine = [c for c in calls if c["case_id"] == case_id]
        row = {"case_id": case_id, "planned": plan["repeats"], "attempted": len(mine),
               "statuses_by_repeat": {str(c["repeat"]): c["status"] for c in mine}, "bullets_checked": 0,
               "d2_flagged_by_repeat": {}, "d3_flagged_by_repeat": {}, "d4_flagged_by_repeat": {}}
        for c in mine:
            ev = eval_index.get(c["call_id"])
            if ev:
                row["bullets_checked"] += ev["bullets_total"]
                for column, check, key in (("d2_flagged_by_repeat", "numbers", "invented"),
                                           ("d3_flagged_by_repeat", "absent_skills", "leaked"),
                                           ("d4_flagged_by_repeat", "technologies", "unsupported")):
                    n, f = _flag_counts([{"evaluation": ev}], check, key)
                    row[column][str(c["repeat"])] = f"{f}/{n}"
        per_case.append(row)
    both_ok = sum(1 for r in per_case if len(r["statuses_by_repeat"]) == plan["repeats"]
                  and all(s == "ok" for s in r["statuses_by_repeat"].values()))
    agree = sum(1 for r in per_case if len(r["statuses_by_repeat"]) == plan["repeats"]
                and len(set(r["statuses_by_repeat"].values())) == 1)
    usage_calls = [c for c in calls if c["usage"]]
    latencies = [c["latency_s"] for c in calls if c["latency_s"] is not None]
    summary = {
        "run_id": plan["run_id"], "split": plan["split"], "status": metadata.get("status"),
        "model_expected": plan["model_expected"], "case_ids": plan["case_ids"], "repeats": plan["repeats"],
        "counts": {
            "planned_calls": plan["planned_calls"], "attempted_calls": len(calls),
            "not_attempted_calls": plan["planned_calls"] - len(calls), "model_call_attempts_including_retries": attempts,
            "responses_received": received, "api_failures": by_status["api_error"],
            "empty_responses": by_status["empty_response"], "json_parse_failures": by_status["parse_failed"],
            "schema_invalid_responses": by_status["schema_invalid"],
            "truncated_responses": truncated, "responses_without_finish_reason": no_finish_reason,
            "checker_errors_response_saved": by_status["evaluation_error"],
            "outputs_eligible_for_bullet_checks": by_status["ok"], "bullets_checked": bullets_checked,
            "bullets_in_label_sheet": rows_total, "bullets_reviewed_by_human": reviewed,
        },
        "rates": {
            "production_failures_over_responses_received": rate(production_failures, received - truncated),
            "truncated_over_responses_received": rate(truncated, received),
            "api_failures_over_attempted_calls": rate(by_status["api_error"], len(calls)),
            "schema_invalid_over_responses_received": rate(by_status["schema_invalid"], received),
            "outputs_eligible_over_responses_received": rate(by_status["ok"], received),
            "bullets_reviewed_over_bullets_checked": rate(reviewed, bullets_checked),
        },
        "production_failures": production_failures,
        "configuration": {
            "evaluation_overrides": plan.get("evaluation_overrides") or {},
            "effective_max_tokens": (plan.get("effective_settings") or {}).get("max_tokens"),
            "production_max_tokens": (plan.get("model_settings") or {}).get("max_tokens"),
            "pace_seconds": (plan.get("pacing") or {}).get("pace_seconds")},
        "finish_reasons": {name: sum(1 for c in calls if c["response_received"]
                                     and (c.get("finish_reason") or "not reported") == name)
                           for name in sorted({(c.get("finish_reason") or "not reported")
                                               for c in calls if c["response_received"]})},
        "checkers": {
            "D2_invented_numbers": {"bullets_checked": d2c, "bullets_flagged": d2f, "rate": rate(d2f, d2c),
                                    "bullets_with_prompt_example_echo": d2e, "bullets_with_unit_mismatch_advisory": d2u},
            "D3_absent_skills": {"bullets_checked": d3c, "bullets_flagged": d3f, "rate": rate(d3f, d3c)},
            "D4_technologies_advisory": {"bullets_checked": d4c, "bullets_flagged": d4f, "rate": rate(d4f, d4c)},
            "D5_sections": sections,
            "D6_duplicates": {"rewritten_bullets_identical_to_a_source_bullet": unchanged,
                              "bullets_checked": bullets_checked, "outputs_with_near_duplicate_pair": near_duplicate_outputs,
                              "outputs": len(evaluations)},
        },
        "repeat_consistency_raw_counts": {
            "cases_where_all_repeats_ok": f"{both_ok}/{len(per_case)}",
            "cases_where_repeats_have_the_same_status": f"{agree}/{len(per_case)}"},
        "per_case": per_case,
        "tokens": {"calls_with_measured_usage": f"{len(usage_calls)}/{received}",
                   "measured_total_tokens": sum(c["usage"].get("total_tokens", 0) for c in usage_calls),
                   "estimates_are_in_plan_json_not_here": True},
        "latency_s": ({"calls": len(latencies), "min": min(latencies), "median": statistics.median(latencies),
                       "max": max(latencies)} if latencies else None),
        "aborted_reason": metadata.get("aborted_reason"),
    }
    return summary


def render_report(summary):
    c, r = summary["counts"], summary["rates"]
    label = "HOLDOUT" if summary["split"] == "holdout" else "DEVELOPMENT"
    lines = [
        f"# ATS bullet rewrite evaluation: {summary['run_id']}",
        "",
        f"**Split: {label}** | status: {summary['status']} | model: {summary['model_expected']} | "
        f"cases: {len(summary['case_ids'])} | repeats: {summary['repeats']}",
        "",
        "Read this first:",
        "- 12 synthetic cases cannot show that the feature works on real resumes. These are screening signals, not accuracy.",
        "- Bullet-level checks run only on responses that parsed and validated, so they ignore every failed response "
        "(selection effect); the reliability rows below keep the failures.",
        "- D3 (declared absent skills) and D4 (vocabulary technologies, advisory) can flag the same bullet. They are "
        "reported separately and must not be added.",
        "- Flags are not human quality judgments. Only the reviewed rows below are.",
    ]
    if summary["split"] == "holdout":
        lines.append("- The vocabulary and absent-skill lists were authored with all 12 cases visible, so the holdouts "
                     "are blind to results but not to dataset design.")
    lines += ["", "## Reliability", "",
              "| Item | Count |", "|---|---|",
              f"| Planned calls | {c['planned_calls']} |", f"| Attempted calls | {c['attempted_calls']} |",
              f"| Not attempted | {c['not_attempted_calls']} |",
              f"| Responses received | {c['responses_received']} |",
              f"| API failures | {r['api_failures_over_attempted_calls']} of attempted calls |",
              f"| Empty responses | {c['empty_responses']} |", f"| JSON parse failures | {c['json_parse_failures']} |",
              f"| Schema-invalid responses | {r['schema_invalid_over_responses_received']} |",
              f"| Truncated (output cap reached) | {r['truncated_over_responses_received']} |",
              f"| Responses with no finish reason reported | {c['responses_without_finish_reason']}/{c['responses_received']} |",
              f"| Checker errors (raw response saved, not evaluated) | {c['checker_errors_response_saved']} |",
              f"| Production failures (would be HTTP 500) | {r['production_failures_over_responses_received']} |",
              f"| Outputs eligible for bullet checks | {r['outputs_eligible_over_responses_received']} |", ""]
    if summary["aborted_reason"]:
        lines += [f"**Run aborted: {summary['aborted_reason']}.**", ""]
    cfg = summary["configuration"]
    if cfg["evaluation_overrides"]:
        lines += [f"**Evaluation-only output cap: max_tokens={cfg['effective_max_tokens']}** (production "
                  f"{cfg['production_max_tokens']}). Truncated responses reached this cap. They are excluded from the "
                  "production-failure rate above (numerator and denominator) and are not model-quality failures "
                  "at the production setting.", ""]
    if c["truncated_responses"]:
        lines += [f"**{c['truncated_responses']} response(s) were truncated: this run is cap-limited and its other "
                  "rates describe only the responses that finished.**", ""]
    if c["responses_without_finish_reason"]:
        lines += [f"Finish reason was not reported for {c['responses_without_finish_reason']} response(s), so "
                  "truncation could not be ruled out for them.", ""]
    k = summary["checkers"]
    lines += ["## Deterministic checks (each with its own denominator)", "",
              "| Check | Bullets flagged / checked |", "|---|---|",
              f"| D2 invented numbers | {k['D2_invented_numbers']['rate']} |",
              f"| D3 declared absent skills | {k['D3_absent_skills']['rate']} |",
              f"| D4 technologies not in resume (advisory) | {k['D4_technologies_advisory']['rate']} |",
              f"| D6 rewrite identical to a source bullet | {k['D6_duplicates']['rewritten_bullets_identical_to_a_source_bullet']}/{k['D6_duplicates']['bullets_checked']} |",
              "", f"D2 extras (advisory): prompt-example echo {k['D2_invented_numbers']['bullets_with_prompt_example_echo']}, "
              f"unit mismatch {k['D2_invented_numbers']['bullets_with_unit_mismatch_advisory']}.",
              f"D5 sections (outputs: {k['D5_sections']['outputs']}): unmatched output section "
              f"{k['D5_sections']['with_unmatched_output_section']}, missing source section "
              f"{k['D5_sections']['with_missing_source_section']}, bullet-count change "
              f"{k['D5_sections']['with_bullet_count_change']}, ambiguous or duplicate group "
              f"{k['D5_sections']['with_ambiguous_or_duplicate_group']}.", "",
              "## Human review", "",
              f"Bullets reviewed: {r['bullets_reviewed_over_bullets_checked']}.", "",
              "## Repeat consistency (raw counts only)", "",
              f"- Cases where every repeat was ok: {summary['repeat_consistency_raw_counts']['cases_where_all_repeats_ok']}",
              f"- Cases where repeats had the same status: {summary['repeat_consistency_raw_counts']['cases_where_repeats_have_the_same_status']}",
              "", "| Case | Status by repeat | D2 | D3 | D4 |", "|---|---|---|---|---|"]
    for row in summary["per_case"]:
        def fmt(mapping):
            return ", ".join(f"r{k_}: {v}" for k_, v in sorted(mapping.items())) or "-"
        statuses = ", ".join(f"r{k_}: {v}" for k_, v in sorted(row["statuses_by_repeat"].items())) or "not attempted"
        lines.append(f"| {row['case_id']} | {statuses} | {fmt(row['d2_flagged_by_repeat'])} | "
                     f"{fmt(row['d3_flagged_by_repeat'])} | {fmt(row['d4_flagged_by_repeat'])} |")
    lines += ["", "## Usage", "",
              f"Measured token usage was returned for {summary['tokens']['calls_with_measured_usage']} responses "
              f"(total {summary['tokens']['measured_total_tokens']}). Plan-time token figures are estimates.", ""]
    if summary["latency_s"]:
        lines.append(f"Latency (s): min {summary['latency_s']['min']}, median {summary['latency_s']['median']}, "
                     f"max {summary['latency_s']['max']}.")
        lines.append("")
    return "\n".join(lines)


def cmd_summarize(run_id, results_root, out=print):
    run = open_run_dir(results_root or RESULTS_DIR, run_id)
    summary = summarize_run(run)
    run.write_json("summary.json", summary)
    run.write_text("report.md", render_report(summary))
    out(f"summary.json and report.md written under eval/ats/results/{run_id}/")
    return summary


def cmd_label_sheet(run_id, results_root, cases_dir=None, out=print):
    run = open_run_dir(results_root or RESULTS_DIR, run_id)
    count = write_label_sheet(run, cases_dir)
    out(f"bullets.csv written with {count} bullet rows (human-review columns are blank)")
    return count


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="offline dry run: validate the dataset and write plan.json (default workflow)")
    plan.add_argument("--split", choices=RUN_SPLITS, default="dev")
    plan.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    plan.add_argument("--pace-seconds", type=int, default=DEFAULT_PACE_SECONDS)
    plan.add_argument("--eval-max-tokens", type=int, nargs="?", const=DEFAULT_EVAL_MAX_TOKENS, default=None,
                      help=f"evaluation-only output cap (lower than production); {DEFAULT_EVAL_MAX_TOKENS} when "
                           "given without a value; omit for no override")
    plan.add_argument("--run-id", default=None)
    plan.add_argument("--frozen-from", default=None, help="holdout plans only: id of a completed dev run")
    run = sub.add_parser("run", help="LIVE model calls (needs --live and --confirm-calls)")
    run.add_argument("--plan", required=True, help="run id created by the plan command")
    run.add_argument("--live", action="store_true")
    run.add_argument("--confirm-calls", type=int, default=None)
    run.add_argument("--confirm-holdout", default=None)
    run.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    run.add_argument("--retry-backoff", type=int, default=DEFAULT_RETRY_BACKOFF)
    run.add_argument("--abort-after", type=int, default=DEFAULT_ABORT_AFTER)
    run.add_argument("--pace-seconds", type=int, default=None, help="must equal the pacing frozen in the plan")
    run.add_argument("--abort-after-truncated", type=int, default=DEFAULT_ABORT_AFTER_TRUNCATED)
    for name, text in (("summarize", "write summary.json and report.md for a run"),
                       ("label-sheet", "write the human-review bullets.csv for a run")):
        item = sub.add_parser(name, help=text)
        item.add_argument("run_id")
    return parser


def main(argv=None, results_root=None, client_factory=None, cases_dir=None, vocab_path=None, source_root=None,
         sleep=time.sleep, clock=time.monotonic, now=utc_now, out=print, environment=None):
    """Returns the process exit code (0 ok, 2 refused). The keyword arguments exist for tests only."""
    args = make_parser().parse_args(argv)
    try:
        if args.command == "plan":
            cmd_plan(args.split, args.repeats, results_root, now(), args.run_id, args.frozen_from, cases_dir,
                     vocab_path, source_root, args.pace_seconds, out, environment, args.eval_max_tokens)
        elif args.command == "run":
            cmd_run(args.plan, args.live, args.confirm_calls, args.confirm_holdout, results_root, client_factory,
                    cases_dir, vocab_path, source_root, args.pace_seconds, args.max_retries, args.retry_backoff,
                    args.abort_after, sleep, clock, now, out, environment,
                    abort_after_truncated=args.abort_after_truncated)
        elif args.command == "summarize":
            cmd_summarize(args.run_id, results_root, out)
        else:
            cmd_label_sheet(args.run_id, results_root, cases_dir, out)
    except (Refused, dataset.DatasetError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
