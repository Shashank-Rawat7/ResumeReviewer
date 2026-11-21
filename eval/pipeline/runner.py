"""Whole-pipeline evaluation runner (Step 6, phase 1): plan, run (LIVE, gated) and summarize.

    python -B -m eval.pipeline.runner plan [--repeats 3] [--pace-seconds 40] [--max-retries 1] [--run-id ID]
    python -B -m eval.pipeline.runner plan --repeats 2 --continues-from RUN_ID      (repeats 2 and 3 after a complete repeat 1)
    python -B -m eval.pipeline.runner run --plan RUN_ID --live --confirm-calls N          (the ONLY model-calling command)
    python -B -m eval.pipeline.runner summarize RUN_ID [--force]

OFFLINE vs LIVE
  plan       offline. Hashes the dataset, prompts, production settings (read as TEXT) and the evaluation code, and freezes
             them with the call schedule in results/<run-id>/plan.json. No network, no key, no model.
  summarize  offline. Runs the real evaluation CLI (cli.py) on every saved outputs_repN.json and aggregates its reports.
             Adds no metric: it shows each repeat's own numerator and denominator and the range across repeats.
  run        LIVE. Refuses unless --live is given AND --confirm-calls equals the plan's planned_calls AND the saved plan
             still matches the current files (every hash). The API key is read ONLY from the process environment
             (GROQ_API_KEY); no .env file is read. The model client is imported lazily inside make_groq_model().

WHAT THE RUN DOES (production-equivalent, without importing production code)
  per repeat, per scenario:  E1 (job description only)  ->  E2 (job description + E1's cleaned output + resume text)
                             ->  E3 (job description + resume text; independent of E1 and E2, always attempted).
  E2 is skipped when the E1 call failed (production keeps both in one try). A failure of E1 or E2 never prevents E3.
  The E2 normalisation is the production function build_prioritized_missing_skills, extracted by stages.py.
  Only transport failures (429, 408, 5xx, timeouts, connection errors) are retried, at most once. A model answer that is
  not valid JSON is a result, not a failure, and is never retried. A call that fails is stored as null, never as correct.

ARTIFACTS  eval/pipeline/results/<run-id>/: plan.json, manifest.json, calls.jsonl (one line per attempt, no response
  text, no error messages), outputs_repN.json (final raw outputs, the CLI input contract), report_repN.json,
  review_rows_repN.json, summary.json, summary.md.

This module never runs the application, never reads secrets or the protected resume artifacts, and never touches the
dataset files. Standard library only (PyYAML is imported lazily by the plan; the model client only by a live run).
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import string
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

PLAN_VERSION = "pipeline-plan-1"
MANIFEST_VERSION = "pipeline-manifest-1"
SUMMARY_VERSION = "pipeline-summary-1"
DEFAULT_REPEATS, MAX_REPEATS = 3, 5
DEFAULT_PACE_SECONDS = 40
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF = 60
DEFAULT_ABORT_AFTER = 3
EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2

STAGES = ("e1", "e2", "e3")
PROMPT_KEYS = {"e1": "job_description_skill_extractor", "e2": "job_resume_comparison", "e3": "resume_ats_optimizer"}
PROMPT_VARIABLES = {"e1": ("job_description",), "e2": ("job_description", "job_skills", "resume_content"),
                    "e3": ("job_description", "resume_content")}
SOURCE_FILES = (("src", "parser.py"), ("src", "prompts.yaml"), ("frontend", "display.py"))
CODE_FILES = ("runner.py", "cli.py", "dataset.py", "stages.py", "score.py", "e3.py", os.path.join("..", "ats", "checks.py"))
EVALUATION_OVERRIDES = {"max_retries": 0, "request_timeout": 120}   # model client only; recorded in the plan
CONTINUATION_REPEATS = 2          # a continuation adds exactly two repeats (numbered 2 and 3) after a complete repeat 1
OUTPUT_FIELDS = ("e1", "e2_raw", "e2_normalized", "e3_raw")
RETRYABLE_NAMES = frozenset({"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError", "Timeout",
                             "ReadTimeout", "ConnectTimeout", "ConnectionError", "TimeoutError"})
API_KEY_VARIABLE = "GROQ_API_KEY"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

CONDITIONS = [
    "Inputs are the scenario's job_description and resume_text, passed to the prompts directly. Production joins overlapping "
    "text chunks into resume_text (open item A6); that duplication is not reproduced.",
    "PDF extraction and the resume-summary and credentials steps are not evaluated.",
    "E2 is COUPLED to E1 as in production: it receives E1's cleaned output. E3 receives only the job description and the "
    "resume text and never E1 or E2 data.",
    "Model settings are those written in the ChatGroq(...) call of src/parser.py. No seed is set, as in production, so "
    "repeats measure run-to-run variation rather than a fixed result.",
    "The runner retries a transport failure at most once. The model client is created with max_retries=0 (an evaluation "
    "override; production leaves the library default), so the runner's retry is the ONLY retry and every HTTP request is "
    "one logged attempt.",
    "The model client is created with a request timeout (evaluation_overrides.request_timeout, seconds). A request that "
    "exceeds it fails as a timeout, which the runner treats as a transport failure (one retry). Production sets no "
    "timeout, and the installed client library applies none by default.",
    "The model name recorded is the CONFIGURED name, echoed back by the client library in the response metadata. It is "
    "not an identifier verified or reported independently by the provider.",
    "Scenario files stay unfrozen (draft); their content hashes are recorded and the run refuses if any changed.",
]
LIMITATIONS = [
    "Ten hand-authored synthetic scenarios, one model, draft-1 labels: a baseline, not a generalisation claim.",
    "Pooled rates add numerators and denominators across scenarios; per-scenario results are kept beside them.",
    "E3 numbers are deterministic screens, not verification that bullets are factually correct.",
    "Ranges across repeats show variation between runs; they are not confidence intervals.",
]


class PlanError(Exception):
    """A planning, validation or execution problem that ends the command (exit 1)."""


class UsageError(Exception):
    """A refused or malformed request (exit 2)."""


class AbortRun(Exception):
    """Too many consecutive failed calls: stop making calls."""


def _load(name, path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = _load("pipeline_cli", os.path.join(HERE, "cli.py"))
stages_module = _load("pipeline_stages", os.path.join(HERE, "stages.py"))
dataset_module = cli.dataset_module


# ---------------------------------------------------------------------------------------------------------------
# Hashing and canonical JSON
# ---------------------------------------------------------------------------------------------------------------

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    """SHA-256 of the file bytes with CRLF turned into LF (the dataset freeze convention), so a checkout on another
    operating system gives the same digest."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as error:
        raise PlanError(f"cannot read {os.path.basename(path)} ({error.__class__.__name__})")
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def clean_response(text):
    """The cleaning production applies to every stored response (src/parser.py): strip one leading ```json or trailing
    ``` marker. Nothing else is changed."""
    return re.sub(r"^```json|```$", "", text.strip()).strip()


# ---------------------------------------------------------------------------------------------------------------
# Production configuration, read as TEXT (src/parser.py is parsed, never imported: it loads .env on import)
# ---------------------------------------------------------------------------------------------------------------

def parse_model_settings(parser_text):
    """The ChatGroq(...) settings written in parser.py. Raises PlanError if they cannot be read as literals."""
    tree = ast.parse(parser_text)
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    if isinstance(item.value, ast.Constant):
                        constants[(node.name, item.targets[0].id)] = item.value.value
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ChatGroq":
            settings = {}
            for keyword in node.keywords:
                if keyword.arg in (None, "api_key"):
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant):
                    settings[keyword.arg] = value.value
                elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                        and (value.value.id, value.attr) in constants:
                    settings[keyword.arg] = constants[(value.value.id, value.attr)]
                else:
                    raise PlanError(f"ChatGroq setting {keyword.arg!r} in parser.py is not a literal")
            for required in ("model", "temperature", "max_tokens", "reasoning_effort"):
                if required not in settings:
                    raise PlanError(f"ChatGroq setting {required!r} not found in parser.py")
            return dict(sorted(settings.items()))
    raise PlanError("no ChatGroq(...) call found in parser.py")


def load_prompt_templates(root):
    """The three stage templates from src/prompts.yaml, with their placeholders checked against production's
    input_variables. Rendering is str.format, which is what a PromptTemplate with the default f-string format does."""
    path = os.path.join(root, "src", "prompts.yaml")
    try:
        import yaml  # a project dependency; imported here so that importing this module needs nothing else
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as error:
        raise PlanError(f"cannot read prompts.yaml ({error.__class__.__name__})")
    templates = {}
    for stage in STAGES:
        template = (data or {}).get(PROMPT_KEYS[stage])
        if not isinstance(template, str):
            raise PlanError(f"prompts.yaml has no text for {PROMPT_KEYS[stage]!r}")
        fields = {name for _lit, name, _spec, _conv in string.Formatter().parse(template) if name}
        if fields != set(PROMPT_VARIABLES[stage]):
            raise PlanError(f"prompt {PROMPT_KEYS[stage]!r} uses placeholders {sorted(fields)}, "
                            f"expected {sorted(PROMPT_VARIABLES[stage])}")
        templates[stage] = template
    return templates


def render_prompt(template, **variables):
    return template.format(**variables)


# ---------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------

def _plan_hash(plan):
    return sha256_text(canonical_json({key: value for key, value in plan.items() if key != "plan_hash"}))


def build_plan(repeats=DEFAULT_REPEATS, pace_seconds=DEFAULT_PACE_SECONDS, max_retries=DEFAULT_MAX_RETRIES,
               retry_backoff=DEFAULT_RETRY_BACKOFF, abort_after=DEFAULT_ABORT_AFTER, run_id=None, root=ROOT,
               scenarios_dir=None, expected_dir=None, continues_from=None, results_dir=None):
    """Build the frozen plan dict. Reads files, writes nothing, contains no timestamp and no absolute path.

    continues_from: id of an earlier run whose repeat 1 is complete. The plan then covers exactly two further repeats
    (numbered 2 and 3) and records the source run's identity and file hashes (results_dir locates the source)."""
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise UsageError(f"--repeats must be a whole number from 1 to {MAX_REPEATS}")
    if max_retries not in (0, 1):
        raise UsageError("--max-retries must be 0 or 1")
    for name, value in (("--pace-seconds", pace_seconds), ("--retry-backoff", retry_backoff)):
        if not isinstance(value, int) or value < 0:
            raise UsageError(f"{name} must be a whole number of seconds, 0 or more")
    if not isinstance(abort_after, int) or abort_after < 1:
        raise UsageError("--abort-after must be 1 or more")
    if run_id is not None and not RUN_ID_PATTERN.match(run_id):
        raise UsageError("--run-id must be 1-64 letters, digits, '.', '_' or '-' and start with a letter or digit")
    if continues_from is not None and repeats != CONTINUATION_REPEATS:
        raise UsageError(f"a continuation plan has exactly {CONTINUATION_REPEATS} repeats: use --repeats {CONTINUATION_REPEATS}")

    scenarios_dir = dataset_module.DEFAULT_SCENARIOS_DIR if scenarios_dir is None else scenarios_dir
    expected_dir = dataset_module.DEFAULT_EXPECTED_DIR if expected_dir is None else expected_dir
    try:
        dataset = dataset_module.load_dataset(scenarios_dir, expected_dir)
    except dataset_module.DatasetError as error:
        raise PlanError(str(error))
    ids = list(dataset["ids"])

    templates = load_prompt_templates(root)
    parser_path = os.path.join(root, "src", "parser.py")
    try:
        with open(parser_path, "r", encoding="utf-8") as handle:
            parser_text = handle.read()
    except OSError as error:
        raise PlanError(f"cannot read parser.py ({error.__class__.__name__})")
    model = parse_model_settings(parser_text)

    per_stage = len(ids) * repeats
    planned = per_stage * len(STAGES)
    plan = {
        "plan_version": PLAN_VERSION,
        "parameters": {"repeats": repeats, "pace_seconds": pace_seconds, "max_retries": max_retries,
                       "retry_backoff_seconds": retry_backoff, "abort_after_consecutive_failures": abort_after},
        "dataset": {
            "ids": ids,
            "scenario_file_sha256": {sid: sha256_file(os.path.join(scenarios_dir, sid + ".json")) for sid in ids},
            "label_file_sha256": {sid: sha256_file(os.path.join(expected_dir, sid + ".json")) for sid in ids},
            "label_frozen_sha256": dict(dataset["verified_hashes"]),
            "scenarios_frozen": False,
        },
        "model": model,
        "evaluation_overrides": dict(EVALUATION_OVERRIDES),
        "prompts": {stage: {"key": PROMPT_KEYS[stage], "variables": list(PROMPT_VARIABLES[stage]),
                            "template_sha256": sha256_text(templates[stage])} for stage in STAGES},
        "source_files": {"/".join(parts): sha256_file(os.path.join(root, *parts)) for parts in SOURCE_FILES},
        "code": {os.path.normpath(name).replace(os.sep, "/"): sha256_file(os.path.join(HERE, name))
                 for name in CODE_FILES},
        "schedule": {
            "order": "repeat-major; within a repeat each scenario in id order; within a scenario E1, then E2, then E3",
            "pacing": "pace_seconds before every logical call except the first; retries wait retry_backoff_seconds",
            "planned_calls_by_stage": {stage: per_stage for stage in STAGES},
            "planned_calls": planned,
            "planned_calls_note": "An upper bound: an E2 call is skipped when its E1 call failed.",
            "max_requests_with_runner_retry": planned * (1 + max_retries),
            "max_requests_note": "Exact upper bound on HTTP requests: provider-library retries are disabled "
                                 "(evaluation_overrides.max_retries = 0).",
            "minimum_pacing_seconds": pace_seconds * (planned - 1),
        },
        "conditions": list(CONDITIONS),
        "limitations": list(LIMITATIONS),
    }
    if continues_from is not None:
        _add_continuation(plan, continues_from, results_dir)
    plan["run_id"] = run_id or "pipeline_" + sha256_text(canonical_json(plan))[:12]
    plan["plan_hash"] = _plan_hash(plan)
    return plan


def verify_source(results_dir, source_id):
    """Check that run `source_id` holds a complete repeat 1 and nothing else, and return what a continuation records.
    Reads only; raises PlanError naming the first problem."""
    if results_dir is None:
        raise PlanError("a continuation needs the results directory to find its source run")
    source_plan = load_plan(results_dir, source_id)
    if source_plan.get("continues") is not None:
        raise PlanError("the source run is itself a continuation; chaining is not supported")
    folder = plan_dir(results_dir, source_id)
    try:
        names = os.listdir(folder)
    except OSError as error:
        raise PlanError(f"cannot list the source run folder ({error.__class__.__name__})")
    present = sorted(int(m.group(1)) for m in (re.match(r"^outputs_rep(\d+)\.json$", n) for n in names) if m)
    if present != [1]:
        raise PlanError(f"the source run must hold outputs for repeat 1 only (found repeats {present})")
    outputs_path, calls_path = os.path.join(folder, "outputs_rep1.json"), os.path.join(folder, "calls.jsonl")
    if not os.path.exists(calls_path):
        raise PlanError("the source run has no calls.jsonl")
    try:
        data, _raw = dataset_module.read_json_strict(outputs_path)
    except ValueError as error:
        raise PlanError(f"source outputs {error}")
    ids = source_plan["dataset"]["ids"]
    outputs = data.get("outputs") if isinstance(data, dict) else None
    if not isinstance(outputs, dict) or data.get("format_version") != cli.OUTPUTS_VERSION or sorted(outputs) != sorted(ids):
        raise PlanError("source outputs_rep1.json does not cover exactly the source plan's scenarios")
    for sid in ids:
        entry = outputs[sid]
        missing = [f for f in OUTPUT_FIELDS if not isinstance(entry, dict) or entry.get(f) is None]
        if missing:
            raise PlanError(f"source repeat 1 is incomplete: {sid} lacks {', '.join(missing)}")
    lines = []
    try:
        with open(calls_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    lines.append(json.loads(raw))
    except (OSError, ValueError) as error:
        raise PlanError(f"cannot read the source call log ({error.__class__.__name__})")
    repeat_one = [line for line in lines if isinstance(line, dict) and line.get("repeat") == 1]
    ok = {(line.get("scenario_id"), line.get("stage")) for line in repeat_one if line.get("status") == "ok"}
    wanted = {(sid, stage) for sid in ids for stage in STAGES}
    if ok != wanted:
        raise PlanError(f"source repeat 1 is not complete in its call log ({len(ok & wanted)} of {len(wanted)} calls ok)")
    return {"plan": source_plan, "plan_hash": source_plan["plan_hash"], "next_repeat": 2,
            "outputs_sha256": sha256_file(outputs_path), "calls_sha256": sha256_file(calls_path),
            "repeat_one_lines": repeat_one, "excluded_log_lines": len(lines) - len(repeat_one)}


def _check_compatible(source_plan, plan, source_id):
    """The continuation must differ from its source only in the runner file, the request timeout, the repeats it covers
    and its own description; anything else would make the repeats incomparable."""
    changed = []
    for section in ("dataset", "model", "prompts", "source_files"):
        changed += _diff(source_plan[section], plan[section], section + ".")
    changed += _diff({k: v for k, v in source_plan["code"].items() if k != "runner.py"},
                     {k: v for k, v in plan["code"].items() if k != "runner.py"}, "code.")
    skip = ("repeats", "first_repeat")
    changed += _diff({k: v for k, v in source_plan["parameters"].items() if k not in skip},
                     {k: v for k, v in plan["parameters"].items() if k not in skip}, "parameters.")
    changed += _diff({k: v for k, v in source_plan["evaluation_overrides"].items() if k != "request_timeout"},
                     {k: v for k, v in plan["evaluation_overrides"].items() if k != "request_timeout"},
                     "evaluation_overrides.")
    if changed:
        raise PlanError(f"the continuation is incompatible with source run {source_id}; changed: " + ", ".join(changed))


def _add_continuation(plan, source_id, results_dir):
    source = verify_source(results_dir, source_id)
    _check_compatible(source["plan"], plan, source_id)
    first = source["next_repeat"]
    last = first + plan["parameters"]["repeats"] - 1
    plan["parameters"]["first_repeat"] = first
    plan["continues"] = {
        "run_id": source_id, "plan_hash": source["plan_hash"], "repeats_reused": [first - 1],
        "outputs_rep1_sha256": source["outputs_sha256"], "calls_jsonl_sha256": source["calls_sha256"],
        "excluded_source_log_lines": source["excluded_log_lines"],
        "excluded_note": "Log lines of the source run outside repeat 1 belong to an interrupted repeat; they are "
                         "excluded from every result and count.",
    }
    timeout = plan["evaluation_overrides"]["request_timeout"]
    plan["conditions"].append(
        f"Repeat {first - 1} comes from source run {source_id}, which set no explicit client request timeout; repeats "
        f"{first}-{last} of this run use request_timeout={timeout} s. The repeats therefore differ in this one client "
        f"setting; no other setting, prompt, dataset file or scoring file differs.")
    plan["limitations"].append(
        f"Repeats are split over two runs ({source_id}: repeat {first - 1}; this run: repeats {first}-{last}). "
        f"{source['excluded_log_lines']} call-log line(s) of the interrupted source run are excluded.")


def _diff(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        found = []
        for key in sorted(set(saved) | set(current)):
            if key not in saved or key not in current:
                found.append(f"{prefix}{key}")
            else:
                found.extend(_diff(saved[key], current[key], f"{prefix}{key}."))
        return found
    return [] if saved == current else [prefix.rstrip(".")]


def plan_dir(results_dir, run_id):
    return os.path.join(results_dir, run_id)


def write_new(path, text):
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError:
        raise PlanError(f"{os.path.basename(path)} already exists")
    except OSError as error:
        raise PlanError(f"cannot write {os.path.basename(path)} ({error.__class__.__name__})")


def write_plan(plan, results_dir):
    """Write plan.json. An identical existing plan is left alone; a different one is refused. Returns (path, created)."""
    folder = plan_dir(results_dir, plan["run_id"])
    path = os.path.join(folder, "plan.json")
    text = cli.to_json(plan)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                same = handle.read() == text
        except OSError:
            same = False
        if same:
            return path, False
        raise PlanError(f"a different plan already exists for run id {plan['run_id']!r}; choose another --run-id")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as error:
        raise PlanError(f"cannot create the run folder ({error.__class__.__name__})")
    write_new(path, text)
    return path, True


def load_plan(results_dir, run_id):
    if not RUN_ID_PATTERN.match(run_id or ""):
        raise UsageError("invalid run id")
    path = os.path.join(plan_dir(results_dir, run_id), "plan.json")
    try:
        plan, _raw = dataset_module.read_json_strict(path)
    except ValueError as error:
        raise PlanError(f"plan file {error}")
    if plan.get("plan_version") != PLAN_VERSION or plan.get("run_id") != run_id:
        raise PlanError("plan file has an unexpected version or run id")
    if plan.get("plan_hash") != _plan_hash(plan):
        raise PlanError("plan file does not match its own plan_hash (it was edited)")
    return plan


def validate_frozen(plan, root=ROOT, scenarios_dir=None, expected_dir=None, results_dir=None):
    """Rebuild the plan from the current files with the saved parameters and compare every field. Raises PlanError
    naming each section that changed. Called by `run` before any model call."""
    params = plan["parameters"]
    extra = {}
    if plan.get("continues") is not None:
        if results_dir is None:
            raise PlanError("a continuation plan needs the results directory to verify its source run")
        extra = {"continues_from": plan["continues"]["run_id"], "results_dir": results_dir}
    try:
        current = build_plan(params["repeats"], params["pace_seconds"], params["max_retries"],
                             params["retry_backoff_seconds"], params["abort_after_consecutive_failures"],
                             plan["run_id"], root, scenarios_dir, expected_dir, **extra)
    except UsageError as error:
        raise PlanError(f"saved plan parameters are invalid: {error}")
    changed = _diff(plan, current)
    if changed:
        raise PlanError("the saved plan no longer matches the current files; changed: " + ", ".join(changed))
    return current


# ---------------------------------------------------------------------------------------------------------------
# Live model access (the only place a model client is created)
# ---------------------------------------------------------------------------------------------------------------

def make_groq_model(settings, api_key):
    """Create the model client with the settings frozen in the plan (the production ChatGroq arguments)."""
    from langchain_groq import ChatGroq  # imported only by a live run
    return ChatGroq(model=settings["model"], api_key=api_key, temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"], reasoning_effort=settings["reasoning_effort"],
                    max_retries=settings["max_retries"],   # 0: the runner's own retry policy is the only one
                    timeout=settings["request_timeout"])   # seconds per request; the library default is no timeout


def is_transport_failure(error):
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429) or status >= 500
    return isinstance(error, (TimeoutError, ConnectionError)) or type(error).__name__ in RETRYABLE_NAMES


class CallResult:
    def __init__(self, ok, text=None):
        self.ok, self.text = ok, text


class Executor:
    """Makes the model calls of one run and logs every attempt (no response text and no error message)."""

    def __init__(self, plan, model, log_handle, sleep):
        params = plan["parameters"]
        self.model, self.log, self.sleep = model, log_handle, sleep
        self.pace, self.max_retries = params["pace_seconds"], params["max_retries"]
        self.backoff, self.abort_after = params["retry_backoff_seconds"], params["abort_after_consecutive_failures"]
        self.sequence = 0
        self.logical_calls = 0
        self.consecutive_failures = 0
        self.tally = {stage: {"logical_calls": 0, "attempts": 0, "ok": 0, "failed": 0, "skipped": 0} for stage in STAGES}
        self.finish_reasons = {stage: {} for stage in STAGES}
        self.configured_names_seen = set()

    def _log(self, **fields):
        self.sequence += 1
        self.log.write(canonical_json(dict(sequence=self.sequence, **fields)) + "\n")
        self.log.flush()

    def skip(self, stage, repeat, sid, reason):
        self.tally[stage]["skipped"] += 1
        self._log(repeat=repeat, scenario_id=sid, stage=stage, attempt=0, status="skipped", reason=reason)

    def call(self, stage, repeat, sid, prompt):
        if self.logical_calls > 0:
            self.sleep(self.pace)
        self.logical_calls += 1
        self.tally[stage]["logical_calls"] += 1
        outcome = CallResult(False)
        for attempt in range(1, self.max_retries + 2):
            self.tally[stage]["attempts"] += 1
            try:
                response = self.model.invoke(prompt)
            except Exception as error:
                retryable = is_transport_failure(error)
                status = getattr(error, "status_code", None)
                self._log(repeat=repeat, scenario_id=sid, stage=stage, attempt=attempt, status="error",
                          error_class=type(error).__name__, status_code=status if isinstance(status, int) else None,
                          retryable=retryable)
                if retryable and attempt <= self.max_retries:
                    self.sleep(self.backoff)
                    continue
                break
            content = getattr(response, "content", None)
            metadata = getattr(response, "response_metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            finish = metadata.get("finish_reason") if isinstance(metadata.get("finish_reason"), str) else None
            model_name = metadata.get("model_name") if isinstance(metadata.get("model_name"), str) else None
            if not isinstance(content, str):
                self._log(repeat=repeat, scenario_id=sid, stage=stage, attempt=attempt, status="invalid_content",
                          finish_reason=finish, configured_model_name=model_name)
                break
            self._log(repeat=repeat, scenario_id=sid, stage=stage, attempt=attempt, status="ok", finish_reason=finish,
                      configured_model_name=model_name, content_chars=len(content))
            if model_name:
                self.configured_names_seen.add(model_name)
            key = finish or "none"
            self.finish_reasons[stage][key] = self.finish_reasons[stage].get(key, 0) + 1
            outcome = CallResult(True, content)
            break
        if outcome.ok:
            self.tally[stage]["ok"] += 1
            self.consecutive_failures = 0
        else:
            self.tally[stage]["failed"] += 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.abort_after:
                raise AbortRun(f"{self.consecutive_failures} consecutive failed calls")
        return outcome


def gather_environment():
    from importlib import metadata
    packages = {}
    for name in ("langchain-groq", "langchain-core", "groq", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except Exception:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def execute_run(plan, dataset, model, run_dir, sleep=time.sleep, now=utc_now, environment=None, root=ROOT):
    """Make every planned call in the planned order and write the run artifacts. Returns the manifest dict.

    Failure isolation: E3 is attempted for every scenario whatever happened to E1 and E2."""
    templates = load_prompt_templates(root)
    production = stages_module.load_parser(root)
    params = plan["parameters"]
    manifest = {"manifest_version": MANIFEST_VERSION, "run_id": plan["run_id"], "plan_hash": plan["plan_hash"],
                "started_at": now(), "environment": environment if environment is not None else gather_environment(),
                "status": "completed", "abort_reason": None, "completed_repeats": [], "normalization_errors": 0}
    status = None
    with open(os.path.join(run_dir, "calls.jsonl"), "x", encoding="utf-8", newline="\n") as log:
        executor = Executor(plan, model, log, sleep)
        try:
            first_repeat = params.get("first_repeat", 1)
            for repeat in range(first_repeat, first_repeat + params["repeats"]):
                outputs = {}
                for sid in plan["dataset"]["ids"]:
                    inputs = dataset["scenarios"][sid]["inputs"]
                    jd, resume = inputs["job_description"], inputs["resume_text"]
                    entry = {"e1": None, "e2_raw": None, "e2_normalized": None, "e3_raw": None}
                    first = executor.call("e1", repeat, sid, render_prompt(templates["e1"], job_description=jd))
                    if first.ok:
                        job_skills = clean_response(first.text)
                        entry["e1"] = job_skills
                        second = executor.call("e2", repeat, sid, render_prompt(
                            templates["e2"], job_description=jd, job_skills=job_skills, resume_content=resume))
                        if second.ok:
                            entry["e2_raw"] = second.text
                            try:
                                entry["e2_normalized"] = production.build_prioritized_missing_skills(
                                    second.text, jd, job_skills, resume)[0]
                            except Exception:
                                manifest["normalization_errors"] += 1
                    else:
                        executor.skip("e2", repeat, sid, "e1_failed")
                    third = executor.call("e3", repeat, sid, render_prompt(
                        templates["e3"], job_description=jd, resume_content=resume))
                    if third.ok:
                        entry["e3_raw"] = clean_response(third.text)
                    outputs[sid] = entry
                write_new(os.path.join(run_dir, f"outputs_rep{repeat}.json"),
                          cli.to_json({"format_version": cli.OUTPUTS_VERSION, "outputs": outputs}))
                manifest["completed_repeats"].append(repeat)
        except AbortRun as stop:
            manifest["status"], manifest["abort_reason"] = "aborted", str(stop)
    manifest["finished_at"] = now()
    manifest["configured_model_names"] = sorted(executor.configured_names_seen)
    manifest["model_name_note"] = ("The client library echoes the configured model name in the response metadata; it is "
                                   "not an identifier verified or reported independently by the provider.")
    manifest["calls"] = executor.tally
    manifest["finish_reasons"] = executor.finish_reasons
    manifest["outputs_sha256"] = {f"outputs_rep{n}.json": sha256_file(os.path.join(run_dir, f"outputs_rep{n}.json"))
                                  for n in manifest["completed_repeats"]}
    write_new(os.path.join(run_dir, "manifest.json"), cli.to_json(manifest))
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# Summarize (offline)
# ---------------------------------------------------------------------------------------------------------------

def _metric_row(metric):
    return {"numerator": metric.get("numerator"), "denominator": metric.get("denominator"), "rate": metric.get("rate"),
            "available": bool(metric.get("available")), "unavailable_reason": metric.get("unavailable_reason"),
            "per_scenario": metric.get("per_scenario", [])}


def aggregate_across_repeats(reports):
    """reports: {repeat number: CLI report}. Shows each repeat's own figures and the range of the available rates.
    No pooling across repeats and no new metric."""
    table = {}
    for repeat in sorted(reports):
        report = reports[repeat]
        groups = []
        if report.get("score"):
            for stage, metrics in report["score"]["aggregate"].items():
                groups.append((stage, metrics))
        if report.get("e3"):
            groups.append(("e3", report["e3"]["aggregate"]["metrics"]))
        for stage, metrics in groups:
            for metric in metrics:
                key = (stage, metric["metric"])
                row = table.setdefault(key, {"stage": stage, "metric": metric["metric"],
                                             "direction": metric.get("direction"), "per_repeat": {}})
                row["per_repeat"][repeat] = _metric_row(metric)
    result = []
    for row in table.values():
        rates = [r["rate"] for r in row["per_repeat"].values() if r["available"] and r["rate"] is not None]
        result.append({"stage": row["stage"], "metric": row["metric"], "direction": row["direction"],
                       "screening_only": row["stage"] == "e3",
                       "per_repeat": [dict(repeat=n, **row["per_repeat"][n]) for n in sorted(row["per_repeat"])],
                       "repeats_available": len(rates), "rate_min": min(rates) if rates else None,
                       "rate_max": max(rates) if rates else None})
    return result


def summarize_calls(lines):
    """Per-stage outcome counts from the calls.jsonl lines (already parsed)."""
    table = {stage: {"logical_calls_ok": 0, "logical_calls_failed": 0, "skipped": 0, "attempts": 0, "retried_attempts": 0,
                     "finish_reasons": {}} for stage in STAGES}
    ok_keys, failed_keys = set(), set()
    for line in lines:
        stage = line.get("stage")
        if stage not in table:
            continue
        row = table[stage]
        key = (line["repeat"], line["scenario_id"], stage)
        if line["status"] == "skipped":
            row["skipped"] += 1
            continue
        row["attempts"] += 1
        if line["attempt"] > 1:
            row["retried_attempts"] += 1
        if line["status"] == "ok":
            ok_keys.add(key)
            reason = line.get("finish_reason") or "none"
            row["finish_reasons"][reason] = row["finish_reasons"].get(reason, 0) + 1
        else:
            failed_keys.add(key)
    for stage in STAGES:
        table[stage]["logical_calls_ok"] = sum(1 for k in ok_keys if k[2] == stage)
        table[stage]["logical_calls_failed"] = sum(1 for k in failed_keys - ok_keys if k[2] == stage)
    return table


def render_markdown(summary):
    lines = [f"# Pipeline evaluation summary: {summary['run_id']}", "",
             f"- plan hash: `{summary['plan_hash']}`",
             f"- run status: {summary['run_status']}; repeats summarised: {summary['repeats_summarized']}",
             f"- configured model: {summary['model'].get('model')} (name echoed by the client library, "
             f"not independently verified by the provider; recorded: "
             f"{', '.join(summary['configured_model_names']) or 'none'})", ""]
    extra = summary.get("continuation")
    if extra:
        lines += ["## Provenance and conditions", "",
                  f"- repeat 1 comes from run `{extra['source_run_id']}` (plan hash `{extra['source_plan_hash']}`), read "
                  f"unchanged; the other repeats come from this run",
                  "- client request timeout: " + "; ".join(f"repeat {n}: {t}" for n, t in
                                                         sorted(extra["client_request_timeout"].items())),
                  f"- {extra['excluded_source_log_lines']} call-log line(s) of the interrupted source run (outside "
                  f"repeat 1) are excluded", ""]
    lines += ["## Metrics", "", "Each repeat shows numerator/denominator; the range is over repeats with an available rate.",
              "E3 rows are deterministic screens only (zero flags is not proof of correctness).", ""]
    repeats = summary["repeats"]
    lines.append("| stage | metric | " + " | ".join(f"rep {n}" for n in repeats) + " | range |")
    lines.append("|---|---|" + "---|" * len(repeats) + "---|")
    for row in summary["metrics"]:
        cells = []
        by_repeat = {r["repeat"]: r for r in row["per_repeat"]}
        for n in repeats:
            r = by_repeat.get(n)
            if r is None:
                cells.append("-")
            elif r["available"]:
                cells.append(f"{r['numerator']}/{r['denominator']}")
            else:
                cells.append("n/a")
        span = "n/a" if row["rate_min"] is None else f"{row['rate_min']} to {row['rate_max']}"
        label = row["metric"] + (" (screen)" if row["screening_only"] else "")
        lines.append(f"| {row['stage']} | {label} | " + " | ".join(cells) + f" | {span} |")
    lines += ["", "## Call outcomes", "", "| stage | ok | failed | skipped | attempts | retried | finish reasons |",
              "|---|---|---|---|---|---|---|"]
    for stage in STAGES:
        row = summary["call_outcomes"][stage]
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(row["finish_reasons"].items())) or "-"
        lines.append(f"| {stage} | {row['logical_calls_ok']} | {row['logical_calls_failed']} | {row['skipped']} | "
                     f"{row['attempts']} | {row['retried_attempts']} | {reasons} |")
    lines += ["", "## Warnings per repeat", ""]
    lines += [f"- repeat {n}: {count} warning(s)" for n, count in sorted(summary["warning_counts"].items())]
    lines += ["", "## Limitations", ""] + [f"- {text}" for text in summary["limitations"]] + [""]
    return "\n".join(lines)


def summarize_run(results_dir, run_id, force=False):
    plan = load_plan(results_dir, run_id)
    folder = plan_dir(results_dir, run_id)
    own = sorted(int(m.group(1)) for m in (re.match(r"^outputs_rep(\d+)\.json$", n) for n in os.listdir(folder)) if m)
    if not own:
        raise PlanError("no outputs_repN.json files in the run folder; nothing to summarize")
    continues, source = plan.get("continues"), None
    if continues is not None:
        source = verify_source(results_dir, continues["run_id"])
        if (source["plan_hash"], source["outputs_sha256"], source["calls_sha256"]) != (
                continues["plan_hash"], continues["outputs_rep1_sha256"], continues["calls_jsonl_sha256"]):
            raise PlanError("the source run no longer matches the continuation plan (its plan, outputs or call log changed)")
        if 1 in own:
            raise PlanError("a continuation run folder must not hold outputs_rep1.json (repeat 1 comes from the source run)")
    numbers = sorted(own + ([1] if source else []))
    source_folder = plan_dir(results_dir, continues["run_id"]) if source else None

    def outputs_file(number):
        return os.path.join(source_folder if source and number == 1 else folder, f"outputs_rep{number}.json")
    manifest = None
    manifest_path = os.path.join(folder, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest, _ = dataset_module.read_json_strict(manifest_path)
        except ValueError as error:
            raise PlanError(f"manifest {error}")
        if manifest.get("plan_hash") != plan["plan_hash"]:
            raise PlanError("manifest.json belongs to a different plan")
    reports, report_hashes, warning_counts = {}, {}, {}
    for number in numbers:
        outputs_path = outputs_file(number)
        report_path = os.path.join(folder, f"report_rep{number}.json")
        rows_path = os.path.join(folder, f"review_rows_rep{number}.json")
        argv = ["--outputs", outputs_path, "--output-report", report_path, "--review-rows", rows_path]
        if force:
            argv.append("--force")
        out, err = _StringSink(), _StringSink()
        code = cli.main(argv, stdout=out, stderr=err)
        if code != 0:
            raise PlanError(f"the evaluation CLI failed for repeat {number} (exit {code}): {err.text.strip()[:300]}")
        with open(report_path, "r", encoding="utf-8") as handle:
            reports[number] = json.load(handle)
        report_hashes[f"report_rep{number}.json"] = sha256_file(report_path)
        warning_counts[number] = len(reports[number]["warnings"])
    calls = []
    calls_path = os.path.join(folder, "calls.jsonl")
    if os.path.exists(calls_path):
        with open(calls_path, "r", encoding="utf-8") as handle:
            calls = [json.loads(line) for line in handle if line.strip()]
    if source:
        calls += source["repeat_one_lines"]
    names = set(manifest.get("configured_model_names", [])) if manifest else set()
    if source:
        names |= {line["configured_model_name"] for line in source["repeat_one_lines"]
                  if line.get("status") == "ok" and isinstance(line.get("configured_model_name"), str)}
    summary = {
        "summary_version": SUMMARY_VERSION, "run_id": run_id, "plan_hash": plan["plan_hash"],
        "run_status": manifest["status"] if manifest else "no manifest",
        "repeats": numbers, "repeats_summarized": len(numbers), "repeats_planned": plan["parameters"].get("first_repeat", 1) - 1 + plan["parameters"]["repeats"],
        "model": plan["model"], "configured_model_names": sorted(names),
        "dataset_ids": plan["dataset"]["ids"],
        "metrics": aggregate_across_repeats(reports),
        "call_outcomes": summarize_calls(calls),
        "warning_counts": warning_counts,
        "outputs_sha256": {f"outputs_rep{n}.json": sha256_file(outputs_file(n)) for n in numbers},
        "report_sha256": report_hashes,
        "conditions": plan["conditions"],
        "limitations": plan["limitations"] + cli_limitations(reports),
    }
    if source:
        timeout = plan["evaluation_overrides"]["request_timeout"]
        summary["continuation"] = {
            "source_run_id": continues["run_id"], "source_plan_hash": continues["plan_hash"],
            "source_outputs_rep1_sha256": continues["outputs_rep1_sha256"],
            "source_calls_jsonl_sha256": continues["calls_jsonl_sha256"],
            "repeat_sources": {str(n): (continues["run_id"] if n == 1 else run_id) for n in numbers},
            "client_request_timeout": {str(n): ("none set (library default)" if n == 1 else f"{timeout} s")
                                       for n in numbers},
            "excluded_source_log_lines": continues["excluded_source_log_lines"],
        }
    texts = {"summary.json": cli.to_json(summary), "summary.md": render_markdown(summary)}
    for name, text in texts.items():
        path = os.path.join(folder, name)
        if os.path.exists(path) and not force:
            raise PlanError(f"{name} already exists (use --force)")
    for name, text in texts.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return summary


def cli_limitations(reports):
    seen = []
    for number in sorted(reports):
        for text in reports[number].get("limitations", []):
            if text not in seen:
                seen.append(text)
    return seen


class _StringSink:
    def __init__(self):
        self.text = ""

    def write(self, value):
        self.text += value


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="python -B -m eval.pipeline.runner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="offline: freeze a run plan")
    plan.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    plan.add_argument("--pace-seconds", type=int, default=DEFAULT_PACE_SECONDS)
    plan.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    plan.add_argument("--retry-backoff", type=int, default=DEFAULT_RETRY_BACKOFF)
    plan.add_argument("--abort-after", type=int, default=DEFAULT_ABORT_AFTER)
    plan.add_argument("--run-id", default=None)
    plan.add_argument("--continues-from", default=None, metavar="RUN_ID",
                      help="add repeats 2 and 3 after the complete repeat 1 of this earlier run (use --repeats 2)")
    plan.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    plan.add_argument("--root", default=ROOT, help="project root holding src/ and frontend/ (for tests)")
    run = sub.add_parser("run", help="LIVE: make the planned model calls")
    run.add_argument("--plan", required=True, metavar="RUN_ID")
    run.add_argument("--live", action="store_true")
    run.add_argument("--confirm-calls", type=int, default=None)
    run.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    run.add_argument("--root", default=ROOT, help="project root holding src/ and frontend/ (for tests)")
    summarize = sub.add_parser("summarize", help="offline: run the CLI on saved outputs and aggregate")
    summarize.add_argument("run_id")
    summarize.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    summarize.add_argument("--force", action="store_true")
    return parser


def plan_lines(plan, path, created):
    schedule, params = plan["schedule"], plan["parameters"]
    return [
        f"run id: {plan['run_id']}", f"plan file: {path} ({'created' if created else 'already existed, identical'})",
        f"plan hash: {plan['plan_hash']}", f"scenarios: {len(plan['dataset']['ids'])}, repeats: {params['repeats']}",
        f"planned_calls: {schedule['planned_calls']}  (E1 {schedule['planned_calls_by_stage']['e1']}, "
        f"E2 {schedule['planned_calls_by_stage']['e2']}, E3 {schedule['planned_calls_by_stage']['e3']})",
        f"client request timeout: {plan['evaluation_overrides']['request_timeout']} s per request (evaluation override)",
        *([f"continues run {plan['continues']['run_id']} (plan hash {plan['continues']['plan_hash']}); this plan runs "
           f"repeats {params['first_repeat']}-{params['first_repeat'] + params['repeats'] - 1}"]
          if plan.get("continues") else []),
        f"max HTTP requests if every call is retried once: {schedule['max_requests_with_runner_retry']} "
        f"(client library retries disabled: max_retries=0)",
        f"pacing: {params['pace_seconds']} s between calls (minimum {schedule['minimum_pacing_seconds']} s in total)",
        f"model: {plan['model']['model']} (temperature {plan['model']['temperature']}, max_tokens "
        f"{plan['model']['max_tokens']}, reasoning_effort {plan['model']['reasoning_effort']})",
        "NOT executed. A live run needs separate authorisation:",
        f"  python -B -m eval.pipeline.runner run --plan {plan['run_id']} --live --confirm-calls {schedule['planned_calls']}",
    ]


def _cmd_plan(args, out):
    plan = build_plan(args.repeats, args.pace_seconds, args.max_retries, args.retry_backoff, args.abort_after,
                      args.run_id, args.root, continues_from=args.continues_from, results_dir=args.results_dir)
    path, created = write_plan(plan, args.results_dir)
    out.write("\n".join(plan_lines(plan, path, created)) + "\n")
    return EXIT_OK


def _cmd_run(args, out, err, model_factory, sleep, now, environment):
    if not args.live:
        raise UsageError("a live run needs --live")
    plan = load_plan(args.results_dir, args.plan)
    expected = plan["schedule"]["planned_calls"]
    if args.confirm_calls != expected:
        raise UsageError(f"--confirm-calls must equal the plan's planned_calls ({expected})")
    validate_frozen(plan, args.root, results_dir=args.results_dir)
    folder = plan_dir(args.results_dir, plan["run_id"])
    first_name = f"outputs_rep{plan['parameters'].get('first_repeat', 1)}.json"
    if any(os.path.exists(os.path.join(folder, name)) for name in ("calls.jsonl", "manifest.json", first_name)):
        raise UsageError("this run folder already holds results; create a new plan with another --run-id")
    api_key = os.environ.get(API_KEY_VARIABLE)
    if not api_key:
        raise UsageError(f"{API_KEY_VARIABLE} is not set in the environment (no .env file is read)")
    dataset = dataset_module.load_dataset()
    model = model_factory(dict(plan["model"], **plan["evaluation_overrides"]), api_key)
    manifest = execute_run(plan, dataset, model, folder, sleep=sleep, now=now, environment=environment, root=args.root)
    out.write(f"run {plan['run_id']}: {manifest['status']}; completed repeats {manifest['completed_repeats']}\n")
    if manifest["status"] != "completed":
        err.write(f"aborted: {manifest['abort_reason']}\n")
        return EXIT_ERROR
    out.write(f"next (offline): python -B -m eval.pipeline.runner summarize {plan['run_id']}\n")
    return EXIT_OK


def main(argv=None, stdout=None, stderr=None, model_factory=None, sleep=None, now=None, environment=None):
    out, err = stdout or sys.stdout, stderr or sys.stderr
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as request:
        return request.code if isinstance(request.code, int) else EXIT_USAGE
    try:
        if args.command == "plan":
            return _cmd_plan(args, out)
        if args.command == "run":
            return _cmd_run(args, out, err, model_factory or make_groq_model, sleep or time.sleep, now or utc_now,
                            environment)
        summary = summarize_run(args.results_dir, args.run_id, args.force)
        out.write(f"summarized {summary['repeats_summarized']} repeat(s): summary.json and summary.md written\n")
        return EXIT_OK
    except UsageError as error:
        err.write(f"error: {error}\n")
        return EXIT_USAGE
    except PlanError as error:
        err.write(f"error: {error}\n")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
