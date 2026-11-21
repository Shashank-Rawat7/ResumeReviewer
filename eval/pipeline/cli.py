"""Offline command-line runner for the whole-pipeline evaluation (Step 5).

A THIN integration layer. It reads a file of PRECOMPUTED outputs, loads and verifies the frozen dataset, calls the
existing scorers (score.py for E1 / E2 / course eligibility, e3.py for E3) and writes one deterministic JSON report.
It adds no metric and re-implements no scoring.

WHAT IT NEVER DOES
  * call a model, the network or any other program; run the application, the parsing pipeline or E2 normalisation;
  * read secrets, a protected resume artifact, or any case file outside the two dataset folders;
  * write anywhere except the two optional files named on the command line (--output-report, --review-rows), and
    never over an existing file without --force, never into the dataset folders, never over the outputs file.

USE
  python -B -m eval.pipeline.cli --outputs OUTPUTS.json [--stages e1,e2,course,e3] [--output-report R.json]
                                 [--review-rows ROWS.json] [--strict] [--force]
  python -B -m eval.pipeline.cli --validate-dataset
  python -B -m eval.pipeline.cli --template

OUTPUTS FILE  {"format_version": "pipeline-outputs-1", "outputs": {scenario_id: {"e1": ..., "e2_normalized": ...,
              "e2_raw": ..., "e3_raw": ...}}}   (the bare {scenario_id: {...}} form is accepted too).
              A key that is absent or null is a MISSING output; it is reported, never counted as correct.

STAGES  e1, e2, course, e3. Only the selected stages' scoring functions run. "course" needs e2_normalized but does not
        run E2 scoring, and e3 never looks at E1/E2 data.

EXIT CODES  0 completed | 1 data-integrity or internal error | 2 usage error | 3 --strict and the report has warnings.
Standard library only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLI_REPORT_VERSION = "pipeline-cli-1"
OUTPUTS_VERSION = "pipeline-outputs-1"
ROWS_VERSION = "pipeline-review-rows-1"
STAGES = ("e1", "e2", "course", "e3")
SCORE_STAGES = ("e1", "e2", "course")
STAGE_INPUT_KEY = {"e1": "e1", "e2": "e2_normalized", "course": "e2_normalized", "e3": "e3_raw"}
KNOWN_KEYS = ("e1", "e2_normalized", "e2_raw", "e3_raw")
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_STRICT = 0, 1, 2, 3


def _load(name, filename):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset_module = _load("pipeline_dataset", "dataset.py")
score = _load("pipeline_score", "score.py")
e3 = _load("pipeline_e3", "e3.py")


class CliError(Exception):
    """A problem that ends the run. `code` is the exit code."""

    def __init__(self, message, code=EXIT_ERROR):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------------------------------------------
# Reading the outputs file
# ---------------------------------------------------------------------------------------------------------------

def read_outputs_file(path):
    """Return (outputs dict, envelope 'wrapped'|'bare', sha256 of the file bytes). Raises CliError (exit 1)."""
    try:
        data, raw = dataset_module.read_json_strict(path)
    except ValueError as error:
        raise CliError(f"outputs file {error}")
    digest = hashlib.sha256(raw).hexdigest()
    if "format_version" in data:
        if data["format_version"] != OUTPUTS_VERSION:
            raise CliError(f"outputs file: format_version is {data['format_version']!r}, expected {OUTPUTS_VERSION!r}")
        extra = sorted(set(data) - {"format_version", "outputs"})
        if extra:
            raise CliError(f"outputs file: unexpected top-level key(s) {extra}")
        outputs = data.get("outputs")
        if not isinstance(outputs, dict):
            raise CliError("outputs file: 'outputs' must be a JSON object keyed by scenario id")
        envelope = "wrapped"
    else:
        outputs, envelope = data, "bare"
    for sid, entry in outputs.items():
        if not isinstance(entry, dict):
            raise CliError(f"outputs file: the entry for scenario {sid!r} must be a JSON object")
    return outputs, envelope, digest


def split_outputs(outputs, ids):
    """Separate known from unknown scenario ids. Unknown ids are excluded from scoring and become warnings."""
    valid = set(ids)
    known = {sid: entry for sid, entry in outputs.items() if sid in valid}
    unknown = sorted(sid for sid in outputs if sid not in known)
    return known, unknown


def input_warnings_and_missing(known, ids, unknown, stages):
    """Warnings about the input itself, plus which selected stages lack an output for which scenarios."""
    warnings = [f"input: unknown scenario id {sid!r} was ignored and not scored" for sid in unknown]
    for sid in sorted(known):
        for key in sorted(set(known[sid]) - set(KNOWN_KEYS)):
            warnings.append(f"input: {sid}: unrecognised key {key!r} was ignored")
    missing = {}
    for stage in stages:
        key = STAGE_INPUT_KEY[stage]
        absent = [sid for sid in ids if (known.get(sid) or {}).get(key) is None]
        missing[stage] = absent
        if absent:
            warnings.append(f"input: stage {stage!r} has no output ({key!r}) for {len(absent)} of {len(ids)} "
                            f"scenario(s): {', '.join(absent)}")
    return warnings, missing


# ---------------------------------------------------------------------------------------------------------------
# Running the selected stages
# ---------------------------------------------------------------------------------------------------------------

def run_score_stages(dataset, outputs, stages):
    """Run score.score_e1 / score_e2 / score_course for the SELECTED stages only, per scenario, and pool them with
    score._aggregate_metric. With all three selected the result equals score.score_dataset (a test pins that)."""
    ids = list(dataset["ids"])
    scenarios, pooled = [], {stage: {} for stage in stages}
    for sid in ids:
        supplied = outputs.get(sid) or {}
        label = dataset["labels"][sid]
        records = {}
        if "e1" in stages:
            records["e1"] = score.score_e1(label, supplied.get("e1"))
        if "e2" in stages:
            records["e2"] = score.score_e2(label, supplied.get("e2_normalized"), supplied.get("e2_raw"))
        if "course" in stages:
            records["course"] = score.score_course(label, supplied.get("e2_normalized"))
        for stage, record in records.items():
            for metric in record["metrics"]:
                pooled[stage].setdefault(metric["metric"], []).append((sid, metric))
        scenarios.append({"scenario_id": sid, "labels_version": label.get("labels_version"),
                          "output_supplied": sid in outputs, "stages": records})
    # report_version and limitations come from score.py itself: an empty dataset scores nothing.
    frame = score.score_dataset({"ids": [], "labels": {}, "scenarios": {}, "verified_hashes": {}}, {})
    warnings = []
    if "e2" in stages:
        warnings = [f"e2:{s['scenario_id']}: {w}" for s in scenarios
                    for w in s["stages"]["e2"]["observations"].get("warnings", [])]
    return {
        "report_version": frame["report_version"],
        "dataset": {"ids": ids, "verified_hashes": dict(dataset.get("verified_hashes") or {})},
        "scenarios": scenarios,
        "aggregate": {stage: [score._aggregate_metric(items) for items in pooled[stage].values()] for stage in stages},
        "unknown_output_ids": [],
        "warnings": warnings,
        "limitations": frame["limitations"],
    }


def _describe(error):
    return f"{error.__class__.__name__}: {str(error)[:200]}"


def assemble_report(dataset, known, unknown, stages, envelope, digest, ids):
    """Build the full report dict. A scorer crash is recorded under 'errors' and the other group still runs."""
    warnings, missing = input_warnings_and_missing(known, ids, unknown, stages)
    errors = []
    score_part = e3_part = None
    selected_score = tuple(stage for stage in SCORE_STAGES if stage in stages)
    if selected_score:
        try:
            score_part = run_score_stages(dataset, known, selected_score)
            warnings.extend(score_part["warnings"])
        except Exception as error:  # reported, not hidden: the run exits 1
            errors.append({"stages": list(selected_score), "error": _describe(error)})
    if "e3" in stages:
        try:
            e3_part = e3.score_e3_dataset(dataset, known)
            warnings.extend(e3_part["warnings"])
        except Exception as error:
            errors.append({"stages": ["e3"], "error": _describe(error)})
    return {
        "cli_report_version": CLI_REPORT_VERSION,
        "dataset": {"ids": list(ids), "verified_hashes": dict(dataset.get("verified_hashes") or {})},
        "stages_run": list(stages),
        "input": {
            "outputs_file_sha256": digest, "envelope": envelope,
            "scenario_ids_supplied": [sid for sid in ids if sid in known],
            "scenario_ids_without_entry": [sid for sid in ids if sid not in known],
            "unknown_ids_ignored": list(unknown),
            "missing_stage_outputs": missing,
        },
        "score": score_part,
        "e3": e3_part,
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "The report scores only the supplied precomputed outputs. It is not a measurement of a live model.",
            "Stage reports are embedded unchanged from score.py and e3.py; read their own limitations.",
            "E3 results are deterministic screens, not verification that bullets are factually correct.",
        ],
    }


def build_review_rows(dataset, known, ids):
    rows = []
    for sid in ids:
        raw = (known.get(sid) or {}).get("e3_raw")
        rows.extend(e3.build_review_rows(dataset["labels"][sid], dataset["scenarios"].get(sid), raw, sid))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------

def to_json(value):
    """Deterministic text: sorted keys, fixed indent, no NaN. Raises ValueError if the value is not serialisable."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def summary_lines(report):
    lines = [f"dataset: {len(report['dataset']['ids'])} scenario(s), labels verified",
             f"stages run: {', '.join(report['stages_run'])}"]

    def metric_line(prefix, metric):
        if metric.get("available"):
            return f"  {prefix}{metric['metric']}: {metric['numerator']}/{metric['denominator']} rate={metric['rate']}"
        return f"  {prefix}{metric['metric']}: unavailable ({metric.get('unavailable_reason')})"

    if report["score"]:
        for stage in report["stages_run"]:
            for metric in report["score"]["aggregate"].get(stage, []):
                lines.append(metric_line(stage + ".", metric))
    if report["e3"]:
        for metric in report["e3"]["aggregate"]["metrics"]:
            lines.append(metric_line("e3.", metric) + "  [screen only]")
    for error in report["errors"]:
        lines.append(f"ERROR in {', '.join(error['stages'])}: {error['error']}")
    lines.append(f"warnings: {len(report['warnings'])}")
    lines.extend("  - " + w for w in report["warnings"])
    return lines


def _inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def check_write_target(path, label, force, protected_folders, outputs_path):
    if outputs_path and os.path.realpath(path) == os.path.realpath(outputs_path):
        raise CliError(f"{label}: refusing to overwrite the outputs file", EXIT_USAGE)
    for folder in protected_folders:
        if _inside(path, folder):
            raise CliError(f"{label}: refusing to write inside the dataset folders", EXIT_USAGE)
    if os.path.isdir(path):
        raise CliError(f"{label}: is a directory", EXIT_USAGE)
    if os.path.exists(path) and not force:
        raise CliError(f"{label}: file already exists (use --force to overwrite)", EXIT_USAGE)


def write_text(path, text, force):
    try:
        with open(path, "w" if force else "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as error:
        raise CliError(f"cannot write {path}: {error.__class__.__name__}")


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="python -B -m eval.pipeline.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--outputs", metavar="FILE", help="JSON file of precomputed outputs to score")
    mode.add_argument("--validate-dataset", action="store_true", help="load and hash-check the dataset only")
    mode.add_argument("--template", action="store_true", help="print an all-missing outputs skeleton")
    parser.add_argument("--stages", default=",".join(STAGES), help="comma list of: " + ", ".join(STAGES))
    parser.add_argument("--output-report", metavar="FILE", help="write the full JSON report here")
    parser.add_argument("--review-rows", metavar="FILE", help="write the E3 human-review rows here (needs e3)")
    parser.add_argument("--scenarios-dir", metavar="DIR", help="override the scenarios folder (for tests)")
    parser.add_argument("--expected-dir", metavar="DIR", help="override the expected-labels folder (for tests)")
    parser.add_argument("--strict", action="store_true", help="exit 3 when the report has any warning")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing output file")
    return parser


def parse_stages(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    bad = [name for name in names if name not in STAGES]
    if bad or not names:
        raise CliError(f"--stages: unknown or empty stage {bad or text!r}; choose from {', '.join(STAGES)}", EXIT_USAGE)
    return tuple(stage for stage in STAGES if stage in names)  # canonical order, duplicates dropped


def _run(args, out, err):
    scenarios_dir = args.scenarios_dir or dataset_module.DEFAULT_SCENARIOS_DIR
    expected_dir = args.expected_dir or dataset_module.DEFAULT_EXPECTED_DIR
    if (args.validate_dataset or args.template) and (args.output_report or args.review_rows):
        raise CliError("--output-report and --review-rows apply only with --outputs", EXIT_USAGE)
    stages = parse_stages(args.stages)
    if args.review_rows and "e3" not in stages:
        raise CliError("--review-rows needs the e3 stage", EXIT_USAGE)
    for path, label in ((args.output_report, "--output-report"), (args.review_rows, "--review-rows")):
        if path:
            check_write_target(path, label, args.force, (scenarios_dir, expected_dir), args.outputs)
    if args.output_report and args.review_rows and \
            os.path.realpath(args.output_report) == os.path.realpath(args.review_rows):
        raise CliError("--output-report and --review-rows name the same file", EXIT_USAGE)

    try:
        dataset = dataset_module.load_dataset(scenarios_dir, expected_dir)
    except dataset_module.DatasetError as error:
        raise CliError(str(error))
    ids = list(dataset["ids"])

    if args.validate_dataset:
        out.write(to_json({"dataset_ok": True, "ids": ids, "verified_hashes": dataset["verified_hashes"]}))
        return EXIT_OK
    if args.template:
        skeleton = {sid: {key: None for key in KNOWN_KEYS} for sid in ids}
        out.write(to_json({"format_version": OUTPUTS_VERSION, "outputs": skeleton}))
        return EXIT_OK

    outputs, envelope, digest = read_outputs_file(args.outputs)
    known, unknown = split_outputs(outputs, ids)
    report = assemble_report(dataset, known, unknown, stages, envelope, digest, ids)
    try:
        text = to_json(report)
        rows_text = to_json({"format_version": ROWS_VERSION, "rows": build_review_rows(dataset, known, ids)}) \
            if args.review_rows else None
    except ValueError as error:
        raise CliError(f"report is not JSON-serialisable: {error}")

    if args.output_report:
        write_text(args.output_report, text, args.force)
    if args.review_rows:
        write_text(args.review_rows, rows_text, args.force)
    if args.output_report:
        out.write("\n".join(summary_lines(report)) + "\n")
    else:
        out.write(text)
        err.write("\n".join(summary_lines(report)) + "\n")
    if report["errors"]:
        return EXIT_ERROR
    if args.strict and report["warnings"]:
        return EXIT_STRICT
    return EXIT_OK


def main(argv=None, stdout=None, stderr=None):
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else EXIT_USAGE
    try:
        return _run(args, out, err)
    except CliError as error:
        err.write(f"error: {error}\n")
        return error.code


if __name__ == "__main__":
    sys.exit(main())
