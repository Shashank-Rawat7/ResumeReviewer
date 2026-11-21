"""Dataset integrity for the whole-pipeline evaluation (eval/pipeline/scenarios and eval/pipeline/expected).

Standard library only: no application import, no .env, no network, no model, no protected resume artifact.
It READS the two dataset folders and never writes, repairs or regenerates a file.

What it checks
  * every *.json file is valid UTF-8 JSON with no BOM and no duplicate key at any depth;
  * the id inside each file (scenario_id) equals its filename and is unique within its folder;
  * scenarios and expected labels pair one-to-one by id (no missing and no extra file);
  * every expected-label file is FROZEN and carries a valid frozen.sha256 that verifies;
  * scenario files are NOT required to be frozen. Their content_hash only has to be present and either null or a
    well-formed digest; it is not verified (no scenario-freeze rule exists yet).

Freeze convention (the one used to freeze the labels, same normalisation as eval/ats/dataset.py:sha256_file):
  SHA-256 over the file bytes with CRLF turned into LF, with the stored frozen.sha256 value replaced by null.

Use:  dataset = load_dataset()   # raises DatasetError listing every problem found
"""
import hashlib
import json
import os
import re

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENARIOS_DIR = os.path.join(PIPELINE_DIR, "scenarios")
DEFAULT_EXPECTED_DIR = os.path.join(PIPELINE_DIR, "expected")

FROZEN_STATUS = "FROZEN"
_BOM = b"\xef\xbb\xbf"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class DatasetError(Exception):
    """One or more dataset problems. `problems` is a list of strings that each name the file and field."""

    def __init__(self, problems):
        self.problems = list(problems)
        lines = "".join("\n  - " + problem for problem in self.problems)
        super().__init__(f"{len(self.problems)} dataset problem(s):{lines}")


# ---------------------------------------------------------------------------------------------------------------
# Strict JSON reading
# ---------------------------------------------------------------------------------------------------------------

def _no_duplicate_keys(pairs):
    """object_pairs_hook: json.loads would silently keep the last of two equal keys; reject instead."""
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
    return dict(pairs)


def read_json_strict(path):
    """Return (data, raw_bytes). Raises ValueError (message without the path) for any unreadable or invalid file."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as error:
        raise ValueError(f"cannot be read ({error.__class__.__name__})")
    if raw.startswith(_BOM):
        raise ValueError("starts with a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("is not valid UTF-8")
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ValueError as error:  # includes json.JSONDecodeError and the duplicate-key error above
        raise ValueError(f"is not valid JSON: {error}")
    if not isinstance(data, dict):
        raise ValueError("top level is not a JSON object")
    return data, raw


# ---------------------------------------------------------------------------------------------------------------
# Frozen-hash convention
# ---------------------------------------------------------------------------------------------------------------

def frozen_digest(raw_bytes, stored_digest):
    """Digest of a frozen file under the freeze convention, or None when the stored digest is not found exactly once
    as the value of a "sha256" key (then the file cannot be hashed under the convention)."""
    normalised = raw_bytes.replace(b"\r\n", b"\n")
    pattern = re.compile(rb'("sha256"\s*:\s*)"' + re.escape(stored_digest.encode("ascii")) + rb'"')
    if len(pattern.findall(normalised)) != 1:
        return None
    return hashlib.sha256(pattern.sub(rb"\1null", normalised)).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Folder loading
# ---------------------------------------------------------------------------------------------------------------

def _load_folder(directory, kind):
    """Read every *.json in `directory`, in filename order. Returns ({stem: record}, problems).

    A record is {"path", "name", "raw", "data"}. Files that cannot be read or parsed are reported and left out."""
    problems = []
    records = {}
    if not os.path.isdir(directory):
        return records, [f"{kind} folder {directory!r} does not exist"]
    names = sorted(name for name in os.listdir(directory)
                   if name.endswith(".json") and os.path.isfile(os.path.join(directory, name)))
    if not names:
        problems.append(f"{kind} folder {directory!r} contains no .json files")
    declared = {}  # scenario_id -> file name, to reject two files that claim the same id
    for name in names:
        path = os.path.join(directory, name)
        label = f"{kind}/{name}"
        try:
            data, raw = read_json_strict(path)
        except ValueError as error:
            problems.append(f"{label}: {error}")
            continue
        stem = name[: -len(".json")]
        scenario_id = data.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            problems.append(f"{label}: field 'scenario_id' is missing or not a non-empty string")
        else:
            if scenario_id != stem:
                problems.append(f"{label}: field 'scenario_id' is {scenario_id!r} but the filename says {stem!r}")
            if scenario_id in declared:
                problems.append(f"{label}: field 'scenario_id' {scenario_id!r} duplicates {kind}/{declared[scenario_id]}")
            declared[scenario_id] = name
        records[stem] = {"path": path, "name": label, "raw": raw, "data": data}
    return records, problems


def _check_scenario(record, problems):
    """Scenario files stay unfrozen; only the shape of content_hash is checked, and it is not verified."""
    data = record["data"]
    if "content_hash" not in data:
        problems.append(f"{record['name']}: field 'content_hash' is missing")
        return
    value = data["content_hash"]
    if value is not None and not (isinstance(value, str) and _SHA256_HEX.match(value)):
        problems.append(f"{record['name']}: field 'content_hash' must be null or a 64-character lowercase hex digest")


def _check_frozen_label(record, problems):
    """The label must be FROZEN and its stored digest must verify under the freeze convention."""
    data, name = record["data"], record["name"]
    if data.get("status") != FROZEN_STATUS:
        problems.append(f"{name}: field 'status' is {data.get('status')!r}, expected {FROZEN_STATUS!r}")
    frozen = data.get("frozen")
    if not isinstance(frozen, dict):
        problems.append(f"{name}: field 'frozen' is missing or not an object")
        return None
    stored = frozen.get("sha256")
    if not isinstance(stored, str) or not _SHA256_HEX.match(stored):
        problems.append(f"{name}: field 'frozen.sha256' is missing or not a 64-character lowercase hex digest")
        return None
    computed = frozen_digest(record["raw"], stored)
    if computed is None:
        problems.append(f"{name}: field 'frozen.sha256' value does not appear exactly once as a \"sha256\" value, "
                        f"so the file cannot be hashed under the freeze convention")
        return None
    if computed != stored:
        problems.append(f"{name}: field 'frozen.sha256' is {stored} but the file hashes to {computed}")
        return None
    return stored


# ---------------------------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------------------------

def load_dataset(scenarios_dir=None, expected_dir=None):
    """Load and validate the whole dataset. Raises DatasetError listing every problem found.

    Returns {"ids": [...], "scenarios": {id: data}, "labels": {id: data}, "verified_hashes": {id: digest}},
    all in filename order. The returned data are the parsed objects; nothing on disk is touched."""
    scenarios_dir = DEFAULT_SCENARIOS_DIR if scenarios_dir is None else scenarios_dir
    expected_dir = DEFAULT_EXPECTED_DIR if expected_dir is None else expected_dir
    scenarios, problems = _load_folder(scenarios_dir, "scenarios")
    labels, label_problems = _load_folder(expected_dir, "expected")
    problems.extend(label_problems)

    for stem in sorted(set(scenarios) - set(labels)):
        problems.append(f"scenarios/{stem}.json: no matching expected/{stem}.json")
    for stem in sorted(set(labels) - set(scenarios)):
        problems.append(f"expected/{stem}.json: no matching scenarios/{stem}.json")

    for stem in sorted(scenarios):
        _check_scenario(scenarios[stem], problems)
    verified = {}
    for stem in sorted(labels):
        digest = _check_frozen_label(labels[stem], problems)
        if digest is not None:
            verified[stem] = digest

    if problems:
        raise DatasetError(problems)
    ids = sorted(scenarios)
    return {
        "ids": ids,
        "scenarios": {i: scenarios[i]["data"] for i in ids},
        "labels": {i: labels[i]["data"] for i in ids},
        "verified_hashes": {i: verified[i] for i in ids},
    }
