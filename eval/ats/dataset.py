"""Shared loader, renderer, validator and fingerprints for the ATS evaluation dataset (Stage 5, checkpoint C).

Standard library only: no network, no LLM, no provider import, no environment access. The checker module
(eval/ats/checks.py) is loaded by file path because eval/ is not a package; it is exposed as `checks`.

The dataset is 12 SYNTHETIC cases (eval/ats/cases/*.json) plus a fixed technology vocabulary
(eval/ats/tech_vocabulary.json). Nothing here is real resume data.

Splits
  "dev"      the 9 cases whose holdout flag is false (the only split the runner uses by default)
  "holdout"  the 3 cases whose holdout flag is true
  "all"      all 12, in file-name order
  Every call validates ALL case files first (so a broken holdout file cannot hide behind the dev split) and then
  returns only the requested split. Ordering is the sorted file-name order, so it is deterministic.

HOLDOUT LIMITATION (accurate statement): the vocabulary and the absent-skill lists were authored with all 12 cases
visible, holdouts included. The holdouts are therefore blind to RESULTS (the checker was never run on any case when
they were written) but NOT blind to dataset design. Treat three holdouts as a small sanity check, not as proof of
generalisation.

Public API (signatures as implemented)
  render_resume(resume) -> str                      deterministic text: the format the tests pin with a literal
  source_sections(case) -> [{"heading", "bullets"}]  the shape check_sections expects
  source_bullets(case) -> [str]                      all source bullets in resume order
  validate_case(case, filename=None) -> [str]        problems (empty list = valid); never raises on bad input
  validate_dataset(cases) -> [str]                   dataset-level problems (12 cases, 3 holdouts, all categories)
  load_cases(split="dev", cases_dir=None, strict=True) -> [case dict]   raises DatasetError listing every problem
  load_vocabulary(path=None) -> {canonical: [aliases]}   ONLY the "terms" mapping (never the whole file)
  sha256_file(path) -> hex                           CRLF is normalised to LF before hashing
  dataset_fingerprint(split="all", cases_dir=None), vocabulary_fingerprint(path=None), checker_fingerprint()
      -> {"sha256": hex, "covers": [{"path", "sha256"}]}   (what each fingerprint covers is in "covers")
"""
import copy
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

ATS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(ATS_DIR, "..", ".."))
CASES_DIR = os.path.join(ATS_DIR, "cases")
VOCAB_PATH = os.path.join(ATS_DIR, "tech_vocabulary.json")
CHECKS_PATH = os.path.join(ATS_DIR, "checks.py")

SPLITS = ("dev", "holdout", "all")
EXPECTED_CASES = 12
EXPECTED_HOLDOUTS = 3
REQUIRED_CATEGORIES = frozenset({
    "strong_evidence", "vague_evidence", "absent_skill", "metric_heavy",
    "no_metric_trap", "inflation_trap", "fresher_backend_ai", "short_mismatch",
})
TRAP_TYPES = frozenset({
    "metric_preservation", "invented_metric", "absent_skill_leak", "vague_evidence", "scale_claim",
    "overstatement", "job_mismatch", "format_robustness",
})
CASE_KEYS = frozenset({
    "id", "category", "holdout", "title", "provenance", "job_description", "resume", "resume_text",
    "absent_skills", "trap_types", "traps",
})
RESUME_KEYS = frozenset({"name", "sections", "skills_line"})
SECTION_KEYS = frozenset({"heading", "dates", "bullets"})
_ID_RE = re.compile(r"^c\d{2}_[a-z0-9_]+$")


class DatasetError(ValueError):
    """The dataset (or the vocabulary) is malformed. The message lists every problem found."""


def _load_checks():
    module = sys.modules.get("ats_checks")
    if module is None:
        spec = importlib.util.spec_from_file_location("ats_checks", CHECKS_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ats_checks"] = module
        spec.loader.exec_module(module)
    return module


checks = _load_checks()


# ---------------------------------------------------------------------------------------------------------------
# Rendering and source views
# ---------------------------------------------------------------------------------------------------------------

def render_resume(resume):
    """Name, a blank line, then per section: heading, optional dates line, '- ' bullets and a blank line; finally
    'Skills: ...'. LF newlines and one trailing newline. The same input always gives the same text."""
    lines = [resume["name"], ""]
    for section in resume["sections"]:
        lines.append(section["heading"])
        if section.get("dates"):
            lines.append(section["dates"])
        lines.extend("- " + bullet for bullet in section["bullets"])
        lines.append("")
    lines.append("Skills: " + resume["skills_line"])
    return "\n".join(lines) + "\n"


def source_sections(case):
    """[{"heading", "bullets"}] in resume order: the `source_sections` argument of checks.check_sections."""
    return [{"heading": s["heading"], "bullets": list(s["bullets"])} for s in case["resume"]["sections"]]


def source_bullets(case):
    return [bullet for section in case["resume"]["sections"] for bullet in section["bullets"]]


# ---------------------------------------------------------------------------------------------------------------
# Validation (returns problems; never silently repairs)
# ---------------------------------------------------------------------------------------------------------------

def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_resume(resume, label, problems):
    """Returns True when the structure is sound enough to render."""
    if not isinstance(resume, dict) or set(resume) != RESUME_KEYS:
        problems.append(f"{label}: resume must be an object with exactly {sorted(RESUME_KEYS)}")
        return False
    ok = True
    if not _nonblank(resume["name"]) or not _nonblank(resume["skills_line"]):
        problems.append(f"{label}: resume name and skills_line must be non-blank strings")
        ok = False
    sections = resume["sections"]
    if not isinstance(sections, list) or not sections:
        problems.append(f"{label}: resume sections must be a non-empty list")
        return False
    headings, seen_bullets = [], set()
    for position, section in enumerate(sections):
        where = f"{label}: section {position}"
        if not isinstance(section, dict) or "heading" not in section or not set(section) <= SECTION_KEYS:
            problems.append(f"{where} must be an object with heading, bullets and optional dates")
            ok = False
            continue
        if not _nonblank(section["heading"]):
            problems.append(f"{where} has a blank heading")
            ok = False
        else:
            headings.append(section["heading"].strip().lower())
        if "dates" in section and not _nonblank(section["dates"]):
            problems.append(f"{where} has blank dates (omit the key instead)")
            ok = False
        bullets = section.get("bullets")
        if not isinstance(bullets, list) or not bullets:
            problems.append(f"{where} needs a non-empty bullets list")
            ok = False
            continue
        for bullet in bullets:
            if not _nonblank(bullet) or bullet != bullet.strip() or "\n" in bullet:
                problems.append(f"{where} has a bullet that is blank, padded or multi-line")
                ok = False
            elif bullet.lower() in seen_bullets:
                problems.append(f"{label}: duplicate source bullet {bullet[:40]!r}")
            else:
                seen_bullets.add(bullet.lower())
    if len(set(headings)) != len(headings):
        problems.append(f"{label}: section headings must be unique")
        ok = False
    for i, first in enumerate(headings):
        for j, second in enumerate(headings):
            if i != j and first in second:
                problems.append(f"{label}: heading {first!r} is a substring of {second!r} (ambiguous matching)")
                ok = False
    return ok


def validate_case(case, filename=None):
    """Problems with one case (empty list = valid). Does not raise on malformed input."""
    if not isinstance(case, dict):
        return ["case is not a JSON object"]
    cid = case.get("id") if isinstance(case.get("id"), str) else "<no id>"
    label = cid
    if set(case) != CASE_KEYS:
        missing, extra = sorted(CASE_KEYS - set(case)), sorted(set(case) - CASE_KEYS)
        return [f"{label}: wrong keys (missing {missing}, unexpected {extra})"]
    problems = []
    if not _ID_RE.match(cid):
        problems.append(f"{label}: id must look like c01_short_name")
    if filename is not None and filename != cid + ".json":
        problems.append(f"{label}: file name {filename!r} does not match the id")
    if case["category"] not in REQUIRED_CATEGORIES:
        problems.append(f"{label}: unknown category {case['category']!r}")
    if not isinstance(case["holdout"], bool):
        problems.append(f"{label}: holdout must be true or false")
    for key in ("title", "provenance", "job_description", "resume_text"):
        if not _nonblank(case[key]):
            problems.append(f"{label}: {key} must be a non-blank string")
    traps = case["traps"]
    if not isinstance(traps, list) or not traps or not all(_nonblank(t) for t in traps):
        problems.append(f"{label}: traps must be a non-empty list of non-blank strings")
    trap_types = case["trap_types"]
    if (not isinstance(trap_types, list) or not trap_types
            or not all(isinstance(t, str) and t in TRAP_TYPES for t in trap_types)):
        problems.append(f"{label}: trap_types must be a non-empty list of known trap types")
    renderable = _validate_resume(case["resume"], label, problems)
    if renderable and _nonblank(case["resume_text"]) and case["resume_text"] != render_resume(case["resume"]):
        problems.append(f"{label}: resume_text differs from the deterministic rendering of the structured resume")
    absent = case["absent_skills"]
    if not isinstance(absent, list) or not absent:
        problems.append(f"{label}: absent_skills must be a non-empty list")
    else:
        shape_ok = True
        for skill in absent:
            if (not isinstance(skill, dict) or set(skill) != {"name", "aliases"}
                    or not _nonblank(skill["name"]) or not isinstance(skill["aliases"], list)
                    or not all(_nonblank(a) for a in skill["aliases"])):
                problems.append(f"{label}: each absent skill needs a name and a list of alias strings")
                shape_ok = False
        if shape_ok and _nonblank(case["resume_text"]) and _nonblank(case["job_description"]):
            for problem in checks.validate_absent_skills(absent, case["resume_text"], case["job_description"]):
                problems.append(f"{label}: {problem}")
    return problems


def validate_dataset(cases):
    """Dataset-level problems: 12 unique ids, 3 holdouts, every required category present."""
    problems = []
    ids = [c["id"] for c in cases]
    if len(cases) != EXPECTED_CASES:
        problems.append(f"expected {EXPECTED_CASES} cases, found {len(cases)}")
    if len(set(ids)) != len(ids):
        problems.append("case ids are not unique")
    holdouts = [c["id"] for c in cases if c["holdout"]]
    if len(holdouts) != EXPECTED_HOLDOUTS:
        problems.append(f"expected {EXPECTED_HOLDOUTS} holdout cases, found {len(holdouts)}")
    missing = sorted(REQUIRED_CATEGORIES - {c["category"] for c in cases})
    if missing:
        problems.append(f"missing categories: {missing}")
    return problems


def load_cases(split="dev", cases_dir=None, strict=True):
    """Cases of one split in file-name order (deep copies). Raises DatasetError if ANY case file is invalid.
    strict=False skips the dataset-level counts (only for tests that build small fixture directories)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    directory = cases_dir or CASES_DIR
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    cases, problems = [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as handle:
                case = json.load(handle)
        except (OSError, ValueError) as error:
            problems.append(f"{name}: cannot be read as JSON ({type(error).__name__})")
            continue
        found = validate_case(case, name)
        problems.extend(found)
        if not found:
            cases.append(case)
    if strict and not problems:
        problems.extend(validate_dataset(cases))
    if problems:
        raise DatasetError("invalid dataset:\n  " + "\n  ".join(problems))
    if split == "dev":
        cases = [c for c in cases if not c["holdout"]]
    elif split == "holdout":
        cases = [c for c in cases if c["holdout"]]
    return copy.deepcopy(cases)


def load_vocabulary(path=None):
    """The "terms" mapping ONLY ({canonical: [aliases]}), the exact argument of checks.check_technologies.
    The file's top level also holds "meta"; passing the whole file to the checker would treat 'meta' and 'terms' as
    technologies, so this function never returns it."""
    try:
        with open(path or VOCAB_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as error:
        raise DatasetError(f"vocabulary cannot be read as JSON ({type(error).__name__})")
    if not isinstance(data, dict) or set(data) != {"meta", "terms"} or not isinstance(data["terms"], dict):
        raise DatasetError('vocabulary must be an object with exactly "meta" and "terms"')
    terms, seen, problems = data["terms"], {}, []
    for name, aliases in terms.items():
        if not _nonblank(name) or not isinstance(aliases, list) or not all(_nonblank(a) for a in aliases):
            problems.append(f"malformed vocabulary entry {name!r}")
            continue
        for term in [name] + aliases:
            key = term.lower()
            if key in seen:
                problems.append(f"{term!r} duplicates {seen[key]!r}")
            seen[key] = name
    if not terms:
        problems.append("vocabulary has no terms")
    if problems:
        raise DatasetError("invalid vocabulary:\n  " + "\n  ".join(problems))
    return {name: list(aliases) for name, aliases in terms.items()}


# ---------------------------------------------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------------------------------------------

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """SHA-256 of the file with CRLF normalised to LF, so a Windows checkout and a Linux checkout agree."""
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read().replace(b"\r\n", b"\n"))


def combine_fingerprint(entries):
    """entries: [(label, sha256 hex)]. Returns {"sha256", "covers"}; the combined hash covers every label and hash
    in sorted label order, so renaming, adding or removing a covered file changes it."""
    ordered = sorted(entries)
    digest = hashlib.sha256()
    for label, sha in ordered:
        digest.update(f"{label}\0{sha}\n".encode("utf-8"))
    return {"sha256": digest.hexdigest(), "covers": [{"path": label, "sha256": sha} for label, sha in ordered]}


def dataset_fingerprint(split="all", cases_dir=None):
    """Covers exactly the case files of the requested split (so editing a holdout file does not invalidate a dev
    plan), labelled 'cases/<file name>'."""
    directory = cases_dir or CASES_DIR
    wanted = {c["id"] + ".json" for c in load_cases(split, cases_dir=directory, strict=False)}
    return combine_fingerprint([(f"cases/{name}", sha256_file(os.path.join(directory, name))) for name in wanted])


def vocabulary_fingerprint(path=None):
    return combine_fingerprint([("tech_vocabulary.json", sha256_file(path or VOCAB_PATH))])


def checker_fingerprint():
    return combine_fingerprint([("checks.py", sha256_file(CHECKS_PATH))])
