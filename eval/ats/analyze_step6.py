"""Step 6: read-only analysis of the ATS development baseline's human review (standard library only).

    python eval/ats/analyze_step6.py --run-id ats_dev_20261029T052205Z

READS (and SHA-256 hashes before and after): bullets.csv, bullets_reviewed.csv, human_review_decisions.csv,
calls.jsonl, plan.json (parsed) and evaluations.jsonl, summary.json, metadata.json, report.md (hashed only).
NEVER reads raw model responses, .env, the protected src/*.txt files or any case file (so no holdout case), and it
makes no model, network, database, Git or Docker call. It modifies no existing file.

WRITES only three NEW files inside <run>/step6/ (the folder must not exist; nothing is ever overwritten):
step6_report.md, step6_metrics.json, step6_repeat_pairs.csv.

Everything it reports is a screening observation on 9 synthetic development cases with provisional human
judgments. Neither the automated checks nor the human labels are treated as ground truth, so no accuracy,
precision or recall is calculated.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone

ATS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ATS_DIR, "results")
SCRIPT_PATH = os.path.abspath(__file__)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
OUTPUT_SUBDIR = "step6"
OUTPUT_FILES = ("step6_report.md", "step6_metrics.json", "step6_repeat_pairs.csv")
# every baseline file this analysis might read; all are hashed before and after
HASHED_FILES = ("bullets.csv", "bullets_reviewed.csv", "human_review_decisions.csv", "calls.jsonl",
                "evaluations.jsonl", "plan.json", "summary.json", "metadata.json", "report.md")
KEY = ("case_id", "repeat", "bullet_index")
HUMAN = ("supported_by_source_evidence", "unsupported_or_invented_claims", "relevance_to_job_description",
         "clarity_and_action_orientation", "overall_rewrite_improvement", "reviewer_notes")
DIMENSIONS = OrderedDict([
    ("supported_by_source_evidence", ("yes", "partially", "no")),
    ("relevance_to_job_description", ("high", "medium", "low")),
    ("clarity_and_action_orientation", ("high", "medium", "low")),
    ("overall_rewrite_improvement", ("improved", "unchanged", "worse"))])
BULLETS_COLUMNS = ("case_id", "split", "repeat", "call_id", "bullet_index", "output_section", "matched_source_heading",
                   "source_bullets", "rewritten_bullet", "checker_findings", "d2_invented_numbers",
                   "d3_absent_skill_leaks", "d4_unsupported_technologies", "d6_unchanged_from_source",
                   "d6_near_duplicate_of") + HUMAN
DECISIONS_COLUMNS = KEY + HUMAN
NON_REVIEW = tuple(c for c in BULLETS_COLUMNS if c not in HUMAN)
PAIR_COLUMNS = ("case_id", "bullet_index", "pair_status", "output_section_r1", "output_section_r2",
                "matched_source_heading_r1", "matched_source_heading_r2", "exact_wording_equal",
                "normalized_wording_equal", "word_overlap_jaccard", "wording_r1", "wording_r2",
                "support_r1", "support_r2", "relevance_r1", "relevance_r2", "clarity_r1", "clarity_r2",
                "overall_r1", "overall_r2", "any_judgment_differs", "d3_r1", "d3_r2", "d4_r1", "d4_r2",
                "d6_unchanged_r1", "d6_unchanged_r2")


class StepError(Exception):
    """A rule stopped the analysis. `code`: 2 refused (setup), 3 integrity failure, 4 baseline changed."""

    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

def rate(k, n):
    """'3/18 (16.7%)'; the denominator is always shown (same format as the evaluation runner)."""
    return f"{k}/{n} ({100.0 * k / n:.1f}%)" if n else f"{k}/0 (n/a)"


def frac(k, n):
    return OrderedDict([("count", k), ("of", n), ("text", rate(k, n))])


def csv_safe(value):
    """Neutralise spreadsheet formulas (same rule as the runner)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def normalise(text):
    """Lower case, punctuation to spaces, whitespace collapsed (same rule as checks.normalise_for_comparison)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def jaccard(a, b):
    sa, sb = set(normalise(a).split()), set(normalise(b).split())
    union = sa | sb
    return round(len(sa & sb) / len(union), 3) if union else None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_baseline(run_dir):
    missing = [n for n in HASHED_FILES if not os.path.isfile(os.path.join(run_dir, n))]
    if missing:
        raise StepError("baseline file(s) missing: " + ", ".join(missing))
    return OrderedDict((n, sha256_file(os.path.join(run_dir, n))) for n in HASHED_FILES)


def read_csv(path):
    """(header, rows). utf-8-sig removes the byte-order mark that Excel adds; newline='' keeps embedded newlines."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


def resolve_run_dir(results_root, run_id):
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise StepError(f"run id {run_id!r} is not a valid run folder name")
    root = os.path.realpath(results_root)
    path = os.path.realpath(os.path.join(root, run_id))
    if os.path.commonpath([root, path]) != root or path == root:
        raise StepError("the run folder must be directly inside the results folder")
    if not os.path.isdir(path):
        raise StepError(f"no such baseline run folder: {run_id}")
    return path


def sort_key(key):
    case_id, repeat, index = key
    return (case_id, int(repeat) if repeat.isdigit() else -1, int(index) if index.isdigit() else -1)


def index_rows(rows, name, problems):
    """OrderedDict keyed by the stable (case_id, repeat, bullet_index) strings; problems records blanks/duplicates."""
    indexed, duplicates, blanks, non_numeric = OrderedDict(), [], 0, 0
    for row in rows:
        key = tuple((row.get(column) or "").strip() for column in KEY)
        if not all(key):
            blanks += 1
            continue
        if not (key[1].isdigit() and key[2].isdigit()):
            non_numeric += 1
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if blanks:
        problems.append(f"{name}: {blanks} row(s) with a blank key part")
    if non_numeric:
        problems.append(f"{name}: {non_numeric} row(s) with a non-numeric repeat or bullet_index")
    if duplicates:
        problems.append(f"{name}: {len(duplicates)} duplicated key(s), e.g. {duplicates[0]}")
    return indexed


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


# ---------------------------------------------------------------------------------------------------------------
# Loading and integrity
# ---------------------------------------------------------------------------------------------------------------

def load_and_check(run_dir):
    """Reads the five parsed inputs, checks structure and cross-file consistency. Structural failures raise
    StepError(code 3); softer observations (label problems, disagreements) go to the returned data_quality block."""
    problems, quality = [], OrderedDict()
    cols_o, rows_o = read_csv(os.path.join(run_dir, "bullets.csv"))
    cols_r, rows_r = read_csv(os.path.join(run_dir, "bullets_reviewed.csv"))
    cols_d, rows_d = read_csv(os.path.join(run_dir, "human_review_decisions.csv"))
    calls = read_jsonl(os.path.join(run_dir, "calls.jsonl"))
    with open(os.path.join(run_dir, "plan.json"), "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    for name, got, expected in (("bullets.csv", cols_o, BULLETS_COLUMNS), ("bullets_reviewed.csv", cols_r, BULLETS_COLUMNS),
                                ("human_review_decisions.csv", cols_d, DECISIONS_COLUMNS)):
        if tuple(got) != tuple(expected):
            problems.append(f"{name}: columns differ from the expected {len(expected)} (got {got})")
    if problems:
        raise StepError("; ".join(problems), 3)  # without the right columns nothing else can be trusted
    orig = index_rows(rows_o, "bullets.csv", problems)
    reviewed = index_rows(rows_r, "bullets_reviewed.csv", problems)
    decisions = index_rows(rows_d, "human_review_decisions.csv", problems)
    if set(orig) != set(reviewed) or set(orig) != set(decisions):
        problems.append("key sets differ: only in bullets.csv %d, only in reviewed %d, only in decisions %d, "
                        "missing from reviewed %d, missing from decisions %d" % (
                            len(set(orig) - set(reviewed) - set(decisions)), len(set(reviewed) - set(orig)),
                            len(set(decisions) - set(orig)), len(set(orig) - set(reviewed)),
                            len(set(orig) - set(decisions))))
    if problems:
        raise StepError("; ".join(problems), 3)
    keys = sorted(orig, key=sort_key)
    changed = Counter()
    changed_ws = Counter()
    for key in keys:
        for column in NON_REVIEW:
            if orig[key][column] != reviewed[key][column]:
                changed[column] += 1
            if " ".join(orig[key][column].split()) != " ".join(reviewed[key][column].split()):
                changed_ws[column] += 1
    if changed_ws:
        problems.append("bullets_reviewed.csv changed original non-review field(s): " + ", ".join(sorted(changed_ws)))
    if problems:
        raise StepError("; ".join(problems), 3)
    # the plan and calls must describe the same dev baseline
    if plan.get("split") != "dev" or {r["split"] for r in rows_o} != {"dev"}:
        problems.append("the baseline is not a development-only run")
    if sorted({k[0] for k in keys}) != sorted(plan.get("case_ids", [])):
        problems.append("case ids in the CSVs differ from plan.json")
    repeats = sorted({int(k[1]) for k in keys})
    if repeats != list(range(1, plan.get("repeats", 0) + 1)):
        problems.append(f"repeats in the CSVs {repeats} differ from plan.json ({plan.get('repeats')})")
    by_call = {c.get("call_id"): c for c in calls}
    if len(by_call) != len(calls):
        problems.append("calls.jsonl has duplicated call_id values")
    for key in keys:
        row = orig[key]
        expected_call = f"{key[0]}__r{int(key[1]):02d}"
        if row["call_id"] != expected_call:
            problems.append(f"call_id {row['call_id']!r} does not match its case/repeat ({expected_call})")
            break
        record = by_call.get(row["call_id"])
        if record is None or record.get("status") != "ok":
            problems.append(f"{row['call_id']}: no matching status-ok record in calls.jsonl")
            break
    if problems:
        raise StepError("; ".join(problems), 3)
    # ---- softer observations (reported, never corrected) ----
    invalid = []
    for source, indexed in (("bullets_reviewed.csv", reviewed), ("human_review_decisions.csv", decisions)):
        for key in keys:
            for field, allowed in DIMENSIONS.items():
                if indexed[key][field] not in allowed:
                    invalid.append(OrderedDict([("file", source), ("case_id", key[0]), ("repeat", key[1]),
                                                ("bullet_index", key[2]), ("field", field),
                                                ("value", indexed[key][field])]))
    disagreements = OrderedDict((field, sum(1 for k in keys if reviewed[k][field] != decisions[k][field]))
                                for field in HUMAN)
    quality["labels_outside_agreed_sets"] = OrderedDict([("count", len(invalid)), ("examples", invalid[:10])])
    quality["reviewed_vs_decisions_disagreements_by_field"] = disagreements
    quality["original_sheet_human_cells_non_empty"] = sum(1 for k in keys for h in HUMAN if orig[k][h].strip())
    quality["reviewed_non_review_cells_changed_exactly"] = sum(changed.values())
    quality["reviewed_non_review_cells_changed_after_whitespace_normalisation"] = sum(changed_ws.values())
    return OrderedDict([("orig", orig), ("reviewed", reviewed), ("decisions", decisions), ("keys", keys),
                        ("calls", calls), ("plan", plan), ("quality", quality),
                        ("sizes", OrderedDict([("bullets.csv", len(rows_o)), ("bullets_reviewed.csv", len(rows_r)),
                                               ("human_review_decisions.csv", len(rows_d)),
                                               ("calls.jsonl", len(calls))]))])


# ---------------------------------------------------------------------------------------------------------------
# Human-review metrics
# ---------------------------------------------------------------------------------------------------------------

def dim_counts(rows):
    """Plain counts per dimension; a supplied value outside the agreed set is counted as other_or_blank."""
    result = OrderedDict()
    for field, allowed in DIMENSIONS.items():
        seen = Counter(r[field] for r in rows)
        entry = OrderedDict((v, seen.get(v, 0)) for v in allowed)
        entry["other_or_blank"] = sum(n for v, n in seen.items() if v not in allowed)
        result[field] = entry
    return result


def claim_kind(text):
    value = (text or "").strip()
    return "blank" if not value else ("none" if value.lower() == "none" else "explanatory_text")


def human_metrics(data):
    labels = [data["decisions"][k] for k in data["keys"]]
    n = len(labels)
    dims = OrderedDict()
    for field, counts in dim_counts(labels).items():
        dims[field] = OrderedDict((value, frac(count, n)) for value, count in counts.items())
    claims = Counter(claim_kind(r["unsupported_or_invented_claims"]) for r in labels)
    claims_block = OrderedDict((name, frac(claims.get(name, 0), n)) for name in ("none", "explanatory_text", "blank"))
    cross = OrderedDict()
    for support in DIMENSIONS["supported_by_source_evidence"]:
        group = [r for r in labels if r["supported_by_source_evidence"] == support]
        cross[support] = OrderedDict([("bullets", len(group))] + [
            (value, sum(1 for r in group if r["overall_rewrite_improvement"] == value))
            for value in DIMENSIONS["overall_rewrite_improvement"]])
    by_case, by_case_repeat = [], []
    for case_id in sorted({k[0] for k in data["keys"]}):
        mine = [data["decisions"][k] for k in data["keys"] if k[0] == case_id]
        by_case.append(OrderedDict([("case_id", case_id), ("bullets", len(mine)), ("counts", dim_counts(mine))]))
        for repeat in sorted({int(k[1]) for k in data["keys"] if k[0] == case_id}):
            part = [data["decisions"][k] for k in data["keys"] if k[0] == case_id and int(k[1]) == repeat]
            by_case_repeat.append(OrderedDict([("case_id", case_id), ("repeat", repeat), ("bullets", len(part)),
                                               ("counts", dim_counts(part))]))
    yes_with_text = [OrderedDict([("case_id", k[0]), ("repeat", int(k[1])), ("bullet_index", int(k[2])),
                                  ("unsupported_or_invented_claims_text", data["decisions"][k]["unsupported_or_invented_claims"].strip()[:160])])
                     for k in data["keys"]
                     if data["decisions"][k]["supported_by_source_evidence"] == "yes"
                     and claim_kind(data["decisions"][k]["unsupported_or_invented_claims"]) == "explanatory_text"]
    not_yes_without_text = sum(1 for r in labels if r["supported_by_source_evidence"] != "yes"
                               and claim_kind(r["unsupported_or_invented_claims"]) != "explanatory_text")
    data_quality = OrderedDict([
        ("support_yes_with_explanatory_claim_text", OrderedDict([
            ("count", len(yes_with_text)), ("of_support_yes", sum(1 for r in labels if r["supported_by_source_evidence"] == "yes")),
            ("rows", yes_with_text),
            ("handling", "reported as a data-quality observation; labels are preserved exactly as supplied and not corrected")])),
        ("support_not_yes_without_explanatory_claim_text", not_yes_without_text),
        ("reviewer_notes_blank", frac(sum(1 for r in labels if not r["reviewer_notes"].strip()), n))])
    return OrderedDict([("bullets", n), ("source_of_labels", "human_review_decisions.csv (identical to the reviewed sheet unless "
                        "data_quality says otherwise)"), ("dimensions", dims), ("unsupported_or_invented_claims_field", claims_block),
                        ("overall_improvement_by_support_counts", cross), ("by_case", by_case),
                        ("by_case_repeat", by_case_repeat), ("data_quality", data_quality)])


# ---------------------------------------------------------------------------------------------------------------
# Automated checks vs human review (counts only)
# ---------------------------------------------------------------------------------------------------------------

def flags(row):
    return OrderedDict([("d2", bool(row["d2_invented_numbers"].strip())), ("d3", bool(row["d3_absent_skill_leaks"].strip())),
                        ("d4", bool(row["d4_unsupported_technologies"].strip())),
                        ("d6_unchanged", row["d6_unchanged_from_source"].strip().lower() == "yes"),
                        ("d6_near_duplicate", bool(row["d6_near_duplicate_of"].strip()))])


def automated_vs_human(data):
    keys, n = data["keys"], len(data["keys"])
    flag = {k: flags(data["orig"][k]) for k in keys}
    human = {k: data["decisions"][k] for k in keys}
    totals = OrderedDict((name, frac(sum(1 for k in keys if flag[k][name]), n)) for name in
                         ("d2", "d3", "d4", "d6_unchanged", "d6_near_duplicate"))
    totals["any_checker_finding_text_in_sheet"] = frac(sum(1 for k in keys if data["orig"][k]["checker_findings"].strip()), n)
    overlap = OrderedDict([
        ("both_d3_and_d4", sum(1 for k in keys if flag[k]["d3"] and flag[k]["d4"])),
        ("d3_only", sum(1 for k in keys if flag[k]["d3"] and not flag[k]["d4"])),
        ("d4_only", sum(1 for k in keys if flag[k]["d4"] and not flag[k]["d3"])),
        ("neither", sum(1 for k in keys if not flag[k]["d3"] and not flag[k]["d4"])),
        ("note", "D3 (declared absent skills) and D4 (vocabulary technologies, advisory) overlap; they are never summed")])

    def split_by(predicate, human_field, values):
        out = OrderedDict()
        for name, selector in (("flagged", predicate), ("not_flagged", lambda k: not predicate(k))):
            group = [k for k in keys if selector(k)]
            out[name] = OrderedDict([("bullets", len(group))] + [
                (v, sum(1 for k in group if human[k][human_field] == v)) for v in values])
        return out

    support, overall = DIMENSIONS["supported_by_source_evidence"], DIMENSIONS["overall_rewrite_improvement"]
    cross = OrderedDict([
        ("d2_vs_support", split_by(lambda k: flag[k]["d2"], "supported_by_source_evidence", support)),
        ("d3_vs_support", split_by(lambda k: flag[k]["d3"], "supported_by_source_evidence", support)),
        ("d4_vs_support", split_by(lambda k: flag[k]["d4"], "supported_by_source_evidence", support)),
        ("d6_unchanged_vs_overall", split_by(lambda k: flag[k]["d6_unchanged"], "overall_rewrite_improvement", overall)),
        ("d6_near_duplicate_vs_overall", split_by(lambda k: flag[k]["d6_near_duplicate"], "overall_rewrite_improvement", overall))])
    any_flag = lambda k: flag[k]["d2"] or flag[k]["d3"] or flag[k]["d4"]
    union = OrderedDict([
        ("definition", "bullets with at least one of D2, D3, D4 (a union of overlapping checks, not a sum)"),
        ("by_support", split_by(any_flag, "supported_by_source_evidence", support)),
        ("support_partially_or_no_total", sum(1 for k in keys if human[k]["supported_by_source_evidence"] != "yes")),
        ("support_partially_or_no_without_any_d2_d3_d4_flag",
         sum(1 for k in keys if human[k]["supported_by_source_evidence"] != "yes" and not any_flag(k))),
        ("human_unchanged_total", sum(1 for k in keys if human[k]["overall_rewrite_improvement"] == "unchanged")),
        ("human_unchanged_without_d6_flag",
         sum(1 for k in keys if human[k]["overall_rewrite_improvement"] == "unchanged" and not flag[k]["d6_unchanged"]))])
    return OrderedDict([
        ("scope", "all bullets from status-ok responses; counts only, because neither the checks nor the human labels are ground truth"),
        ("flag_totals", totals), ("d3_d4_overlap", overlap), ("crosstabs_counts", cross), ("union_d2_d3_d4_counts", union),
        ("not_comparable", (["D2 flagged no bullets, so it cannot be compared with human labels"]
                            if not any(flag[k]["d2"] for k in keys) else [])
         + ["D1/D5 detail is not in the label sheet; only matched_source_heading is (non-blank on "
            f"{sum(1 for k in keys if data['orig'][k]['matched_source_heading'].strip())}/{n} bullets)"]),
        ("not_calculated", "accuracy, precision and recall (no defensible ground truth exists)")])


# ---------------------------------------------------------------------------------------------------------------
# Repeat comparison
# ---------------------------------------------------------------------------------------------------------------

def repeat_comparison(data):
    if data["plan"].get("repeats") != 2:
        raise StepError("repeat comparison is implemented for exactly two repeats", 3)
    by_position = defaultdict(dict)
    for key in data["keys"]:
        by_position[(key[0], int(key[2]))][int(key[1])] = key
    pairs, statuses = [], Counter()
    for (case_id, index) in sorted(by_position):
        slot = by_position[(case_id, index)]
        k1, k2 = slot.get(1), slot.get(2)
        r1 = data["orig"][k1] if k1 else None
        r2 = data["orig"][k2] if k2 else None
        if r1 is None or r2 is None:
            status = "unmatched_missing_repeat"
        elif r1["output_section"] != r2["output_section"]:
            status = "section_mismatch"
        elif not r1["matched_source_heading"].strip() or not r2["matched_source_heading"].strip():
            status = "ambiguous_no_matched_heading"
        elif r1["matched_source_heading"] != r2["matched_source_heading"]:
            status = "heading_mismatch"
        else:
            status = "paired"
        statuses[status] += 1
        pairs.append(OrderedDict([("case_id", case_id), ("bullet_index", index), ("status", status), ("r1", r1), ("r2", r2),
                                  ("h1", data["decisions"][k1] if k1 else None), ("h2", data["decisions"][k2] if k2 else None)]))
    paired = [p for p in pairs if p["status"] == "paired"]
    for p in paired:
        a, b = p["r1"]["rewritten_bullet"], p["r2"]["rewritten_bullet"]
        p["exact"], p["normalized"] = a == b, normalise(a) == normalise(b)
        p["jaccard"] = jaccard(a, b)
        p["differs"] = [f for f in DIMENSIONS if p["h1"][f] != p["h2"][f]]
    exact = sum(1 for p in paired if p["exact"])
    norm_equal = sum(1 for p in paired if p["normalized"])
    changed = [p for p in paired if not p["normalized"]]
    n_paired = len(paired)
    dim_diff = OrderedDict((f, frac(sum(1 for p in paired if f in p["differs"]), n_paired)) for f in DIMENSIONS)
    claim_diff = sum(1 for p in paired if claim_kind(p["h1"]["unsupported_or_invented_claims"]) != claim_kind(p["h2"]["unsupported_or_invented_claims"]))

    def judgment_split(group):
        return OrderedDict([("positions", len(group)), ("any_of_four_dimensions_differs", sum(1 for p in group if p["differs"]))])

    flag_diff = OrderedDict((name, sum(1 for p in paired if flags(p["r1"])[name] != flags(p["r2"])[name]))
                            for name in ("d3", "d4", "d6_unchanged"))
    per_case_counts = defaultdict(Counter)
    for key in data["keys"]:
        per_case_counts[key[0]][int(key[1])] += 1
    count_differs = [c for c, v in sorted(per_case_counts.items()) if v[1] != v[2]]
    by_call = {c["call_id"]: c for c in data["calls"]}
    case_status = OrderedDict()
    for case_id in sorted(per_case_counts):
        s1 = by_call.get(f"{case_id}__r01", {}).get("status")
        s2 = by_call.get(f"{case_id}__r02", {}).get("status")
        case_status[case_id] = OrderedDict([("r1", s1), ("r2", s2), ("same", s1 == s2)])
    all_statuses = Counter(c.get("status") for c in data["calls"])
    summary = OrderedDict([
        ("method", "positions are matched by (case_id, bullet_index); a pair is compared only when output_section and "
                   "matched_source_heading agree and the heading is non-blank; anything else is reported, never forced"),
        ("positions_total", len(pairs)), ("pair_status_counts", OrderedDict(sorted(statuses.items()))),
        ("cases_with_different_bullet_counts_between_repeats", count_differs),
        ("paired_positions", n_paired),
        ("wording", OrderedDict([
            ("exact_equal", frac(exact, n_paired)),
            ("normalized_equal_but_not_exact", frac(norm_equal - exact, n_paired)),
            ("normalized_equal_total_including_exact", frac(norm_equal, n_paired)),
            ("changed_after_normalisation", frac(len(changed), n_paired)),
            ("word_overlap_jaccard_of_changed_pairs", OrderedDict(
                [("pairs", len(changed))] + ([("min", min(p["jaccard"] for p in changed)),
                                              ("median", statistics.median(p["jaccard"] for p in changed)),
                                              ("max", max(p["jaccard"] for p in changed))] if changed else [])))])),
        ("human_judgments_differing_between_repeats", dim_diff),
        ("unsupported_claims_field_none_vs_text_differs", frac(claim_diff, n_paired)),
        ("judgments_by_wording_class", OrderedDict([
            ("exact_wording", judgment_split([p for p in paired if p["exact"]])),
            ("normalized_equal_not_exact", judgment_split([p for p in paired if p["normalized"] and not p["exact"]])),
            ("wording_changed", judgment_split(changed))])),
        ("automated_flags_differing_between_repeats", flag_diff),
        ("response_status", OrderedDict([
            ("status_counts_all_calls", OrderedDict(sorted(all_statuses.items(), key=lambda x: str(x[0])))),
            ("cases_with_same_status_in_both_repeats", frac(sum(1 for v in case_status.values() if v["same"]), len(case_status))),
            ("finish_reasons", OrderedDict(sorted(Counter(c.get("finish_reason") or "not reported" for c in data["calls"]).items()))),
            ("truncated_calls", sum(1 for c in data["calls"] if c.get("truncated"))),
            ("calls_with_more_than_one_runner_attempt", sum(1 for c in data["calls"] if len(c.get("attempts") or []) > 1)),
            ("note", "every call had the same status, so status consistency carries no information here"
                     if len(all_statuses) == 1 else "statuses varied; see status_counts_all_calls")]))])
    return summary, pairs


def pairs_csv(pairs):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PAIR_COLUMNS)
    yn = lambda value: "" if value is None else ("yes" if value else "no")
    for p in pairs:
        r1, r2, h1, h2 = p["r1"], p["r2"], p["h1"], p["h2"]
        get = lambda row, name: row[name] if row else ""
        writer.writerow([csv_safe(v) for v in (
            p["case_id"], p["bullet_index"], p["status"], get(r1, "output_section"), get(r2, "output_section"),
            get(r1, "matched_source_heading"), get(r2, "matched_source_heading"), yn(p.get("exact")),
            yn(p.get("normalized")), p.get("jaccard", ""), get(r1, "rewritten_bullet"), get(r2, "rewritten_bullet"),
            get(h1, "supported_by_source_evidence"), get(h2, "supported_by_source_evidence"),
            get(h1, "relevance_to_job_description"), get(h2, "relevance_to_job_description"),
            get(h1, "clarity_and_action_orientation"), get(h2, "clarity_and_action_orientation"),
            get(h1, "overall_rewrite_improvement"), get(h2, "overall_rewrite_improvement"),
            yn(bool(p["differs"])) if "differs" in p else "", get(r1, "d3_absent_skill_leaks"),
            get(r2, "d3_absent_skill_leaks"), get(r1, "d4_unsupported_technologies"),
            get(r2, "d4_unsupported_technologies"), get(r1, "d6_unchanged_from_source"),
            get(r2, "d6_unchanged_from_source"))])
    return buffer.getvalue()


# ---------------------------------------------------------------------------------------------------------------
# Disclosures, report, metrics
# ---------------------------------------------------------------------------------------------------------------

def disclosures(data):
    plan = data["plan"]
    override = (plan.get("evaluation_overrides") or {}).get("max_tokens")
    production = (plan.get("model_settings") or {}).get("max_tokens")
    effective = (plan.get("effective_settings") or {}).get("max_tokens")
    return OrderedDict([
        ("dataset", f"{len(plan['case_ids'])} synthetic development cases, {plan['repeats']} repeats each, "
                    f"{len(data['keys'])} generated bullets; not real resumes"),
        ("human_judgments", "provisional: the files record one labelling source, not the reviewer's identity or "
                            "independence; the decisions file consolidates the reviewed sheet, so their agreement is "
                            "not independent corroboration"),
        ("holdouts", "the three holdout cases were excluded from the baseline plan (plan split is dev); this analysis "
                     "reads no case file, no raw response and no holdout data"),
        ("evaluation_only_cap", f"the baseline ran with an evaluation-only max_tokens of {effective} "
                                f"(override {override}); production uses {production}. Results describe the capped "
                                "configuration, not production"),
        ("model", plan.get("model_expected")),
        ("effective_max_tokens", effective), ("production_max_tokens", production), ("override_max_tokens", override)])


def build_metrics(data, hashes, human, auto, repeats, now):
    integrity = OrderedDict([
        ("row_counts", data["sizes"]), ("keys_unique_and_identical_across_three_csvs", True),
        ("unique_keys", len(data["keys"])), ("column_names_match_expected", True),
        ("reviewed_preserves_every_non_review_field", True), ("split_is_dev_only", True),
        ("calls_all_status_ok_and_linked_by_call_id", True), ("details", data["quality"])])
    return OrderedDict([
        ("step6_analysis_version", 1), ("run_id", data["plan"].get("run_id")), ("generated_utc", now.isoformat()),
        ("analysis_script_sha256", sha256_file(SCRIPT_PATH)), ("disclosures", disclosures(data)),
        ("baseline_hashes", hashes), ("integrity", integrity), ("human_review", human),
        ("automated_vs_human", auto), ("repeat_comparison", repeats),
        ("not_calculated", ["accuracy", "precision", "recall", "inter-rater reliability", "generalisation to real resumes",
                            "production-equivalent quality (the run used a lower evaluation-only token cap)"])])


def render_report(metrics):
    d, h, a, r, i = (metrics["disclosures"], metrics["human_review"], metrics["automated_vs_human"],
                     metrics["repeat_comparison"], metrics["integrity"])
    n = h["bullets"]
    L = [f"# Step 6: human review and repeat analysis of {metrics['run_id']}", "",
         f"Generated {metrics['generated_utc']} by `eval/ats/analyze_step6.py` (read-only).", "",
         "## Read this first", "",
         f"- **Synthetic, development-only data:** {d['dataset']}.",
         f"- **Provisional human judgments:** {d['human_judgments']}.",
         f"- **Holdouts:** {d['holdouts']}.",
         f"- **Evaluation-only cap:** {d['evaluation_only_cap']}.",
         "- **Not ground truth:** neither the automated checks (D2-D6) nor the human labels are treated as ground truth. "
         "No accuracy, precision or recall is calculated, and nothing here shows the feature works on real resumes.",
         "- **D3 and D4 overlap** and are always reported separately, never summed.", "",
         "## 1. Integrity", ""]
    L += table(["Check", "Result"], [
        ["Rows (bullets / reviewed / decisions)", f"{i['row_counts']['bullets.csv']} / {i['row_counts']['bullets_reviewed.csv']} / {i['row_counts']['human_review_decisions.csv']}"],
        ["Keys (case_id, repeat, bullet_index) unique and identical in all three files", f"yes ({i['unique_keys']} keys)"],
        ["Reviewed sheet preserves every original non-review field (by key)", f"yes ({i['details']['reviewed_non_review_cells_changed_exactly']} cells differ)"],
        ["Labels outside the agreed sets", str(i["details"]["labels_outside_agreed_sets"]["count"])],
        ["Reviewed vs decisions disagreements (all six fields)", str(sum(i["details"]["reviewed_vs_decisions_disagreements_by_field"].values()))],
        ["Original sheet cells already filled by a human", str(i["details"]["original_sheet_human_cells_non_empty"])],
        ["Calls: status ok and linked by call_id", f"yes ({i['row_counts']['calls.jsonl']} calls)"]])
    L += ["", f"## 2. Human review: all {n} bullets", ""]
    for field, values in h["dimensions"].items():
        L += [f"**{field}**", ""] + table(["Value", "Bullets / total"], [[v, f["text"]] for v, f in values.items()]) + [""]
    L += ["**unsupported_or_invented_claims** (field content type)", ""] + table(
        ["Content", "Bullets / total"], [[k, f["text"]] for k, f in h["unsupported_or_invented_claims_field"].items()])
    L += ["", "**overall_rewrite_improvement by support** (counts)", ""] + table(
        ["Support", "Bullets", "improved", "unchanged", "worse"],
        [[s, v["bullets"], v["improved"], v["unchanged"], v["worse"]] for s, v in h["overall_improvement_by_support_counts"].items()])
    L += ["", "## 3. By case and repeat (counts; each row shows its own bullet total)", ""]
    L += table(["Case", "Bullets", "support yes/partially/no", "relevance high/medium/low", "clarity high/medium/low",
                "improved/unchanged/worse"],
               [[c["case_id"], c["bullets"]] + ["/".join(str(c["counts"][f][v]) for v in vals) for f, vals in DIMENSIONS.items()]
                for c in h["by_case"]])
    L += ["", "Partial support, worse rewrites and unchanged wording by case and repeat:", ""]
    L += table(["Case", "Repeat", "Bullets", "support partially", "overall worse", "overall unchanged"],
               [[c["case_id"], c["repeat"], c["bullets"], c["counts"]["supported_by_source_evidence"]["partially"],
                 c["counts"]["overall_rewrite_improvement"]["worse"], c["counts"]["overall_rewrite_improvement"]["unchanged"]]
                for c in h["by_case_repeat"]])
    dq = h["data_quality"]["support_yes_with_explanatory_claim_text"]
    L += ["", "## 4. Data-quality observations (labels preserved as supplied, not corrected)", "",
          f"- **{dq['count']} of {dq['of_support_yes']} bullets rated support `yes` also carry explanatory text in "
          "`unsupported_or_invented_claims`.** Rows: " + ", ".join(f"{x['case_id']} r{x['repeat']} #{x['bullet_index']}" for x in dq["rows"]) +
          ". This may be a labelling inconsistency or a deliberate caveat; it needs the reviewer's decision.",
          f"- Rated `partially` or `no` without claim text: {h['data_quality']['support_not_yes_without_explanatory_claim_text']}.",
          f"- Reviewer notes blank: {h['data_quality']['reviewer_notes_blank']['text']}.", "",
          "## 5. Automated checks vs human review (counts only)", ""]
    L += table(["Automated flag", "Bullets flagged / total"], [[k, f["text"]] for k, f in a["flag_totals"].items()])
    o = a["d3_d4_overlap"]
    L += ["", f"D3/D4 overlap (a partition of {n}): both {o['both_d3_and_d4']}, D3 only {o['d3_only']}, D4 only {o['d4_only']}, neither {o['neither']}. D3 and D4 are never summed.", ""]
    for name, split in a["crosstabs_counts"].items():
        cols = [k for k in split["flagged"] if k != "bullets"]
        L += [f"**{name}**", ""] + table(["Group", "Bullets"] + cols, [[g, v["bullets"]] + [v[c] for c in cols] for g, v in split.items()]) + [""]
    u = a["union_d2_d3_d4_counts"]
    L += [f"Union of D2/D3/D4 (not a sum): of {u['support_partially_or_no_total']} bullets rated partially or no, "
          f"{u['support_partially_or_no_without_any_d2_d3_d4_flag']} had no D2/D3/D4 flag. Of {u['human_unchanged_total']} "
          f"rated `unchanged`, {u['human_unchanged_without_d6_flag']} were not flagged by D6.", "",
          "These are counts, not accuracy. "
          + ("D2 flagged no bullets, so it cannot be compared with the human labels. " if a["flag_totals"]["d2"]["count"] == 0 else "")
          + "Accuracy, precision and recall are not calculated because no defensible ground truth exists.", "",
          "## 6. Repeat comparison", "", f"Method: {r['method']}.", ""]
    L += table(["Pair status", "Positions"], [[k, v] for k, v in r["pair_status_counts"].items()])
    w = r["wording"]
    L += ["", f"Cases with different bullet counts between repeats: {r['cases_with_different_bullet_counts_between_repeats'] or 'none'}.", "",
          f"Paired positions: {r['paired_positions']}.", ""]
    L += table(["Wording (paired positions)", "Count / paired"], [
        ["Exactly equal", w["exact_equal"]["text"]],
        ["Equal only after normalisation (case/punctuation/spacing)", w["normalized_equal_but_not_exact"]["text"]],
        ["Equal after normalisation (total)", w["normalized_equal_total_including_exact"]["text"]],
        ["Changed after normalisation", w["changed_after_normalisation"]["text"]]])
    L += ["", "Human judgments that differ between repeats:", ""] + table(
        ["Dimension", "Positions / paired"], [[f, v["text"]] for f, v in r["human_judgments_differing_between_repeats"].items()])
    L += ["", "Judgment differences by wording class (positions where any of the four dimensions differs):", ""] + table(
        ["Wording class", "Positions", "Any dimension differs"],
        [[k, v["positions"], v["any_of_four_dimensions_differs"]] for k, v in r["judgments_by_wording_class"].items()])
    rs = r["response_status"]
    L += ["", f"Response status: {dict(rs['status_counts_all_calls'])}; cases with the same status in both repeats "
          f"{rs['cases_with_same_status_in_both_repeats']['text']}; truncated calls {rs['truncated_calls']}. {rs['note']}.", "",
          "## 7. What this analysis does not show", ""]
    L += [f"- {x}" for x in metrics["not_calculated"]]
    L += ["- Two repeats per case is too few to describe variability beyond these raw counts.", "",
          "## 8. Baseline hash verification", ""]
    L += table(["File", "SHA-256 (first 16)", "Unchanged after analysis"],
               [[name, hex_[:16], "yes"] for name, hex_ in metrics["baseline_hashes"]["before"].items()])
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------------------

def write_new_outputs(run_dir, contents):
    """Creates step6/ (fails if it exists) and each file exclusively ('x'): nothing is ever overwritten."""
    target = os.path.join(run_dir, OUTPUT_SUBDIR)
    try:
        os.mkdir(target)
    except FileExistsError:
        raise StepError(f"{OUTPUT_SUBDIR}/ already exists in the run folder; refusing to overwrite. Stop and ask the owner.")
    for name in OUTPUT_FILES:
        with open(os.path.join(target, name), "x", encoding="utf-8", newline="") as handle:
            handle.write(contents[name])
    return target


def analyze(run_id, results_root=None, now=None):
    """Returns (output_dir, metrics). Raises StepError."""
    run_dir = resolve_run_dir(results_root or RESULTS_DIR, run_id)
    if os.path.lexists(os.path.join(run_dir, OUTPUT_SUBDIR)):
        raise StepError(f"{OUTPUT_SUBDIR}/ already exists in the run folder; refusing to overwrite. Stop and ask the owner.")
    before = hash_baseline(run_dir)
    data = load_and_check(run_dir)
    human = human_metrics(data)
    auto = automated_vs_human(data)
    repeats, pairs = repeat_comparison(data)
    after = hash_baseline(run_dir)
    if after != before:
        raise StepError("a baseline file changed while the analysis was running; nothing was written", 4)
    metrics = build_metrics(data, OrderedDict([("before", before), ("after_analysis", after), ("unchanged", True)]),
                            human, auto, repeats, now or datetime.now(timezone.utc))
    contents = {"step6_report.md": render_report(metrics),
                "step6_metrics.json": json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                "step6_repeat_pairs.csv": pairs_csv(pairs)}
    target = write_new_outputs(run_dir, contents)
    if hash_baseline(run_dir) != before:
        raise StepError("a baseline file changed after the outputs were written", 4)
    return target, metrics


def main(argv=None, results_root=None, out=print, now=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="baseline run folder name under eval/ats/results/")
    args = parser.parse_args(argv)
    try:
        target, metrics = analyze(args.run_id, results_root, now)
    except StepError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        return error.code
    out(f"Step 6 outputs written to {target}")
    for name in OUTPUT_FILES:
        out(f"  {name}")
    out("Baseline files verified unchanged (SHA-256 before, after analysis, and after writing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
