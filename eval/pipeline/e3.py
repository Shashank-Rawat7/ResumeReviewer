"""Offline screening of supplied E3 (ATS bullet rewrite) output against the authored labels (whole-pipeline evaluation, Step 4).

E3 is independent of E1 and E2: production gives it only the job description and the resume text. This module therefore
takes only an E3 raw output string, the scenario inputs and the label. It never receives, reads or needs any E1 or E2
output. It does not call a model, run the application, write a file or read anything but the data passed to it.

DETERMINISTIC SCREENING IS NOT FACTUAL VERIFICATION
  Every quality metric here is a screening signal built from eval/ats/checks.py (D1, D2, D3, D6) and the authored
  ground truth. A rate of 0 means "the screen found nothing", never "the bullets are factually correct". An
  overclaim worded without a number or a listed term ("led the migration", unsupported scale) is NOT detected. The
  authored supported_claims, prohibited_claims and acceptance_rule are natural-language sentences that cannot be matched
  deterministically; build_review_rows pairs them with each bullet for a HUMAN to judge. No claim-support metric is
  computed.

METRICS (quality; lower is better; denominator = bullets checked)
  bullets_with_unsupported_number     a number not in the authored ats.supported_numbers   (PRIMARY ground truth)
  bullets_with_number_not_in_resume_text   a number not in the scenario resume text         (CROSS-CHECK)
  bullets_mentioning_absent_skill     a skill the label declares absent from the resume
  bullets_with_guard_phrase           a guarded phrase/term on a non-absent skill (for example "designed GraphQL")
  scenario_has_flagged_bullet         1 when the scenario has any bullet flagged by the three primary screens (denominator 1)
  A metric with nothing to screen for (no absent skills, no guard terms defined) is UNSCORABLE, not 0/N.
  An invalid, empty, wrong-shaped or missing output gets NO quality score: its metrics are unavailable.

DESCRIPTIVE COUNTS (kept apart from the metrics; they are not quality scores)
  output status, bullets, source bullets, bullets identical to a source bullet, near-duplicate pairs, unit-mismatch
  advisories. A verbatim copy of the resume scores zero on every screen; the report therefore warns when every bullet is
  unchanged or when there are fewer bullets than the source.

Deferred: section-structure evaluation (assumption A4 is unresolved).
Standard library only. Reuses eval/ats/checks.py (loaded by path) and two metric helpers of score.py.
"""
import importlib.util
import json
import os
import sys

REPORT_VERSION = "pipeline-e3-1"
HERE = os.path.dirname(os.path.abspath(__file__))
ATS_CHECKS_PATH = os.path.join(HERE, "..", "ats", "checks.py")
LOWER = "lower_is_better"

SCREENING_NOTICE = (
    "Deterministic screening, not factual verification. A rate of 0 means the screens found nothing; it does not show "
    "that the bullets are accurate. Claims, overstated responsibility and scale worded without a number or a listed term "
    "are not detected. A human applies the authored acceptance rule (see build_review_rows).")
ZERO_MEANS = "Nothing was flagged by this screen. It is not evidence that the bullets are factually correct."

NOTES = {
    "bullets_with_unsupported_number":
        "PRIMARY. Bullets containing a number (compared by value, D2) that is not in the authored ats.supported_numbers.",
    "bullets_with_number_not_in_resume_text":
        "CROSS-CHECK. As the primary number screen, but against the scenario resume text (the existing D2 check).",
    "bullets_mentioning_absent_skill":
        "Bullets that mention a skill the label declares absent from the resume (any mention, including a negated one), "
        "using the skill name, its authored aliases and its authored prohibited terms (D3 word-boundary matching).",
    "bullets_with_guard_phrase":
        "Bullets containing an authored prohibited term on a skill that is NOT absent (for example an overclaim phrase "
        "or an unearned qualification).",
    "scenario_has_flagged_bullet":
        "1 when any of the three primary screens flagged at least one bullet in the scenario. Pooled, it is the number "
        "of scenarios with a flag over the scenarios scored, so a long output cannot hide a failure.",
}
MATCHING_ASSUMPTIONS = [
    "Numbers are compared by value after normalisation (eval/ats/checks.py D2); units are only an advisory.",
    "Skill and guard terms are matched case-insensitively at word boundaries (D3). Authored aliases and prohibited terms "
    "are an evaluation aid; any mention counts, including a negated one.",
    "No semantic or paraphrase matching is done.",
]
LIMITATIONS = [
    "Screening only: unsupported claims, inflated responsibility or scale without a number or listed term are not "
    "detected, and a flagged mention can be legitimate. Human review is the reference.",
    "D2 blind spots apply: numbers glued to letters, units outside the whitelist and figures derived from the resume.",
    "The prompt asks the model to exclude skills 'marked missing' but production never passes that list to E3, and it "
    "asks for a separate metric suggestion that the output format has no field for. Neither is scored.",
    "Scores describe only the supplied output, on 10 synthetic scenarios with draft-1 labels. They say nothing about a live "
    "model in general. Section structure is not evaluated (A4 unresolved).",
]


def _load(name, path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = _load("ats_checks", ATS_CHECKS_PATH)
_score = _load("pipeline_score", os.path.join(HERE, "score.py"))
# The only two score.py helpers reused. Both are private; tests/test_pipeline_e3.py pins the record shape they return.
_metric = _score._metric
_aggregate_metric = _score._aggregate_metric


# ---------------------------------------------------------------------------------------------------------------
# Metric records
# ---------------------------------------------------------------------------------------------------------------

def _tag(record):
    """Mark a metric as a deterministic screen and say what zero means."""
    record["screening_only"] = True
    record["zero_means"] = ZERO_MEANS
    return record


def _unscored(name, reason):
    """No quality score: the output is missing/invalid, or there was nothing to screen."""
    return _tag(_metric("e3", name, None, None, LOWER, unscorable={reason: 1}, note=NOTES[name],
                        unavailable_reason=f"not scored: {reason}"))


def _scored(name, flagged, checked):
    return _tag(_metric("e3", name, flagged, checked, LOWER, note=NOTES[name]))


# ---------------------------------------------------------------------------------------------------------------
# Authored screens
# ---------------------------------------------------------------------------------------------------------------

def _dedupe(values):
    seen, out = set(), []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip().casefold() not in seen:
            seen.add(value.strip().casefold())
            out.append(value.strip())
    return out


def build_screens(label):
    """Authored absent-skill and guard-phrase screens for one label.

    absent: [{"name", "aliases", "skill_ids"}]  from ats.absent_skills and every skill with ats.absent_from_resume true
            (aliases = the skill's authored aliases and prohibited terms).
    guards: [{"name", "aliases", "skill_ids"}]  from the prohibited terms of the remaining skills. `name` is the first
            term only because check_absent_skills needs one; skill ids tell which skill the guard belongs to.
    numbers_text: the authored supported_numbers joined by newlines, or None when the list is not usable."""
    ats = label.get("ats") if isinstance(label, dict) else None
    ats = ats if isinstance(ats, dict) else {}
    skills = [s for s in (label.get("skills") if isinstance(label, dict) else None) or [] if isinstance(s, dict)]
    listed = _dedupe(ats.get("absent_skills") if isinstance(ats.get("absent_skills"), list) else [])
    absent = {name.casefold(): {"name": name, "aliases": [], "skill_ids": []} for name in listed}
    guards = []
    for skill in skills:
        skill_ats = skill.get("ats") if isinstance(skill.get("ats"), dict) else {}
        terms = _dedupe(skill_ats.get("prohibited_claim_terms") if isinstance(skill_ats.get("prohibited_claim_terms"),
                                                                                list) else [])
        canonical = skill.get("canonical") if isinstance(skill.get("canonical"), str) else ""
        key = canonical.strip().casefold()
        is_absent = skill_ats.get("absent_from_resume") is True or key in absent
        if is_absent and key:
            entry = absent.setdefault(key, {"name": canonical.strip(), "aliases": [], "skill_ids": []})
            entry["aliases"] = _dedupe(entry["aliases"] + list(skill.get("aliases") or []) + terms)
            entry["skill_ids"].append(skill.get("skill_id"))
        elif terms:
            guards.append({"name": terms[0], "aliases": terms[1:], "skill_ids": [skill.get("skill_id")]})
    for entry in absent.values():
        entry["aliases"] = [a for a in entry["aliases"] if a.casefold() != entry["name"].casefold()]
    numbers = ats.get("supported_numbers")
    numbers_text = "\n".join(numbers) if isinstance(numbers, list) and all(isinstance(n, str) for n in numbers) else None
    return {"absent": list(absent.values()), "guards": guards, "numbers_text": numbers_text}


def _source_bullets(scenario):
    structure = ((scenario or {}).get("inputs") or {}).get("resume_structure")
    out = []
    for section in (structure or {}).get("sections") or []:
        for entry in (section.get("entries") or []) if isinstance(section, dict) else []:
            if isinstance(entry, dict):
                out.extend(b for b in entry.get("bullets") or [] if isinstance(b, str))
    return out


def _resume_text(scenario):
    text = ((scenario or {}).get("inputs") or {}).get("resume_text")
    return text if isinstance(text, str) else None


# ---------------------------------------------------------------------------------------------------------------
# Analysis of one output
# ---------------------------------------------------------------------------------------------------------------

def _analyse(label, scenario, raw_output):
    """Parse one raw E3 string and run the screens. -> dict (no scoring decisions)."""
    if raw_output is None:
        return {"status": "output_missing", "detail": None}
    parsed = checks.parse_ats_output(raw_output)
    if parsed["status"] != "ok":
        return {"status": parsed["status"], "detail": "; ".join(parsed["problems"]) or None,
                "would_500": parsed["production_would_500"]}
    flat = [(section["section"], bullet) for section in parsed["sections"] for bullet in section["bullets"]]
    bullets = [bullet for _section, bullet in flat]
    screens = build_screens(label)
    resume_text = _resume_text(scenario)
    primary = (checks.check_invented_numbers(bullets, screens["numbers_text"])
               if screens["numbers_text"] is not None else None)
    cross = checks.check_invented_numbers(bullets, resume_text) if resume_text is not None else None
    absent = checks.check_absent_skills(bullets, screens["absent"]) if screens["absent"] else None
    guard = checks.check_absent_skills(bullets, screens["guards"]) if screens["guards"] else None
    source = _source_bullets(scenario)
    unchanged = checks.find_unchanged(bullets, source) if source else None
    per_bullet = []
    for index, (section, _bullet) in enumerate(flat):
        per_bullet.append({
            "index": index, "section": section,
            "unsupported_numbers": primary["results"][index]["invented"] if primary else None,
            "numbers_not_in_resume_text": cross["results"][index]["invented"] if cross else None,
            "absent_skills_mentioned": absent["results"][index]["leaked"] if absent else None,
            "guard_terms_hit": ([name for name in guard["results"][index]["leaked"]] if guard else None),
            "identical_to_source_bullet": (index in unchanged) if unchanged is not None else None})
    return {"status": "ok", "detail": None, "flat": flat, "bullets": bullets, "screens": screens,
            "primary": primary, "cross": cross, "absent": absent, "guard": guard,
            "unchanged": unchanged, "source_bullets": source, "per_bullet": per_bullet,
            "duplicates": checks.find_duplicates(bullets)}


def _flagged(per_bullet, field):
    return sum(1 for record in per_bullet if record[field])


def score_e3(label, scenario, raw_output):
    """Screen one scenario's supplied E3 raw output. No quality score is given to a non-ok output."""
    analysis = _analyse(label, scenario, raw_output)
    status = analysis["status"]
    names = list(NOTES)
    if status != "ok":
        reason = "output_missing" if status == "output_missing" else f"output_{status}"
        metrics = [_unscored(name, reason) for name in names]
        descriptive = {"output_status": status, "bullets_total": 0, "source_bullets_total": None,
                       "identical_to_source_bullets": None, "near_duplicate_pairs": None,
                       "unit_mismatch_advisory_bullets": None}
        observations = {"production_would_500": analysis.get("would_500")} if status != "output_missing" else {}
        return _stage(status, analysis["detail"], metrics, descriptive, [], observations, [])

    per_bullet, n = analysis["per_bullet"], len(analysis["bullets"])
    screens = analysis["screens"]
    metrics, scorable_flags = [], {}
    if analysis["primary"] is None:
        metrics.append(_unscored(names[0], "supported_numbers_unavailable"))
    else:
        flagged = _flagged(per_bullet, "unsupported_numbers")
        metrics.append(_scored(names[0], flagged, n))
        scorable_flags["unsupported_number"] = flagged
    if analysis["cross"] is None:
        metrics.append(_unscored(names[1], "resume_text_unavailable"))
    else:
        metrics.append(_scored(names[1], _flagged(per_bullet, "numbers_not_in_resume_text"), n))
    if analysis["absent"] is None:
        metrics.append(_unscored(names[2], "no_absent_skills_defined"))
    else:
        flagged = _flagged(per_bullet, "absent_skills_mentioned")
        metrics.append(_scored(names[2], flagged, n))
        scorable_flags["absent_skill"] = flagged
    if analysis["guard"] is None:
        metrics.append(_unscored(names[3], "no_guard_terms_defined"))
    else:
        flagged = _flagged(per_bullet, "guard_terms_hit")
        metrics.append(_scored(names[3], flagged, n))
        scorable_flags["guard_phrase"] = flagged
    if scorable_flags:
        any_flag = 1 if any(scorable_flags.values()) else 0
        metrics.append(_tag(_metric("e3", names[4], any_flag, 1, LOWER, note=NOTES[names[4]],
                                    counts={"flagged_bullets_by_screen": dict(sorted(scorable_flags.items()))})))
    else:
        metrics.append(_unscored(names[4], "no_primary_screen_scorable"))

    unchanged = analysis["unchanged"]
    # Units come from the resume text: the authored supported_numbers carry no units, so a mismatch against them would
    # flag every "90 minutes" written against a bare "90".
    mismatch = analysis["cross"]["bullets_with_unit_mismatch"] if analysis["cross"] else None
    descriptive = {"output_status": "ok", "bullets_total": n,
                   "source_bullets_total": len(analysis["source_bullets"]) if analysis["source_bullets"] else None,
                   "identical_to_source_bullets": len(unchanged) if unchanged is not None else None,
                   "near_duplicate_pairs": len(analysis["duplicates"]),
                   "unit_mismatch_advisory_bullets": mismatch}
    warnings = []
    if unchanged is not None and n and len(unchanged) == n:
        warnings.append("verbatim_copy: every bullet is identical to a source bullet, so nothing was rewritten and zero "
                        "flags says nothing about rewriting quality.")
    if descriptive["source_bullets_total"] and n < descriptive["source_bullets_total"]:
        warnings.append(f"fewer_bullets_than_source: {n} bullets returned for {descriptive['source_bullets_total']} "
                        "source bullets; bullets that were dropped are not screened.")
    observations = {"screens": {"absent_skills": [{"name": e["name"], "aliases": e["aliases"]} for e in screens["absent"]],
                                "guard_terms": [{"name": g["name"], "aliases": g["aliases"],
                                                 "skill_ids": g["skill_ids"]} for g in screens["guards"]]},
                    "production_would_500": False}
    return _stage("scored", None, metrics, descriptive, per_bullet, observations, warnings)


def _stage(status, detail, metrics, descriptive, per_bullet, observations, warnings):
    return {"stage": "e3", "status": status, "status_detail": detail, "screening_notice": SCREENING_NOTICE,
            "metrics": metrics, "descriptive": descriptive, "bullets": per_bullet, "observations": observations,
            "warnings": warnings, "matching_assumptions": list(MATCHING_ASSUMPTIONS), "limitations": list(LIMITATIONS)}


# ---------------------------------------------------------------------------------------------------------------
# Human review rows
# ---------------------------------------------------------------------------------------------------------------

def build_review_rows(label, scenario, raw_output, scenario_id=None):
    """One row per returned bullet for a HUMAN reviewer. Pure: nothing is written.

    Each row pairs the bullet with the automated findings (screening signals) and the authored supported claims,
    prohibited claims and acceptance rule. The human fields are None. Returns [] for a missing or non-ok output."""
    analysis = _analyse(label, scenario, raw_output)
    if analysis["status"] != "ok":
        return []
    ats = label.get("ats") if isinstance(label.get("ats"), dict) else {}
    sid = scenario_id if scenario_id is not None else label.get("scenario_id")
    claims = [{"id": c.get("id"), "claim": c.get("claim")} for c in ats.get("supported_claims") or []
              if isinstance(c, dict)]
    prohibited = [p for p in ats.get("prohibited_claims") or [] if isinstance(p, str)]
    rows = []
    for (section, bullet), record in zip(analysis["flat"], analysis["per_bullet"]):
        rows.append({
            "scenario_id": sid, "row_id": f"{sid}:{record['index']}", "section": section,
            "bullet_index": record["index"], "bullet": bullet,
            "automated_findings": {key: record[key] for key in (
                "unsupported_numbers", "numbers_not_in_resume_text", "absent_skills_mentioned", "guard_terms_hit",
                "identical_to_source_bullet")},
            "authored_supported_claims": claims, "authored_prohibited_claims": prohibited,
            "acceptance_rule": ats.get("acceptance_rule"),
            "human_verdict": None, "human_notes": None,
            "reviewer_note": "Automated findings are screening signals. Apply the acceptance rule yourself."})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Dataset report
# ---------------------------------------------------------------------------------------------------------------

DESCRIPTIVE_SUMS = ("bullets_total", "source_bullets_total", "identical_to_source_bullets", "near_duplicate_pairs",
                    "unit_mismatch_advisory_bullets")


def score_e3_dataset(dataset, outputs):
    """Screen every scenario in `dataset` (dataset.load_dataset result) against `outputs`.

    outputs: {scenario_id: {"e3_raw": raw E3 string}}. Any other key (for example E1/E2 data) is ignored: E3 is
    independent of them. Missing outputs are reported as missing, never as clean. Returns a deterministic,
    JSON-serialisable report; nothing is written."""
    if not isinstance(outputs, dict):
        raise ValueError("outputs must be a dict keyed by scenario id")
    ids = list(dataset["ids"])
    unknown = sorted(set(outputs) - set(ids))
    scenarios, pooled, distribution = [], {}, {}
    totals = {key: 0 for key in DESCRIPTIVE_SUMS}
    warnings = []
    for sid in ids:
        supplied = outputs.get(sid)
        if supplied is not None and not isinstance(supplied, dict):
            raise ValueError(f"outputs[{sid!r}] must be a dict")
        raw = (supplied or {}).get("e3_raw")
        record = score_e3(dataset["labels"][sid], dataset["scenarios"].get(sid), raw)
        for metric in record["metrics"]:
            pooled.setdefault(metric["metric"], []).append((sid, metric))
        status = record["descriptive"]["output_status"]
        distribution[status] = distribution.get(status, 0) + 1
        for key in DESCRIPTIVE_SUMS:
            value = record["descriptive"].get(key)
            totals[key] += value if isinstance(value, int) else 0
        warnings.extend(f"e3:{sid}: {w}" for w in record["warnings"])
        scenarios.append({"scenario_id": sid, "output_supplied": raw is not None, "e3": record})
    aggregate = [_tag(_aggregate_metric(items)) for items in pooled.values()]
    return {
        "report_version": REPORT_VERSION, "stage": "e3", "screening_notice": SCREENING_NOTICE,
        "dataset": {"ids": ids, "verified_hashes": dict(dataset.get("verified_hashes") or {})},
        "scenarios": scenarios,
        "aggregate": {"metrics": aggregate},
        "descriptive": {"output_status_distribution": dict(sorted(distribution.items())), "totals": totals,
                        "note": "Counts for context, not quality scores. Totals add bullets across scenarios."},
        "unknown_output_ids": unknown, "warnings": warnings,
        "limitations": LIMITATIONS + [
            "Pooled rates sum numerators and denominators across scenarios with a usable output; scenarios without one "
            "are excluded from the sums and counted in scenarios_without_usable_output. Per-scenario results are listed "
            "beside them.",
            "Pooled bullet counts weight verbose outputs more; scenario_has_flagged_bullet counts each scenario once."],
    }
