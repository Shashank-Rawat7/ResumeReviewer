"""Offline scoring of supplied E1, E2 and course-eligibility outputs against the frozen expected labels (Step 3).

This module scores DATA. It never calls a model, never runs the application or the parsing pipeline, never writes a
file and never reads a scenario or label itself: the caller passes the dataset (see dataset.load_dataset) and the
outputs. The production logic it needs (skill-key normalisation, the course-eligibility function and the frontend
validity check) comes from eval/pipeline/stages.py, which executes pure definitions extracted from the source text.

WHAT IS SCORED, AND WHAT IS NOT
  E1  recall of the authored skills, with true-positive and false-negative counts. Precision, F1 and false positives are
      reported as UNAVAILABLE: the labels do not establish that the authored skill list is exhaustive, so a predicted
      skill without a label is not known to be wrong. Such predictions are listed as unlabelled_predictions.
  E2  status agreement on determinate labels, gap-detection recall, false-flag rate, a missing/partial confusion table,
      tier agreement (determinate and ambiguous tier sets scored in SEPARATE metrics), JD-quote and resume-quote support,
      and how each absence arose (genuine, or a raw entry the parser rejected).
  Course  conformance of the production eligibility rule on the supplied normalised output, and agreement with the
      authored eligible_expected flags. Recommendation relevance and quality are not scored.

EVALUATION-ONLY MATCHING
  Production matches skills with _skill_key (case-folding, punctuation stripping, one trailing "s" dropped) and nothing
  else. This module uses that same key function, but compares it against the AUTHORED alias and accepted-form lists of
  each label. Those lists are an evaluation aid; the production pipeline does not resolve aliases or acronyms (A7).
  No substring, fuzzy or semantic matching is used.

HONEST ACCOUNTING
  * zero and unavailable are different: a rate is None (available=False) when there is nothing to divide or no output;
  * every rate carries its numerator and denominator, and unscorable items are counted with a reason;
  * a missing, invalid or failed-normalisation output is never counted as correct;
  * a raw entry rejected by the parser (for example a raw "no_gap") is never credited as a model prediction of absence;
  * per-scenario results are always kept next to the pooled figures.
Standard library only.
"""
import importlib.util
import json
import os
import re
import sys

REPORT_VERSION = "pipeline-score-1"
HERE = os.path.dirname(os.path.abspath(__file__))

E1_CATEGORIES = ("technical_skills", "soft_skills", "academic_qualifications")
GAP_TRUTHS = ("missing", "partial")
HIGHER, LOWER = "higher_is_better", "lower_is_better"

MATCHING_E1 = [
    "A prediction matches a label only when _skill_key(prediction) equals _skill_key of one of the label's authored "
    "extraction.accepted_forms (case-folding, punctuation stripping, one trailing 's' dropped). No substring, semantic "
    "or acronym matching is used.",
    "The authored accepted forms are an evaluation aid written for this dataset. The production pipeline does not "
    "resolve aliases or acronyms (A7).",
    "A prediction is matched only inside the label's own category; a hit in another category is recorded as "
    "found_in_other_category and is not counted.",
]
MATCHING_E2 = [
    "A normalised entry matches a label when _skill_key(entry skill) equals _skill_key of the label's canonical name or "
    "one of its authored aliases (evaluation aid only; production does not resolve aliases or acronyms, A7).",
    "ABSENT means no valid normalised entry matches. EMITTED:no_gap is a retained draft token in the labels; a valid "
    "normalised entry can only be missing or partial, so a normalised status outside those two is reported as an "
    "anomaly and excluded from matching.",
    "Raw output, when supplied, is inspected by the scorer (not by the parser) only to tell a genuine absence from an "
    "entry the parser rejected. That reading may differ from the production JSON extraction.",
]
LIMITATIONS_E1 = [
    "Precision, F1 and false positives are unavailable: the labels do not establish that the authored skill list is "
    "exhaustive. Predictions without a label are listed in unlabelled_predictions for review, not counted as errors.",
    "The score describes only the supplied output. It says nothing about how a live model behaves in general.",
]
LIMITATIONS_E2 = [
    "The score describes only the supplied output. It says nothing about how often a live model emits an invalid "
    "status, a duplicate or a given tier.",
    "Labels with outcome_class 'ambiguous' are not scored; their observed outcomes are listed in ambiguous_labels.",
    "Tier agreement is scored only where the authored label has a tier set. Ambiguous tier sets accept any member and "
    "are reported in a separate metric so they cannot inflate the determinate one.",
    "jd_quote_matches_authored_anchor compares the entry's quote with the authored JD anchor by whole-phrase "
    "containment. A mismatch means the quote differs from the authored anchor, not necessarily that it is wrong.",
]
LIMITATIONS_COURSE = [
    "Only the eligibility rule is scored, on the supplied normalised output. Whether recommended courses are relevant "
    "or good is not measured and no recommender or YouTube call is made.",
    "label_eligibility_agreement is downstream of E2: a gap the model did not flag shows up there as well.",
]

# ---------------------------------------------------------------------------------------------------------------
# Production logic (extracted by stages.py; nothing is imported from the application)
# ---------------------------------------------------------------------------------------------------------------

_PROD = {}


def _load_sibling(name, filename):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _production():
    """(parser slice, display slice), extracted once from src/parser.py and frontend/display.py as text."""
    if not _PROD:
        stages = _load_sibling("pipeline_stages", "stages.py")
        _PROD["parser"] = stages.load_parser()
        _PROD["display"] = stages.load_display()
    return _PROD["parser"], _PROD["display"]


def _key(text):
    if not isinstance(text, str):
        return ""
    return _production()[0]._skill_key(text)


def _keys(strings):
    return {key for key in (_key(text) for text in strings if isinstance(text, str)) if key}


def _tiers():
    return tuple(_production()[0].PRIORITY_TIERS)


# ---------------------------------------------------------------------------------------------------------------
# Metric records
# ---------------------------------------------------------------------------------------------------------------

def _metric(stage, name, numerator, denominator, direction, unscorable=None, note="", counts=None,
            unavailable_reason=None, permanent=False):
    unscorable = dict(sorted((unscorable or {}).items()))
    available = (unavailable_reason is None and numerator is not None and denominator is not None and denominator > 0)
    record = {
        "stage": stage, "metric": name, "numerator": numerator, "denominator": denominator,
        "rate": round(numerator / denominator, 6) if available else None,
        "available": available, "direction": direction,
        "unscorable_count": sum(unscorable.values()), "unscorable_reasons": unscorable, "note": note,
    }
    if not available:
        if unavailable_reason is None:
            unavailable_reason = ("no output to score" if numerator is None and denominator is not None
                                  else "nothing scorable: the denominator is 0")
        record["unavailable_reason"] = unavailable_reason
    if permanent:
        record["permanently_unavailable"] = True
    if counts is not None:
        record["counts"] = counts
    return record


def _never(stage, name, reason):
    """A metric the labels cannot support, whatever the output."""
    return _metric(stage, name, None, None, HIGHER, unavailable_reason=reason, permanent=True)


def _blocked(stage, name, denominator, reason, direction, note=""):
    """A metric that could not be computed because the stage output is missing or unusable."""
    unscorable = {reason: denominator} if denominator else {}
    return _metric(stage, name, None, denominator, direction, unscorable=unscorable, note=note,
                   unavailable_reason=f"stage output unusable: {reason}")


def _tally(stage, name, items, judge, direction, note="", counts=None):
    """judge(item) -> (True|False, None) when scored, or (None, reason) when unscorable."""
    correct = total = 0
    unscorable = {}
    for item in items:
        verdict, reason = judge(item)
        if verdict is None:
            unscorable[reason] = unscorable.get(reason, 0) + 1
            continue
        total += 1
        correct += 1 if verdict else 0
    return _metric(stage, name, correct, total, direction, unscorable=unscorable, note=note, counts=counts)


def _stage_record(stage, status, detail, metrics, skills, assumptions, limitations, observations=None):
    return {"stage": stage, "status": status, "status_detail": detail, "metrics": metrics, "skills": skills,
            "observations": observations or {}, "matching_assumptions": list(assumptions),
            "limitations": list(limitations)}


def _label_skills(label):
    skills = label.get("skills") if isinstance(label, dict) else None
    return [skill for skill in skills if isinstance(skill, dict)] if isinstance(skills, list) else []


# ---------------------------------------------------------------------------------------------------------------
# E1 - JD skill extraction
# ---------------------------------------------------------------------------------------------------------------

def _read_e1(output):
    """-> (parsed or None, status, detail). Accepts the extractor's JSON object or its JSON text."""
    if output is None:
        return None, "output_missing", None
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None, "invalid_output", "not valid JSON"
    if not isinstance(output, dict):
        return None, "invalid_output", "top level is not an object"
    categories, notes, bad_items = {}, [], 0
    for category in E1_CATEGORIES:
        if category not in output:
            categories[category] = []
            notes.append(f"category_missing:{category}")
            continue
        value = output[category]
        if not isinstance(value, list):
            return None, "invalid_output", f"'{category}' is not a list"
        items, seen = [], set()
        for item in value:
            key = _key(item)
            if not key:
                bad_items += 1
                continue
            if key not in seen:
                seen.add(key)
                items.append((item, key))
        categories[category] = items
    return {"categories": categories, "notes": notes, "bad_items": bad_items}, "scored", None


def score_e1(label, output):
    """Score one scenario's E1 (JD skill extraction) output against its label. Recall only; see the module docstring."""
    skills = _label_skills(label)
    parsed, status, detail = _read_e1(output)
    precision_reason = ("the labels do not establish that the authored skill list is exhaustive, so false positives "
                        "cannot be identified")
    fixed = [_never("e1", "precision", precision_reason), _never("e1", "f1", precision_reason)]
    if parsed is None:
        metrics = [_blocked("e1", "skill_recall", len(skills), status, HIGHER)] + fixed
        return _stage_record("e1", status, detail, metrics, [], MATCHING_E1, LIMITATIONS_E1)

    all_label_keys = set()
    for skill in skills:
        all_label_keys |= _keys((skill.get("extraction") or {}).get("accepted_forms") or [])
    records = []
    for skill in skills:
        forms = _keys((skill.get("extraction") or {}).get("accepted_forms") or [])
        category = skill.get("category")
        record = {"skill_id": skill.get("skill_id"), "category": category, "matched": None,
                  "matched_form": None, "matched_by_canonical_form": None, "found_in_other_category": [],
                  "unscorable_reason": None}
        if not forms:
            record["unscorable_reason"] = "no_usable_accepted_form"
            records.append(record)
            continue
        hits = [item for item, key in parsed["categories"].get(category, []) if key in forms]
        record["matched"] = bool(hits)
        if hits:
            record["matched_form"] = hits[0]
            record["matched_by_canonical_form"] = _key(hits[0]) == _key(skill.get("canonical"))
        record["found_in_other_category"] = sorted(
            other for other, items in parsed["categories"].items()
            if other != category and any(key in forms for _item, key in items))
        records.append(record)

    true_positives = sum(1 for r in records if r["matched"] is True)
    false_negatives = sum(1 for r in records if r["matched"] is False)
    unscorable = {}
    for record in records:
        if record["unscorable_reason"]:
            unscorable[record["unscorable_reason"]] = unscorable.get(record["unscorable_reason"], 0) + 1
    unlabelled = [{"category": category, "item": item}
                  for category in E1_CATEGORIES for item, key in parsed["categories"][category]
                  if key not in all_label_keys]
    counts = {"true_positives": true_positives, "false_negatives": false_negatives, "false_positives": None,
              "unlabelled_predictions": len(unlabelled),
              "matched_by_non_canonical_form": sum(1 for r in records if r["matched_by_canonical_form"] is False)}
    recall = _metric("e1", "skill_recall", true_positives, true_positives + false_negatives, HIGHER,
                     unscorable=unscorable, counts=counts,
                     note="Share of authored JD skills found in the supplied extraction.",
                     unavailable_reason=None if (true_positives + false_negatives) else
                     "no labelled skill could be scored (empty ground truth or no usable accepted form)")
    observations = {"input_notes": parsed["notes"], "non_string_or_empty_items": parsed["bad_items"],
                    "unlabelled_predictions": unlabelled}
    return _stage_record("e1", "scored", None, [recall] + fixed, records, MATCHING_E1, LIMITATIONS_E1, observations)


# ---------------------------------------------------------------------------------------------------------------
# E2 - skill-gap comparison, normalised output
# ---------------------------------------------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _read_raw(raw):
    """Evaluation-side reading of the raw model string -> (entries or None, state)."""
    if raw is None:
        return None, "not_supplied"
    if not isinstance(raw, str):
        return None, "not_a_string"
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError:
        return None, "not_valid_json"
    if not isinstance(data, dict):
        return None, "not_an_object"
    statuses = tuple(_production()[0].SKILL_STATUSES)
    entries = []
    for tier in _tiers():
        for item in data.get(tier) if isinstance(data.get(tier), list) else []:
            if isinstance(item, dict) and isinstance(item.get("skill"), str):
                status = item.get("status")
                entries.append({"tier": tier, "key": _key(item["skill"]),
                                "raw_status": status if isinstance(status, str) else None,
                                "valid": isinstance(status, str) and status.strip().lower() in statuses})
    return entries, "parsed"


def _read_normalized(normalized):
    """-> (entries or None, status, detail, anomalies, observations)."""
    if normalized is None:
        return None, "output_missing", None, [], {}
    tiers = _tiers()
    if (not isinstance(normalized, dict) or not isinstance(normalized.get("normalization_ok"), bool)
            or not all(isinstance(normalized.get(tier), list) for tier in tiers)):
        return None, "invalid_output", "not a normalised comparison object", [], {}
    diagnostics = normalized.get("diagnostics")
    observations = {"normalization_ok": normalized["normalization_ok"],
                    "diagnostics": [d for d in diagnostics if isinstance(d, str)] if isinstance(diagnostics, list) else []}
    if normalized["normalization_ok"] is not True:
        return None, "normalization_failed", "normalization_ok is false (fallback result)", [], observations
    valid_statuses = tuple(_production()[0].SKILL_STATUSES)
    entries, anomalies = [], []
    for tier in tiers:
        for index, entry in enumerate(normalized[tier]):
            if not isinstance(entry, dict) or not isinstance(entry.get("skill"), str) or not _key(entry["skill"]):
                anomalies.append({"type": "malformed_entry", "tier": tier, "index": index})
            elif entry.get("status") not in valid_statuses:
                anomalies.append({"type": "invalid_normalized_status", "tier": tier, "index": index,
                                  "status": entry.get("status") if isinstance(entry.get("status"), str) else None})
            else:
                entries.append({"tier": tier, "key": _key(entry["skill"]), "entry": entry})
    return entries, "scored", None, anomalies, observations


def _norm_text(text):
    return " ".join(text.casefold().split())


def _quote_matches_anchor(label_skill, entry):
    quote = entry.get("jd_evidence")
    anchors = [m.get("anchor") for m in (label_skill.get("jd") or {}).get("mentions") or []
               if isinstance(m, dict) and isinstance(m.get("anchor"), str)]
    if not isinstance(quote, str) or not quote.strip() or not anchors:
        return None
    quote = _norm_text(quote)
    return any(_norm_text(anchor) in quote or quote in _norm_text(anchor) for anchor in anchors)


def _absence_provenance(forms, raw_entries, raw_state):
    """Why a label has no normalised entry -> (provenance, raw statuses seen)."""
    if raw_entries is None:
        return "unknown", []
    seen = [e for e in raw_entries if e["key"] in forms]
    if not seen:
        return "genuine_absence", []
    statuses = sorted({e["raw_status"] if e["raw_status"] is not None else "<missing>" for e in seen})
    if any(not e["valid"] for e in seen):
        return "rejected_raw_entry", statuses
    return "dropped_raw_entry_other", statuses


def _label_record(skill, entries, raw_entries, raw_state):
    forms = _keys([skill.get("canonical")] + list(skill.get("aliases") or []))
    matched = [e for e in entries if e["key"] in forms]
    observed = sorted({"EMITTED:" + e["entry"]["status"] for e in matched}) or ["ABSENT"]
    record = {
        "skill_id": skill.get("skill_id"), "category": skill.get("category"),
        "authoring_truth": skill.get("authoring_truth"), "outcome_class": skill.get("outcome_class"),
        "acceptable_outcomes": list(skill.get("acceptable_outcomes") or []),
        "observed_outcomes": observed, "absence_provenance": None, "raw_statuses_for_absence": [],
        "matched_entries": [{"tier": e["tier"], "skill": e["entry"]["skill"], "status": e["entry"]["status"],
                             "evidence_found": e["entry"].get("evidence_found") if isinstance(
                                 e["entry"].get("evidence_found"), bool) else None,
                             "jd_quote_matches_anchor": _quote_matches_anchor(skill, e["entry"]),
                             "resume_evidence_found": e["entry"].get("resume_evidence_found") if isinstance(
                                 e["entry"].get("resume_evidence_found"), bool) else None} for e in matched],
        "status_label_pass": None, "status_counted_correct": None, "interpretation": None,
    }
    if not matched:
        record["absence_provenance"], record["raw_statuses_for_absence"] = _absence_provenance(
            forms, raw_entries, raw_state)
    record["_forms"] = forms
    return record


def _interpret(record):
    """Fill status_label_pass / status_counted_correct / interpretation for a determinate label."""
    passes = all(outcome in record["acceptable_outcomes"] for outcome in record["observed_outcomes"])
    provenance = record["absence_provenance"]
    record["status_label_pass"] = passes
    rejected = record["observed_outcomes"] == ["ABSENT"] and provenance in ("rejected_raw_entry",
                                                                             "dropped_raw_entry_other")
    record["status_counted_correct"] = bool(passes and not rejected)
    if rejected:
        record["interpretation"] = ("absent_because_parser_dropped_a_raw_entry: the label accepts ABSENT, but this is a "
                                    "rejected entry, not a model prediction of absence; not counted correct")
    elif record["observed_outcomes"] == ["ABSENT"]:
        record["interpretation"] = {"genuine_absence": "genuine_absence: no raw or normalised entry for the skill",
                                    "unknown": "absence_provenance_unknown: raw output not supplied or unreadable"
                                    }.get(provenance, "absent")
    else:
        record["interpretation"] = ("emitted_outcome_is_acceptable" if passes else "emitted_outcome_not_acceptable")


def _tier_item(record, skill):
    """(passes?, reason) for the tier of one label; the caller picks determinate or ambiguous labels."""
    allowed = (skill.get("jd") or {}).get("expected_tier_set")
    tiers = sorted({e["tier"] for e in record["matched_entries"]})
    if not tiers:
        return None, "no_emitted_entry"
    if not isinstance(allowed, list) or not allowed:
        return None, "no_expected_tier_set"
    return all(tier in allowed for tier in tiers), None


def score_e2(label, normalized, raw=None):
    """Score one scenario's E2 normalised output (and, optionally, its raw string) against its label."""
    skills = _label_skills(label)
    entries, status, detail, anomalies, observations = _read_normalized(normalized)
    raw_entries, raw_state = _read_raw(raw)
    names = ("status_agreement", "status_agreement_strict", "gap_detection_recall", "false_flag_rate",
             "status_exact_match_given_flagged",
             "tier_agreement_determinate", "tier_agreement_ambiguous_set", "jd_evidence_found",
             "jd_quote_matches_authored_anchor", "resume_evidence_quote_found")
    directions = {"false_flag_rate": LOWER}
    gap_scope = [s for s in skills if s.get("authoring_truth") in GAP_TRUTHS and s.get("outcome_class") == "determinate"]
    demo_scope = [s for s in skills if s.get("authoring_truth") == "demonstrated"
                  and s.get("outcome_class") == "determinate"]
    if entries is None:
        dens = {"status_agreement": len(skills), "status_agreement_strict": len(skills),
                "gap_detection_recall": len(gap_scope),
                "false_flag_rate": len(demo_scope)}
        metrics = [_blocked("e2", name, dens.get(name), status, directions.get(name, HIGHER)) for name in names]
        observations = dict(observations, raw_reading=raw_state)
        return _stage_record("e2", status, detail, metrics, [], MATCHING_E2, LIMITATIONS_E2, observations)

    records = [_label_record(skill, entries, raw_entries, raw_state) for skill in skills]
    by_id = {id(skill): record for skill, record in zip(skills, records)}
    for skill, record in zip(skills, records):
        if skill.get("outcome_class") == "determinate":
            _interpret(record)
        else:
            record["interpretation"] = "label_not_scored: outcome_class is not 'determinate'"

    all_forms = [record["_forms"] for record in records]
    unlabelled_entries = [{"tier": e["tier"], "skill": e["entry"]["skill"], "status": e["entry"]["status"]}
                          for e in entries if not any(e["key"] in forms for forms in all_forms)]
    multi_label = sorted({e["entry"]["skill"] for e in entries if sum(e["key"] in forms for forms in all_forms) > 1})

    def status_judge(skill):
        record = by_id[id(skill)]
        if skill.get("outcome_class") != "determinate":
            return None, ("ambiguous_label_excluded" if skill.get("outcome_class") == "ambiguous"
                          else "unrecognised_outcome_class")
        return record["status_counted_correct"], None

    def strict_status_judge(skill):
        """As status_judge, but an ABSENT whose provenance is unknown is unscorable instead of credited.

        Only the cases that would be CREDITED are excluded: an ABSENT the label does not accept is a miss whatever its
        provenance, and excluding it would hide the failure."""
        record = by_id[id(skill)]
        if (skill.get("outcome_class") == "determinate" and record["observed_outcomes"] == ["ABSENT"]
                and record["absence_provenance"] == "unknown" and record["status_counted_correct"]):
            return None, "absence_provenance_unknown"
        return status_judge(skill)

    def flagged(record):
        return any(outcome.startswith("EMITTED:") for outcome in record["observed_outcomes"])

    confusion = {}
    for skill in gap_scope:
        record = by_id[id(skill)]
        observed = ("ABSENT" if not flagged(record) else
                    record["observed_outcomes"][0].split(":", 1)[1] if len(record["observed_outcomes"]) == 1 else "mixed")
        row = confusion.setdefault(skill["authoring_truth"], {})
        row[observed] = row.get(observed, 0) + 1
    confusion = {truth: dict(sorted(row.items())) for truth, row in sorted(confusion.items())}

    def exact_judge(skill):
        record = by_id[id(skill)]
        if not flagged(record):
            return None, "not_flagged"
        return record["observed_outcomes"] == ["EMITTED:" + skill["authoring_truth"]], None

    determinate_tier = [s for s in skills if (s.get("jd") or {}).get("tier_class") == "determinate"]
    ambiguous_tier = [s for s in skills if (s.get("jd") or {}).get("tier_class") == "ambiguous"]
    matched_entries = [(skill, entry) for skill, record in zip(skills, records) for entry in record["matched_entries"]]

    metrics = [
        _tally("e2", "status_agreement", skills, status_judge, HIGHER,
               note="QUALIFIED agreement, determinate labels only. An ABSENT counts correct when it is a genuine "
                    "absence OR its provenance is unknown (raw output not supplied/readable), so it can include "
                    "absences that were really parser-dropped entries. Read it with status_agreement_strict. "
                    "ABSENT caused by a rejected raw entry is not counted correct.",
               counts={"absence_provenance": dict(sorted(_count(
                   r["absence_provenance"] for r in records
                   if r["absence_provenance"] and r["outcome_class"] == "determinate"))),
                   "unknown_provenance_counted_correct": sum(
                       1 for r in records if r["outcome_class"] == "determinate"
                       and r["absence_provenance"] == "unknown" and r["status_counted_correct"])}),
        _tally("e2", "status_agreement_strict", skills, strict_status_judge, HIGHER,
               note="STRICT agreement: as status_agreement, but an ABSENT with unknown provenance is excluded from "
                    "the denominator (reason absence_provenance_unknown) instead of being credited. Supply the raw "
                    "string to make those cases scorable."),
        _tally("e2", "gap_detection_recall", gap_scope, lambda s: (flagged(by_id[id(s)]), None), HIGHER,
               note="Labels authored missing or partial: share the model flagged at all (any status).",
               counts={"truth_vs_observed": confusion}),
        _tally("e2", "false_flag_rate", demo_scope, lambda s: (flagged(by_id[id(s)]), None), LOWER,
               note="Labels authored demonstrated that the model flagged as a gap anyway (unsupported flags). "
                    "A rejected raw no_gap is not a flag. It is 0 for an output that flags nothing, so read it "
                    "with gap_detection_recall."),
        _tally("e2", "status_exact_match_given_flagged", gap_scope, exact_judge, HIGHER,
               note="Missing versus partial, strict: the emitted status equals the authored truth. Lenient "
                    "alternatives in acceptable_outcomes are NOT applied here; see status_agreement."),
        _tally("e2", "tier_agreement_determinate", determinate_tier, lambda s: _tier_item(by_id[id(s)], s),
               HIGHER, note="The emitted tier is the single authored tier."),
        _tally("e2", "tier_agreement_ambiguous_set", ambiguous_tier, lambda s: _tier_item(by_id[id(s)], s),
               HIGHER, note="The emitted tier is any member of an authored ambiguous set. Kept apart from the "
                            "determinate metric."),
        _tally("e2", "jd_evidence_found", matched_entries,
               lambda pair: (None, "evidence_found_not_boolean") if pair[1]["evidence_found"] is None
               else (pair[1]["evidence_found"], None), HIGHER,
               note="Matched entries whose JD quote was found in the job description (the parser's own flag)."),
        _tally("e2", "jd_quote_matches_authored_anchor", matched_entries,
               lambda pair: (None, "quote_or_anchor_unavailable") if pair[1]["jd_quote_matches_anchor"] is None
               else (pair[1]["jd_quote_matches_anchor"], None), HIGHER,
               note="Matched entries whose JD quote contains (or is contained in) an authored JD anchor."),
        _tally("e2", "resume_evidence_quote_found", matched_entries,
               lambda pair: (None, "resume_evidence_not_annotated") if pair[1]["resume_evidence_found"] is None
               else (pair[1]["resume_evidence_found"], None), HIGHER,
               note="Matched entries that carry a resume quote and whether the parser found it in the resume text."),
    ]
    ambiguous_labels = [{"skill_id": r["skill_id"], "observed_outcomes": r["observed_outcomes"],
                         "absence_provenance": r["absence_provenance"]}
                        for r in records if r["outcome_class"] != "determinate"]
    warnings = []
    if not entries and gap_scope:
        warnings.append("empty_comparison: the output flags no skill at all, so its ABSENT results and a false_flag_rate "
                        "of 0 are vacuous; gap_detection_recall shows what was missed.")
    observations = dict(observations, raw_reading=raw_state, anomalies=anomalies,
                        unlabelled_entries=unlabelled_entries, entries_matching_several_labels=multi_label,
                        ambiguous_labels=ambiguous_labels, warnings=warnings)
    for record in records:
        record.pop("_forms", None)
    return _stage_record("e2", "scored", None, metrics, records, MATCHING_E2, LIMITATIONS_E2, observations)


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts.items()


# ---------------------------------------------------------------------------------------------------------------
# Course eligibility
# ---------------------------------------------------------------------------------------------------------------

def _expected_by_rule(normalized, display):
    """The documented rule, restated independently of get_recommendation_options (A3).

    -> (usable, offered skill names in priority order, skipped count). Usability uses the frontend's own validity check."""
    state, data = display.get_prioritized_view({"prioritized_missing_skills": normalized})
    if state != "ok":
        return False, [], 0
    seen, offered, skipped = set(), [], 0
    for tier in _tiers():
        for entry in data[tier]:
            key = _key(entry["skill"])
            if key in seen:
                continue
            seen.add(key)
            if entry["category"] != "technical_skills":
                continue
            if entry["evidence_found"] is not True:
                skipped += 1
                continue
            offered.append(entry["skill"].strip())
    return True, offered, skipped


def score_course(label, normalized):
    """Score eligibility-rule conformance (not relevance) on a scenario's supplied normalised E2 output."""
    skills = _label_skills(label)
    names = ("offered_options_match_rule", "skipped_count_match_rule", "label_eligibility_agreement")
    if normalized is None or not isinstance(normalized, dict):
        status = "output_missing" if normalized is None else "invalid_output"
        detail = None if normalized is None else "not an object"
        metrics = [_blocked("course", name, 1 if name != names[2] else len(skills), status, HIGHER) for name in names]
        return _stage_record("course", status, detail, metrics, [], [], LIMITATIONS_COURSE)

    _parser, display = _production()
    options, skipped = display.get_recommendation_options(normalized)
    usable, expected_offered, expected_skipped = _expected_by_rule(normalized, display)
    observed_offered = list(options)
    offered_ok = observed_offered == expected_offered
    skipped_ok = skipped == expected_skipped
    offered_keys = _keys(observed_offered)

    records = []
    for skill in skills:
        forms = _keys([skill.get("canonical")] + list(skill.get("aliases") or []))
        expected = (skill.get("course") or {}).get("eligible_expected")
        record = {"skill_id": skill.get("skill_id"), "eligible_expected": expected if isinstance(expected, bool) else None,
                  "eligible_observed": bool(forms & offered_keys), "outcome_class": skill.get("outcome_class")}
        records.append(record)

    def agreement(pair):
        skill, record = pair
        if not usable:
            return None, "comparison_output_unusable"
        if skill.get("outcome_class") != "determinate":
            return None, "ambiguous_label_excluded"
        if record["eligible_expected"] is None:
            return None, "eligible_expected_missing"
        return record["eligible_expected"] == record["eligible_observed"], None

    if usable:
        conformance = [
            _metric("course", names[0], 1 if offered_ok else 0, 1, HIGHER,
                    note="The offered skills equal the skills the documented rule selects (technical, quote found, "
                         "one per skill key, priority order). A regression check on the rule, not a quality measure."),
            _metric("course", names[1], 1 if skipped_ok else 0, 1, HIGHER,
                    note="The skipped count equals the technical entries whose quote was not found."),
        ]
    else:  # an unusable E2 result yields ({}, 0) by contract, which is not evidence that the rule was conformed to
        conformance = [_blocked("course", name, 1, "comparison_output_unusable", HIGHER) for name in names[:2]]
    metrics = conformance + [
        _tally("course", names[2], list(zip(skills, records)), agreement, HIGHER,
               note="Determinate labels: whether the skill was offered matches the authored eligible_expected."),
    ]
    observations = {"comparison_output_usable": usable, "offered": observed_offered, "skipped": skipped,
                    "expected_offered_by_rule": expected_offered, "expected_skipped_by_rule": expected_skipped}
    return _stage_record("course", "scored" if usable else "comparison_output_unusable",
                         None if usable else "the E2 result is unusable, so no eligibility conformance is reported",
                         metrics, records, [], LIMITATIONS_COURSE, observations)


# ---------------------------------------------------------------------------------------------------------------
# Dataset report
# ---------------------------------------------------------------------------------------------------------------

def _merge_counts(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: _merge_counts(left.get(key), right.get(key)) for key in sorted(set(left) | set(right))}
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (left or 0) + (right or 0)
    return left if right is None else right


def _aggregate_metric(per_scenario):
    """Pool one metric over scenarios. Pooling is descriptive (a micro-average); per-scenario figures stay listed."""
    first = per_scenario[0][1]
    numerator = denominator = 0
    scored = missing = 0
    unscorable, counts = {}, None
    for _sid, metric in per_scenario:
        for reason, count in metric["unscorable_reasons"].items():
            unscorable[reason] = unscorable.get(reason, 0) + count
        if metric["numerator"] is not None and metric["denominator"] is not None:
            numerator += metric["numerator"]
            denominator += metric["denominator"]
            scored += 1 if metric["denominator"] > 0 else 0
        else:
            missing += 1
        if isinstance(metric.get("counts"), dict):
            counts = _merge_counts(counts or {}, metric["counts"])
    permanent = all(m.get("permanently_unavailable") for _s, m in per_scenario)
    reason = None
    if permanent:
        reason = first.get("unavailable_reason")
    elif denominator == 0:
        reason = "nothing scorable in any scenario" if not missing else "no scenario had a usable output to score"
    record = _metric(first["stage"], first["metric"], None if permanent else numerator,
                     None if permanent else denominator, first["direction"], unscorable=unscorable,
                     note=first["note"], counts=counts, unavailable_reason=reason, permanent=permanent)
    record["scenarios_scored"] = scored
    record["scenarios_without_usable_output"] = missing
    record["per_scenario"] = [{"scenario_id": sid, "numerator": m["numerator"], "denominator": m["denominator"],
                               "rate": m["rate"]} for sid, m in per_scenario]
    return record


def score_dataset(dataset, outputs):
    """Score every scenario in `dataset` (the dict returned by dataset.load_dataset) against `outputs`.

    outputs: {scenario_id: {"e1": E1 output (dict or JSON text), "e2_normalized": prioritised dict,
              "e2_raw": raw model string (optional)}}. Anything absent is reported as missing, never as correct.
    Returns a deterministic, JSON-serialisable report. Nothing is written to disk."""
    if not isinstance(outputs, dict):
        raise ValueError("outputs must be a dict keyed by scenario id")
    ids = list(dataset["ids"])
    unknown = sorted(set(outputs) - set(ids))
    scenarios, pooled = [], {"e1": {}, "e2": {}, "course": {}}
    for sid in ids:
        supplied = outputs.get(sid)
        if supplied is not None and not isinstance(supplied, dict):
            raise ValueError(f"outputs[{sid!r}] must be a dict")
        supplied = supplied or {}
        label = dataset["labels"][sid]
        stages = {"e1": score_e1(label, supplied.get("e1")),
                  "e2": score_e2(label, supplied.get("e2_normalized"), supplied.get("e2_raw")),
                  "course": score_course(label, supplied.get("e2_normalized"))}
        for stage, record in stages.items():
            for metric in record["metrics"]:
                pooled[stage].setdefault(metric["metric"], []).append((sid, metric))
        scenarios.append({"scenario_id": sid, "labels_version": dataset["labels"][sid].get("labels_version"),
                          "output_supplied": sid in outputs, "stages": stages})
    aggregate = {stage: [_aggregate_metric(items) for items in metrics.values()] for stage, metrics in pooled.items()}
    warnings = [f"e2:{s['scenario_id']}: {w}" for s in scenarios for w in s["stages"]["e2"]["observations"].get(
        "warnings", [])]
    return {
        "report_version": REPORT_VERSION,
        "dataset": {"ids": ids, "verified_hashes": dict(dataset.get("verified_hashes") or {})},
        "scenarios": scenarios,
        "aggregate": aggregate,
        "unknown_output_ids": unknown,
        "warnings": warnings,
        "limitations": [
            "Scores describe only the supplied outputs. They are not a measurement of a live model unless the outputs "
            "were produced by one, and then only for these hand-authored scenarios.",
            "Pooled rates sum numerators and denominators across 10 hand-authored scenarios. They are descriptive; "
            "per-scenario results are listed beside them and must be read with them.",
            "Precision and F1 are unavailable for E1 because the labels do not establish an exhaustive skill list.",
        ],
    }
