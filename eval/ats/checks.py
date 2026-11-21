"""Deterministic screening checks for ATS bullet rewrites (Stage 5, checkpoint A).

Pure functions on strings and plain dicts: no network, no LLM, no file access, no LangChain. Standard library only.

IMPORTANT: these are SCREENING SIGNALS, not proof of factual accuracy.
  * False positives: a legitimate reformatted or derived number, or a technology the resume names in other words,
    can be flagged.
  * False negatives: an unsupported claim worded without a number or a listed technology is NOT detected, and the
    number blind spots listed under D2 below exist.
Human fidelity review remains the reference.

Checks:
  D1 parse_ats_output        schema / shape of the model output, parsed the way production parses it
  D2 check_invented_numbers  numbers in a rewrite that the source resume does not contain (numbers only)
  D3 check_absent_skills     skills a case declares as absent from the resume that a rewrite mentions
  D4 check_technologies      vocabulary technologies in a rewrite that the resume does not mention (advisory)
  D5 check_sections          output sections versus the resume's sections, and bullet-count changes
  D6 find_duplicates / find_unchanged   near-duplicate rewrites; rewrites identical to a source bullet

D2 number rules (what is read as a number, and what is not)
  Detected, compared by VALUE after normalisation:
    * integers and decimals (3.50 = 3.5), dotted versions (3.11, 3.11.2, compared as written), years, ranges (each
      end separately), and the parts of dates (2023-10-27 gives 2023, 10, 27);
    * comma grouping, international (1,000,000) and Indian (1,00,000; 12,34,567.89);
    * magnitude: attached k/K (thousand) and M (million), and the words thousand, million, billion, lakh, crore
      (optional plural, one optional space), so 1.5 million = 1.5M = 1500000 and 2 lakh = 2,00,000 = 200k;
    * a multiplier x/X (3.5x), a percent sign (35% or 35 %), and the word percent;
    * units, attached or separated by one space: ms, msec(s), sec(s), min(s), hr(s), the words millisecond(s),
      second(s), minute(s), hour(s), day(s), week(s), month(s), year(s), and KB/MB/GB/TB/PB (any case of the
      B: 50ms = 50 ms, 5GB = 5 GB);
    * attached only: s (5s), h (2h) and the ordinals st, nd, rd, th (1st, 95th).
  Units and magnitudes are different things: a magnitude changes the value, a unit is only RECORDED (see
  extract_quantities) and is not part of the comparison. Equal values with different units (35 hours vs 35%) pass
  the invented-number check; this is NOT a claim that they are equivalent. check_invented_numbers reports them
  separately as the advisory "unit_mismatch".
  Not detected (blind spots) or deliberately conservative:
    * technology and product names with digits, which are never read as numbers: S3, OAuth2, GPT-4, GPT-4o, Llama-3,
      Neo4j, Top-3, COVID-19, ISO-27001, v2, p99 (a number glued to a letter, or letter-hyphen-digit, is part of
      a name). The spaced forms (GPT 4, Python 3, Llama 3) ARE read as numbers;
    * units outside the whitelist (kg, km, mph, rps, Mbps, ...) are never recognised as units. ATTACHED to a
      number (3.5kg, 60mph, 5000rps, 20Mbps, 5G) the whole token is ignored; SEPARATED by a space (3.5 kg, 60 mph)
      the digits are read as a plain number with no unit. So the two spellings are NOT treated consistently:
      source "3.5kg" with rewrite "3.5 kg" flags 3.5 as invented (false positive), and an invented "60mph" is
      never flagged (false negative) while an invented "60 mph" is. Only the whitelisted units above are
      consistent across the attached and spaced spellings. An attached lowercase m (5m: minutes, metres or
      millions?) is ignored as well;
    * space grouping (10 000) and comma decimals (1,5) are not merged: they give separate tokens (10 and 0, 1 and
      5), so equal values written in another format can be flagged;
    * a European-style X.YYY (1.000, 2.500: thousand or decimal?) is ambiguous and is compared as written, so it
      matches only the identical text;
    * "5 M" and "5 k" (spaced single letters), spelled-out numbers above twenty ("thirty", "twenty-five" reads as
      20 and 5, "hundred") and other languages;
    * a derived figure (a duration or percentage computed from the source) counts as invented unless the source
      states it, so "(2 years)" computed from "2020-2022" is flagged.
"""
import json
import re
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------------------------------------------
# D1: parsing and schema
# ---------------------------------------------------------------------------------------------------------------


def strip_fences(text):
    """Exactly the cleaning src/parser.py applies to the ATS response before it is stored."""
    return re.sub(r"^```json|```$", "", text.strip()).strip()


def parse_ats_output(raw_output):
    """Parse a model response and validate the {"optimized_bullets": [{"section", "bullets"}]} shape.

    Returns {"status", "production_would_500", "sections", "problems"}.
      status: "ok" | "invalid_json" | "wrong_shape" | "empty"
        invalid_json  the cleaned text is not JSON (production's json.loads would fail -> HTTP 500)
        wrong_shape   JSON, but not a dict with an "optimized_bullets" list of valid items
        empty         "optimized_bullets" is an empty list
      sections: [{"section": str, "bullets": [str, ...]}] when status is "ok", otherwise [].
    Never raises; problems are short fixed descriptions and never contain the model's text.
    """
    result = {"status": "invalid_json", "production_would_500": True, "sections": [], "problems": []}
    if not isinstance(raw_output, str):
        result["problems"].append("output is not a string")
        return result
    try:
        data = json.loads(strip_fences(raw_output))
    except (ValueError, RecursionError):
        result["problems"].append("output is not valid JSON")
        return result
    result["production_would_500"] = False
    if not isinstance(data, dict):
        result.update(status="wrong_shape", problems=["JSON is not an object"])
        return result
    items = data.get("optimized_bullets")
    if not isinstance(items, list):
        result.update(status="wrong_shape", problems=['"optimized_bullets" is missing or not a list'])
        return result
    if not items:
        result.update(status="empty", problems=['"optimized_bullets" is an empty list'])
        return result
    sections, problems = [], []
    for position, item in enumerate(items):
        label = f"item {position}"
        if not isinstance(item, dict):
            problems.append(f"{label} is not an object")
            continue
        section = item.get("section")
        bullets = item.get("bullets")
        if not isinstance(section, str) or not section.strip():
            problems.append(f"{label} has a missing or blank section")
        if not isinstance(bullets, list) or not bullets:
            problems.append(f"{label} has a missing or empty bullets list")
            continue
        if not all(isinstance(bullet, str) and bullet.strip() for bullet in bullets):
            problems.append(f"{label} has a blank or non-string bullet")
            continue
        if isinstance(section, str) and section.strip():
            sections.append({"section": section, "bullets": list(bullets)})
    if problems:
        result.update(status="wrong_shape", problems=problems)
        return result
    result.update(status="ok", sections=sections)
    return result


# ---------------------------------------------------------------------------------------------------------------
# D2: invented numbers (numbers only; NOT a general unsupported-claim detector)
# ---------------------------------------------------------------------------------------------------------------

# A standalone number: not glued to a letter, digit or underscore before it, not part of letter-hyphen-digit
# names (GPT-4), and not the tail of a dotted version. Digit groups: international (1,000), Indian (1,00,000) or
# plain/dotted (3.5, 3.11.2). Group repetition is capped at 6 (more than that is not a realistic number and an
# unbounded repeat made long runs of comma groups quadratic). It may be followed by ONE of: a percent sign, a magnitude word, a whitelisted unit
# (attached or after one space), or an attached-only suffix (k, K, M, x, X, ordinals, s, h). Anything else glued to
# the digits (S3, 5kg, 5m) means no match at all; the guard at the end also stops "3.5kg" from degrading to "3".
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?<![A-Za-z]-)"
    r"(?P<num>\d{1,3}(?:,\d{3}){1,6}(?:\.\d+)?|\d{1,2}(?:,\d{2}){1,6},\d{3}(?:\.\d+)?|\d+(?:\.\d+)*)"
    r"(?:"
    r"(?P<pct>\s?%)"
    r"|\s?(?P<mag>(?i:thousand|million|billion|lakh|crore)s?)"
    r"|\s?(?P<unit>(?i:percent|milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
    r"|ms|msecs?|secs?|mins?|hrs?|[KkMmGgTtPp][Bb])"
    r"|(?P<att>[kK]|M|[xX]|st|nd|rd|th|s|h)"
    r")?"
    r"(?!\w|\.\d)"
)
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-•*–])\s+")

_MAGNITUDES = {"thousand": 1000, "million": 1000000, "billion": 1000000000, "lakh": 100000, "crore": 10000000}
_ATTACHED_MULTIPLIERS = {"k": 1000, "K": 1000, "M": 1000000}
_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")
_UNIT_LABELS = {
    "ms": "ms", "msec": "ms", "msecs": "ms", "millisecond": "ms", "milliseconds": "ms",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
    "year": "year", "years": "year", "percent": "%",
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
# Advisory only (never counted as invented numbers): words that state a quantity without digits. A magnitude word
# that directly follows a digit string (2 million) is part of that number and is not reported here.
_VAGUE_QUANTITY_WORDS = (
    "dozen", "dozens", "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
    "billions", "double", "doubled", "triple", "tripled", "half",
)


def _unit_label(text):
    """One label per unit family (hrs, hour and h are all "h"); storage units are lower case (GB -> gb)."""
    lowered = text.lower()
    return _UNIT_LABELS.get(lowered, lowered)


def _canonical_value(num, multiplier):
    """(canonical string, ambiguous). Commas removed, magnitude applied, trailing zeros dropped.

    A dotted version (3.11.2) and an ambiguous European-style X.YYY (1.000: one, or one thousand?) are compared as
    written, so they only match identical text; ambiguous is True for the latter.
    """
    if num.count(".") > 1:
        return num, False
    if multiplier == 1 and re.fullmatch(r"[1-9]\d{0,2}\.\d{3}", num):
        return num, True
    try:
        value = Decimal(num.replace(",", "")) * multiplier
    except InvalidOperation:
        return num, False
    return format(value.normalize(), "f"), False


def extract_quantities(text):
    """[{"raw", "value", "unit", "ambiguous"}, ...] for every standalone number in text.

    value: canonical string of the number with any magnitude (k, M, million, lakh, ...) applied.
    unit: "" (none or not whitelisted), "%", "x", "ordinal", or a unit label (ms, s, min, h, day, week, month,
    year, kb, mb, gb, tb, pb). A magnitude is NOT a unit: it changes value and leaves unit empty.
    A leading list marker ("1. ", "- ") is ignored. Non-string input gives [].
    """
    if not isinstance(text, str):
        return []
    text = _LIST_MARKER_RE.sub("", text, count=1)
    quantities = []
    for match in _NUMBER_RE.finditer(text):
        multiplier, unit = 1, ""
        if match.group("pct"):
            unit = "%"
        elif match.group("mag"):
            multiplier = _MAGNITUDES[match.group("mag").lower().rstrip("s")]
        elif match.group("unit"):
            unit = _unit_label(match.group("unit"))
        elif match.group("att"):
            attached = match.group("att")
            if attached in _ATTACHED_MULTIPLIERS:
                multiplier = _ATTACHED_MULTIPLIERS[attached]
            elif attached in ("x", "X"):
                unit = "x"
            elif attached in _ORDINAL_SUFFIXES:
                unit = "ordinal"
            else:
                unit = _unit_label(attached)
        value, ambiguous = _canonical_value(match.group("num"), multiplier)
        quantities.append({"raw": match.group(0).strip(), "value": value, "unit": unit, "ambiguous": ambiguous})
    return quantities


def extract_numbers(text):
    """[(as written, canonical value), ...]: extract_quantities without the unit."""
    return [(quantity["raw"], quantity["value"]) for quantity in extract_quantities(text)]


def source_numbers(source_text):
    """Canonical values the source states, including number words one..twenty ("three" allows 3)."""
    return set(source_units(source_text))


def source_units(source_text):
    """{canonical value: {units seen with it}} for the source; a number word has the empty unit."""
    units = {}
    if not isinstance(source_text, str):
        return units
    for quantity in extract_quantities(source_text):
        units.setdefault(quantity["value"], set()).add(quantity["unit"])
    for word in _WORD_RE.findall(source_text):
        units.setdefault(str(_NUMBER_WORDS[word.lower()]), set()).add("")
    return units


def example_numbers_from_prompt(prompt_text):
    """Canonical numbers inside the prompt's own JSON example (to detect few-shot echo). Empty if not found."""
    if not isinstance(prompt_text, str):
        return set()
    start = prompt_text.find("JSON Output Format Example")
    end = prompt_text.find("Inputs", start) if start != -1 else -1
    if start == -1 or end == -1:
        return set()
    return {canonical for _raw, canonical in extract_numbers(prompt_text[start:end])}


def check_invented_numbers(bullets, source_text, example_numbers=()):
    """Flag bullets containing a number the source text does not contain.

    The comparison is by VALUE (see the module header for what is read as a number). A year, a version or any
    other digit string must appear in the source too (no exemption). A figure computed from the source counts as
    invented unless the source states it.
    Per bullet: invented (raw tokens whose value is absent from the source), echo (invented values equal to a
    number in the prompt's example), unit_mismatch and quantity_words (both ADVISORY, not part of the invented
    count). unit_mismatch lists tokens whose value is in the source but only with a different unit than the one
    written (source "35 hours" or "35 users", rewrite "35%"): equal values with different units are not equivalent,
    and only this advisory shows it. Tokens without a recognised unit never produce a unit_mismatch.
    example_numbers: canonical numbers of the prompt's example.
    """
    units = source_units(source_text)
    allowed = set(units)
    source_lower = source_text.lower() if isinstance(source_text, str) else ""
    example = set(example_numbers)
    results, with_invented, with_echo, with_unit_mismatch, invalid = [], 0, 0, 0, 0
    for index, bullet in enumerate(bullets):
        if not isinstance(bullet, str):
            invalid += 1
            results.append({"index": index, "invalid": True, "invented": [], "echo": [], "unit_mismatch": [],
                            "quantity_words": []})
            continue
        quantities = extract_quantities(bullet)
        invented = [q["raw"] for q in quantities if q["value"] not in allowed]
        echo = [q["raw"] for q in quantities if q["value"] not in allowed and q["value"] in example]
        mismatch = [q["raw"] for q in quantities
                    if q["value"] in allowed and q["unit"] and q["unit"] not in units[q["value"]]]
        residual = _NUMBER_RE.sub(" ", _LIST_MARKER_RE.sub("", bullet, count=1)).lower()
        words = []
        for word in _VAGUE_QUANTITY_WORDS:
            if re.search(r"\b" + word + r"\b", residual) and not re.search(r"\b" + word + r"\b", source_lower):
                words.append(word)
        for word in _WORD_RE.findall(bullet):
            value = str(_NUMBER_WORDS[word.lower()])
            if value != "1" and value not in allowed:
                words.append(word.lower())
        with_invented += bool(invented)
        with_echo += bool(echo)
        with_unit_mismatch += bool(mismatch)
        results.append({"index": index, "invalid": False, "invented": invented, "echo": echo,
                        "unit_mismatch": mismatch, "quantity_words": words})
    return {"bullets_checked": len(results) - invalid, "bullets_invalid": invalid,
            "bullets_with_invented": with_invented, "bullets_with_example_echo": with_echo,
            "bullets_with_unit_mismatch": with_unit_mismatch, "results": results}


# ---------------------------------------------------------------------------------------------------------------
# D3: declared absent skills; D4: vocabulary technologies
# ---------------------------------------------------------------------------------------------------------------

_AMBIGUOUS_SHORT_NAMES = {"go", "r", "c", "d", "it", "ai", "ml", "bi", "rest"}


def _term_in(text, term):
    """Case-insensitive match of term bounded by non-alphanumeric characters (so Go does not match Google)."""
    if not isinstance(text, str) or not isinstance(term, str) or not term.strip():
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term.strip()) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def _skill_terms(skill):
    terms = [skill.get("name")] + list(skill.get("aliases") or [])
    return [term for term in terms if isinstance(term, str) and term.strip()]


def check_absent_skills(bullets, absent_skills):
    """Flag bullets that mention a skill the case declares absent from the resume.

    absent_skills: [{"name": str, "aliases": [str, ...]}]. Any mention counts (the output format has no place for
    "you lack X"), so a negated mention would also be flagged; human review sorts out such cases.
    """
    results, with_leak = [], 0
    for index, bullet in enumerate(bullets):
        leaked = []
        for skill in absent_skills or []:
            if isinstance(skill, dict) and any(_term_in(bullet, term) for term in _skill_terms(skill)):
                leaked.append(skill.get("name"))
        with_leak += bool(leaked)
        results.append({"index": index, "leaked": leaked})
    return {"bullets_checked": len(results), "bullets_with_leak": with_leak, "results": results}


def validate_absent_skills(absent_skills, resume_text, job_description):
    """Dataset sanity problems for a case's absent_skills (empty list = valid). Used by the case validator.

    Names or aliases shorter than three characters are rejected as ambiguous, which also rejects unambiguous ones
    such as C#: declare those with a longer alias or leave them out of the dataset.
    """
    problems = []
    seen = set()
    for position, skill in enumerate(absent_skills or []):
        label = f"absent skill {position}"
        if not isinstance(skill, dict) or not isinstance(skill.get("name"), str) or not skill["name"].strip():
            problems.append(f"{label} has no name")
            continue
        name = skill["name"].strip()
        if name.lower() in seen:
            problems.append(f"{label} duplicates '{name}'")
        seen.add(name.lower())
        for term in _skill_terms(skill):
            if term.strip().lower() in _AMBIGUOUS_SHORT_NAMES or len(term.strip()) < 3:
                problems.append(f"'{term}' is too ambiguous to match reliably")
            if _term_in(resume_text, term):
                problems.append(f"'{term}' appears in the resume, so it is not absent")
        if not any(_term_in(job_description, term) for term in _skill_terms(skill)):
            problems.append(f"'{name}' is not requested by the job description")
    return problems


def check_technologies(bullets, source_text, vocabulary):
    """Advisory: vocabulary technologies in a bullet that the source text never mentions.

    vocabulary: {canonical name: [alias, ...]}; the canonical name is always one of its own terms. A technology
    counts as supported if any of its terms appears in the source. Only vocabulary terms are seen.
    """
    results, with_unsupported = [], 0
    supported = {
        name for name, aliases in (vocabulary or {}).items()
        if any(_term_in(source_text, term) for term in [name] + list(aliases or []))
    }
    for index, bullet in enumerate(bullets):
        found = [
            name for name, aliases in (vocabulary or {}).items()
            if name not in supported and any(_term_in(bullet, term) for term in [name] + list(aliases or []))
        ]
        with_unsupported += bool(found)
        results.append({"index": index, "unsupported": found})
    return {"bullets_checked": len(results), "bullets_with_unsupported": with_unsupported, "results": results}


# ---------------------------------------------------------------------------------------------------------------
# D5: sections; D6: duplicates and unchanged copies
# ---------------------------------------------------------------------------------------------------------------


def _squash(text):
    return " ".join(text.lower().split())


def check_sections(output_sections, source_sections):
    """Match output sections to the resume's sections by heading. All results are evaluation SIGNALS.

    output_sections: [{"section", "bullets"}] (from parse_ats_output). source_sections: [{"heading", "bullets"}]
    (dataset configuration: a malformed entry raises KeyError/TypeError/AttributeError; validate it first).
    Matching, per output group, in order:
      1. an output section equal to a heading (case and whitespace ignored) matches it;
      2. otherwise the LONGEST heading that the output section contains matches (an earlier heading wins a tie);
      3. but if that heading is itself contained in another, longer source heading (a generic prefix such as
         "Projects" next to "Projects - Task API"), the group is NOT assigned: it is reported in ambiguous_output.
    A second group for a heading that was already matched is reported in duplicate_groups; its bullets are added
    to that heading's output count, so a section split in two is compared as a whole.
    Returns: matched (distinct headings matched), unmatched_output (section texts), ambiguous_output and
    duplicate_groups ([{"section", "heading"}]), missing_from_output (source headings never matched) and
    bullet_count_changes ([{"heading", "source_count", "output_count"}]). A changed bullet count can be
    legitimate (merged or split bullets) and is a signal for human review, not proof of a wrong rewrite.
    """
    headings = [_squash(section["heading"]) for section in source_sections]
    output_counts, unmatched, ambiguous, duplicates = {}, [], [], []
    for output in output_sections:
        label = _squash(output["section"])
        candidates = [i for i, heading in enumerate(headings) if heading and heading in label]
        if not candidates:
            unmatched.append(output["section"])
            continue
        exact = [i for i in candidates if headings[i] == label]
        if exact:
            hit = exact[0]
        else:
            hit = max(candidates, key=lambda i: (len(headings[i]), -i))
            if any(j != hit and headings[hit] in headings[j] and headings[hit] != headings[j]
                   for j in range(len(headings))):
                ambiguous.append({"section": output["section"], "heading": source_sections[hit]["heading"]})
                continue
        if hit in output_counts:
            duplicates.append({"section": output["section"], "heading": source_sections[hit]["heading"]})
            output_counts[hit] += len(output["bullets"])
        else:
            output_counts[hit] = len(output["bullets"])
    count_changes = [
        {"heading": source_sections[i]["heading"], "source_count": len(source_sections[i]["bullets"]),
         "output_count": output_counts[i]}
        for i in sorted(output_counts) if len(source_sections[i]["bullets"]) != output_counts[i]
    ]
    missing = [source_sections[i]["heading"] for i in range(len(source_sections)) if i not in output_counts]
    return {"matched": len(output_counts), "unmatched_output": unmatched, "ambiguous_output": ambiguous,
            "duplicate_groups": duplicates, "missing_from_output": missing, "bullet_count_changes": count_changes}


def normalise_for_comparison(text):
    """Lower case, punctuation to spaces, whitespace collapsed."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def find_duplicates(bullets, threshold=0.8):
    """Pairs of bullets (positions a < b) whose word sets overlap by Jaccard similarity >= threshold."""
    token_sets = [set(normalise_for_comparison(b).split()) if isinstance(b, str) else set() for b in bullets]
    pairs = []
    for a in range(len(token_sets)):
        for b in range(a + 1, len(token_sets)):
            union = token_sets[a] | token_sets[b]
            if not union:
                continue
            similarity = len(token_sets[a] & token_sets[b]) / len(union)
            if similarity >= threshold:
                pairs.append({"a": a, "b": b, "similarity": round(similarity, 3)})
    return pairs


def find_unchanged(bullets, source_bullets):
    """Positions of rewritten bullets identical to a source bullet after normalise_for_comparison (informational)."""
    originals = {normalise_for_comparison(b) for b in source_bullets if isinstance(b, str)}
    return [i for i, b in enumerate(bullets) if isinstance(b, str) and normalise_for_comparison(b) in originals]


# ---------------------------------------------------------------------------------------------------------------
# Combined evaluation of one model response
# ---------------------------------------------------------------------------------------------------------------


def evaluate_output(raw_output, source_text, source_sections=None, absent_skills=None, vocabulary=None,
                    example_numbers=()):
    """Run every applicable check on one response. JSON-serialisable; checks whose inputs were not given are None.

    MODEL OUTPUT is never a reason to raise: malformed output is reported by the schema check, and then no
    bullet-level check runs (there are no usable bullets).
    CASE CONFIGURATION is not validated here. A malformed source_sections (an entry that is not a dict, or has no
    "heading"/"bullets", or a non-string heading) raises KeyError, TypeError or AttributeError; a malformed
    absent_skills or vocabulary entry is tolerated (it simply matches nothing). Validate the dataset first.
    """
    parsed = parse_ats_output(raw_output)
    result = {
        "schema": {"status": parsed["status"], "production_would_500": parsed["production_would_500"],
                   "problems": parsed["problems"]},
        "bullets_total": 0, "numbers": None, "absent_skills": None, "technologies": None,
        "sections": None, "duplicates": None, "unchanged": None,
    }
    if parsed["status"] != "ok":
        return result
    flat, labels = [], []
    for section in parsed["sections"]:
        for bullet in section["bullets"]:
            flat.append(bullet)
            labels.append(section["section"])
    result["bullets_total"] = len(flat)
    result["numbers"] = check_invented_numbers(flat, source_text, example_numbers)
    if absent_skills is not None:
        result["absent_skills"] = check_absent_skills(flat, absent_skills)
    if vocabulary is not None:
        result["technologies"] = check_technologies(flat, source_text, vocabulary)
    if source_sections is not None:
        result["sections"] = check_sections(parsed["sections"], source_sections)
        result["unchanged"] = find_unchanged(flat, [b for s in source_sections for b in s["bullets"]])
    result["duplicates"] = find_duplicates(flat)
    for check in ("numbers", "absent_skills", "technologies"):
        if result[check] is not None:
            for entry in result[check]["results"]:
                entry["section"] = labels[entry["index"]]
    return result
