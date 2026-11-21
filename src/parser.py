import os
import re
import logging
import json
import unicodedata
from typing import Optional, Tuple, List
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import yaml
from langchain_groq import ChatGroq


load_dotenv()

def load_prompts():
    filepath = os.path.join(os.path.dirname(__file__), "prompts.yaml")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    
class ResumeParserConfig:
    """Configuration class for ResumeParser."""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    RESUME_SUMMARY_FILE_PATH = os.path.join(CURRENT_DIR, "resume_summary.txt")
    RESUME_SKILLS_FILE_PATH = os.path.join(CURRENT_DIR, "resume_skills.txt")
    MISSING_SKILLS_FILE_PATH = os.path.join(CURRENT_DIR, "missing_skills.txt")
    JOB_SKILLS_FILE_PATH = os.path.join(CURRENT_DIR, "job_skills.txt")
    RESUME_REWRITTEN_FILE_PATH = os.path.join(CURRENT_DIR, "resume_rewritten.txt")
    ATS_OPTIMIZED_BULLETS_FILE_PATH = os.path.join(CURRENT_DIR, "ats_optimized_bullets.txt")
    RESUME_TEXT = os.path.join(CURRENT_DIR, "resume_text.txt")
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    GROQ_MODEL = "openai/gpt-oss-20b"
   
logger = logging.getLogger(__name__)

# --- Prioritised missing-skills normalisation -------------------------------
PRIORITY_TIERS = ("crucial", "good_to_have", "optional")
SKILL_CATEGORIES = ("technical_skills", "soft_skills", "academic_qualifications")
SKILL_STATUSES = ("missing", "partial")

_MAX_SKILL_LEN = 100
_MAX_REASON_LEN = 600
_MAX_EVIDENCE_LEN = 400
_MIN_EVIDENCE_LEN = 6
_MAX_DIAGNOSTICS = 50
_STOP_TOKENS = {
    "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "the",
    "experience", "skills", "skill", "knowledge", "using", "use", "based",
    "strong", "good", "working", "understanding", "ability",
}
_CHAR_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",
}


def _empty_legacy_missing_skills() -> dict:
    """Legacy-shaped missing_skills dict with no entries."""
    return {category: [] for category in SKILL_CATEGORIES}


def _fallback_prioritized(message: str) -> dict:
    """Prioritised result used when normalisation did not succeed."""
    result = {tier: [] for tier in PRIORITY_TIERS}
    result["normalization_ok"] = False
    result["diagnostics"] = [message]
    return result


def _strip_json_fence(text) -> str:
    """Remove one surrounding ``` / ```json code fence, if present."""
    text = (text or "").strip()
    match = re.match(r"^```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?```\s*$", text, re.S)
    return match.group(1).strip() if match else text


def _fold_text(text: str, keep_newlines: bool = False) -> str:
    """Case-fold and normalise quotes, dashes and whitespace for comparison."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_FOLD).lower()
    if keep_newlines:
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _skill_key(name: str) -> str:
    """Comparison key: 'REST APIs' and 'rest api' give the same key."""
    return _singular(re.sub(r"[^a-z0-9+#]", "", _fold_text(name)))


def _tokens(text: str) -> list:
    all_tokens = re.findall(r"[a-z0-9+#]+", _fold_text(text))
    kept = [_singular(t) for t in all_tokens if t not in _STOP_TOKENS]
    return kept or [_singular(t) for t in all_tokens]


def _tokens_match(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 5 and long_.startswith(short)


def _locate_evidence(evidence: str, jd_folded: str):
    """Return (start, end) of the evidence phrase inside the folded JD, or None.

    The phrase must occur as whole words. An ellipsis ('...') in the evidence
    splits it into fragments that must all occur, in order.
    """
    folded = _fold_text(evidence).strip(" \"'`.,;:!?()[]{}-")
    fragments = [f.strip(" \"'`.,;:!?()[]{}-") for f in folded.split("...")]
    fragments = [f for f in fragments if len(f) >= 3]
    if not fragments or sum(len(f) for f in fragments) < _MIN_EVIDENCE_LEN:
        return None
    position, first_start = 0, None
    for fragment in fragments:
        pattern = r"(?<![a-z0-9])" + r"\s+".join(
            re.escape(word) for word in fragment.split(" ")
        ) + r"(?![a-z0-9])"
        match = re.compile(pattern).search(jd_folded, position)
        if not match:
            return None
        if first_start is None:
            first_start = match.start()
        position = match.end()
    return first_start, position


def _sentence_window(jd_folded: str, start: int, end: int) -> str:
    """The JD line/sentence that contains jd_folded[start:end]."""
    left = max(jd_folded.rfind(m, 0, start) for m in ("\n", ". ", "; "))
    rights = [i for i in (jd_folded.find(m, end) for m in ("\n", ". ", "; ")) if i != -1]
    return jd_folded[left + 1: min(rights) if rights else len(jd_folded)]


def _evidence_relates_to_skill(skill: str, window: str) -> bool:
    """Heuristic: at least half of the skill's words appear in the evidence sentence."""
    skill_tokens = _tokens(skill)
    window_tokens = _tokens(window)
    if not skill_tokens:
        return False
    hits = sum(1 for s in skill_tokens if any(_tokens_match(s, w) for w in window_tokens))
    return hits >= (len(skill_tokens) + 1) // 2


def _parse_helper_skills(job_skills_text):
    """{skill_key: category} from the extracted JD skills JSON, or None if unusable."""
    try:
        data = json.loads(_strip_json_fence(job_skills_text))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    index = {}
    for category in SKILL_CATEGORIES:
        items = data.get(category, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and _skill_key(item):
                index.setdefault(_skill_key(item), category)
    return index


def _validate_entry(raw_entry):
    """Return (clean_entry, None) or (None, problem). Never invents or repairs fields."""
    if not isinstance(raw_entry, dict):
        return None, "entry is not an object"
    values = {}
    for field in ("skill", "category", "status", "reason", "jd_evidence"):
        value = raw_entry.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, f"missing or empty '{field}'"
        values[field] = value.strip()
    values["category"] = values["category"].lower()
    values["status"] = values["status"].lower()
    if values["category"] not in SKILL_CATEGORIES:
        return None, f"invalid category '{values['category']}'"
    if values["status"] not in SKILL_STATUSES:
        return None, f"invalid status '{values['status']}'"
    if not _skill_key(values["skill"]):
        return None, "skill name has no usable characters"
    if len(values["skill"]) > _MAX_SKILL_LEN:
        return None, "skill name too long"
    if len(values["reason"]) > _MAX_REASON_LEN:
        return None, f"reason too long for '{values['skill']}'"
    if len(values["jd_evidence"]) > _MAX_EVIDENCE_LEN:
        return None, f"jd_evidence too long for '{values['skill']}'"
    return values, None


def _build_prioritized(raw_text, job_description, job_skills_text):
    diagnostics = []

    def note(message):
        if len(diagnostics) < _MAX_DIAGNOSTICS:
            diagnostics.append(message)
        elif len(diagnostics) == _MAX_DIAGNOSTICS:
            diagnostics.append("further diagnostics omitted")

    try:
        data = json.loads(_strip_json_fence(raw_text))
    except (ValueError, TypeError):
        return _fallback_prioritized("Comparison output is not valid JSON."), _empty_legacy_missing_skills()
    if not isinstance(data, dict) or not any(tier in data for tier in PRIORITY_TIERS):
        return (
            _fallback_prioritized("Comparison output has none of the expected tiers."),
            _empty_legacy_missing_skills(),
        )

    jd_folded = _fold_text(job_description or "", keep_newlines=True)
    helper = _parse_helper_skills(job_skills_text)
    if helper is None:
        note("Extracted JD skill list is unavailable; helper-list membership is unknown "
             "and the legacy missing_skills is left empty.")

    candidates = []  # (tier_index, order, entry)
    order = 0
    for tier_index, tier in enumerate(PRIORITY_TIERS):
        if tier not in data:
            note(f"Tier '{tier}' missing from output; treated as empty.")
            continue
        if not isinstance(data[tier], list):
            note(f"Tier '{tier}' is not a list; ignored.")
            continue
        for raw_entry in data[tier]:
            order += 1
            entry, problem = _validate_entry(raw_entry)
            if entry is None:
                note(f"Dropped malformed entry in '{tier}': {problem}.")
                continue
            # Optional resume-side quote. Kept only if it is a non-empty string; anything else is
            # ignored (the entry stays valid). It is checked against the resume later, in
            # annotate_resume_evidence, because this function does not receive the resume.
            quote = raw_entry.get("resume_evidence")
            if isinstance(quote, str):
                if len(quote.strip()) > _MAX_RESUME_EVIDENCE_LEN:
                    note(f"'{entry['skill']}': resume_evidence is longer than {_MAX_RESUME_EVIDENCE_LEN} "
                         f"characters; ignored.")
                elif quote.strip():
                    entry["resume_evidence"] = quote.strip()
            elif quote is not None:
                note(f"'{entry['skill']}': resume_evidence is not a non-empty string; ignored.")
            # evidence_found: the quote exists in the JD (after safe normalisation).
            # evidence_related: advisory only - the quoted sentence shares words with the
            # skill. It is None when the quote was not found (not evaluated).
            span = _locate_evidence(entry["jd_evidence"], jd_folded)
            entry["evidence_found"] = span is not None
            if span is None:
                entry["evidence_related"] = None
                note(f"'{entry['skill']}': jd_evidence was not found in the job description "
                     f"(evidence_found=False); excluded from legacy missing_skills.")
            else:
                entry["evidence_related"] = _evidence_relates_to_skill(
                    entry["skill"], _sentence_window(jd_folded, *span)
                )
                if not entry["evidence_related"]:
                    note(f"'{entry['skill']}': the quote exists in the job description but its sentence "
                         f"does not mention the skill (evidence_related=False, advisory); relevance unconfirmed.")
            entry["in_extracted_list"] = None if helper is None else _skill_key(entry["skill"]) in helper
            if entry["evidence_found"] and entry["in_extracted_list"] is False:
                note(f"'{entry['skill']}' has a quote found in the JD but is absent from the extracted skill list; "
                     f"kept in the prioritised output, excluded from legacy missing_skills.")
            candidates.append((tier_index, order, entry))

    # One entry per skill: prefer a found evidence quote, then the highest-priority tier,
    # then first seen. evidence_related is advisory and does not affect the choice.
    best = {}
    for tier_index, position, entry in candidates:
        key = _skill_key(entry["skill"])
        rank = (not entry["evidence_found"], tier_index, position)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, tier_index, position)
    result = {tier: [] for tier in PRIORITY_TIERS}
    legacy = _empty_legacy_missing_skills()
    for tier_index, position, entry in candidates:
        key = _skill_key(entry["skill"])
        if best[key][2] != position:
            note(f"Duplicate '{entry['skill']}' in '{PRIORITY_TIERS[tier_index]}' removed; "
                 f"kept the occurrence in '{PRIORITY_TIERS[best[key][1]]}'.")
            continue
        result[PRIORITY_TIERS[tier_index]].append(entry)
        # Legacy eligibility: status "missing", a found evidence quote, listed in the extracted
        # JD skills, and the same category as that list.
        if entry["status"] == "missing" and entry["evidence_found"] and entry["in_extracted_list"]:
            if helper[key] != entry["category"]:
                note(f"'{entry['skill']}': category '{entry['category']}' differs from the extracted "
                     f"list ('{helper[key]}'); excluded from legacy missing_skills.")
            elif entry["skill"] not in legacy[entry["category"]]:
                legacy[entry["category"]].append(entry["skill"])

    result["normalization_ok"] = True
    result["diagnostics"] = diagnostics
    return result, legacy


# Advisory check: does a skill the model called "missing" appear in the resume text at all?
# A mention is NOT proof of proficiency and a False is NOT proof of absence. It never changes status,
# tier, reason, quote or any score.
_MENTION_SEPARATOR = r"[\s\-_./]*"      # between the parts of a multiword name (zero allowed: PDF text drops spaces)
_MENTION_EDGE = "a-z0-9+#"               # characters that may not touch the start or end of a match
_MIN_MENTION_ALNUM = 3                   # shorter names (Go, R, C, AI) are too ambiguous unless they contain + or #


def resume_mention_found(skill, resume_text) -> bool:
    """True only if the skill name appears in the resume text as a whole term. Never raises.

    Case-insensitive; whitespace is normalised. A multiword name matches when its parts appear in order,
    separated by nothing, whitespace, '-', '_', '.' or '/'. The match must not touch another letter, digit,
    '+' or '#', so "Java" does not match "JavaScript" and "C++" does not match "C". Names with fewer than
    3 letters/digits and no '+' or '#' (Go, R, C, AI) always give False. No synonyms, acronym expansion or
    plural handling. Any unexpected input gives False.
    """
    try:
        if not isinstance(skill, str) or not isinstance(resume_text, str):
            return False
        parts = re.findall(r"[a-z0-9+#]+", _fold_text(skill))
        if not parts:
            return False
        joined = "".join(parts)
        if len(re.sub(r"[^a-z0-9]", "", joined)) < _MIN_MENTION_ALNUM and not re.search(r"[+#]", joined):
            return False
        pattern = (
            "(?<![" + _MENTION_EDGE + "])"
            + _MENTION_SEPARATOR.join(re.escape(part) for part in parts)
            + "(?![" + _MENTION_EDGE + "])"
        )
        return re.search(pattern, _fold_text(resume_text)) is not None
    except Exception:
        return False


def annotate_resume_mentions(prioritized, resume_text) -> None:
    """Add a boolean 'resume_mention_found' to every entry whose status is 'missing'. Changes nothing else.

    Does nothing if resume_text is not a string or the result has no tiers (nothing was checked, so
    nothing is claimed). Logs counts only.
    """
    if not isinstance(prioritized, dict) or not isinstance(resume_text, str):
        return
    checked = found = 0
    for tier in PRIORITY_TIERS:
        entries = prioritized.get(tier)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("status") == "missing":
                entry["resume_mention_found"] = resume_mention_found(entry.get("skill"), resume_text)
                checked += 1
                found += entry["resume_mention_found"]
    if checked:
        logger.info("resume mention check: %d of %d 'missing' skills have a name match in the resume text "
                    "(advisory only)", found, checked)


# Advisory check of the optional "resume_evidence" quote the model may give for an entry. It only tells
# you (a) whether the quote exists in the resume text and (b) whether the quote itself names the skill.
# Neither is proof the candidate used the skill, and neither changes status, tier, reason, quote, score
# or the legacy dict.
_MAX_RESUME_EVIDENCE_LEN = 300       # longer quotes are not accepted as evidence (and are not stored)
_MIN_RESUME_EVIDENCE_CHARS = 15      # fewer non-whitespace characters (after folding) is too short to count
_MAX_EVIDENCE_DIAGNOSTICS = 10       # fixed, separate budget for the evidence diagnostics added after the general ones
_MIN_GLUE_WORD_CHARS = 4             # a space may be missing in the resume only between two words this long (letters/digits)
_DENSE_GLUE_TOKEN_CHARS = 25         # a resume "word" this long means spaces were dropped wholesale; glue is then tolerated


def _word_chars(word) -> int:
    """Number of letters/digits in a word (punctuation does not count)."""
    return sum(ch.isalnum() for ch in word)


def resume_quote_found(quote, resume_text) -> bool:
    """True only if the quote appears word for word in the resume text. Never raises.

    Both sides are folded (NFKC, lower-case, curly quotes/dashes, whitespace collapsed). Punctuation and word
    order must match exactly, so a paraphrase fails. A quote under 15 non-whitespace characters or over 300
    characters is never accepted. How spaces are treated:
      - Any amount or kind of whitespace (spaces, tabs, newlines) between two words matches.
      - A space that is MISSING in the resume (PDF extraction glues words) is tolerated only (1) between two
        words of at least 4 letters/digits, or (2) when the matched text sits in a resume "word" of 25+
        characters, meaning spaces were dropped wholesale.
      - Otherwise a short word cannot be glued to its neighbour, and the quote cannot start or end in the
        middle of a resume word. So "the rapist organised files" does NOT match "therapist organisedfiles".
      - A quote that is glued where the resume has spaces is not accepted.
    Missing spaces cannot be told apart from real words without a dictionary, so this is a trade-off: it
    prefers a false "not found" over a false "found".
    """
    try:
        if not isinstance(quote, str) or not isinstance(resume_text, str):
            return False
        if len(quote) > _MAX_RESUME_EVIDENCE_LEN:
            return False
        words = _fold_text(quote).split()
        if len("".join(words)) < _MIN_RESUME_EVIDENCE_CHARS:
            return False
        text = _fold_text(resume_text)
        escaped = [re.escape(word) for word in words]
        # Pass 1: word for word. A missing space needs two long neighbours; the quote must start and end on
        # word boundaries of the resume (punctuation next to the quote is fine).
        pattern = escaped[0]
        for before, after, escaped_after in zip(words, words[1:], escaped[1:]):
            long_pair = min(_word_chars(before), _word_chars(after)) >= _MIN_GLUE_WORD_CHARS
            pattern += (r"\s*" if long_pair else r"\s+") + escaped_after
        if re.search(r"(?<!\w)" + pattern + r"(?!\w)", text):
            return True
        # Pass 2: the same words with spaces optional everywhere, accepted only where the resume itself shows
        # wholesale space loss (a very long "word" under the match).
        for match in re.finditer(r"\s*".join(escaped), text):
            token_start = text.rfind(" ", 0, match.start()) + 1
            token_end = text.find(" ", match.end())
            token_end = len(text) if token_end == -1 else token_end
            if max(len(token) for token in text[token_start:token_end].split()) >= _DENSE_GLUE_TOKEN_CHARS:
                return True
        return False
    except Exception:
        return False


def annotate_resume_evidence(prioritized, resume_text) -> None:
    """Add 'resume_evidence_found' and 'resume_evidence_names_skill' to every entry. Changes nothing else.

    - resume_evidence_found: None when the entry has no quote (not provided); otherwise True/False, whether
      the quote exists in the resume text (see resume_quote_found).
    - resume_evidence_names_skill: None unless the quote was found AND the skill name is one that can be
      matched reliably (names under 3 letters such as Go or AI cannot); then True/False, whether the quote
      itself contains the skill name (same rule as resume_mention_found). A quote that only names the skill
      is a mention, not evidence of use; whether it shows use is the model's unverified judgment.
    A 'missing' entry with a found quote, and any quote that is not found, add a diagnostic. These are
    added after the general diagnostics with their own fixed budget (_MAX_EVIDENCE_DIAGNOSTICS), contradictions
    first; any further ones are replaced by one "omitted" line. The per-entry flags are always complete.
    Status is never changed. Does nothing if resume_text is not a string. Logs counts only.
    """
    if not isinstance(prioritized, dict) or not isinstance(resume_text, str):
        return
    contradictions = []
    unverified = []
    quoted = verified = 0
    for tier in PRIORITY_TIERS:
        entries = prioritized.get(tier)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry["resume_evidence_found"] = None
            entry["resume_evidence_names_skill"] = None
            quote = entry.get("resume_evidence")
            if not isinstance(quote, str):
                continue
            quoted += 1
            skill = entry.get("skill")
            entry["resume_evidence_found"] = resume_quote_found(quote, resume_text)
            if not entry["resume_evidence_found"]:
                unverified.append(f"'{skill}': resume_evidence was not found in the resume text; unverified.")
                continue
            verified += 1
            # A name that cannot find itself cannot be checked reliably, so the answer stays None.
            if resume_mention_found(skill, skill):
                entry["resume_evidence_names_skill"] = resume_mention_found(skill, quote)
            if entry.get("status") == "missing":
                detail = {True: "names the skill", False: "does not contain the skill name",
                          None: "could not be checked for the skill name"}[entry["resume_evidence_names_skill"]]
                contradictions.append(
                    f"'{skill}' is marked missing but its resume_evidence quote exists in the resume text "
                    f"and {detail} (advisory; status unchanged; not proof of proficiency).")
    notes = contradictions + unverified
    diagnostics = prioritized.get("diagnostics")
    if isinstance(diagnostics, list) and notes:
        diagnostics.extend(notes[:_MAX_EVIDENCE_DIAGNOSTICS])
        if len(notes) > _MAX_EVIDENCE_DIAGNOSTICS:
            diagnostics.append(f"{len(notes) - _MAX_EVIDENCE_DIAGNOSTICS} further resume_evidence diagnostics "
                               f"omitted; see each entry's resume_evidence_found flag.")
    if quoted:
        logger.info("resume evidence check: %d of %d quotes found in the resume text (advisory only)",
                    verified, quoted)


# NOTE ON SCORING AND KNOWN LIMITATIONS (read before relying on the legacy dict or the score):
# - The legacy missing_skills dict built here holds only entries whose status is "missing" that
#   have a found evidence quote, are listed in the extracted JD skills, and have the same
#   category as that list. "partial" entries - and anything else excluded by those rules - are
#   NOT in it. The existing frontend skill-match bar and ATS score treat every JD skill absent
#   from the legacy dict as matched, so they currently count partial skills as matched.
# - evidence_related is advisory only. An entry with evidence_found=True and
#   evidence_related=False can still enter the legacy dict, so the bar and ATS score can be
#   wrong in either direction. They are NOT fully reliable; the scoring model has not been
#   reviewed yet.
# - Acronym-to-expansion matching (e.g. "NLP" vs "natural language processing") is not
#   supported: such an entry is kept in the prioritised output but stays out of the legacy dict
#   unless its name matches the extracted skill list.
# - Failure signal: when normalization_ok is False, the legacy dict is an EMPTY valid dict.
#   That means "analysis unavailable", not "nothing missing"; display code must check
#   normalization_ok in the prioritised result before showing a skill-match bar or ATS score.
def build_prioritized_missing_skills(raw_text, job_description, job_skills_text, resume_text=None):
    """Validate the LLM's prioritised missing-skills JSON.

    Returns (prioritized, legacy_missing_skills). Never raises: on malformed
    output it returns a result with normalization_ok=False and an empty,
    legacy-shaped dict so the rest of the analysis can continue.

    If resume_text is given, 'missing' entries also get an advisory 'resume_mention_found' flag, and every
    entry gets advisory 'resume_evidence_found' / 'resume_evidence_names_skill' flags.
    """
    try:
        prioritized, legacy = _build_prioritized(raw_text, job_description, job_skills_text)
        try:
            annotate_resume_mentions(prioritized, resume_text)
            annotate_resume_evidence(prioritized, resume_text)
        except Exception:
            logger.exception("Resume mention/evidence check failed; continuing without it")
    except Exception:
        logger.exception("Prioritised missing-skills normalisation failed")
        return (
            _fallback_prioritized("Normalisation raised an unexpected error; see the server log."),
            _empty_legacy_missing_skills(),
        )
    for message in prioritized["diagnostics"]:
        logger.warning("missing-skills normalisation: %s", message)
    return prioritized, legacy


# A bullet shorter than this (non-whitespace characters) is never reported as verified: very short
# strings can match unrelated text by coincidence.
_MIN_BULLET_CHARS = 20


def _remove_whitespace(text: str) -> str:
    """Remove every whitespace character (spaces, tabs, newlines, ...). Used only for comparison."""
    return "".join(text.split())


def verify_project_bullets(summary_text: str, resume_text: str) -> str:
    """Record which project bullets cannot be found in the extracted resume text. Never raises.

    For each project whose "bullets" is a list, every bullet is searched for in resume_text after
    removing ALL whitespace from both (nothing else is normalised): an exact, case-sensitive,
    punctuation-exact substring match. Whitespace is ignored because PDF extraction often drops
    the spaces between words, while the model restores them. No fuzzy matching is used, so a
    paraphrase is never counted as found, and a bullet with fewer than _MIN_BULLET_CHARS
    non-whitespace characters is never counted as found. The project gets
    "bullets_unverified": the sorted zero-based positions, in the original "bullets" list, of the
    bullets that were NOT found ([] when all were). A blank, too-short or non-string item is also reported.
    Bullets themselves are never rewritten, reordered or removed, and any "bullets_unverified"
    value supplied by the model is overwritten.

    A bullet that is not found is only "not verified": PDF extraction (ligatures, hyphenation,
    odd spacing) or the chunk joins in resume_text can cause that. It is not proof of a mistake.
    A summary that is not valid JSON, or has no project with a "bullets" list, is returned
    unchanged (the key's absence means "not checked"). Only counts are logged, never text.
    """
    try:
        summary = json.loads(summary_text)
        projects = summary.get("key_projects") if isinstance(summary, dict) else None
        if not isinstance(projects, list):
            return summary_text
        source = _remove_whitespace(resume_text) if isinstance(resume_text, str) else ""
        checked = unverified_total = 0
        touched = False
        for project in projects:
            bullets = project.get("bullets") if isinstance(project, dict) else None
            if not isinstance(bullets, list):
                continue
            unverified = []
            for position, bullet in enumerate(bullets):
                squashed = _remove_whitespace(bullet) if isinstance(bullet, str) else ""
                if len(squashed) < _MIN_BULLET_CHARS or squashed not in source:
                    unverified.append(position)
            project["bullets_unverified"] = unverified
            touched = True
            checked += len(bullets)
            unverified_total += len(unverified)
        if not touched:
            return summary_text
        if unverified_total:
            logger.warning("Project bullets: %d of %d could not be verified against the extracted resume text",
                           unverified_total, checked)
        return json.dumps(summary, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Project bullet verification skipped (%s)", type(exc).__name__)
        return summary_text


class ResumeParserError(Exception):
    """Custom exception for ResumeParser errors."""
    pass


class _NoStage:
    """Stand-in used when no metrics collector is passed: ``config`` is None, so invoke() behaves as before."""
    config = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


_NO_STAGE = _NoStage()


def _note_normalization(metrics, prioritized_json):
    """Tell the collector the normalisation outcome; a failing collector must never affect the pipeline."""
    if metrics is None:
        return
    try:
        metrics.set_normalization(prioritized_json)
    except Exception:
        pass


def _metrics_stage(metrics, name):
    """Observability hook: the collector's timing/usage context for one model call, or a no-op."""
    if metrics is None:
        return _NO_STAGE
    try:
        return metrics.stage(name)
    except Exception:
        return _NO_STAGE

class ResumeParser:
    """A class to parse and analyze resumes using LLMs."""
    
    def __init__(self):
        """Initialize the ResumeParser with Groq API key."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model=ResumeParserConfig.GROQ_MODEL,
            api_key=groq_api_key,
            temperature=0.2,
            max_tokens=8192,
            reasoning_effort="low"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ResumeParserConfig.CHUNK_SIZE,
            chunk_overlap=ResumeParserConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.output_parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="Technical skills", description="List of technical skills mentioned."),
            ResponseSchema(name="Soft skills", description="List of soft skills mentioned."),
            ResponseSchema(name="Academic qualifications", description="List of academic qualifications mentioned.")
        ])
        self._setup_prompts()

    def _setup_prompts(self):
        """Set up prompt templates for resume and job analysis."""
        prompts = load_prompts()
        self.resume_prompt = PromptTemplate(
            input_variables=["text"],
            template=prompts['resume_summary_generator']
        )

        self.credentials_prompt = PromptTemplate(
            input_variables=["resume_content"],
            template=prompts['resume_skill_extractor']
            # output_parser=self.output_parser
        )

        self.missing_skills_prompt = PromptTemplate(
                input_variables=["job_description", "job_skills", "resume_content"],
                template=prompts['job_resume_comparison']
            # output_parser=self.output_parser
        )

        self.job_prompt = PromptTemplate(
            input_variables=["job_description"],
            template=prompts['job_description_skill_extractor']
            # output_parser=self.output_parser
        )

        self.ats_bullet_prompt = PromptTemplate(
            input_variables=["job_description", "resume_content"],
            template=prompts['resume_ats_optimizer']
        )

    @staticmethod
    def extract_text_from_resume(file_path: str) -> Optional[str]:
        """Extract PDF text and embedded hyperlink URLs."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text_parts = []
                hyperlink_urls = []

                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)

                    for annotation_ref in page.get("/Annots", []):
                        annotation = annotation_ref.get_object()
                        action = annotation.get("/A")

                        if action:
                            action = action.get_object()
                            url = action.get("/URI")

                            if url and url not in hyperlink_urls:
                                hyperlink_urls.append(url)

                if hyperlink_urls:
                    text_parts.append(
                        "\n\nEmbedded hyperlinks:\n" +
                        "\n".join(hyperlink_urls)
                    )

                return "\n".join(text_parts)

        except FileNotFoundError:
            raise ResumeParserError(f"Resume file not found: {file_path}")
        except Exception as e:
            raise ResumeParserError(f"Error during PDF processing: {e}")

    @staticmethod
    def save_text_to_file(file_path: str, text_content: str) -> None:
        """Save text content to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        except Exception as e:
            raise ResumeParserError(f"Error saving text to {file_path}: {e}")

    @staticmethod
    def extract_dictionary_from_llm_output(text: str) -> str:
        """Extract and validate a dictionary string from LLM output.

        Args:
            text (str): Raw LLM output containing a dictionary.

        Returns:
            str: Cleaned dictionary string or empty string if invalid.
        """
        try:
            # Find the first '{' and the last matching '}'
            start_idx = text.find('{')
            if start_idx == -1:
                return "{}"

            # Track brace count to find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = start_idx + i + 1
                        break
            else:
                return "{}"

            # Extract dictionary string
            dict_str = text[start_idx:end_idx].strip()

            # Clean the string (remove backslashes, newlines, etc.)
            dict_str = dict_str.replace('\\', '').replace('\n', ' ').replace('\r', '')

            # Validate dictionary using json.loads
            json.loads(dict_str)

            return dict_str
        except json.JSONDecodeError as e:
            return "{}"
        except Exception as e:
            return "{}"

    def preprocess_resume(self, resume_path: str) -> List[Document]:
        """Preprocess resume by extracting and splitting text into chunks."""
        if not os.path.exists(resume_path):
            raise ResumeParserError(f"Resume file not found: {resume_path}")

        text = self.extract_text_from_resume(resume_path)
        if not text:
            raise ResumeParserError("Failed to extract text from resume.")

        texts = self.text_splitter.split_text(text)
        resume_content = [Document(page_content=t) for t in texts]
        return resume_content


    def parse_resume(self, resume_path: str, job_description: str, metrics=None) -> Tuple[str, str, str, str, str, str]:
        """Parse resume and return summary, credentials, missing skills, job skills,
        ATS bullets and the prioritised missing skills (JSON strings, in that order).

        ``metrics`` is an optional request-scoped collector (see src/metrics.py). It only observes timing and token
        usage of the five model calls; it never changes prompts, inputs, ordering or the returned tuple."""
        
        # Preprocess resume
        resume_content = self.preprocess_resume(resume_path)
        resume_text = " ".join(doc.page_content for doc in resume_content)
        self.save_text_to_file(ResumeParserConfig.RESUME_TEXT, resume_text)

        # Generate summary
        try:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=self.resume_prompt,
                verbose=False
            )
            with _metrics_stage(metrics, "summary") as stage:
                summary = chain.invoke({"input_documents": resume_content}, config=stage.config)
            summary_text = summary['output_text']
            summary_text = re.sub(r"^```json|```$", "", summary_text.strip()).strip()
            summary_text = verify_project_bullets(summary_text, resume_text)
            self.save_text_to_file(ResumeParserConfig.RESUME_SUMMARY_FILE_PATH, summary_text)
        except Exception as e:
            logging.exception("Resume summarization failed")
            raise ResumeParserError(f"Resume summarization failed: {e}") from e

        # Extract credentials
        try:
            llm_chain = self.credentials_prompt | self.llm
            with _metrics_stage(metrics, "credentials") as stage:
                credentials_response = llm_chain.invoke({"resume_content": resume_text}, config=stage.config).content
            credentials_response = re.sub(r"^```json|```$", "", credentials_response.strip()).strip()
            # credentials_response = self.extract_dictionary_from_llm_output(credentials_response)
            self.save_text_to_file(ResumeParserConfig.RESUME_SKILLS_FILE_PATH, credentials_response)
        except Exception as e:
            credentials_response = "Error extracting credentials."
            self.save_text_to_file(ResumeParserConfig.RESUME_SKILLS_FILE_PATH, credentials_response)

        # Generate job-specific analysis
        prioritized_missing_skills_response = json.dumps(
            _fallback_prioritized("Missing-skills comparison did not run.")
        )
        try:
            job_llm_chain = self.job_prompt | self.llm
            with _metrics_stage(metrics, "e1") as stage:
                job_response = job_llm_chain.invoke({"job_description": job_description}, config=stage.config).content
            job_response = re.sub(r"^```json|```$", "", job_response.strip()).strip()

            missing_skill_llm_chain = self.missing_skills_prompt | self.llm
            with _metrics_stage(metrics, "e2") as stage:
                missing_skills_raw = missing_skill_llm_chain.invoke({
                    "job_description": job_description,
                    "job_skills": job_response,
                    "resume_content": resume_text
                }, config=stage.config).content
            prioritized_missing_skills, legacy_missing_skills = build_prioritized_missing_skills(
                missing_skills_raw, job_description, job_response, resume_text
            )
            missing_skills_response = json.dumps(legacy_missing_skills, ensure_ascii=False)
            prioritized_missing_skills_response = json.dumps(
                prioritized_missing_skills, ensure_ascii=False
            )
            _note_normalization(metrics, prioritized_missing_skills_response)  # read-only observation
            self.save_text_to_file(ResumeParserConfig.MISSING_SKILLS_FILE_PATH, missing_skills_response)
            self.save_text_to_file(ResumeParserConfig.JOB_SKILLS_FILE_PATH, job_response)
        except Exception as e:
            logging.exception("Job-specific analysis failed")
            missing_skills_response = "Error generating missing skills."
            job_response = "Error generating job skills."
            prioritized_missing_skills_response = json.dumps(
                _fallback_prioritized("Missing-skills comparison failed; see the server log.")
            )
            self.save_text_to_file(ResumeParserConfig.MISSING_SKILLS_FILE_PATH, missing_skills_response)
            self.save_text_to_file(ResumeParserConfig.JOB_SKILLS_FILE_PATH, job_response)

        # Rewrite resume bullets for ATS
        ats_chain = self.ats_bullet_prompt | self.llm
        with _metrics_stage(metrics, "e3") as stage:
            improved_response = ats_chain.invoke({
                    "job_description": job_description,
                    "resume_content": resume_text
                }, config=stage.config).content

        improved_response = re.sub(r"^```json|```$", "", improved_response.strip()).strip()
        # Save improved bullets to file
        self.save_text_to_file(ResumeParserConfig.ATS_OPTIMIZED_BULLETS_FILE_PATH, improved_response)

        return (
            summary_text,
            credentials_response,
            missing_skills_response,
            job_response,
            improved_response,
            prioritized_missing_skills_response,
        )


job_description = """Tower Research Capital is a leading quantitative trading firm founded in 1998. Tower has built its business on a high-performance platform and independent trading teams. We have a 25+ year track record of innovation and a reputation for discovering unique market opportunities.

Tower is home to some of the world’s best systematic trading and engineering talent. We empower portfolio managers to build their teams and strategies independently while providing the economies of scale that come from a large, global organization.

Engineers thrive at Tower while developing electronic trading infrastructure at a world class level. Our engineers solve challenging problems in the realms of low-latency programming, FPGA technology, hardware acceleration and machine learning. Our ongoing investment in top engineering talent and technology ensures our platform remains unmatched in terms of functionality, scalability and performance.

At Tower, every employee plays a role in our success. Our Business Support teams are essential to building and maintaining the platform that powers everything we do — combining market access, data, compute, and research infrastructure with risk management, compliance, and a full suite of business services. Our Business Support teams enable our trading and engineering teams to perform at their best.

At Tower, employees will find a stimulating, results-oriented environment where highly intelligent and motivated colleagues inspire each other to reach their greatest potential.

Responsibilities:

Developing data pipelines for cleaning, collecting, processing, and analyzing diverse datasets.
Designing and implement machine learning and LLM-based NLP models.
Conducting experiments and tests to evaluate model performance.
Contributing to optimizing and fine-tuning existing machine learning models.

Qualifications:

A bachelor’s, master's, or PhD (ongoing or complete) degree or equivalent from a top university, available to join for an in-office 6 month Internship starting Jan 2025/Feb 2025.
Prior experience with training, building, and deploying models via Tensorflow/Pytorch (or similar) is mandatory.
Experience with CI/CD pipelines and MLOps for automating model deployments.
Skilled in using Linux, SQL, Git, and BASH scripting.
Strong knowledge of Python and hands-on."""
