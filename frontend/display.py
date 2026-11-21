import streamlit as st
import requests
import os
import re
import html
import unicodedata



def display_resume_summary(summary):
    st.subheader("👤 Personal Info & Education")
    info = summary.get("personal_info_and_education", {})
    st.markdown(f"**Name:** {info.get('name', 'N/A')}")
    st.markdown(f"**Email:** {info.get('email', 'N/A')}")
    st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    st.markdown("**Profiles:**")
    for link in info.get("profiles", []):
        st.markdown(f"- [{link}]({link})")

    st.markdown("### 🎓 Education")
    for edu in info.get("education", []):
        st.markdown(f"**{edu.get('degree', '')}**, {edu.get('institution', '')} ({edu.get('graduation_year', '')})")
        for ach in edu.get("achievements", []):
            st.markdown(f"- {ach}")

    st.markdown("---")

    st.subheader("💼 Work Experience")
    for exp in summary.get("work_experience", []):
        st.markdown(f"**{exp.get('job_title', '')}**, {exp.get('company_name', '')} ({exp.get('employment_duration', '')})")
        st.markdown("**Responsibilities:**")
        for resp in exp.get("responsibilities", []):
            st.markdown(f"- {resp}")
        st.markdown("**Technologies Used:**")
        st.markdown(", ".join(exp.get("technologies_used", [])))
        st.markdown("")

    st.markdown("---")

    st.subheader("🛠️ Key Projects")
    for proj in summary.get("key_projects", []):
        st.markdown(f"**{proj.get('project_name', '')}**")
        st.markdown(f"*Objective:* {proj.get('objective', '')}")
        st.markdown("**Technologies Used:**")
        st.markdown(", ".join(proj.get("technologies_used", [])))
        st.markdown("**Description:**")
        st.markdown(", ".join(proj.get("outcomes", [])))
        st.markdown("")

    st.markdown("---")

    st.subheader("🌟 Extracurricular Activities")
    for activity in summary.get("extracurricular_and_soft_skills", {}).get("activities", []):
        st.markdown(f"-{activity}")
    
    st.subheader("🤝 Soft Skills")
    soft_skills = summary.get("extracurricular_and_soft_skills", {}).get("soft_skills", [])
    if soft_skills:
        st.markdown("**Soft Skills:** " + ", ".join(soft_skills))

def display_skills(title, skills):
    st.subheader(title)
    st.markdown("### 🔧 Technical Skills")
    st.markdown(", ".join(skills.get("technical_skills", [])) or "_None_")

    st.markdown("### 🤝 Soft Skills")
    st.markdown(", ".join(skills.get("soft_skills", [])) or "_None_")

    st.markdown("### 🎓 Academic Qualifications")
    st.markdown(", ".join(skills.get("academic_qualifications", [])) or "_None_")

def display_skill_match(missing_skills_len, job_skills_len):
    st.subheader("🎯 Skill Match Overview")
    matched_skills_len = job_skills_len - missing_skills_len
    if job_skills_len == 0:
        match_percentage = 0
    else:
        match_percentage = int((matched_skills_len / job_skills_len) * 100)
    # Clamp to 100%
    match_percentage = min(match_percentage, 100)
    st.markdown(f"**Matched {matched_skills_len} out of {job_skills_len} required skills ({match_percentage}%)**")
    # Progress bar expects 0.0–1.0
    st.progress(match_percentage / 100.0)

def display_missing_skills(missing_skills):
    st.subheader("❗ Missing Skills")
    all_missing_technical_skills = missing_skills.get("technical_skills", [])
    all_missing_soft_skills = missing_skills.get("soft_skills", [])
    all_missing_academic_skills = missing_skills.get("academic_qualifications", [])

    
    if all_missing_technical_skills:
        st.markdown("**Technical Skills**")
        # Create a list of styled skill strings
        styled_skills = [
            f"<span style='background-color:#FF6961;color:white;padding:5px 10px;border-radius:20px;margin:3px;display:inline-block'>{skill}</span>"
            for skill in all_missing_technical_skills
        ]
        # Join them and display within a single markdown call for better grouping
        st.markdown(" ".join(styled_skills), unsafe_allow_html=True)
 

    
    if all_missing_soft_skills:
        st.markdown("**Soft Skills**")
        # Create a list of styled skill strings
        styled_skills = [
            f"<span style='background-color:#FF6961;color:white;padding:5px 10px;border-radius:20px;margin:3px;display:inline-block'>{skill}</span>"
            for skill in all_missing_soft_skills
        ]
        # Join them and display within a single markdown call for better grouping
        st.markdown(" ".join(styled_skills), unsafe_allow_html=True)
   


    if all_missing_academic_skills:
        st.markdown("**Academic Skills**")
        # Create a list of styled skill strings
        styled_skills = [
            f"<span style='background-color:#FF6961;color:white;padding:5px 10px;border-radius:20px;margin:3px;display:inline-block'>{skill}</span>"
            for skill in all_missing_academic_skills
        ]
        # Join them and display within a single markdown call for better grouping
        st.markdown(" ".join(styled_skills), unsafe_allow_html=True)
  

PRIORITY_TIER_LABELS = (
    ("crucial", "Crucial"),
    ("good_to_have", "Good to Have"),
    ("optional", "Optional"),
)
_CATEGORY_LABELS = {
    "technical_skills": "Technical",
    "soft_skills": "Soft",
    "academic_qualifications": "Academic",
}
_EXTRACTED_CATEGORIES = ("technical_skills", "soft_skills", "academic_qualifications")
# Allowed values are tuples on purpose: `x in tuple` is safe for unhashable bad values (list, dict).
_ENTRY_STATUSES = ("missing", "partial")
SKILL_ANALYSIS_UNAVAILABLE_MESSAGE = (
    "⚠️ The skill analysis could not be validated, so the job skill coverage overview, the skill gaps "
    "by priority and the skill recommendations are not shown. Please run the analysis again."
)


def _skill_key(name):
    """Normalised skill name used to match flagged skills to the extracted JD skills.

    Same rule as src.parser._skill_key (duplicated on purpose: the frontend must not import
    the parser): NFKC, lower-case, keep only a-z 0-9 + #, drop a trailing plural "s" on names
    longer than three characters. "REST APIs" and "rest api" match. An acronym and its
    expansion ("NLP" / "natural language processing") do NOT match - a known limitation.
    """
    key = re.sub(r"[^a-z0-9+#]", "", unicodedata.normalize("NFKC", str(name)).lower())
    return key[:-1] if len(key) > 3 and key.endswith("s") else key


def _extracted_skills(job_skills):
    """Distinct skills extracted from the JD as [(key, display name, category)], in extraction order.

    Technical, soft and academic lists are read in that order; a skill is kept once, by normalised
    name (_skill_key), at its first occurrence, even if it appears in several lists or with different
    spelling or punctuation. Anything that is not a list of strings is ignored.
    """
    found = {}
    if isinstance(job_skills, dict):
        for category in _EXTRACTED_CATEGORIES:
            items = job_skills.get(category)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        key = _skill_key(item)
                        if key and key not in found:
                            found[key] = (key, item.strip(), category)
    return list(found.values())


def _safe_text(value):
    """HTML-escape model-written text and collapse whitespace so it stays inside one HTML block."""
    return html.escape(" ".join(str(value).split()))


def _entry_problem(entry):
    """Why a prioritised entry is unusable, or None if it is valid.

    Mirrors backend.main._entry_problem (the frontend is a separate service and cannot import
    it), plus one check the backend leaves to the parser: the skill name must have a non-empty
    normalised key, because matching and de-duplication are done on that key.
    """
    if not isinstance(entry, dict):
        return "entry is not an object"
    for field in ("skill", "reason", "jd_evidence"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            return f"missing or empty '{field}'"
    if not _skill_key(entry["skill"]):
        return "skill name has no usable characters"
    if entry.get("category") not in _EXTRACTED_CATEGORIES:
        return "invalid 'category'"
    if entry.get("status") not in _ENTRY_STATUSES:
        return "invalid 'status'"
    if not isinstance(entry.get("evidence_found"), bool):
        return "'evidence_found' is not a boolean"
    for field in ("evidence_related", "in_extracted_list"):
        if field not in entry or not (entry[field] is None or isinstance(entry[field], bool)):
            return f"'{field}' is missing or not a boolean/null"
    return None


def get_prioritized_view(analysis):
    """Decide how the saved analysis should be shown. Never raises.

    Returns (state, data):
      ("legacy", None)      - no 'prioritized_missing_skills' key (saved before this feature)
      ("unavailable", None) - key present but not a usable result (see below)
      ("ok", data)          - usable prioritised analysis (all tiers may be empty)
    A missing key is NOT a failure; a present-but-unusable value is. Unusable means: not an
    object; normalization_ok not exactly True; a tier that is not a list; diagnostics that are
    not a list of strings; or any entry failing _entry_problem. The checks match the backend's
    _load_prioritized_missing_skills, so a result the backend accepts is accepted here and
    stale or hand-edited data is still caught.
    """
    if not isinstance(analysis, dict) or "prioritized_missing_skills" not in analysis:
        return "legacy", None
    data = analysis["prioritized_missing_skills"]
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("normalization_ok"), bool)
        or not all(isinstance(data.get(tier), list) for tier, _label in PRIORITY_TIER_LABELS)
        or not isinstance(data.get("diagnostics"), list)
        or not all(isinstance(item, str) for item in data["diagnostics"])
    ):
        return "unavailable", None
    if data["normalization_ok"] is not True:
        return "unavailable", None
    for tier, _label in PRIORITY_TIER_LABELS:
        if any(_entry_problem(entry) for entry in data[tier]):
            return "unavailable", None
    return "ok", data


def unique_prioritized_entries(prioritized):
    """{tier: [entries]} with each skill kept once, by normalised name (_skill_key).

    The first occurrence in priority order (Crucial, then Good to Have, then Optional, then list
    order) wins; later duplicates are dropped, even in another tier or category. Every count and
    list shown for the analysis is built from this, so they always agree.
    """
    seen = set()
    unique = {}
    for tier, _label in PRIORITY_TIER_LABELS:
        unique[tier] = []
        for entry in prioritized.get(tier, []):
            key = _skill_key(entry["skill"])
            if key not in seen:
                seen.add(key)
                unique[tier].append(entry)
    return unique


# ---- Canonical skill-gap representation -------------------------------------------------------
# Every section that shows skill gaps (Job Skill Coverage Overview, Skill Gaps by Priority and the
# Skill Recommendations dropdown) is built from the records below, so their counts reconcile.
#
# A record describes one distinct skill (by _skill_key) and has these fields:
#   key, skill (display name), tier (crucial / good_to_have / optional, or None when the model did not
#   flag it), category, status (missing / partial, or None), state (see STATE_*), evidence_found,
#   evidence_related, in_extracted_list (frontend name match against the extracted JD skills) and
#   entry (the model's entry, or None).
#
# State rule for a flagged entry: a quote that cannot be found in the job description
# (evidence_found is not True) makes it STATE_UNVERIFIED, whatever its status, so it is never also
# counted as not demonstrated or partially demonstrated. Otherwise status "missing" is
# STATE_NOT_DEMONSTRATED and "partial" is STATE_PARTIAL. An extracted skill with no flagged entry is
# STATE_NO_GAP. That is the model not reporting a gap; it does not confirm proficiency.
STATE_NO_GAP = "no_gap"
STATE_PARTIAL = "partially_demonstrated"
STATE_NOT_DEMONSTRATED = "not_demonstrated"
STATE_UNVERIFIED = "evidence_not_verified"
STATE_LABELS = {
    STATE_NO_GAP: "No Gap Identified",
    STATE_PARTIAL: "Partially Demonstrated",
    STATE_NOT_DEMONSTRATED: "Not Demonstrated",
    STATE_UNVERIFIED: "Evidence Not Verified",
}
_FLAGGED_STATES = (STATE_NOT_DEMONSTRATED, STATE_PARTIAL, STATE_UNVERIFIED)
_STATE_COLORS = {STATE_NOT_DEMONSTRATED: "#FF6961", STATE_PARTIAL: "#E6A100", STATE_UNVERIFIED: "#808080"}
_STATUS_TO_STATE = {"missing": STATE_NOT_DEMONSTRATED, "partial": STATE_PARTIAL}
_TIER_LABELS = dict(PRIORITY_TIER_LABELS)
SKILL_CAVEAT = (
    "No Gap Identified means the model reported no gap; it does not confirm the candidate's proficiency. "
    "Not Demonstrated means the skill was not found demonstrated in the resume text, which is not proof "
    "that the candidate lacks it. Evidence Not Verified means the job-description quote the model cited "
    "could not be found, so the entry is not counted as Not Demonstrated or Partially Demonstrated. "
    "This is the model's comparison, not a guarantee of fit."
)


def classify_entry(entry):
    """State of one valid flagged entry (see the rule above): evidence first, then status."""
    if entry["evidence_found"] is not True:
        return STATE_UNVERIFIED
    return _STATUS_TO_STATE[entry["status"]]


def _flagged_records(prioritized):
    """Records for the model's flagged skills, in priority order, or None when the result is unusable.

    Validated by get_prioritized_view and de-duplicated by unique_prioritized_entries (the first
    occurrence in Crucial, Good to Have, Optional order wins, across tiers and categories). Extracted-list
    membership is not known here; build_skill_gap_records adds it.
    """
    state, data = get_prioritized_view({"prioritized_missing_skills": prioritized})
    if state != "ok":
        return None
    unique = unique_prioritized_entries(data)
    return [
        {
            "key": _skill_key(entry["skill"]),
            "skill": entry["skill"].strip(),
            "tier": tier,
            "category": entry["category"],
            "status": entry["status"],
            "state": classify_entry(entry),
            "evidence_found": entry["evidence_found"],
            "evidence_related": entry["evidence_related"],
            "entry": entry,
        }
        for tier, _label in PRIORITY_TIER_LABELS
        for entry in unique[tier]
    ]


def build_skill_gap_records(prioritized, job_skills):
    """The canonical skill-gap data: {"records": [...], "extracted_total": n}, or None when unusable.

    Flagged records come first (priority order), each marked with in_extracted_list. Every distinct
    extracted JD skill that has no flagged entry follows as a STATE_NO_GAP record (tier, status and
    entry are None), so skills the model did not flag are represented, not omitted. A flagged skill
    whose name is not among the extracted skills keeps in_extracted_list=False: it is never added to
    the extracted total.
    """
    flagged = _flagged_records(prioritized)
    if flagged is None:
        return None
    extracted = _extracted_skills(job_skills)
    extracted_keys = {key for key, _name, _category in extracted}
    flagged_keys = {record["key"] for record in flagged}
    records = [dict(record, in_extracted_list=record["key"] in extracted_keys) for record in flagged]
    for key, name, category in extracted:
        if key not in flagged_keys:
            records.append({
                "key": key, "skill": name, "tier": None, "category": category, "status": None,
                "state": STATE_NO_GAP, "evidence_found": None, "evidence_related": None,
                "in_extracted_list": True, "entry": None,
            })
    return {"records": records, "extracted_total": len(extracted)}


def summarize_skill_gaps(prioritized, job_skills):
    """Job Skill Coverage Overview counts, or None when the result is unusable.

    Every distinct extracted JD skill is in exactly one state, so no_gap + partially_demonstrated +
    not_demonstrated + evidence_not_verified == total. Flagged skills that are not among the extracted
    skills are not added to any of those counts; they are reported as "outside".
    """
    view = build_skill_gap_records(prioritized, job_skills)
    if view is None:
        return None
    counts = {state: 0 for state in STATE_LABELS}
    outside = 0
    for record in view["records"]:
        if record["in_extracted_list"]:
            counts[record["state"]] += 1
        else:
            outside += 1
    counts["total"] = view["extracted_total"]
    counts["outside"] = outside
    return counts


def summarize_priority_gaps(prioritized):
    """Skill Gaps by Priority counts over ALL flagged skills (every category, inside or outside the
    extracted list), or None when the result is unusable.

    Returns {"tiers": {tier: {state counts..., "total": n}}, "total": n, "technical": n,
    "categories": {category: n}}. "technical" equals categories["technical_skills"].
    """
    flagged = _flagged_records(prioritized)
    if flagged is None:
        return None
    tiers = {tier: dict({state: 0 for state in _FLAGGED_STATES}, total=0) for tier, _label in PRIORITY_TIER_LABELS}
    categories = {category: 0 for category in _EXTRACTED_CATEGORIES}
    for record in flagged:
        tiers[record["tier"]][record["state"]] += 1
        tiers[record["tier"]]["total"] += 1
        categories[record["category"]] += 1
    return {
        "tiers": tiers,
        "total": len(flagged),
        "technical": categories["technical_skills"],
        "categories": categories,
    }


def get_recommendation_options(prioritized):
    """Return (options, skipped) for the Skill Recommendations dropdown.

    Built from the same flagged records as the overview and the priority section. Only technical skills
    are offered. One whose quote could not be found (STATE_UNVERIFIED) is counted in skipped and never
    offered; non-technical entries are neither offered nor counted. options: {skill: label} in priority
    order. A label says when the quote's relevance is unconfirmed (evidence_related is False). An
    unusable result gives ({}, 0). So options + skipped == the technical count of summarize_priority_gaps.
    """
    options = {}
    skipped = 0
    for record in _flagged_records(prioritized) or []:
        if record["category"] != "technical_skills":
            continue
        if record["state"] == STATE_UNVERIFIED:
            skipped += 1
            continue
        label = f"{record['skill']} — {_TIER_LABELS[record['tier']]}, {STATE_LABELS[record['state']]}"
        if record["evidence_related"] is False:
            label += " · quote relevance unconfirmed"
        options[record["skill"]] = label
    return options, skipped


def display_skill_gap_summary(prioritized, job_skills):
    """Job Skill Coverage Overview (see summarize_skill_gaps). Shows nothing for an unusable result."""
    counts = summarize_skill_gaps(prioritized, job_skills)
    if counts is None:
        return
    st.subheader("🎯 Job Skill Coverage Overview")
    states = (STATE_NO_GAP, STATE_PARTIAL, STATE_NOT_DEMONSTRATED, STATE_UNVERIFIED)
    for column, state in zip(st.columns(len(states)), states):
        column.metric(STATE_LABELS[state], counts[state])
    note = f"Out of {counts['total']} distinct skills extracted from the job description."
    if counts["outside"]:
        note += (
            f" {counts['outside']} further flagged item(s) are not in the extracted list (names are matched "
            "after normalisation, so an acronym and its expansion count as different skills). They are not "
            "part of these counts; they are included in Skill Gaps by Priority below."
        )
    st.caption(note + " " + SKILL_CAVEAT)


def _evidence_notes(entry, outside_extracted):
    notes = []
    if entry.get("evidence_found") is True:
        notes.append(f"JD quote: “{_safe_text(entry['jd_evidence'])}”")
        if entry.get("evidence_related") is True:
            notes.append("quote relates to the skill")
        else:
            notes.append("⚠ relevance of the quote is unconfirmed")
    else:
        notes.append("⚠ quote not found in the job description (unverified)")
        notes.append(f"reported status (not counted): {STATE_LABELS[_STATUS_TO_STATE[entry['status']]]}")
    if outside_extracted:
        notes.append("outside the extracted skill list")
    return notes


def display_prioritized_missing_skills(prioritized, job_skills):
    """Skill Gaps by Priority: a summary table, then the flagged skills grouped by priority and state.

    Covers every valid flagged skill (technical, soft and academic), each once, with its category
    labelled. Empty state groups are omitted. Every model-written string is HTML-escaped. Model
    diagnostics are intentionally not displayed. Shows nothing for an unusable result.
    """
    view = build_skill_gap_records(prioritized, job_skills)
    summary = summarize_priority_gaps(prioritized)
    if view is None or summary is None:
        return
    st.subheader("❗ Skill Gaps by Priority")
    if summary["total"] == 0:
        st.info(
            "The model did not flag any skill gaps for this job description. "
            "This does not guarantee the resume meets the role."
        )
        return
    rows = [
        "| Priority | Not Demonstrated | Partially Demonstrated | Evidence Not Verified | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for tier, tier_label in PRIORITY_TIER_LABELS:
        counts = summary["tiers"][tier]
        rows.append(
            f"| {tier_label} | {counts[STATE_NOT_DEMONSTRATED]} | {counts[STATE_PARTIAL]} | "
            f"{counts[STATE_UNVERIFIED]} | {counts['total']} |"
        )
    totals = {state: sum(summary["tiers"][tier][state] for tier, _label in PRIORITY_TIER_LABELS) for state in _FLAGGED_STATES}
    rows.append(
        f"| **All priorities** | {totals[STATE_NOT_DEMONSTRATED]} | {totals[STATE_PARTIAL]} | "
        f"{totals[STATE_UNVERIFIED]} | {summary['total']} |"
    )
    st.markdown("\n".join(rows))
    categories = summary["categories"]
    st.markdown(
        f"**Flagged skills: {summary['total']}** · Technical skills: {summary['technical']} · "
        f"Soft skills: {categories['soft_skills']} · Academic qualifications: {categories['academic_qualifications']}"
    )
    st.caption(
        "Counts cover every flagged skill, each once, including skills that are not in the extracted job "
        "skill list. Recommendations are offered for technical skills with a verified quote only."
    )
    flagged = [record for record in view["records"] if record["tier"] is not None]
    for tier, tier_label in PRIORITY_TIER_LABELS:
        tier_records = [record for record in flagged if record["tier"] == tier]
        st.markdown(f"**{tier_label}** ({len(tier_records)})")
        if not tier_records:
            st.markdown("_None identified._")
            continue
        for state in _FLAGGED_STATES:
            group = [record for record in tier_records if record["state"] == state]
            if not group:
                continue
            st.markdown(f"*{STATE_LABELS[state]}* ({len(group)})")
            for record in group:
                entry = record["entry"]
                chip = (
                    f"<span style='background-color:{_STATE_COLORS[state]};color:white;padding:5px 10px;"
                    f"border-radius:20px;margin:3px;display:inline-block'>{_safe_text(record['skill'])}</span>"
                )
                category = _CATEGORY_LABELS.get(record["category"])
                line = chip + (f" · {category}" if category else "")
                line += f"<br>{_safe_text(entry['reason'])}"
                notes = _evidence_notes(entry, not record["in_extracted_list"])
                line += "<br><small style='opacity:0.75'>" + " · ".join(notes) + "</small>"
                st.markdown(f"<div style='margin-bottom:10px'>{line}</div>", unsafe_allow_html=True)


def build_session_options(sessions):
    """([session id, ...], {session id: "Name (#id)"}) for the session picker.

    A session is identified by its unique database id, so two sessions with the same name stay distinct
    ("Analysis (#4)" and "Analysis (#6)"). Entries that are not objects, or whose id is not an integer,
    are skipped; a blank name is shown as "Untitled"; the result is ordered by id. Anything that is not a
    list gives ([], {}).
    """
    entries = []
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, dict):
                continue
            session_id = item.get("id")
            if not isinstance(session_id, int) or isinstance(session_id, bool):
                continue
            name = " ".join(str(item.get("name") or "").split()) or "Untitled"
            entries.append((session_id, f"{name} (#{session_id})"))
    entries.sort(key=lambda entry: entry[0])
    labels = dict(entries)
    return list(labels), labels


def display_work_experience(experiences):
    st.subheader("💼 Work Experience")

    for i, exp in enumerate(experiences):
        # Normalize technologies_used (blank and non-string items are dropped)
        technologies = _text_items(exp.get('technologies_used'))
        # Fallback if technologies is empty
        tech_display = ', '.join(technologies) if technologies else "No technologies used"

        # Normalize responsibilities (blank and non-string items are dropped)
        responsibilities = _text_items(exp.get('responsibilities')) or ["No responsibilities specified."]

        with st.expander(f"**{exp.get('job_title', 'N/A')} at {exp.get('company_name', 'N/A')}** ({exp.get('employment_duration', 'N/A')})"):
            st.markdown(f"""
                <div style="background: #f8f8f8; color: #333; padding: 15px; border-radius: 10px;">
                    <strong>Responsibilities:</strong>
                    <ul>
                        {''.join([f"<li>{_safe_text(resp)}</li>" for resp in responsibilities])}
                    </ul>
                    <strong>Technologies:</strong> {_safe_text(tech_display)}
                </div>
            """, unsafe_allow_html=True)


UNVERIFIED_BULLET_WARNING = "⚠ Could not verify against extracted resume text"
UNVERIFIED_BULLET_NOTE = (
    "A warning means the bullet could not be found in the text extracted from the PDF. "
    "It does not mean the bullet is wrong."
)


def _indexed_bullets(value):
    """(position, text) for each non-blank string in a bullets list, else [] (legacy/malformed).

    Position is the index in the ORIGINAL list, so it still matches the parser's
    "bullets_unverified" indices when blank or non-string items are skipped.
    """
    if not isinstance(value, list):
        return []
    return [(position, item.strip()) for position, item in enumerate(value) if isinstance(item, str) and item.strip()]


def _usable_bullets(value):
    """Project bullets that can be shown: the non-blank strings of a list, else [] (legacy/malformed)."""
    return [text for _position, text in _indexed_bullets(value)]


def _text_items(value):
    """The non-blank strings (stripped) of a string or a list. A non-string item, or any other type, gives nothing."""
    if isinstance(value, str):
        value = [value]
    return _usable_bullets(value)


def _unverified_indices(value):
    """Set of integer positions from a project's "bullets_unverified"; anything malformed is ignored."""
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, int) and not isinstance(item, bool)}


def display_projects(projects):
    st.subheader("🚀 Key Projects")

    for i, proj in enumerate(projects):
        # Get outcomes and normalize (blank and non-string items are dropped)
        # Fallback if no usable outcome remains
        outcomes = _text_items(proj.get('outcomes')) or ["No outcomes specified."]

        # Get technologies and normalize (blank and non-string items are dropped)
        technologies = _text_items(proj.get('technologies_used'))
        # Fallback if technologies is empty
        tech_display = ', '.join(technologies) if technologies else "N/A"

        # Original resume bullets (new format); a missing, empty or malformed list falls back to the legacy view
        indexed_bullets = _indexed_bullets(proj.get('bullets'))

        with st.expander(f"**{proj.get('project_name', 'N/A')}**"):
            if indexed_bullets:
                # Every model-written string is HTML-escaped here; the legacy view below is unchanged
                unverified = _unverified_indices(proj.get('bullets_unverified'))
                bullets_html = "".join(
                    f"<li>{_safe_text(text)}"
                    + (f" <small>{UNVERIFIED_BULLET_WARNING}</small>" if position in unverified else "")
                    + "</li>"
                    for position, text in indexed_bullets
                )
                note_html = (
                    f"<p><small>{UNVERIFIED_BULLET_NOTE}</small></p>"
                    if any(position in unverified for position, _text in indexed_bullets) else ""
                )
                st.markdown(f"""
                <div style="background: #f8f8f8; color: #333; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p><strong>Objective:</strong> <em>{_safe_text(proj.get('objective', 'N/A'))}</em></p>
                    <p><strong>Bullets:</strong></p>
                    <ul>{bullets_html}</ul>{note_html}
                    <strong>Technologies:</strong> {_safe_text(tech_display)}
                </div>
                """, unsafe_allow_html=True)
                continue
            st.markdown(f"""
                <div style="background: #f8f8f8; color: #333; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <p><strong>Objective:</strong> <em>{_safe_text(proj.get('objective', 'N/A'))}</em></p>
                    <p><strong>Outcome:</strong></p>
                    <ul>
                        {''.join([f"<li>{_safe_text(resp)}</li>" for resp in outcomes])}
                    </ul>
                    <strong>Technologies:</strong> {_safe_text(tech_display)}
                </div>
            """, unsafe_allow_html=True)

# Example Usage:
# display_projects_expander_in_columns(projects_data)

def display_personal_info(info):
    st.subheader("👤 Personal Information")
    st.markdown(f"**Name:** {info.get('name', 'N/A')}")
    st.markdown(f"**Email:** {info.get('email', 'N/A')}")
    st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    st.markdown("### 🎓 Education")
    for edu in info.get("education", []):
        st.markdown(
            f"**{edu.get('degree', '')}**, "
            f"{edu.get('institution', '')} "
            f"({edu.get('graduation_year', '')})"
        )
        for achievement in edu.get("achievements", []):
            st.markdown(f"- {achievement}")
    for profile in info.get('profiles', []):
        profile = profile.strip()
        profile_lower = profile.lower()

        if profile_lower.startswith(("https://", "http://")):
            if "linkedin.com/" in profile_lower:
                label = "LinkedIn"
            elif "github.com/" in profile_lower:
                label = "GitHub"
            elif "kaggle.com/" in profile_lower:
                label = "Kaggle"
            else:
                label = "Other Link"

            st.markdown(f"🔗 [{label}]({profile})")
        else:
            st.markdown(f"🔗 {profile}")

import urllib.parse

# Shown when /recommend_courses reports a provider failure. It must never read like "no results".
COURSE_PROVIDER_ERROR_MESSAGE = (
    "YouTube could not be reached or rejected the request, so no recommendations could be loaded. "
    "This is not the same as having no results. Please try again later."
)
_YOUTUBE_HOSTS = ("www.youtube.com", "youtube.com", "m.youtube.com")
_VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")


def _youtube_video_id(link):
    """The video ID of an https YouTube watch link, or None. Only a well-formed 11-character ID passes."""
    try:
        parsed = urllib.parse.urlparse(str(link))
        if parsed.scheme != "https" or parsed.hostname not in _YOUTUBE_HOSTS or parsed.path != "/watch":
            return None
        video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
    except ValueError:
        return None
    return video_id if _VIDEO_ID_PATTERN.fullmatch(video_id) else None


def display_youtube_courses(courses, skill_name):
    """One card per video: thumbnail, title, channel and a Watch link.

    courses: {"title", "channel", "link"} dicts (a plain link string from older callers also works, and
    then shows the generic "<skill> Course" heading). The skill name, title and channel are HTML-escaped,
    and the link is rebuilt from a validated video ID, so no model- or provider-written text reaches the
    HTML unescaped. An entry without a valid YouTube watch link is skipped.
    """
    safe_skill = _safe_text(skill_name)
    skipped = 0
    for course in courses:
        if isinstance(course, dict):
            link, title, channel = course.get("link"), course.get("title"), course.get("channel")
        else:
            link, title, channel = course, None, None
        video_id = _youtube_video_id(link)
        if not video_id:
            skipped += 1
            continue

        title_html = _safe_text(title) if title else f"{safe_skill} Course"
        channel_html = (
            f'<div style="color:#444;font-size:14px;line-height:1.3;margin-bottom:6px;">Channel: {_safe_text(channel)}</div>'
            if channel else ""
        )
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        watch_url = f"https://www.youtube.com/watch?v={video_id}"

        # Display Card
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 14px; background-color: #f9f9f9; color: #111; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; border: 1px solid #ddd;">
            <img src="{thumbnail_url}" width="150" style="border-radius: 8px; flex-shrink: 0;">
            <div style="min-width: 0; flex: 1;">
                <div style="color:#555;font-size:12px;line-height:1.3;">For skill: {safe_skill}</div>
                <div style="color:#111;font-size:16px;font-weight:600;line-height:1.3;margin:2px 0 4px 0;overflow-wrap:anywhere;">{title_html}</div>{channel_html}
                <a href="{watch_url}" target="_blank" style="text-decoration: none;">
                    <button style="background-color:#FF0000;color:white;padding:6px 14px;border:none;border-radius:5px;cursor:pointer;">
                        ▶️ Watch on YouTube
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    if skipped:
        st.caption(f"{skipped} result(s) were skipped because their video link was not a valid YouTube link.")


def display_course_recommendations(status_code, body, skill):
    """Show the outcome of /recommend_courses. A provider failure is never shown as an empty result."""
    if status_code == 200 and isinstance(body, dict) and isinstance(body.get("courses"), list):
        st.subheader(f"🎓 Recommended YouTube Courses for {skill}")
        if body["courses"]:
            display_youtube_courses(body["courses"], skill)
        else:
            st.info("The search worked but found no matching YouTube videos for this skill.")
    elif status_code == 502:
        st.error(COURSE_PROVIDER_ERROR_MESSAGE)
    else:
        st.error("Failed to fetch recommendations.")


def display_ats_content(ats_content_data):
    st.subheader("💡 Optimized Resume Bullet Points")
    # Check if data is available
  
    if not ats_content_data or "optimized_bullets" not in ats_content_data:
        st.info("No ATS optimized content available.")
        return

    # --- Custom CSS for Styling the Expanders ---
    # This CSS will make each expander look like a distinct card.
    st.markdown("""
        <style>
        /* Styles for the expander container (the card itself) */
        .ats-section-card .stExpander {
            border: 1px solid #e0e0e0; /* Light grey border */
            border-radius: 12px; /* Rounded corners */
            margin-bottom: 20px; /* Space between cards */
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Soft shadow for depth */
            background-color: #ffffff; /* White background for the card */
            overflow: hidden; /* Ensures content respects border-radius */
        }

        /* Styles for the expander header (the clickable part) */
        .ats-section-card .stExpander > button {
            font-size: 1.15em; /* Slightly larger font for title */
            font-weight: bold;
            color: #2c3e50; /* Darker text for readability */
            padding: 18px 20px; /* More padding */
            width: 100%; /* Full width button */
            text-align: left; /* Align text to the left */
            background-color: #f8f9fa; /* Light background for header */
            border-bottom: 1px solid #e0e0e0; /* Separator line */
            border-top-left-radius: 12px; /* Match card radius */
            border-top-right-radius: 12px;
            display: flex; /* Use flexbox for alignment */
            align-items: center; /* Vertically center icon and text */
            gap: 10px; /* Space between icon and text */
        }
        /* Style for the Streamlit expander icon (caret) */
        .ats-section-card .stExpander > button svg {
            color: #3498db; /* Blue color for expander icon */
            transition: transform 0.3s ease; /* Smooth rotation transition */
        }
        /* Rotate icon when expander is open */
        .ats-section-card .stExpander > button[aria-expanded="true"] svg {
            transform: rotate(90deg); /* Rotates the caret right when expanded */
        }

        /* Styles for the expanded content area */
        .ats-section-card .streamlit-expanderContent {
            background-color: #fdfdfd; /* Very light background for content */
            padding: 15px 20px; /* Padding for content */
            border-bottom-left-radius: 12px; /* Match card radius */
            border-bottom-right-radius: 12px;
        }

        /* Styling for the unordered list of bullets */
        .ats-section-card ul {
            list-style-type: disc; /* Standard disc bullets */
            padding-left: 25px; /* Indent bullets */
            margin-top: 10px; /* Space above the list */
            margin-bottom: 10px; /* Space below the list */
        }

        /* Styling for individual bullet points */
        .ats-section-card ul li {
            margin-bottom: 8px; /* Space between bullet points */
            line-height: 1.6; /* Better readability */
            color: #495057; /* Slightly darker text for content */
        }
        </style>
    """, unsafe_allow_html=True)

    # Loop through each section in the ATS content
    for item in ats_content_data["optimized_bullets"]:
        section_title = item.get("section", "Untitled Section")
        bullets = item.get("bullets", [])

        # Create a custom container for each expander to apply our CSS classes
        with st.container():
            st.markdown(f'<div class="ats-section-card">', unsafe_allow_html=True) # Start custom wrapper div
            
            # Use st.expander for interactive collapsing
            with st.expander(f"**{section_title}**", expanded=False): # 'expanded=False' makes them initially collapsed
                if bullets:
                    # Construct an HTML unordered list for the bullet points
                    bullets_html = "".join([f"<li>{bullet}</li>" for bullet in bullets])
                    st.markdown(f"<ul>{bullets_html}</ul>", unsafe_allow_html=True)

                    # Add a Streamlit button to copy all bullets for the section
                    section_bullets_text = "\n".join(bullets)
                    # Using hash() for key to ensure uniqueness even if section titles are identical
                    if st.button(f"📋 Copy Bullets", key=f"copy_btn_{section_title.replace(' ', '_').replace('|', '')}_{hash(section_title)}"):
                        st.code(section_bullets_text, language='text')
                        st.success("The bullet points are now in the code block above. Please copy them manually from there.")
                else:
                    st.info("No bullet points available for this section.")
            st.markdown('</div>', unsafe_allow_html=True) # End custom wrapper div
