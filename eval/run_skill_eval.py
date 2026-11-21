"""Local Stage 2 evaluation runner (not part of the application)."""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parser import (  # noqa: E402
    PRIORITY_TIERS,
    ResumeParser,
    ResumeParserConfig,
    build_prioritized_missing_skills,
)

JOB_NAMES = ("strong", "partial", "poor")
MAX_REPEAT = 5
ENTRY_FIELDS = (
    "skill", "category", "status", "jd_evidence",
    "evidence_found", "evidence_related", "in_extracted_list",
    "resume_mention_found", "resume_evidence_found", "resume_evidence_names_skill",
)
PACKAGES = ("langchain", "langchain-core", "langchain-groq", "groq", "PyPDF2")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repeat_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number")
    if not 1 <= count <= MAX_REPEAT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_REPEAT}")
    return count


def load_job_texts(jobs_dir: Path, names) -> dict:
    jobs, problems = {}, []
    for name in names:
        path = jobs_dir / f"{name}.txt"
        if not path.is_file():
            problems.append(f"{name}.txt is missing")
            continue
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}.txt is not valid UTF-8")
            continue
        if not text.strip():
            problems.append(f"{name}.txt is empty")
            continue
        jobs[name] = (data, text.rstrip("\r\n"))
    if problems:
        sys.exit("Job-description preflight failed: " + "; ".join(problems))
    return jobs


def analyse_job(parser, resume_text, jd_text, progress):
    progress["stage"] = "job_skills_chain"
    job_response = (parser.job_prompt | parser.llm).invoke(
        {"job_description": jd_text}
    ).content
    job_response = re.sub(r"^```json|```$", "", job_response.strip()).strip()
    progress["stage"] = "comparison_chain"
    raw = (parser.missing_skills_prompt | parser.llm).invoke({
        "job_description": jd_text,
        "job_skills": job_response,
        "resume_content": resume_text,
    }).content
    progress["stage"] = "normalization"
    prioritized, legacy = build_prioritized_missing_skills(
        raw, jd_text, job_response, resume_text
    )
    return job_response, prioritized, legacy


def build_record(name, jd_bytes, job_response, prioritized, legacy):
    entries = [
        {"tier": tier, **{field: entry.get(field) for field in ENTRY_FIELDS}}
        for tier in PRIORITY_TIERS
        for entry in prioritized.get(tier, [])
    ]
    try:
        extracted = json.loads(job_response)
    except ValueError:
        extracted = None
    return {
        "job": name,
        "jd_chars": len(jd_bytes.decode("utf-8")),
        "jd_sha256": sha256_hex(jd_bytes),
        "extracted_jd_skills": extracted,
        "extracted_jd_skills_parse_ok": extracted is not None,
        "normalization_ok": prioritized.get("normalization_ok"),
        "entries": entries,
        "counts": {
            "total": len(entries),
            "by_tier_status": dict(Counter(f"{e['tier']}/{e['status']}" for e in entries)),
        },
        "legacy_missing_skills": legacy,
        "diagnostics": prioritized.get("diagnostics", []),
    }


def package_versions() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--resume", required=True)
    args.add_argument("--jobs", default=str(ROOT / "eval" / "jobs"))
    args.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args.add_argument("--only", choices=JOB_NAMES)
    args.add_argument("--repeat", type=repeat_count, default=1)
    opts = args.parse_args()
    if opts.repeat > 1 and not opts.only:
        args.error("--repeat greater than 1 requires --only (choose one job: strong, partial or poor)")

    resume_path = Path(opts.resume)
    if not resume_path.is_file() or resume_path.suffix.lower() != ".pdf":
        sys.exit("Resume must be an existing .pdf file.")
    selected = (opts.only,) if opts.only else JOB_NAMES
    custom_run = bool(opts.only) or opts.repeat != 1
    jobs = load_job_texts(Path(opts.jobs), selected)
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (it is read from .env or the environment).")

    parser = ResumeParser()
    chunks = parser.preprocess_resume(str(resume_path))
    resume_text = " ".join(chunk.page_content for chunk in chunks)

    now = datetime.now(timezone.utc)
    run_dir = Path(opts.out) / now.strftime("run_%Y%m%dT%H%M%SZ")
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        sys.exit("A results folder for this second already exists; wait a second and run again.")
    pdf_bytes = resume_path.read_bytes()
    src_dir = Path(ResumeParserConfig.CURRENT_DIR)
    metadata = {
        "utc_time": now.isoformat(),
        "model": ResumeParserConfig.GROQ_MODEL,
        "temperature": parser.llm.temperature,
        "parser_py_sha256": sha256_hex((src_dir / "parser.py").read_bytes()),
        "prompts_yaml_sha256": sha256_hex((src_dir / "prompts.yaml").read_bytes()),
        "resume_pdf_bytes": len(pdf_bytes),
        "resume_pdf_sha256": sha256_hex(pdf_bytes),
        "resume_text_chars": len(resume_text),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    if custom_run:
        metadata["selected_jobs"] = list(selected)
        metadata["repeat"] = opts.repeat
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if custom_run:
        print(f"Planned: {len(selected) * opts.repeat} analyses, "
              f"{2 * len(selected) * opts.repeat} model calls (more if the provider retries).")

    for name in selected:
        jd_bytes, jd_text = jobs[name]
        for index in range(1, opts.repeat + 1):
            label = name if opts.repeat == 1 else f"{name}_r{index}"
            progress = {"stage": "not_started"}
            try:
                job_response, prioritized, legacy = analyse_job(
                    parser, resume_text, jd_text, progress
                )
                record = build_record(name, jd_bytes, job_response, prioritized, legacy)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                record = {
                    "job": name,
                    "error_type": type(exc).__name__,
                    "failed_stage": progress["stage"],
                    "http_status": status if isinstance(status, int) else None,
                }
                print(f"{label}: FAILED at {progress['stage']} ({type(exc).__name__}, "
                      f"http_status={record['http_status']})")
            else:
                print(f"{label}: {record['counts']['total']} entries, "
                      f"normalization_ok={record['normalization_ok']}")
            (run_dir / f"{label}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
