# Job-description inputs for the Stage 2 skill-gap evaluation

**Public-release note:** the posting texts described below (`strong.txt`, `partial.txt`, `poor.txt`) are **not included in this repository**, because they are third-party content that cannot be redistributed. This file records only source metadata (titles, URLs, status, what was removed) and short hash prefixes of the local copies. It reproduces no posting text.

The notes below describe the maintainer's local, unpublished working copies, which must not be published or redistributed. They were retrieved on 2026-10-26 with the Claude desktop app's built-in browser. Only page boilerplate was removed, the requirements were not edited, summarised, reordered or reworded, and blank lines between sections and list items were normalised (no words changed). These statements compare the local copies with the original web pages; they say nothing about content in this repository.

Fit labels (strong / partial / poor) describe the posting against the one test resume and are provisional until
the ground-truth comparison is done. Seniority is recorded here separately from technical skill gaps.

## strong.txt
- Posting: Python AI Engineer / GenAI Backend Developer (1 - 2 Years), BitCot
- URL: https://www.bitcot.com/job-openings/python-ai-engineer/
- Status when retrieved: live; no posting date shown
- Stated experience: the body gives a range of roughly one to one and a half years and the header gives one to two years (the posting is inconsistent)
- Seniority: entry level
- Removed: recruiter email addresses and phone number, the apply-now lines, the application-form fields,
  the data-processing consent checkboxes, and two stray form labels
- Added: the first line is the page title taken from the browser tab (site suffix removed)
- Documented conflicts: none beyond the experience range above

## partial.txt
- Posting: Generative AI Engineer (Python + RAG + Agentic AI), Scrobits Technologies, via Peerlist
- URL: https://peerlist.io/company/scrobits/careers/generative-ai-engineer-python--rag--agentic-ai/jobha9ee7rdmbelqnikkmpogeoal6r
- Status when retrieved: ARCHIVED (the page carried an archived notice), so no longer accepting applications; no posting date shown
- Stated experience: described only as experienced, no number of years
- Seniority: unspecified, but the wording asks for strong, deep and production-level experience
- Removed: site navigation, the archived banner (recorded above instead), apply buttons, tab labels,
  the block listing people at the company, promotional blocks and the similar-jobs list
- Kept on purpose: the separate must-have skills list (Python, Langchain, FastAPI, MCP, RAG, Agentic AI)
- Documented conflict: the must-have skills list names MCP, but the description also places familiarity with
  FastMCP and other model-context-protocol integrations under its preferred skills.
  Either crucial or good-to-have is accepted for MCP.

## poor.txt
- Posting: AI ML Platform Engineer, NYXSES India Private Limited, via Weekday
- URL: https://jobs.weekday.works/nyxses-india-private-limited-ai-ml-platform-engineer
- Status when retrieved: live; no posting date shown
- Stated experience: 5+ yrs; salary shown as a range in INR (kept as part of the header block)
- Seniority: Senior Software Engineer or Technical Lead; requires technical-lead, architect or mentor experience.
  This is a large seniority mismatch for the test resume and is tracked separately from skill gaps.
- Removed: breadcrumb and repeated title lines, the posted-by attribution line, sign-in, all-jobs and report
  links, the description / application tab labels, hide / apply / save buttons and the similar-jobs list
- Kept on purpose: the key-skills line and the separate must-have skills list (GPU Computing, CUDA,
  Distributed ML Systems)
- Documented conflicts: the must-have skills list includes CUDA and GPU Computing, but the description also
  places CUDA, GPU kernel and profiling knowledge and performance optimisation under its preferred qualifications
  (either crucial or good-to-have is accepted). Containers, Kubernetes, CI/CD and observability appear in both
  the required and the preferred lists.

## File integrity of the local copies (SHA-256, first 12 characters; UTF-8, LF line endings)
- strong.txt   d13ec67c14e3
- partial.txt  644fe6948d4b
- poor.txt     dc8b4b32094a
