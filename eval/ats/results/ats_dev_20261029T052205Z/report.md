# ATS bullet rewrite evaluation: ats_dev_20261029T052205Z

**Split: DEVELOPMENT** | status: complete | model: openai/gpt-oss-20b | cases: 9 | repeats: 2

Read this first:
- 12 synthetic cases cannot show that the feature works on real resumes. These are screening signals, not accuracy.
- Bullet-level checks run only on responses that parsed and validated, so they ignore every failed response (selection effect); the reliability rows below keep the failures.
- D3 (declared absent skills) and D4 (vocabulary technologies, advisory) can flag the same bullet. They are reported separately and must not be added.
- Flags are not human quality judgments. Only the reviewed rows below are.

## Reliability

| Item | Count |
|---|---|
| Planned calls | 18 |
| Attempted calls | 18 |
| Not attempted | 0 |
| Responses received | 18 |
| API failures | 0/18 (0.0%) of attempted calls |
| Empty responses | 0 |
| JSON parse failures | 0 |
| Schema-invalid responses | 0/18 (0.0%) |
| Truncated (output cap reached) | 0/18 (0.0%) |
| Responses with no finish reason reported | 0/18 |
| Checker errors (raw response saved, not evaluated) | 0 |
| Production failures (would be HTTP 500) | 0/18 (0.0%) |
| Outputs eligible for bullet checks | 18/18 (100.0%) |

**Evaluation-only output cap: max_tokens=4096** (production 8192). Truncated responses reached this cap. They are excluded from the production-failure rate above (numerator and denominator) and are not model-quality failures at the production setting.

## Deterministic checks (each with its own denominator)

| Check | Bullets flagged / checked |
|---|---|
| D2 invented numbers | 0/102 (0.0%) |
| D3 declared absent skills | 6/102 (5.9%) |
| D4 technologies not in resume (advisory) | 6/102 (5.9%) |
| D6 rewrite identical to a source bullet | 8/102 |

D2 extras (advisory): prompt-example echo 0, unit mismatch 0.
D5 sections (outputs: 18): unmatched output section 0, missing source section 0, bullet-count change 0, ambiguous or duplicate group 0.

## Human review

Bullets reviewed: 0/102 (0.0%).

## Repeat consistency (raw counts only)

- Cases where every repeat was ok: 9/9
- Cases where repeats had the same status: 9/9

| Case | Status by repeat | D2 | D3 | D4 |
|---|---|---|---|---|
| c01_strong_task_api | r1: ok, r2: ok | r1: 0/9, r2: 0/9 | r1: 0/9, r2: 0/9 | r1: 0/9, r2: 0/9 |
| c03_vague_backend_intern | r1: ok, r2: ok | r1: 0/5, r2: 0/5 | r1: 3/5, r2: 3/5 | r1: 3/5, r2: 3/5 |
| c04_vague_data_scripts | r1: ok, r2: ok | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 |
| c05_absent_k8s_kafka | r1: ok, r2: ok | r1: 0/6, r2: 0/6 | r1: 0/6, r2: 0/6 | r1: 0/6, r2: 0/6 |
| c07_metric_heavy_formats | r1: ok, r2: ok | r1: 0/7, r2: 0/7 | r1: 0/7, r2: 0/7 | r1: 0/7, r2: 0/7 |
| c08_no_metric_scale_trap | r1: ok, r2: ok | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 |
| c09_no_metric_perf_trap | r1: ok, r2: ok | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 | r1: 0/5, r2: 0/5 |
| c11_fresher_llm_rag | r1: ok, r2: ok | r1: 0/7, r2: 0/7 | r1: 0/7, r2: 0/7 | r1: 0/7, r2: 0/7 |
| c12_short_mismatch | r1: ok, r2: ok | r1: 0/2, r2: 0/2 | r1: 0/2, r2: 0/2 | r1: 0/2, r2: 0/2 |

## Usage

Measured token usage was returned for 18/18 responses (total 21037). Plan-time token figures are estimates.

Latency (s): min 0.656, median 0.875, max 1.125.
