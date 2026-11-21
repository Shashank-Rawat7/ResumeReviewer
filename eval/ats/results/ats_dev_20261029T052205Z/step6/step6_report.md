# Step 6: human review and repeat analysis of ats_dev_20261029T052205Z

Generated 2026-10-29T08:03:05.645818+00:00 by `eval/ats/analyze_step6.py` (read-only).

## Read this first

- **Synthetic, development-only data:** 9 synthetic development cases, 2 repeats each, 102 generated bullets; not real resumes.
- **Provisional human judgments:** provisional: the files record one labelling source, not the reviewer's identity or independence; the decisions file consolidates the reviewed sheet, so their agreement is not independent corroboration.
- **Holdouts:** the three holdout cases were excluded from the baseline plan (plan split is dev); this analysis reads no case file, no raw response and no holdout data.
- **Evaluation-only cap:** the baseline ran with an evaluation-only max_tokens of 4096 (override 4096); production uses 8192. Results describe the capped configuration, not production.
- **Not ground truth:** neither the automated checks (D2-D6) nor the human labels are treated as ground truth. No accuracy, precision or recall is calculated, and nothing here shows the feature works on real resumes.
- **D3 and D4 overlap** and are always reported separately, never summed.

## 1. Integrity

| Check | Result |
|---|---|
| Rows (bullets / reviewed / decisions) | 102 / 102 / 102 |
| Keys (case_id, repeat, bullet_index) unique and identical in all three files | yes (102 keys) |
| Reviewed sheet preserves every original non-review field (by key) | yes (0 cells differ) |
| Labels outside the agreed sets | 0 |
| Reviewed vs decisions disagreements (all six fields) | 0 |
| Original sheet cells already filled by a human | 0 |
| Calls: status ok and linked by call_id | yes (18 calls) |

## 2. Human review: all 102 bullets

**supported_by_source_evidence**

| Value | Bullets / total |
|---|---|
| yes | 70/102 (68.6%) |
| partially | 32/102 (31.4%) |
| no | 0/102 (0.0%) |
| other_or_blank | 0/102 (0.0%) |

**relevance_to_job_description**

| Value | Bullets / total |
|---|---|
| high | 72/102 (70.6%) |
| medium | 30/102 (29.4%) |
| low | 0/102 (0.0%) |
| other_or_blank | 0/102 (0.0%) |

**clarity_and_action_orientation**

| Value | Bullets / total |
|---|---|
| high | 96/102 (94.1%) |
| medium | 6/102 (5.9%) |
| low | 0/102 (0.0%) |
| other_or_blank | 0/102 (0.0%) |

**overall_rewrite_improvement**

| Value | Bullets / total |
|---|---|
| improved | 70/102 (68.6%) |
| unchanged | 18/102 (17.6%) |
| worse | 14/102 (13.7%) |
| other_or_blank | 0/102 (0.0%) |

**unsupported_or_invented_claims** (field content type)

| Content | Bullets / total |
|---|---|
| none | 66/102 (64.7%) |
| explanatory_text | 36/102 (35.3%) |
| blank | 0/102 (0.0%) |

**overall_rewrite_improvement by support** (counts)

| Support | Bullets | improved | unchanged | worse |
|---|---|---|---|---|
| yes | 70 | 52 | 18 | 0 |
| partially | 32 | 18 | 0 | 14 |
| no | 0 | 0 | 0 | 0 |

## 3. By case and repeat (counts; each row shows its own bullet total)

| Case | Bullets | support yes/partially/no | relevance high/medium/low | clarity high/medium/low | improved/unchanged/worse |
|---|---|---|---|---|---|
| c01_strong_task_api | 18 | 16/2/0 | 14/4/0 | 18/0/0 | 14/4/0 |
| c03_vague_backend_intern | 10 | 2/8/0 | 8/2/0 | 10/0/0 | 4/0/6 |
| c04_vague_data_scripts | 10 | 0/10/0 | 6/4/0 | 8/2/0 | 8/0/2 |
| c05_absent_k8s_kafka | 12 | 10/2/0 | 10/2/0 | 12/0/0 | 10/2/0 |
| c07_metric_heavy_formats | 14 | 14/0/0 | 10/4/0 | 14/0/0 | 12/2/0 |
| c08_no_metric_scale_trap | 10 | 4/6/0 | 8/2/0 | 8/2/0 | 6/0/4 |
| c09_no_metric_perf_trap | 10 | 6/4/0 | 4/6/0 | 8/2/0 | 6/2/2 |
| c11_fresher_llm_rag | 14 | 14/0/0 | 12/2/0 | 14/0/0 | 6/8/0 |
| c12_short_mismatch | 4 | 4/0/0 | 0/4/0 | 4/0/0 | 4/0/0 |

Partial support, worse rewrites and unchanged wording by case and repeat:

| Case | Repeat | Bullets | support partially | overall worse | overall unchanged |
|---|---|---|---|---|---|
| c01_strong_task_api | 1 | 9 | 0 | 0 | 2 |
| c01_strong_task_api | 2 | 9 | 2 | 0 | 2 |
| c03_vague_backend_intern | 1 | 5 | 4 | 3 | 0 |
| c03_vague_backend_intern | 2 | 5 | 4 | 3 | 0 |
| c04_vague_data_scripts | 1 | 5 | 5 | 1 | 0 |
| c04_vague_data_scripts | 2 | 5 | 5 | 1 | 0 |
| c05_absent_k8s_kafka | 1 | 6 | 1 | 0 | 1 |
| c05_absent_k8s_kafka | 2 | 6 | 1 | 0 | 1 |
| c07_metric_heavy_formats | 1 | 7 | 0 | 0 | 1 |
| c07_metric_heavy_formats | 2 | 7 | 0 | 0 | 1 |
| c08_no_metric_scale_trap | 1 | 5 | 3 | 2 | 0 |
| c08_no_metric_scale_trap | 2 | 5 | 3 | 2 | 0 |
| c09_no_metric_perf_trap | 1 | 5 | 2 | 1 | 1 |
| c09_no_metric_perf_trap | 2 | 5 | 2 | 1 | 1 |
| c11_fresher_llm_rag | 1 | 7 | 0 | 0 | 2 |
| c11_fresher_llm_rag | 2 | 7 | 0 | 0 | 6 |
| c12_short_mismatch | 1 | 2 | 0 | 0 | 0 |
| c12_short_mismatch | 2 | 2 | 0 | 0 | 0 |

## 4. Data-quality observations (labels preserved as supplied, not corrected)

- **4 of 70 bullets rated support `yes` also carry explanatory text in `unsupported_or_invented_claims`.** Rows: c03_vague_backend_intern r1 #1, c03_vague_backend_intern r2 #1, c08_no_metric_scale_trap r1 #1, c08_no_metric_scale_trap r2 #1. This may be a labelling inconsistency or a deliberate caveat; it needs the reviewer's decision.
- Rated `partially` or `no` without claim text: 0.
- Reviewer notes blank: 0/102 (0.0%).

## 5. Automated checks vs human review (counts only)

| Automated flag | Bullets flagged / total |
|---|---|
| d2 | 0/102 (0.0%) |
| d3 | 6/102 (5.9%) |
| d4 | 6/102 (5.9%) |
| d6_unchanged | 8/102 (7.8%) |
| d6_near_duplicate | 0/102 (0.0%) |
| any_checker_finding_text_in_sheet | 14/102 (13.7%) |

D3/D4 overlap (a partition of 102): both 6, D3 only 0, D4 only 0, neither 96. D3 and D4 are never summed.

**d2_vs_support**

| Group | Bullets | yes | partially | no |
|---|---|---|---|---|
| flagged | 0 | 0 | 0 | 0 |
| not_flagged | 102 | 70 | 32 | 0 |

**d3_vs_support**

| Group | Bullets | yes | partially | no |
|---|---|---|---|---|
| flagged | 6 | 0 | 6 | 0 |
| not_flagged | 96 | 70 | 26 | 0 |

**d4_vs_support**

| Group | Bullets | yes | partially | no |
|---|---|---|---|---|
| flagged | 6 | 0 | 6 | 0 |
| not_flagged | 96 | 70 | 26 | 0 |

**d6_unchanged_vs_overall**

| Group | Bullets | improved | unchanged | worse |
|---|---|---|---|---|
| flagged | 8 | 0 | 8 | 0 |
| not_flagged | 94 | 70 | 10 | 14 |

**d6_near_duplicate_vs_overall**

| Group | Bullets | improved | unchanged | worse |
|---|---|---|---|---|
| flagged | 0 | 0 | 0 | 0 |
| not_flagged | 102 | 70 | 18 | 14 |

Union of D2/D3/D4 (not a sum): of 32 bullets rated partially or no, 26 had no D2/D3/D4 flag. Of 18 rated `unchanged`, 10 were not flagged by D6.

These are counts, not accuracy. D2 flagged no bullets, so it cannot be compared. accuracy, precision and recall (no defensible ground truth exists).

## 6. Repeat comparison

Method: positions are matched by (case_id, bullet_index); a pair is compared only when output_section and matched_source_heading agree and the heading is non-blank; anything else is reported, never forced.

| Pair status | Positions |
|---|---|
| paired | 51 |

Cases with different bullet counts between repeats: none.

Paired positions: 51.

| Wording (paired positions) | Count / paired |
|---|---|
| Exactly equal | 25/51 (49.0%) |
| Equal only after normalisation (case/punctuation/spacing) | 1/51 (2.0%) |
| Equal after normalisation (total) | 26/51 (51.0%) |
| Changed after normalisation | 25/51 (49.0%) |

Human judgments that differ between repeats:

| Dimension | Positions / paired |
|---|---|
| supported_by_source_evidence | 2/51 (3.9%) |
| relevance_to_job_description | 0/51 (0.0%) |
| clarity_and_action_orientation | 0/51 (0.0%) |
| overall_rewrite_improvement | 4/51 (7.8%) |

Judgment differences by wording class (positions where any of the four dimensions differs):

| Wording class | Positions | Any dimension differs |
|---|---|---|
| exact_wording | 25 | 0 |
| normalized_equal_not_exact | 1 | 0 |
| wording_changed | 25 | 6 |

Response status: {'ok': 18}; cases with the same status in both repeats 9/9 (100.0%); truncated calls 0. every call had the same status, so status consistency carries no information here.

## 7. What this analysis does not show

- accuracy
- precision
- recall
- inter-rater reliability
- generalisation to real resumes
- production-equivalent quality (the run used a lower evaluation-only token cap)
- Two repeats per case is too few to describe variability beyond these raw counts.

## 8. Baseline hash verification

| File | SHA-256 (first 16) | Unchanged after analysis |
|---|---|---|
| bullets.csv | cd39b7bf7a3feedf | yes |
| bullets_reviewed.csv | d3c9fe877f10002a | yes |
| human_review_decisions.csv | 3f74ea6aabce1d3f | yes |
| calls.jsonl | 09d81363578e5fe3 | yes |
| evaluations.jsonl | c117548828d4644e | yes |
| plan.json | 5fa407e4b3046fbd | yes |
| summary.json | 175945392f0b1a6c | yes |
| metadata.json | 73bf6b11eb09d14f | yes |
| report.md | 74308c56161c6e8d | yes |
