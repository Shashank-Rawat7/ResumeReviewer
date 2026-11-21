# Step 5 — ATS Bullet Rewrite Human Review

## Run and scope

- Run ID: `ats_dev_20261029T052205Z`
- Split: development only
- Synthetic cases reviewed: 9
- Model repeats: 2 per case
- Generated bullets reviewed: 102
- Live application-level invocations: 18
- Holdout cases: excluded
- Production pipeline changes: none

## Human-review results

### Source-evidence support

| Judgment | Count | Percentage |
|---|---:|---:|
| Yes | 70/102 | 68.6% |
| Partially | 32/102 | 31.4% |
| No | 0/102 | 0.0% |

A partially supported bullet may preserve its core claim while adding an unverified detail, implementation choice, outcome, or benefit.

### Overall rewrite improvement

| Judgment | Count | Percentage |
|---|---:|---:|
| Improved | 70/102 | 68.6% |
| Unchanged | 18/102 | 17.6% |
| Worse | 14/102 | 13.7% |

Percentages are rounded. Counts and denominators are authoritative.

## Automated-check comparison

- D2 invented-number flags: 0/102.
- D3 absent-skill flags: 6/102.
- D4 unsupported-technology flags: 6/102.
- All D3/D4 flags came from `c03_vague_backend_intern`.
- D3 and D4 overlap; their counts must not be summed.
- D6 identical-to-source flags: 8/102, as recorded in the baseline summary.

Automated checks and human judgments measure different properties. Zero D2 flags does not establish complete factual correctness.

## Validation and artifact integrity

- Review decisions: 102 rows with unique `(case_id, repeat, bullet_index)` keys.
- Original non-review fields compared against the reviewed copy: unchanged.
- Original `bullets.csv`: not overwritten.
- Reviewed output: `bullets_reviewed.csv`.
- Consolidated decisions: `human_review_decisions.csv`.
- Label values: validated against the agreed vocabulary.

## Interpretation and limitations

This is a small synthetic development evaluation covering nine cases and two repeats. It is useful for finding failure patterns, but it does not establish general accuracy on real-world resumes or job descriptions.

Human judgments are provisional and were consolidated from case-by-case review. They should not be treated as independent annotator agreement or a statistically representative estimate.

Partially supported bullets require attention because a correct central claim can still contain unsupported details. Unchanged wording may be factually correct but provide little rewriting value.

The three holdout cases remain excluded from tuning and this report. No production prompt or pipeline changes were made during this evaluation.

## Step 5 conclusion

Human review and artifact-integrity validation are complete for the development baseline. The reviewed copy and structured decisions are preserved separately from the original baseline artifacts.

Step 5 may be marked complete. Proceed to Step 6 only under the existing project approval rules.
