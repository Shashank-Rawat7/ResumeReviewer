# Pipeline evaluation summary: pipeline_918e73269cf5

- plan hash: `e178ce2e87927dc53387b88fff6bcdc428a8e5344c5d40e619f9e1ca05224f28`
- run status: completed; repeats summarised: 3
- configured model: openai/gpt-oss-20b (name echoed by the client library, not independently verified by the provider; recorded: openai/gpt-oss-20b)

## Provenance and conditions

- repeat 1 comes from run `pipeline_8a8a6a446c17` (plan hash `1115e1a8aeb8d7cd5dbebf5ab9975345b92a7c4f15a12482bba791d7cbc51072`), read unchanged; the other repeats come from this run
- client request timeout: repeat 1: none set (library default); repeat 2: 120 s; repeat 3: 120 s
- 23 call-log line(s) of the interrupted source run (outside repeat 1) are excluded

## Metrics

Each repeat shows numerator/denominator; the range is over repeats with an available rate.
E3 rows are deterministic screens only (zero flags is not proof of correctness).

| stage | metric | rep 1 | rep 2 | rep 3 | range |
|---|---|---|---|---|---|
| course | offered_options_match_rule | 10/10 | 10/10 | 10/10 | 1.0 to 1.0 |
| course | skipped_count_match_rule | 10/10 | 10/10 | 10/10 | 1.0 to 1.0 |
| course | label_eligibility_agreement | 58/65 | 58/65 | 63/65 | 0.892308 to 0.969231 |
| e1 | skill_recall | 64/68 | 65/68 | 64/68 | 0.941176 to 0.955882 |
| e1 | precision | n/a | n/a | n/a | n/a |
| e1 | f1 | n/a | n/a | n/a | n/a |
| e2 | status_agreement | 57/65 | 57/65 | 60/65 | 0.876923 to 0.923077 |
| e2 | status_agreement_strict | 57/65 | 57/65 | 60/65 | 0.876923 to 0.923077 |
| e2 | gap_detection_recall | 30/37 | 30/37 | 34/37 | 0.810811 to 0.918919 |
| e2 | false_flag_rate | 0/28 | 0/28 | 0/28 | 0.0 to 0.0 |
| e2 | status_exact_match_given_flagged | 29/30 | 29/30 | 32/34 | 0.941176 to 0.966667 |
| e2 | tier_agreement_determinate | 29/29 | 27/28 | 28/30 | 0.933333 to 1.0 |
| e2 | tier_agreement_ambiguous_set | 2/2 | 3/3 | 4/5 | 0.8 to 1.0 |
| e2 | jd_evidence_found | 31/32 | 32/32 | 36/36 | 0.96875 to 1.0 |
| e2 | jd_quote_matches_authored_anchor | 29/32 | 30/32 | 34/36 | 0.90625 to 0.944444 |
| e2 | resume_evidence_quote_found | 5/5 | 2/2 | 4/4 | 1.0 to 1.0 |
| e3 | bullets_with_unsupported_number (screen) | 0/58 | 0/59 | 0/59 | 0.0 to 0.0 |
| e3 | bullets_with_number_not_in_resume_text (screen) | 0/58 | 0/59 | 0/59 | 0.0 to 0.0 |
| e3 | bullets_mentioning_absent_skill (screen) | 0/58 | 0/59 | 0/59 | 0.0 to 0.0 |
| e3 | bullets_with_guard_phrase (screen) | 0/18 | 0/19 | 0/19 | 0.0 to 0.0 |
| e3 | scenario_has_flagged_bullet (screen) | 0/10 | 0/10 | 0/10 | 0.0 to 0.0 |

## Call outcomes

| stage | ok | failed | skipped | attempts | retried | finish reasons |
|---|---|---|---|---|---|---|
| e1 | 30 | 0 | 0 | 30 | 0 | stop=30 |
| e2 | 30 | 0 | 0 | 30 | 0 | stop=30 |
| e3 | 30 | 0 | 0 | 31 | 1 | stop=30 |

## Warnings per repeat

- repeat 1: 1 warning(s)
- repeat 2: 0 warning(s)
- repeat 3: 0 warning(s)

## Limitations

- Ten hand-authored synthetic scenarios, one model, draft-1 labels: a baseline, not a generalisation claim.
- Pooled rates add numerators and denominators across scenarios; per-scenario results are kept beside them.
- E3 numbers are deterministic screens, not verification that bullets are factually correct.
- Ranges across repeats show variation between runs; they are not confidence intervals.
- Repeats are split over two runs (pipeline_8a8a6a446c17: repeat 1; this run: repeats 2-3). 23 call-log line(s) of the interrupted source run are excluded.
- The report scores only the supplied precomputed outputs. It is not a measurement of a live model.
- Stage reports are embedded unchanged from score.py and e3.py; read their own limitations.
- E3 results are deterministic screens, not verification that bullets are factually correct.
