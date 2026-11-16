# Human audits

Labels and agreement statistics of the four human audits reported in the paper.
Three annotators labelled each audit independently and are anonymized as
Annotator A, B and C. Their individual labels are released; the annotation apps
and the raw per-annotator result files are not.

## At a glance

| Audit | What is labelled | Items labelled by all three | Fleiss κ |
|-|-|-|-|
| Judge agreement | One claim of a hop: `supported` / `unsupported` / `contradicted` | 121 (main 101 from 50 trajectories, calibration 20) | main 0.531, calibration 0.582 |
| Abstention parser | Whether a hop's output is an abstention: `abstain` / `answer` / `unclear` | 80 (overlap) of 915 | 0.858 |
| Retrieval proxy | Cross-index hop-0 abstentions where the answer-bearing proxy fired | 106 (main 90, calibration 16) | main 0.498, calibration 0.400 |
| Question templates | Whether a Korean–English template pair asks the same thing | 37 (main 27, calibration 10) | main 0.642 (Gwet's AC1 0.894), calibration 0.127 |

For the judge, proxy and template audits the paper reports the `main` items;
their `calibration` items were labelled first to align the annotators and are
kept separate. In the parser audit the 80 `calibration` hops are the
three-way overlap on which agreement is measured.

## Files

| File | Contents |
|-|-|
| `iaa_labels.csv` | Judge agreement: the three labels, their `majority` (`tie` when all differ) and the LLM judge's `judge_label`, keyed by `model`, `qid`, `hop`, `claim_idx` |
| `iaa_score.json` | Fleiss and pairwise Cohen κ, and the judge's agreement with the majority (88/98 three-way excluding ties, 89/101 supported vs not) |
| `parser_audit_labels.csv` | Abstention parser: 915 hops sampled by failure-risk stratum. All three annotators label the 80 `calibration` hops; each labels a disjoint shard of the rest, so the other two columns are blank there. `majority` is the gold label |
| `parser_audit_score.json` | Agreement on the 80-hop overlap |
| `parser_majority/` | The gold label of every audited hop with its stratum, population size and model output, one file per model and parser decision — the input of the precision/recall scripts below |
| `proxy_audit_labels.csv` | Retrieval proxy: `proxy_fp` (the answer entity appears, but the passage does not answer the question), `overabstain` (the passage answers it and the model abstained anyway), `ambiguous`, `parser_fp` (the output is not an abstention) |
| `proxy_audit_score.json` | Agreement, label distribution and rates with intervals |
| `template_audit_labels.csv` | Question templates: `equivalent`, `relation_mismatch`, `scope_mismatch`, `unnatural_ko`, `unnatural_en`. `author_draft` is the author's draft label for the calibration items, for reference only |
| `template_audit_relabels.csv` | Three `main` labels one annotator changed (`unnatural_ko` → `scope_mismatch`) after the label definitions were clarified; the reported agreement uses the changed labels |
| `template_audit_score.json` | Fleiss κ, Gwet's AC1 and specific agreement |
| `template_main.jsonl` | The audited template pairs with example questions |
| `instruments/` | The guidelines and rubric the annotators worked from (Korean) |

## Recomputing the statistics

Every Fleiss κ above can be recomputed from the `label_A`, `label_B` and
`label_C` columns of the corresponding CSV with a standard implementation,
restricting to the `section` named in the table.

```bash
# abstention parser precision / recall against the gold labels (point estimates 0.942 / 0.987)
python3 audit/parser_abstain_audit_prep.py score --filled 'human_audit/parser_majority/*.csv'
# the intervals reported in the paper (column B′: precision [0.921, 0.948], recall [0.897, 0.994])
python3 audit/parser_audit_followup.py ci
# template audit agreement, with and without the three relabels
bash scripts/camera_ready/run_template_audit.sh /tmp/template_audit
```

`audit/parser_fp_yield_impact.py` reads `parser_majority/` to bound the effect
of the 82 identified parser false positives on the yield gap
(`results/camera_ready/canonical_ci/parser_fp_impact.txt`).
