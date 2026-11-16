# DATASHEET — K-CrossHop

## 1. What the data measures

Whether a **cross-index hop** — where a Korean query has to search the English
index — produces less faithful answers than a **same-index hop** searching the
index in its own language, and how far that loss propagates into the next hop.
The query is always Korean, so **what is manipulated is the language of the
evidence, not the language of the query.**

## 2. Provenance and construction

| Stage | Description |
|-|-|
| Path collection | Two-hop (S)→(B)→(A) paths extracted from Wikidata via SPARQL, organized by relation schema |
| Passages | Korean and English Wikipedia articles split into paragraph-level chunks |
| Lexical gate | Only paths whose gold entity aliases actually occur in the evidence passages pass |
| Oracle gate | An LLM must be able to solve the hop from that evidence alone. Only triples that pass **all four conditions** (the 2×2 of hop0 and hop1 evidence languages) are kept — this is what makes the balanced design possible |
| Question generation | Korean and English question templates are instantiated per relation schema (dump them all with `dataset/oracle_gate_lib.py --dump-templates`) |

**Provenance of the question templates**: the Korean and English templates are
not translations of one another. They are constant dictionaries authored in
parallel in each language; only the instantiation is performed by deterministic
code. The command above prints the full set.

**Intermediate construction records**: the dataset repository's `construction/`
holds the oracle-gate decisions for all 693 candidate paths under the four
evidence-language patterns, the 410 paths left after single-answer cleanup,
the recorded cap outcome, the 62 paths of the manual audit as reviewed, and
the answer-type labels the group cap uses. `audit/construction_audit.py`
recomputes from them the gate pass rates per pattern, the stage counts per
collection round, the manual-audit tallies and the 410 and 263 identifier
lists, and checks the recorded cap outcome against the cap rules;
`dataset/apply_caps.py` is an order-independent version of the cap step. The
recorded 263 came from a run that depended on set iteration order, so the
order-independent version returns a different set (266 paths) under the same
rules. Outputs are in `results/camera_ready/construction/`
(`bash scripts/camera_ready/run_construction_audit.sh` regenerates them).

## 3. Composition

* **250** entity paths (triples), each crossed with the 4 evidence-language
  patterns → **1,000** questions
* Exactly **250** per pattern (a: KO→KO→KO / b: KO→EN→KO / c: KO→KO→EN /
  d: KO→EN→EN)
* **2,000** hops
* 250 `path_cluster_id` values = the matched 2×2 groups; **249**
  `bridge_cluster_id` values = groups sharing a hop0 answer. The latter is used
  for the cluster-bootstrap robustness check
* 6 models × 3 protocols = 18 trajectory files and 18 annotation files
  (1,000 rows each)

| Model | Identifier | Class |
|-|-|-|
| aya | `aya-expanse:8b` | open 8B |
| exaone | `exaone3.5` | open 8B |
| llama | `llama3.1:8b` | open 8B |
| qwen | `qwen3:8b` | open 8B |
| sonnet | `claude-sonnet-4-6` | frontier |
| gemini | `gemini-3.5-flash` | frontier |

**Excluded by design**: a separate set expanding one bridge entity into several
questions was built, but sharing a bridge breaks independence across questions,
so it is excluded from both the paper's analysis and this release. The release
contains only the balanced 2×2 core of 1,000.

## 4. Annotation

* Judge: `gpt-5.4-mini`
* The `답변:` ("Answer:") section of `generated_text` is split into
  sentence-level claims, and each claim is checked against the passages that
  hop retrieved, then labelled `supported` / `unsupported` / `contradicted`
* One API call per hop (claims × passages batched)
* Abstention decisions treat explicit abstention markers and the markerless
  fallback under the **same criterion**
  (`pipeline/normalize_abstention_labels.py`)

## 5. Human audits

`human_audit/` contains human verification of the LLM judge and of the automatic
decisions. The three annotators are anonymized as Annotator A / B / C.

| Audit | Target | Completed items | Fleiss κ |
|-|-|-|-|
| IAA | The judge's claim labels | 121 | main 0.531 / calibration 0.582 |
| Abstention parser | Precision/recall of the abstention classifier | 80 | calibration 0.858 |
| Retrieval proxy | The answer-bearing hit decision | 106 | main 0.498 / calibration 0.400 |

"Completed items" counts every item labelled by all three annotators, across
both the `main` and `calibration` strata; the paper reports statistics on the
`main` stratum (101 IAA claims from 50 trajectories, 90 proxy instances). For
the abstention parser, agreement is computed on the 80 calibration items; the
remaining sampled rows in that file are partially labelled. Each
`*_score.json` records `n_complete` and `n_dropped_partial`.

The label definitions and rubrics are in `human_audit/instruments/`.
`human_audit/README.md` lists every file, including the question-template
audit, and how to recompute the statistics.

## 6. answer-bearing hit@5 (retrieval proxy)

Whether the answer entity for that hop **actually appears** in the top-5
retrieved passages, decided in three stages:

1. exact — string match on the gold surface form
2. alias — expanded with Wikidata aliases and sitelinks
3. surface-normalized — retried after surface-form normalization

The files in `data/retrieval/` hold the final decision after all three stages,
with a field recording which stage matched. This is a **proxy**: the entity can
appear without the passage stating the relation the question asks about. The
proxy audit in §5 is a direct human measurement of that error.

## 7. Known limitations

1. **12 hops of abstention-parser false positives** — proper nouns such as
   "Unknown Worlds Entertainment" match the `unknown` token in the English
   abstention pattern. The effect on AFR (the paper's abstention rate) is under 0.1 percentage
   points and it ships unfixed.
2. **Asymmetric maximum output tokens** — 128 for the open-weight models, 256
   for the frontier models, because frontier responses are longer and get
   truncated at 128. Every other decoding setting and the prompts are identical.
3. **The low hallucination rate of the two frontier models is entangled with
   their tendency to abstain.** FaithRate (the paper's faithfulness), which
   drops abstentions from the
   denominator, therefore overstates them; the yield-axis metrics, which count
   an abstention as zero, are reported alongside it
   (`analysis/yield_decomposition.py`).
4. **The three query conditions are not interchangeable.** The released
   suffixes `ecological` / `query_matched` / `normalized` are the paper's
   Controlled, template-matched, and surface-normalized conditions; every
   headline gap is computed on `ecological`, and the ablations are meaningful
   only as successive differences from it, not as standalone estimates.
5. **No regeneration snapshot survives for the open-weight `query_matched` and
   `normalized` trajectories.** They reproduce from the released annotation
   files only.
6. The corpus is a Wikipedia snapshot and will diverge from later edits to the
   articles.

## 8. Licensing

| Artifact | Licence |
|-|-|
| Code | MIT |
| Wikipedia passage text (`corpus/passages.jsonl` and the copies embedded in the trajectories) | **CC BY-SA 4.0** (derived from Wikipedia — required, not chosen) |
| `dataset/triples.jsonl`, Wikidata paths | CC0 1.0 |
| Questions, human audit labels, derived metric outputs | CC BY-SA 4.0 |
| Stored model outputs in `trajectories/` and `annotated/` | Terms of the model that produced them — see the dataset repository's `README.md` |

The passage corpus is in `corpus/` of the dataset repository. The FAISS indices
(~2.6 GB) are not distributed; `pipeline/04_build_faiss.py` rebuilds them from
the corpus.
