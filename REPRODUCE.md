# REPRODUCE

Commands that regenerate the numbers in the paper. All paths are relative to the
repository root and assume `data/` holds the dataset repository (see the
README). **Everything except §8 is computed from released files, so none of it
incurs API cost or needs a GPU.**

`{model}` ∈ `aya` `exaone` `llama` `qwen` `sonnet` `gemini`

## Everything at once

```bash
bash tools/verify_baseline.sh /tmp/k-crosshop-out
```

This runs every command in §1–§6 and every added analysis in §9, writes the
outputs to the given directory, and compares each one byte for byte with the
copy in `results/`. It prints `OK` or a diff per output and takes about 90
minutes on CPU. `CR=0 bash tools/verify_baseline.sh …` runs §1–§6 only (about
17 minutes).

Three of the added analyses read `data/corpus/` (all-hop screening, path
exclusion, E5 truncation), and the E5-truncation analysis downloads the
`intfloat/multilingual-e5-large` tokenizer from the Hugging Face Hub.

---

## 0. Preflight

```bash
python3 -m pytest tests/ -q
```

## 1. Core metrics — FaithRate / LSHD / CLEPR / AFR / PDR

In the paper's terms: faithfulness, the faithfulness gap Δf and the yield
gap Δy, the error-persistence rate, the abstention rate, and the parametric
fallback rate. The name mapping is in the README; the released `ecological`
files are the paper's Controlled condition.

```bash
python3 analysis/07_analyze.py \
    --annotated data/annotated/{model}_ecological.jsonl \
    --qa-pairs  data/dataset/qa_pairs.jsonl
```

For Sonnet and Gemini, compare against `results/07_{model}_ecological.txt`.

## 2. Two-stage QMC decomposition

QMC is a **chained comparison**: stage 1 is the template-language component,
stage 2 the bridge-surface component.

```bash
# stage 1: ecological vs query_matched
python3 analysis/11_ablation_compare.py \
    --main-annotated     data/annotated/{model}_ecological.jsonl \
    --ablation-annotated data/annotated/{model}_query_matched.jsonl \
    --qa-pairs data/dataset/qa_pairs.jsonl

# stage 2: query_matched vs normalized
python3 analysis/11_ablation_compare.py \
    --main-annotated     data/annotated/{model}_query_matched.jsonl \
    --ablation-annotated data/annotated/{model}_normalized.jsonl \
    --qa-pairs data/dataset/qa_pairs.jsonl
```

The paper's decomposition intervals come from a joint bootstrap in which the
two components and the residual add up to the total in every draw, resampling
the 250 entity paths. `--axis yield` gives the yield-axis decomposition and
`--axis faith` the faithfulness-axis one:

```bash
python3 analysis/yield_qmc_joint_ci.py \
    --controlled    data/annotated/{model}_ecological.jsonl \
    --query-matched data/annotated/{model}_query_matched.jsonl \
    --normalized    data/annotated/{model}_normalized.jsonl \
    --qa-pairs      data/dataset/qa_pairs.jsonl \
    --cluster-key path_cluster_id --n-boot 10000 --seed 42 --axis yield
```

All six models on both axes are in
`results/camera_ready/canonical_ci/qmc_{yield,faith}.txt`.

## 3. Retrieval-conditioned LSHD (the paper's retrieval-conditioned faithfulness gap)

```bash
python3 analysis/12_conditioned_analysis.py \
    --annotated       data/annotated/{model}_ecological.jsonl \
    --retrieval-hits  data/retrieval/{model}_ecological_hits.jsonl
```

## 4. Bootstrap 95% CI

Percentile bootstrap, B = 10,000, seed = 42. **The paper resamples the 250
entity paths**: each draw takes all four evidence-language patterns of a path
together.

```bash
python3 analysis/13_bootstrap_ci.py \
    --annotated      data/annotated/{model}_ecological.jsonl \
    --qa-pairs       data/dataset/qa_pairs.jsonl \
    --retrieval-hits data/retrieval/{model}_ecological_hits.jsonl \
    --trajectories   data/trajectories/{model}_ecological.jsonl \
    --alias-cache    data/retrieval/alias_cache.json \
    --cluster-bootstrap --cluster-key path_cluster_id
```

The output equals `results/camera_ready/canonical_ci/13_{model}.txt`. Without
the last two flags the script resamples questions instead
(`results/13_{model}_ecological_ci.txt`); `--cluster-bootstrap` alone
resamples the 249 clusters that share a bridge entity.

> **`--retrieval-hits` must be a file that contains hop1.** Every released hit
> file has 2,000 rows covering hop0 and hop1. Passing a hop0-only file empties
> §6 of the output entirely and changes the reference set of its §8.
>
> The intervals depend on the global RNG stream and on the order in which the
> sections run. Modifying the script or skipping a section changes every CI
> after §1 relative to the published values.

## 5. Cross-model comparison / behavioral regime

```bash
python3 analysis/14_model_comparison.py --qa-pairs data/dataset/qa_pairs.jsonl --annotated \
    data/annotated/aya_ecological.jsonl    data/annotated/exaone_ecological.jsonl \
    data/annotated/llama_ecological.jsonl  data/annotated/qwen_ecological.jsonl \
    data/annotated/sonnet_ecological.jsonl data/annotated/gemini_ecological.jsonl
```

## 6. PDR (the paper's parametric fallback rate) + pooled ΔCLEPR

```bash
python3 analysis/16_pdr_analysis.py \
    --trajectories   data/trajectories/{model}_ecological.jsonl \
    --retrieval-hits data/retrieval/{model}_ecological_hits.jsonl \
    --alias-cache    data/retrieval/alias_cache.json \
    --annotated      data/annotated/{model}_ecological.jsonl

python3 analysis/pooled_clepr_meta.py --qa-pairs data/dataset/qa_pairs.jsonl \
    --cluster-bootstrap --cluster-key path_cluster_id \
    --annotated data/annotated/{all six models}_ecological.jsonl …
```

The pooled ΔCLEPR with path resampling is
`results/camera_ready/canonical_ci/pooled_clepr.txt` (all six models) and
`pooled_clepr_8b.txt` (the four open-weight models). The confidence interval
of the parametric-fallback difference is in §8 of the §4 output.

## 7. Recomputing answer-bearing hits (optional — calls the Wikidata API)

The decisions are already in the release
(`data/retrieval/{model}_ecological_hits.jsonl`). To rebuild one model's file,
run 08 → 09 → 10 in order. Without arguments the three scripts rebuild Aya's
file; for another model, pass its files:

```bash
m=llama   # aya / exaone / llama / qwen / sonnet / gemini
python3 analysis/08_retrieval_quality.py --model $m \
    --annotated data/annotated/${m}_ecological.jsonl \
    --trajectories data/trajectories/${m}_ecological.jsonl        # exact match, hop 0
python3 analysis/09_alias_retrieval.py \
    --hop0-path data/retrieval/hop0_retrieval_$m.jsonl \
    --trajectories data/trajectories/${m}_ecological.jsonl        # + Wikidata aliases
python3 analysis/10_retrieval_proxy.py \
    --trajectories data/trajectories/${m}_ecological.jsonl \
    --v2-path data/retrieval/hop0_retrieval_$m.jsonl \
    --output data/retrieval/hits_rebuilt_$m.jsonl                 # + surface-normalized retry, hops 0 and 1
```

The rebuilt `answer_bearing_hit` equals the shipped file for all 2,000
(path, hop) pairs of every model. With the shipped alias cache
(`data/retrieval/alias_cache.json`) no Wikidata call is made. If you delete the
cache, 09 fetches aliases from the Wikidata API (set `WIKI_USER_AGENT` first),
and aliases added to Wikidata since our snapshot can change a few decisions.

## 8. Regenerating trajectories from scratch (needs a GPU and API keys)

The passage corpus is `data/corpus/passages.jsonl`. Rebuild the indices, then
generate and judge:

```bash
python3 pipeline/04_build_faiss.py                        # BGE-M3 indices → data/faiss_index (GPU)
python3 pipeline/05_pilot_runner.py --retrieval-mode dynamic_subq \
    --qa-pairs data/dataset/qa_pairs.jsonl --model {…} --output traj.jsonl
python3 pipeline/06_annotate.py --trajectories traj.jsonl --output ann.jsonl
```

The added runs have their own scripts in `scripts/camera_ready/`:
`build_e5_index.sh` (the E5 index, GPU), `analysis/retriever_robustness.py`
(the E5 retriever appendix — Table 15 — whose outputs are in
`results/camera_ready/retriever_robustness/answer_bearing/`; GPU), `run_local.sh` (open-weight
generation through Ollama, GPU), `run_frontier.sh` (frontier generation
through batch APIs, billed), `run_judge.sh` (judging, billed) and
`run_F_screening.sh` (the all-hop answerability screen, billed).

Generation settings are in `config.yaml`. **The maximum output tokens differ
between the open-weight and frontier models — 128 against 256** — because
frontier responses are longer and get truncated at 128; every other decoding
setting and the prompts themselves are identical. This is stated in the paper's
limitations.

## 9. Added analyses

Each script writes to `results/camera_ready/` by default and takes an output
directory as its first argument. All run on CPU from released files.

| Script (`scripts/camera_ready/`) | Output (`results/camera_ready/`) | What it reproduces |
|-|-|-|
| `run_canonical_ci.sh` | `canonical_ci/` | Every bootstrap interval in the paper with path resampling: §4 per model, the QMC decomposition on both axes, pooled ΔCLEPR, the parser false-positive impact, and the surface-normalization audit |
| `run_b_qmc.sh` | `qwen_ladder/` | Qwen3 model-size ladder (1.7B, 4B, 8B, 14B) and precision (8B fp16 vs Q4_K_M) |
| `run_e5_qmc.sh` | `e5/` | The decomposition with the E5 retriever (Llama, Sonnet) beside BGE-M3 |
| `run_2x2.sh` | `2x2/` | Template × bridge 2×2, adding the bridge-only condition |
| `run_bridge_type_qmc.sh` | `bridge_type/` | The yield decomposition by bridge-entity type, the type checked against each bridge entity's Wikidata P31 (shipped cache, no network), and for human bridges by hop1 relation and with the educated-at (P69) paths excluded |
| `run_screening_sensitivity.sh` | `screening/` | Excluding paths with a hop that fails the all-hop answerability screen (needs the corpus) |
| `run_exclusion_sensitivity.sh` | `exclusion/` | Excluding the 44 paths that use a scope-mismatched template, and the screening failures, each compared with excluding the same number of random paths (needs the corpus) |
| `run_qualitative.sh` | `qualitative/` | Error-type rates and example cases per model |
| `run_self_translate.sh` | `self_translate/` | The self-translation condition |
| `run_e5_truncation.sh` | `retriever_robustness/e5_truncation/` | How many passages E5's 512-token limit truncates, and whether evidence is cut (needs the corpus) |
| `run_gold_entity_ub.sh` | `retriever_robustness/gold_entity_ub/` | The gold-entity upper bound against controlled hit@5, and the BGE-M3 against E5 upper-bound comparison (per-entity inputs were computed on a GPU and are shipped) |
| `run_matched.sh` | `matched/` | Matched-pair analysis: hop0 and hop1 Δf/Δy on entity paths where both sides retrieved answer-bearing evidence, its comparison across the three conditions, and the retrieval-quality covariate |
| `run_judge_leniency.sh` | `judge_leniency/` | Judge leniency: among hops the judge scored faithful but the answer matcher scored wrong, the share whose answer names a different entity (§2 of the output gives the per-model rates) |
| `run_construction_audit.sh` | `construction/` | Dataset construction: gate pass rates, stage counts, the manual audit sample, the 410 and 263 path lists, and the cap checks |
| `run_template_audit.sh` | `template_audit/` | Agreement of the question-template audit |
| `run_correctness.sh` | `correctness/` | Answer accuracy by the answer matcher: the same/cross accuracy gap, the decomposition on the accuracy axis beside the yield axis, final-answer (hop1) accuracy, two-hop trajectory success by pattern, and faithful × correct (`tables.md` collects the tables) |
| `run_hop_qmc.sh` | `hop_qmc/` | The decomposition at each hop position separately (hop0, hop1), on both axes |

---

## Known reproduction limits

* **No regeneration snapshot survives for the open-weight `query_matched` and
  `normalized` trajectories.** The QMC of §2 reproduces from the released
  annotation files only; rerunning from 05 will not give the same numbers,
  because decoding is not bit-reproducible across environments.
* The two frontier models were called through undated API aliases, so
  regenerated frontier outputs can differ if an alias has since moved.
* The abstention parser retains 12 hops of `unknown` false positives
  ("Unknown Worlds Entertainment" matches the English abstention pattern). The
  effect on AFR (the abstention rate) is under 0.1 percentage points and it
  ships unfixed.
* The LLM judge (the annotation step) is non-deterministic on rerun. The
  released annotations are the run used in the paper.

---

## Appendix — which command produced which file in `results/`

The 27 text files at the top of `results/` are the outputs of §1–§6 as
written there, with question-level resampling in 13 and the pooled ΔCLEPR.
The added analyses are in `results/camera_ready/` (§9).

| File | Command |
|-|-|
| `07_{sonnet,gemini}_ecological.txt` | §1 |
| `11_{model}_qmc_stage1.txt` / `_stage2.txt` (6 models) | §2 |
| `12_{sonnet,gemini}_conditioned.txt` | §3 |
| `13_{model}_ecological_ci.txt` (6 models) | §4 without the cluster flags |
| `14_all6_comparison.txt` / `14_8b_vs_sonnet.txt` | §5 |
| `16_{sonnet,gemini}_pdr.txt` / `16_pooled_clepr_meta_all6.txt` | §6 without the cluster flags |
