# PVSD: Privilege Vector Self-Distillation

On-policy self-distillation with privileged information has a structural problem: the
privileged reference sits in the teacher's context from layer 1 onward, so the teacher
distribution is contaminated before anything can be filtered. Every fix in this
lineage — OPSD, RLSD, CEPO, TRACE, HDPO, AVSD — intervenes on the teacher's *output
distribution*, downstream of where the leakage physically enters the forward pass.

**PVSD moves the intervention upstream.** The privileged reference is read out of a
small set of causally-identified attention heads as a single residual-stream vector,
purified against matched corrupted contexts, fused across views, and injected into the
student's own hidden state. The teacher is the *steered student*: privileged text never
appears in the context of the pass that produces the teacher distribution.

```
rollout y ~ P_student(.|x)                          on-policy, vLLM
  every N steps:  A^(m) = TopK(PIE^(m))             Stage 1: causal localisation
  v_r^(m)     = sum_{(l,j) in A^(m)} W_O[:,j] a_lj(last(x, r^(m)))
  v_transfer  = v_r^(m) - mean_k v_{r~_k}^(m)       per-example contrastive purification
  v*          = (1/M) sum_m v_transfer^(m)          activation-space fusion
  q*          = sg[ P_theta(. | h_l* + alpha v*) ]  the steered teacher
  loss        = D_KL( p_student || q* )             reverse KL
```

This repository implements PVSD on top of the AVSD codebase, which it uses as both the
infrastructure and the primary baseline. AVSD's own code paths are unchanged.

---

## Setup

```bash
conda env create -f environment.yml
conda activate pvsd
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

On the shared EC2 node, `scripts/remote/setup_env.sh` does all four steps (plus
miniconda itself) and is idempotent — see [Running on the remote runner](#running-on-the-remote-runner).

Nothing large is ever written inside this folder. `scripts/remote/paths.sh` puts
checkpoints, results and the HuggingFace cache under `~/outputs` instead:

| Path | Contents | Override |
|---|---|---|
| `~/outputs/checkpoints/` | LoRA checkpoints | `PVSD_CKPT_ROOT` |
| `~/outputs/results/` | eval and probe JSON | `PVSD_RESULTS_ROOT` |
| `~/outputs/hf_cache/` | models and datasets pulled from the Hub | `HF_HOME` |

## Running on the remote runner

Jobs are driven by `commands.sh` at the repository root: push to GitHub, a monitor
pulls the repo onto the EC2 node (Ubuntu, 8xH200) and runs that file inside a screen
session. Line 1 is the mode, line 2 the job name, the rest is the job.

```bash
#1 +1800+500     # sync and run, wait 1800s, then pull the last 500 log lines
#pvsd-smoke
...
```

The checked-in `commands.sh` bootstraps the environment (`scripts/remote/setup_env.sh`
— miniconda, the `pvsd` env, `pip install -e .`, flash-attn; a no-op on later runs),
kills any leftover job, and then runs the plumbing check. The other jobs are ready to
uncomment one at a time: probe, PVSD training, AVSD baseline, ablations, evaluation,
tests.

Fetching results is a separate job — the runner never pushes anything out:

```bash
#2 -f-~/outputs/results/ +a          # pull every eval / probe report
#pvsd-results

#2 +a                                # pull the full log of the last job
#pvsd-log

#3                                   # list available logs
#pvsd-logs
```

Three runner rules the repository already keeps:

* nothing large is written into the code folder — everything goes to `~/outputs`;
* models and datasets are downloaded on the instance, never committed (the repo is
  ~2MB against a 25MB limit);
* results leave the instance only through a `#2` job.

Two that are on you: **never commit the runner's instruction file** — `GUIDE.md` lives
outside this repository and is in `.gitignore` as well, and pushing it can get the
account banned — and never commit `HF_TOKEN`; hand it over separately so it can be
exported on the instance.

---

## Run the full test suite

```bash
python -m pytest
```

---

## Running PVSD

```bash
bash scripts/math/train_pvsd_qwen3_4b.sh
```

Qwen3-4B on OpenThoughts math, 500 steps, LoRA r=64, 4 processes (DeepSpeed ZeRO-2),
vLLM colocate for rollouts. Method settings: `l* = L/4 = 9`, `alpha = 1.0`, 3 views,
`K_c = 2` corrupted contexts, PIE top-10 heads per view refreshed every 100 steps,
reverse KL (`--beta 1.0`) with no token clipping.

Everything is overridable from the environment, and extra flags pass through:

```bash
RUN_CONFIG=qwen3_4b_pvsd_a2 PVSD_ALPHA=2.0 bash scripts/math/train_pvsd_qwen3_4b.sh
MODEL_NAME_OR_PATH=Qwen/Qwen3-8B OUTPUT_DIR=outputs/pvsd/qwen3_8b \
  RUN_CONFIG=qwen3_8b_pvsd bash scripts/math/train_pvsd_qwen3_4b.sh
bash scripts/math/train_pvsd_qwen3_4b.sh --pvsd_top_k_heads 20
```

Two constraints the script enforces for you: `per_device_train_batch_size >=
pvsd_num_corrupt + 1` (corrupted contexts come from other examples of the same device
batch), and `dataloader_drop_last`, so a trailing batch of one cannot break a run.

On an 8-GPU node, `NUM_PROCESSES=8 GRADIENT_ACCUMULATION_STEPS=1` uses the whole
machine at an unchanged effective batch size.

While it trains, the metrics that matter are not the loss:

| Metric | Read it as |
|---|---|
| `pvsd/steer_advantage` | `log q*(y_t) − log p(y_t)` on the rollout; must be > 0 |
| `pvsd/steer_advantage_positive_frac` | fraction of tokens where steering helps |
| `pvsd/<view>/cos_raw_corrupt` | ≈1 means the read-out is almost pure template |
| `pvsd/<view>/transfer_ratio` | how much of the raw vector survives purification |
| `pvsd/<view>/corrupt_len_delta` | how well the corrupted context matches in length |
| `pvsd/head_jaccard/<a>__<b>` | cross-view head overlap (30–70% motivates fusion) |
| `pvsd/fused_norm` | scale of the injected signal (interacts with `alpha`) |

If `pvsd/steer_advantage` sits at zero, the teacher is indistinguishable from the
student and the loss carries no information no matter how it looks.

### Evaluate a checkpoint

```bash
CHECKPOINTS=~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd/checkpoint-500 \
  bash scripts/math/eval_math.sh
```

Avg@8 on AIME 2024, AIME 2025 and HMMT 2025, then one summary table across every run
found under `results/math/`. Pass several checkpoints at once to compare arms:

```bash
CHECKPOINTS="~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd_main/checkpoint-500
             ~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd_no_purification/checkpoint-500" \
  bash scripts/math/eval_math.sh
```

Already-computed dataset/checkpoint pairs are skipped unless `OVERWRITE=1`.

---

## Sanity checks before a long run

Both take minutes on one GPU, and each rules out a different way of wasting GPU-days.

### 1. Plumbing

```bash
bash scripts/math/smoke_pvsd.sh
```

3 optimizer steps with short rollouts and a cut-down PIE. It verifies that trl,
DeepSpeed, vLLM, LoRA and the PVSD hooks compose, that the model topology is detected
correctly (`head_dim=128` for Qwen3-4B, **not** `hidden_size // n_heads`), and that the
steered teacher actually differs from the student. The script prints exactly what to
look for and the three symptoms that mean it does not work.

### 2. Does the vector carry content?

```bash
bash scripts/math/probe_pvsd.sh
```

The decisive pre-training experiment. For held-out problems it steers with

* `correct` — the purified vector from the reference that belongs to the problem,
* `mismatched` — another problem's reference, purified the same way (same format, same
  construction, wrong content),
* `random` — a random direction of the same norm,

and reports the change in `log P(gold solution)` across the three candidate injection
layers (`L/4, L/3, L/2`) plus a no-PIE control. It prints one table with a verdict per
row:

| Outcome | Meaning |
|---|---|
| `correct` >> `mismatched` ≈ `random` ≈ 0 | the vector transports content — proceed |
| `correct` ≈ `mismatched` > 0 | the vector transports the view *template* — purification is not isolating content |
| every arm ≈ 0 | no transportable trace at any of these layers |

This is Experiment 2 of `pvsd.md` §9, and it is also the honest re-test of the pilot in
`pvsd_motivating_observation.ipynb`: that pilot used the crude `h_teacher − h_student`
difference and gave a weak result, which is *not* informative about the PIE-localised,
purified construction. This probe is.

---

## Ablations

```bash
DRY_RUN=1 bash scripts/math/ablate_pvsd.sh          # print the plan
ONLY=main,no_purification,template_only bash scripts/math/ablate_pvsd.sh
bash scripts/math/ablate_pvsd.sh                     # all 14 arms, sequentially
```

| Arm | What it isolates |
|---|---|
| `main` | the method as specified |
| `no_purification` | steering without template subtraction |
| `template_only` | the discarded component alone — should hurt |
| `single_view` | multi-view fusion vs one view |
| `frozen_calibration` | online PIE vs one-time calibration (staleness) |
| `no_pie` | causal localisation vs all heads of the layer |
| `layer_third`, `layer_half` | injection depth |
| `alpha_0p5`, `alpha_2p0` | steering strength |
| `heads_5`, `heads_20` | head budget |
| `corrupt_cycle` | corruption donor choice vs length matching |
| `jsd_beta0p5` | JSD instead of reverse KL |

The purification arms still extract the corrupted contexts, so they are cost-matched
with the method and differ only in the injected signal. They also decompose exactly:
`contrast = none − template_only`.

Entropy-weighted fusion is deliberately **not** implemented: `fuse_view_vectors`
accepts weights, but the weighting rule is not defined in `pvsd.md` or the paper, so
nothing was invented. Uniform `w = 1/M` is what runs.

### AVSD baseline

```bash
bash scripts/math/train_avsd_qwen3_8b.sh
bash scripts/math/train_avsd_deepseek_r1_distill_qwen_7b.sh
```

`avsd.math.train` without `--pvsd_enable` is exactly the previous AVSD behaviour. Only
two hooks were added to `AVSDTrainer` (nine lines), so baseline numbers stay
comparable. AVSD's reported Qwen3-4B Avg@8 figures — Base 55.0, SFT 47.6, GRPO 56.6,
OPSD 58.2, AVSD 59.9 — are what the PVSD table should sit next to; `eval_math.sh`
prints them under every summary.

---

## Repository structure

```text
.
├── PVSD.md                              implementation reference: file map, metrics, tests
├── pvsd.md                              the full method proposal
├── PVSD_IMPLEMENTATION.md               superseded design (kept for history)
├── commands.sh                          remote-runner entry point (required at root)
├── configs/accelerate.yaml              DeepSpeed ZeRO-2, bf16, 4 processes
├── scripts/remote/
│   ├── setup_env.sh                     miniconda + conda env + install, idempotent
│   └── paths.sh                         keeps every output under ~/outputs
├── scripts/math/
│   ├── smoke_pvsd.sh                    3-step plumbing check
│   ├── probe_pvsd.sh                    does the vector carry content?
│   ├── pvsd_vector_probe.py             the probe itself
│   ├── train_pvsd_qwen3_4b.sh           PVSD training entry point
│   ├── ablate_pvsd.sh                   the ablation matrix
│   ├── eval_math.sh                     Avg@8 + summary table
│   └── pvsd_decisive_position_audit.py  standalone diagnostic (not in the training path)
├── scripts/code/                        AVSD code-domain experiments (Codeforces)
├── src/pvsd/
│   ├── common/privilege_vectors.py      head read-out, purification, fusion, injection
│   ├── common/pie.py                    Stage 1: PIE causal mediation
│   ├── common/token_log_probs.py        memory-bounded log-prob read-out
│   ├── math/pvsd_trainer.py             PVSDTrainer
│   ├── math/data_collator.py            privileged + corrupted prompt sets
│   └── math/{train,trainer,evaluate}.py AVSD infrastructure (baseline, unchanged)
└── tests/                               CPU tests, no GPU, no network
```

`PVSD.md` is the reference for *what runs where* — the method-to-code correspondence,
the full metric dictionary, the test inventory, and notes on which older documents in
this repository have been superseded.

---

## Acknowledgements

Built on the AVSD codebase (which itself builds on the OPSD implementation at
<https://github.com/siyan-zhao/OPSD>), and on the Function Vectors line of work for the
head read-out and causal-mediation machinery.

> Note for anonymous submission: this README is method-first and carries no author
> names. `PVSD_IMPLEMENTATION.md` and the AVSD citation history do reference the
> upstream repository and its authors — strip or redact those before submitting this
> directory as supplementary material.
