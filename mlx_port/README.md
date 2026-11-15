# MLX A1 port

Run commands from the workspace root with `.venv/bin/python -m mlx_port.…`.
Start with `scripts/setup.sh` and `scripts/check.sh`; see the [root README](../README.md)
for fresh-Mac setup, the pinned reference submodule and optional parity extra.
Training does not import the course reference or PyTorch. Parity tools use the
shared lookup in `mlx_port.paths` (environment override, submodule, legacy sibling).

## Correctness

Install `uv sync --locked --extra parity` first. Tests use normal MLX's default
device. Set `MLX_DEVICE=cpu` to request CPU execution on a Mac with an accessible
Metal runtime; a sandbox without Metal may still fail at MLX import.

```sh
.venv/bin/python -m mlx_port.parity_check
```

This loads actual initialized PyTorch reference weights, checks default d8
logits and loss, checks mask/position semantics and weight-decay membership,
then compares 20 steps of the reference AdamW/scheduler with the port on a fixed
batch. Tests use short sequences. The tiny architecture also exercises
compilation, checkpointing, chunked cross-entropy and both mixed precision paths.
Results are written to `results/mlx_parity.json`.

## Coordinator GPU commands (outside sandbox)

These commands use the normal pip MLX installation.

```sh
.venv/bin/python -m mlx_port.bench --device gpu --smoke --output results/mlx.jsonl
```

Smoke runs a tiny architecture and 512-square matmuls with a 55-second total
worker deadline. Its numbers are explicitly ineligible for d8 projections.
Then measure the actual recipe and roofline:

```sh
.venv/bin/python -m mlx_port.bench --device gpu \
  --precisions fp32,mp,bf16 --micro-batches 1,2,4,8 \
  --compiled true --chunk-sizes 0,1024 --boundaries micro \
  --output results/mlx.jsonl
```

This is 24 d8 configurations, plus 4096-square fp32 and bf16 matmul rooflines.
Refine around the best micro-batch/precision/chunk size:

```sh
.venv/bin/python -m mlx_port.bench --device gpu \
  --precisions mp,bf16 --micro-batches 2,4 \
  --compiled true,false --chunk-sizes 0,1024 --boundaries micro,step \
  --checkpoints false,true --output results/mlx_refine.jsonl
```

Each configuration has its own child process, at least five warmup steps and
20 timed optimizer steps, and a configurable 1,200-second timeout. OOM, native
crashes, unavailable Metal, and timeouts produce JSON rows and the sweep
continues. Output files append. `--device mps` aliases `gpu`.

Every non-smoke optimizer step processes **64 × 1024** input tokens. AdamW is
applied once after averaging micro-gradients, with global norm clipping at 1.0.
The two compiled functions cover the differentiated micro-step and update.
`--boundaries micro` evaluates accumulated gradients after each micro-step;
`step` defers materialization until the optimizer update (potentially much more
memory). Both always evaluate model weights and optimizer state before timing
the next step. Chunking projects the language-model head by token chunks and
checkpoints those chunks, avoiding a retained full vocabulary-logit tensor.

All modes keep **fp32 master parameters, gradients and Adam moments**. `fp32`
uses fp32 throughout; `mp` uses bf16 linear/attention operations and fp32
embeddings, normalization scales and residual streams; `bf16` uses bf16
activation/residual/normalization scale computations. All losses are fp32.
These last two are explicitly different compute policies, to be selected by
measured speed and numerical stability. Neither stores trainable weights in
bf16 or applies tiny updates to bf16 master weights.

## Full training

```sh
.venv/bin/python -m mlx_port.train_mlx --device gpu --precision mp \
  --micro-batch 2 --chunk-size 1024 --compiled --boundary micro \
  --tokens path/in/workspace/train.npy --val-tokens path/in/workspace/val.npy
```

The arrays must contain the reference seed-42 pre-shuffled dataset: at least
600,000 train sequences and exactly 1,000 validation sequences, context 1024.
This command is a conservative candidate, **not a measured fastest setting**.
It runs 9,375 updates, evaluates every 94 updates and on the final update,
and optionally saves final fp32 weights with `--save-weights`. This legacy
array runner does not download data or implement W&B,
module-RMS logging, periodic checkpoint/resume, or match PyTorch's RNG stream.
For identical initial weights use `Model.load_reference` rather than the
distribution-matched MLX initializer.

## Timing interpretation

Training timings use resident synthetic integer tokens and real model forward,
backward, clipping and AdamW updates. They exclude data loading, logging and
checkpoint I/O. Evaluation overhead is estimated from measured forward times
for a full micro-batch and any final partial micro-batch, scaled to 1,000 rows
and 100 passes. Initial compilation and startup are excluded from the projected
steady-state total. The benchmark records both MLX peak allocation and process
peak RSS (bytes on macOS); the former excludes cached buffers and graph objects.
The memory limit is advisory; the wired limit bounds pinned memory, and the
process timeout bounds slow configurations.

Both `bench` and `run` default to an allocation/wired cap of 75% of
`mx.device_info()['max_recommended_working_set_size']`; `--memory-gb N` explicitly
overrides it in GiB. `queue` and `sweep` forward the same option to each worker.
The effective byte limit is recorded, and no fixed Mac memory size is assumed.

FLOPs per update use the stated approximation
`6*N*B*S + 12*L*B*S*S*d`, including both embedding matrices in `N` and dense
attention forward plus backward. For d8 this is 16.9088 TFLOP/update. This
conventional dense-attention estimate counts masked attention work and
overestimates embedding work; its utilization is not a hardware-counter result.
MFU divides model FLOP/s by the measured 4096-square matmul rate in the
corresponding arithmetic precision. It is relative to an achieved GEMM rate,
not vendor peak silicon TFLOPS. A checkpointed run includes extra recomputation
time but the model FLOP numerator is unchanged.

## Real DCLM experiments (T3)

The `run` command defaults to the explicit **local** preset: d4 (5,901,056
parameters), 20,480 sequences / 20.97M tokens, context 1024, global batch 64,
micro-batch 8, mp precision, linear LR 0.003 with 1% warmup, AdamW decay 0.1,
and four complete 1,000-sequence validation passes. This is a proposed
15–20 minute M4 GPU experiment, pending measurement at d4. It is a deliberately
smaller recipe than Assignment 1's default. `--preset reference` retains the
supported reference TrainConfig defaults, including d8 / 600k sequences /
100 evaluations / num_micro_batches=1; it needs a larger data prefix and
likely a smaller micro-batch on this Mac.

Download the exact seed-42 global training prefix plus the reference validation
split (about 280 MB successful range payload, about 208 MB final cache):

```sh
.venv/bin/python -m mlx_port.data --num-train-sequences 100000 --data-seed 42
```

The source release is pinned by commit. It contains a 19.66 GB raw train file;
the downloader fetches only selected byte ranges and checks every response's
status, length and Content-Range. It verifies publisher SHA256 digests for the
source metadata and the full validation split, and records local train/index
SHA256 digests. A partial training file cannot be verified against the
publisher's whole-file digest. `data/dclm/train_ds42/source_rows.npy` records
its exact original row IDs. Downloads checkpoint every 1,000 ranges and can
be resumed with the same command/row count/merge-gap. Defaults fetch at most
1.9 GB of successful payload; a server ignoring ranges fails before reading
its body. Cache files stay under `data/`, outside `.venv`.

Reference order is `default_rng(seed).permutation(9600000)[:N]`, **not** a
shuffle of the N downloaded rows. A cache is seed-specific. To change data
seed, first run the downloader with `--data-seed SEED`, then the runner with
that seed. `--data-seed none` fetches an unshuffled source prefix. Validation
always uses the original 1,000 rows, without a shuffle. An existing completed
cache cannot be extended in place: choose another `--data-dir` for a larger
prefix. A 600k-row cache fits the download cap with `--merge-gap 0`.

On a Mac with Metal access:

```sh
# Real-data smoke: d4, 16 updates, one full validation pass; estimated 76–94 s.
.venv/bin/python -m mlx_port.run --preset smoke --device gpu
# Local experiment: d4, 320 updates, four full validation passes.
.venv/bin/python -m mlx_port.run --preset local --device gpu
# 16 sequential experiments; roughly 7–13 h including explicit dropout attention.
.venv/bin/python -m mlx_port.sweep --device gpu --preset local \
  --depths 4,5 --learning-rates 0.001,0.003 --schedules linear,cos --dropouts 0,0.1
MPLCONFIGDIR="$PWD/.build/matplotlib" .venv/bin/python -m mlx_port.plot_runs --runs-dir runs/p2-local
```

Every run writes `config.json`, per-update `metrics.jsonl`, `summary.json`
(including final eval loss, throughput, memory, train/eval/wall times), and optional
`final.safetensors` (`--save-weights`) under `runs/<knob-name>-<config-hash>/`. Existing runs are
protected from overwriting; use `--run-name-suffix repeat1` or `--output-dir`.
The full config digest prevents collisions between knobs omitted from the
readable name. The sequential sweep accepts a JSON **list** of override objects
with `--configs`, captures each run's progress in `_sweep_configs/*.log`, and
emits one JSON result per trial, continuing after errors, OOMs or timeouts.
Individual runs print the first update, every `--log-every N`, and every eval.

CLI knobs mirror TrainConfig names (use hyphens): `--model-name d4` or
`--depth 4`, `--learning-rate`, `--lr-schedule`, `--warmup-percent`,
`--weight-decay`, `--dropout`, `--batch-size`, `--num-micro-batches`,
`--num-train-sequences`, `--num-epochs`, `--num-evals`, `--data-seed`,
`--model-seed`, `--precision`, `--beta1`, `--beta2`, `--grad-norm`.
`--micro-batch 8` is an alias for the appropriate num_micro_batches;
`--eval-interval N` explicitly overrides the reference `(steps//num_evals)+1`.
`--config file.json` loads a JSON object of these fields over the preset;
explicit CLI flags win. JSON field names use underscores.

All reference schedules work: `linear`/`lin`, `cos`,
`constant`/`const`/`flat`, and `wsdFRACTION` (e.g. `wsd0.1`). Constant LR
ignores warmup, matching the reference. The optimizer uses the LR indexed by
completed steps, so warmup begins at LR 0. Dropout applies to attention
softmax probabilities and both residual branches, with inverted scaling.
Nonzero attention dropout uses explicit attention because MLX fast SDPA lacks
a dropout argument; expect lower throughput and higher memory than dropout=0.
Evaluation disables dropout and weights mean NLL by next-token count,
including the last partial batch. Activation checkpointing with dropout is
rejected because RNG replay is not implemented. MLX's RNG differs from
PyTorch's; seeds are reproducible within this port, not across frameworks.

CPU verification with normal MLX (requires Metal to be available at import):

```sh
MLX_DEVICE=cpu \
  .venv/bin/python -m mlx_port.test_t3
MLX_DEVICE=cpu \
  .venv/bin/python -m mlx_port.run --device cpu --preset local --max-steps 3 \
  --verification-context 32 --verification-val-sequences 16 --output-dir runs/T3-cpu
```

CPU-only shape overrides and truncated runs are tagged as verification in
names, config and summary; the plot excludes them by default. These do not
measure full-context GPU throughput or establish the local preset's final loss.
`python -m mlx_port.estimate_local` reproduces FLOP estimates from the existing
coordinator GPU measurements. Module RMS logs, checkpoint/resume, custom
model builders, SGD, QK-norm removal and tied embeddings are outside this runner.

## Lecture pop-quiz (T4)

`run` now defaults to W&B **online** in project `dl-alchemy-popquiz`, default
entity. Use `--wandb offline` or `--wandb disabled` for local verification.
Online initialization has a 20-second timeout and falls back to offline.
Each update logs loss, LR, pre-clip gradient norm, cumulative/step tokens,
synchronized tok/s and wall time; evals log validation NLL. Config includes
variant, preset and SHA256 of the package's Python source without using git.
Summaries include final/best validation loss, status and elapsed time.

Final weights are now **off by default**; opt in with `--save-weights`.
`--variant` accepts `reference` (unchanged), `no_prenorm`, `no_residuals`,
`no_mlp`, `no_attention`, `no_rope`, or `mixed_residual`. All parameters are
retained. Bypassed parameters have zero MLX gradients but are skipped by AdamW,
including decay, matching PyTorch's `grad=None` behavior. A nonfinite training
loss stops before the optimizer update and further micro-batches; a nonfinite
loss/norm/evaluation produces a `diverged` summary and still finishes W&B.
Python JSON files intentionally use `NaN` for nonfinite values.

The ordered plan has 37 unique runs: six Q1 variants at five LRs, mixed
residual at those LRs plus 1e-4, and reference at 1e-4. Q2 shares the other
reference runs. The local preset remains d4/context 1024/global batch 64,
8 micro-batches, 20,480 sequences and four full validation passes.

```sh
.venv/bin/python -m mlx_port.queue --dry-run
# One full-context update + full validation; pipeline check, projected <60 s.
.venv/bin/python -m mlx_port.run --smoke --device gpu --wandb offline \
  --output-dir runs/popquiz-smoke
# Full ordered queue; choose a sequence budget if different from 20480.
.venv/bin/python -m mlx_port.queue --device gpu --token-budget-sequences 20480 \
  --timeout 3600 --wandb online
MPLCONFIGDIR="$PWD/.build/matplotlib" .venv/bin/python -m mlx_port.popquiz_summary
```

The queue uses fresh subprocesses, per-run timeouts and existing-summary
skips, appends to `results/popquiz.jsonl`, and prints progress/ETA. Worker logs
and W&B files stay in each run directory. A timeout/error receives a terminal
summary too. To retry it, choose a new output directory (and results file),
or archive its directory first. For a one-run queue check use `--limit 1
--max-steps 1 --output-dir runs/popquiz-check --results results/popquiz_check.jsonl`.
Checks are excluded from scientific summaries. Use separate results files for
incompatible recipes/budgets. The summary writes measured-only Q1 rankings,
Q2 ΔL, a run status table, and two PNGs. Missing or diverged runs never become
finite loss values; repeated resume records are deduplicated.

```sh
MLX_DEVICE=cpu \
  .venv/bin/python -m mlx_port.test_variants
MLX_DEVICE=cpu \
  MPLCONFIGDIR="$PWD/.build/matplotlib" .venv/bin/python -m mlx_port.test_t4
```

`test_variants` patches only in-memory PyTorch reference instances, loads
identical weights, and checks fp32 logits/loss/gradients, masks/positions,
10 AdamW updates, unused-parameter semantics, and compiled activation
checkpoint gradients for each variant. `test_t4` verifies offline logging,
mocked online fallback, divergence/error finalization, timeout/restart behavior
and summary calculations on isolated synthetic data.
