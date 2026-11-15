# T3: real DCLM data + fast small-scale A1 config for the MLX port

Read tasks/COMMON.md first (rules still apply: sandbox has no Metal GPU; CPU-only verification; the coordinator runs GPU jobs).
Read tasks/T2.report.md and mlx_port/ — you are extending that port. Keep parity_check.py passing.

## Goal
Let the user run real Assignment 1 experiments locally on the M4 Air with the MLX port: real DCLM tokens, the
reference's training knobs, and a small default config that finishes fast (target ~10–20 min on the GPU) while still
giving meaningful signal (loss well below unigram entropy, LR/schedule/dropout differences visible).

## 1. Data
- Reference: ../dl-alchemy-a1/data.py, download_data.py (HF dataset, manifest).
  Network is available in your sandbox. Download ONLY what is needed: the validation split + enough train sequences for
  ~100k sequences (disk is tight: 22 GB free; stay under ~2 GB total). Store under dl-alchemy-mac/data/ (not in .venv).
- Implement mlx_port/data.py that yields exactly the same sequences in the same order as the reference loader for a given
  seed / num_train_sequences prefix (the reference uses a seed-42 preshuffled order — reproduce its semantics; read the
  code). Prove it: a CPU test comparing the first N batches (token-exact) against the reference data.py.
- Validation: the reference's 1000 validation sequences, same eval loss definition and eval interval logic.

## 2. Training knobs (mirror reference TrainConfig names/semantics)
Support at least: depth via depth_model_config (d4..d9), learning_rate, lr_schedule (all schedules in lr_schedules.py),
warmup, weight_decay, dropout (where the reference applies it), global batch size, micro batch, num_train_sequences,
eval interval, seed(s), precision (mp default). Read train.py TrainConfig for defaults; unchanged defaults must equal the
reference. Add CPU tests for new pieces (dropout placement, schedules vs reference lr_schedules output).

## 3. Runner
mlx_port/run.py: one CLI entry that trains from a config (CLI flags and/or a small JSON/YAML), writes
runs/<run_name>/metrics.jsonl (step, train loss, lr, grad norm, tokens, tok/s, wall time, eval loss when computed),
config.json, final eval loss, and final weights. Run name encodes non-default knobs (like the reference).
Must print a line per N steps so progress is visible. Must work with `--device gpu` (coordinator) and `--device cpu` (you).
Also mlx_port/sweep.py: run a list of configs sequentially (e.g. a p2-style depth x {lr, schedule, dropout} grid) and a
tiny plot/summary script for final eval loss vs config (matplotlib is fine to install).

## 4. Pick the small default ("local") config
FLOP-estimate (state formula) using measured d8 GPU numbers in results/mlx_sweep.jsonl (bf16/mp, micro-batch 8:
~7.2–7.5k tok/s at d8 ~1.9 model TFLOPS; smaller models will get lower MFU — be conservative) and propose:
- `local` preset: smallest depth/token budget expected to finish in ~10–20 min on GPU incl. eval, keeping context 1024 and
  global batch 64 unless you justify otherwise. Explain why it still gives real signal (tokens/param, expected loss).
- `smoke` preset: < 3 min, for checking the pipeline end to end.
Do not run them on GPU (you can't); run both presets on CPU for a handful of steps to prove they start and log.

## Report: tasks/T3.report.md
Exact coordinator commands for: smoke on GPU, local on GPU, and a p2-style sweep at local scale (with total estimated time).
List all tests and their results, data bytes downloaded, and any deviations from the reference.
