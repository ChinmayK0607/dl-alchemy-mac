# dl-alchemy-mac — project context

Running Stanford CS 312 **Deep Learning Alchemy** (Fall 2026, <https://deep-learning-alchemy.github.io/>) experiments
locally on Apple Silicon instead of the course's H100s on Modal. This file is the "where are we" doc; the per-task
history is in `tasks/` (specs `T*_*.md`, agent reports `T*.report.md`).

## Workflow

- Claude Code coordinates: writes task specs, reviews, runs GPU jobs. Codex CLI agents (`codex exec -m gpt-6.1-sol
  -s workspace-write`) implement. The Codex sandbox has **no Metal GPU**, so agents verify on CPU and the coordinator runs
  every GPU benchmark/training job outside the sandbox. (Running Codex with the sandbox disabled was refused by
  Claude Code's auto-mode safety check, so this split is deliberate.)
- Course reference code: `reference/` submodule = `deep-learning-alchemy/assignments` @ `a1_release`
  (`baafc3f2`). Read-only; only parity tests import it.

## What exists

| Path | What |
|---|---|
| `mlx_port/` | MLX re-implementation of the course d4–d18 Llama-style model + training loop (AdamW w/ reference decay mask, clipping, all LR schedules, dropout, grad accumulation, compiled step), exact DCLM data order, eval, W&B logging, run/sweep/queue runners, pop-quiz architecture variants. |
| `torch_mps/` | PyTorch-MPS port (benchmarks only; ~2x slower than MLX, abandoned). |
| `plans/popquiz.json` | 37-run lecture-1 pop-quiz plan. |
| `results/` | Benchmarks (`mlx_sweep.jsonl`, `torch_mps.jsonl`), parity proofs (`mlx_parity.json`, `T4_variants.json`), pop-quiz results (`popquiz.jsonl`, `popquiz_summary.md`, PNGs). |
| `tasks/` | Specs and agent reports T1–T5. |

Correctness evidence (CPU, fp32): MLX d8 matches the reference PyTorch model (loss diff 3e-6 over 20 AdamW steps,
loaded from identical weights); each pop-quiz variant matches the corresponding patched PyTorch model (loss < 1e-6,
grads < 3e-5, 10-step curves). Data loader is token-exact vs the reference `data.py` seed-42 order.

## Hardware findings (MacBook Air M4, 10-core GPU, 24 GB, fanless)

- Measured 4096³ matmul ceiling: **2.67 TFLOPS fp32, 3.11 TFLOPS bf16** (MLX). **Low Power Mode cuts this ~4x** —
  turn it off. Plug in; use `caffeinate` for long runs. A training run pins the GPU and makes the UI laggy.
- MLX wired-memory limit must stay below `max_recommended_working_set_size` (~17.7 GB on 24 GB); 18 GB failed.
- d8 default recipe (35M params, 64×1024 batch, 9,375 steps, 614M tokens):

| Config (MLX, compiled, CE chunk 1024) | s/step | tok/s | MFU vs bf16 matmul | Default run incl. 100 evals |
|---|---|---|---|---|
| mp (fp32 master, bf16 matmuls), micro 8 | 9.07 | 7,228 | 60% | 24.7 h |
| bf16, micro 8 (fastest) | 8.78 | 7,464 | 62% | 23.8 h |
| PyTorch MPS bf16-autocast, micro 16 | — | 3,640 | — | ~49 h |

  Same run on one H100 ≈ 10 min. Compute: 6·N·D + attention ≈ 16.9 TFLOP/step ≈ 1.6e17 FLOPs total.
- Re-measured 2026-11-15 on an idle machine: d8 mp micro 8 = **8.0k tok/s (8.2 s/step, 66% MFU) → ~22.5 h**.
- d4 (5.9M params): **~40k tok/s cool, ~35k after 2 min of sustained load** (fanless thermal drop). The overnight
  pop-quiz runs only got ~16.5k tok/s all night (no-attention ~32k, no-MLP ~24k) and the final run jumped to 39.7k
  mid-run at ~09:50 when the user woke the machine: some overnight power/thermal state throttled the GPU ~2.4x. Not
  code (identical results/packages). On a new machine, watch tok/s in long unattended runs.

## Data

`python -m mlx_port.data --num-train-sequences N --data-seed 42` fetches exactly the reference's seed-42 first N
rows of `kothasuhas/dl_alchemy_seq9p6m_context1024` via HTTP ranges (100k rows ≈ 200 MB, ~21 min on home internet) plus
the 1,000 validation rows. The full A1 run needs N = 600,000 (~1.2 GB).

## Lecture-1 pop quiz (done overnight 2026-11-15)

Q1: rank {reference, no prenorm, no residuals, no MLP, no attention, no RoPE} by best final val loss over LR
{3e-4,1e-3,3e-3,1e-2,3e-2}. Q2: ΔL = best(mixed residual ½x_l+½x_{l−1}+a+m) − best(reference) over LR {1e-4…3e-2}.
We ran it **downscaled: d4, 12,800 sequences (13M tokens, 200 steps)**, 37 runs, ~16 min each, W&B project
`ChinmayK0604/dl-alchemy-popquiz`.

| Variant | Course d8 (9,375 steps) | Ours d4 (200 steps) |
|---|---|---|
| reference | **2.927** (1) @3e-3 | 5.117 (3) @1e-3 |
| no RoPE | 3.067 (2) | 5.205 (4) |
| no prenorm | 3.077 (3) @1e-3, unstable higher | **4.690** (1) @3e-3, diverges ≥1e-2 |
| no MLP | 3.220 (4) | 5.278 (5) |
| no attention | 4.864 (5) | 4.997 (2) |
| no residuals | 6.875 (6) | 6.757 (6) |
| Q2 ΔL | −0.001 | +0.028 |

**Lesson:** 200 steps is too short — we measured *early learning speed*, not final quality. In the course's curves,
no-attention plateaus at ~4.9–5.0 and the baseline only overtakes it after ~150–200 steps; we stopped right at the
crossover. No-prenorm also learns fastest early. The safe downscale must go well past the crossover (≥1,000 steps).
Q2's ±0.03 at our scale is likely within seed noise (unmeasured).

## Next steps (on a bigger Mac, e.g. M5 Pro)

1. `scripts/setup.sh`, then `scripts/check.sh` → compare roofline/d8 tok/s with the M4 numbers above.
   Expectation (estimate, unverified): M5-generation GPU Neural Accelerators (MLX, macOS ≥ 26.2) + 2x cores → ~3–5x the
   M4 Air; memory size doesn't matter for these models (d8 uses ~4 GB).
2. Step-matched check: d8 baseline, full recipe (needs N = 600,000 data) → should land near the course's 2.9267; this also
   validates the port end to end. Or d4 baseline vs no-attention at ≥1,000 steps to confirm the crossover explanation.
3. Re-run the pop quiz at ≥1,000 steps; measure seed noise (a few seeds at the best LR) before trusting small gaps.

## Commands

```sh
python -m mlx_port.run --preset smoke --device gpu                 # ~1.5 min pipeline check (real data)
python -m mlx_port.run --preset local --device gpu --variant no_rope --learning-rate 1e-3
python -m mlx_port.queue --device gpu --token-budget-sequences 12800 --wandb online \
  --output-dir runs/popquiz --results results/popquiz.jsonl      # skips completed runs; resumable
python -m mlx_port.popquiz_summary --results results/popquiz.jsonl --output-dir results
python -m mlx_port.bench --device gpu --precisions mp,bf16 --micro-batches 8 --compiled true --chunk-sizes 1024 --boundaries micro
```
W&B uses the machine's `~/.netrc` login (this account logs to `forge.coreweave.com`). Weights are not saved unless
`--save-weights`.
