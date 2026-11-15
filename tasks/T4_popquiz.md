# T4: lecture-1 pop-quiz variants, W&B logging, overnight queue (MLX port)

Read tasks/COMMON.md (sandbox: no Metal GPU, CPU-only verification; coordinator runs GPU jobs) and tasks/T3.report.md.
You extend mlx_port/ (run.py, model.py, train_mlx.py, config.py, ...). Keep ALL existing tests (parity_check, test_t3) passing.

## 1. Architecture variants (flag `--variant`, recorded in config + run name; default `reference` = unchanged model)
Implement exactly these lecture diffs (block = pre-norm Llama block; attention_update(x) = attn(input_layernorm(x)),
mlp_update(h) = mlp(post_attention_layernorm(h)); dropout placement as already implemented):
- `no_prenorm`:    input_layernorm and post_attention_layernorm -> identity. QK-norm and the final norm stay.
- `no_residuals`:  a = attention_update(x); return mlp_update(a)   (no skip additions at all).
- `no_mlp`:        mlp_update returns zeros (h = x + attention_update(x); out = h).
- `no_attention`:  attention_update returns zeros (out = x + mlp_update(x)).
- `no_rope`:       identity rotations (cos = 1, sin = 0) for queries and keys; everything else unchanged.
- `mixed_residual` (backbone): first block unchanged. For later blocks with block input x_l and previous block input
  x_{l-1}: a = A(x_l), m = M(x_l + a) (A, M include prenorm; i.e. m = mlp_update(x_l + a));
  x_{l+1} = 0.5*x_l + 0.5*x_{l-1} + a + m. Gradients flow through both skip inputs.
Correctness (required, CPU): for each variant, patch a copy of the reference PyTorch model the same way (monkeypatch in the
test, never edit the reference repo), load identical weights into MLX, and show loss + logits + grads match (fp32, small
context), plus a 10-step AdamW curve match. Put in mlx_port/test_variants.py; write results to results/T4_variants.json.
The same variant must not change parameter count (no params removed; bypassed modules just go unused — say if MLX
grads for unused params are zero and AdamW then only applies weight decay, which matches PyTorch semantics; verify).

## 2. Divergence handling
If train loss becomes non-finite, stop the run immediately, record status "diverged", final_val_loss = NaN, and still
finish logging. Never crash the queue.

## 3. Weights & Biases
Log every run to W&B (project `dl-alchemy-popquiz`, default entity): full config (incl. variant, lr, preset, git-free
code hash), per-step train loss / lr / grad norm / tokens / tok/s / wall time, each eval's val loss, and summary
(final_val_loss, best_val_loss, status, wall time). Group = question id (q1/q2), tags = [variant, lr]. Use `wandb.init(dir=runs/...)`.
Flag `--wandb {online,offline,disabled}` (default online). In YOUR tests use offline or disabled only — do not create
online runs. If online init fails at runtime, fall back to offline and keep training.

## 4. No checkpoints
Weight saving off by default (`--save-weights` to opt in). Don't leave large files in runs/.

## 5. Overnight queue
- plans/popquiz.json: ordered list of run configs (preset `local`, variant, learning_rate, question group).
  Q1: variants {reference,no_prenorm,no_residuals,no_mlp,no_attention,no_rope} x LR {3e-4,1e-3,3e-3,1e-2,3e-2}.
  Q2: mixed_residual x LR {1e-4,3e-4,1e-3,3e-3,1e-2,3e-2} plus reference at 1e-4 (other reference LRs shared with Q1).
  ORDER so partial completion is still useful: all 7 variants at 3e-3, then 1e-3, then 1e-2, then 3e-4, then 3e-2, then 1e-4.
- mlx_port/queue.py: runs the plan sequentially, each in a fresh subprocess with a per-run timeout, skips runs whose
  summary already exists, continues past failures/timeouts, appends one line per run to results/popquiz.jsonl,
  prints progress (i/N, ETA from measured run times), supports `--token-budget-sequences N` to override the local preset's
  num_train_sequences for all runs (coordinator will pick it), and `--dry-run`.
- mlx_port/popquiz_summary.py: from results/popquiz.jsonl make: per-variant best final val loss over the LR grid (Q1
  ranking, diverged marked non-finite), Q2 ΔL = min_lr L_mixed − min_lr L_reference, a val-loss-vs-LR plot per variant
  and val-loss curves at the best LR; write results/popquiz_summary.md + PNGs. Do NOT state "expected answers".

## Report: tasks/T4.report.md — commands for the coordinator (dry-run, a 1-run GPU check, full queue), tests + results.
