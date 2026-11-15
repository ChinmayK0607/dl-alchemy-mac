# T2: fastest MLX port of the A1 default run

Read tasks/COMMON.md first.

Write mlx_port/ in the workspace:
1. mlx_port/model.py: an exact MLX re-implementation of the reference d8 model (RMSNorm eps, RoPE theta and layout
   (rotate_half style as in the reference), QK-norm placement, SwiGLU, init, untied head). Prefer mx.fast kernels
   (mx.fast.rms_norm, mx.fast.rope with the right `traditional` flag, mx.fast.scaled_dot_product_attention with causal mask).
2. Correctness (required): load the reference PyTorch model's initialized weights (CPU fp32, reference code) into the MLX
   model, feed the same tokens, and show logits max-abs-diff is small (fp32) and the loss matches to ~1e-4. Also show
   that ~20 AdamW steps on a fixed batch track the reference loss curve. Put this in mlx_port/parity_check.py.
3. mlx_port/train_mlx.py: training step with value_and_grad, AdamW matching the reference hyperparameters (betas, eps,
   weight decay and which params get decay — check optimizers.py/train.py), mx.compile'd step, grad accumulation to 64.
4. You cannot measure GPU speed yourself: implement each option behind a flag for bench.py to sweep. Optimize: fp32 vs bf16 params/activations (with fp32 master weights or loss-safe alternative), micro-batch sizes,
   compiled vs not, chunked cross-entropy, lazy-eval graph boundaries (mx.eval once per step). Keep what helps.
5. Benchmark script mlx_port/bench.py printing a JSON line per config; save to results/mlx.jsonl.

Report: tasks/T2.report.md
