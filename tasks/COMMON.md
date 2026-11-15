# Common context (read first)

Goal: get an honest, measured estimate of how long the Stanford CS 312 "Deep Learning Alchemy" Assignment 1 default
training run takes on THIS Mac (Apple M4, 10-core GPU, 24 GB unified memory), with a port that is as fast as possible.

Reference code (read-only, do NOT modify): `reference/` (override with `DL_ALCHEMY_REFERENCE`; legacy fallback `../dl-alchemy-a1`)
  - modeling.py (model), model_config.py (d8 config), train.py (training loop, loss, optimizer use), optimizers.py,
    lr_schedules.py, utils.py, MACHINE_LEARNING.md (recipe).
Default recipe: model d8 (8 layers, d=512, 8 heads x 64, SwiGLU intermediate 1792, RMSNorm, RoPE theta 500000, QK-norm,
no biases, untied embeddings, vocab 4096), context 1024, global batch 64 sequences, AdamW lr 3e-3, linear decay w/ 1% warmup,
bf16 mixed precision ("mp"), 9,375 steps = 614.4M tokens. Verify every one of these against the code; the code wins.

Workspace (you may write ONLY here): .
Python: .venv/bin/python (torch 2.15 w/ MPS, mlx 0.32.4).
You may `uv pip install -p .venv <pkg>` if truly needed; note it in your report.

RULES
- Your sandbox has NO Metal GPU access (MPS / mlx gpu will fail). Do all correctness work on CPU (torch device "cpu",
  mlx: mx.set_default_device(mx.cpu)) with small shapes where needed. The coordinator will run your GPU benchmarks
  outside the sandbox and send results back, so make bench scripts that run unattended on GPU with --device gpu/mps,
  sweep a config grid given by flags, never hang, and print one JSON line per config (catch OOM per config, continue).
  Also make a quick `--smoke` mode (< 60 s total on GPU) the coordinator can run first.
- Timing (implement in bench scripts): warm up >= 5 steps, then time >= 20 steps with proper synchronization (torch.mps.synchronize() / mx.eval).
  Report tokens/sec, sec/step at the FULL global batch of 64 x 1024 (use gradient accumulation if needed and say so),
  peak memory, and projected wall-clock for 9,375 steps (also include eval overhead if the reference does periodic eval).
- Also measure a roofline: achieved TFLOPS of a large bf16 and fp32 matmul (e.g. 4096^3) in your framework, and report
  model FLOPs utilization (use 6*N*tokens + attention FLOPs; state your formula).
- Correctness matters more than speed: a fast wrong model is worthless. Prove numerical equivalence to the reference.
- No git commits, no reading ~/.codex or any .env, nothing outside the workspace besides reading the reference repo.
- Do not "optimize" by changing the recipe (smaller batch/context/model, fewer steps, skipping QK-norm etc.). Precision
  changes (bf16/fp16 autocast, fused kernels) are fine if loss stays sane; report which you used.
- Finish with a report file (path given in your task) containing: the final command, a table of every configuration you
  tried with tok/s, the best configuration, the projected default-run time, MFU, and any caveats.
