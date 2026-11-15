# PyTorch MPS benchmark

Run from the workspace root using `.venv/bin/python`, after `uv sync --locked --extra parity`.
The reference repository is imported read-only through `mlx_port.paths`: environment
`DL_ALCHEMY_REFERENCE`, then the pinned `reference/` submodule, then the legacy
sibling `../dl-alchemy-a1`. Missing or incomplete checkouts produce setup guidance.
See the [root README](../README.md) for fresh-Mac setup. No optional dataset or
logging dependencies are required by the PyTorch tools.

```sh
# CPU correctness: 32 loss/attention/accumulation gradient comparisons,
# seven 20-step optimizer/precision trajectories, optimizer/schedule assertions,
# and a high-position RoPE autocast regression.
.venv/bin/python torch_mps/verify_cpu.py

# Outside the sandbox: quick GPU smoke, then real default-recipe sweep.
.venv/bin/python torch_mps/bench.py --device mps --smoke --output results/torch_mps_smoke.jsonl
.venv/bin/python torch_mps/bench.py --device mps --suite recommended --roofline
.venv/bin/python torch_mps/report.py
```

The recommended suite runs 29 unique training configurations and nine 4096³ matmul rooflines (three dtypes × three matmul precision settings). The default grid is the four precisions crossed with micro-batches 8/16/32/64. Every non-smoke training case runs a CPU-fp32 versus selected-device 20-step correctness gate, then measures d8 with global batch 64, context 1024, five warmup steps and twenty timed optimizer steps. Evaluation timing is included in the runtime projection. GPU numerical or stability failures cannot win selection. All output lines are JSON records, saved by default to `results/torch_mps.jsonl`. Errors, OOMs and hard timeouts become records and the sweep continues. Each child starts fresh; CPU fallback is disabled.

Use comma-separated flags to sweep any Cartesian grid: `--precisions`, `--micro-batches`, `--attentions`, `--compiles`, `--optimizers`, `--matmul-precisions`, `--losses`, `--chunk-tokens`. See `--help` for choices. `--device gpu` aliases `mps`. The precision `bf16` uses bf16 compute weights plus fp32 masters, AdamW state and RoPE frequency buffers. `fp16-autocast` uses a GradScaler. Every loss computes in fp32. `chunked` checkpoints and recomputes each output-head/CE chunk, bounding logits memory. `ce_channels` explicitly transposes and makes the channel-first loss input contiguous. `reference` runs the exact reference loss.

A focused second sweep should combine the first sweep's fastest viable settings. Use `--append` to retain previous records, and include `--roofline` for matching precision-setting MFU denominators. `--warmup` and `--steps` can only increase the measurement minima. `--timeout` is a hard limit per child (default 1800 seconds). Smoke always uses a tiny shape, at most 45 seconds per child, and is excluded from performance selection.

MPS memory is sampled every 10 ms because this installed PyTorch exposes no public peak-memory counter; this is a lower bound. Process high-water RSS is also recorded. Synthetic data is preloaded on the device. Projections exclude data loading, W&B/module metrics, serialization, startup/compilation and sustained thermal changes. `train_mps.py` is the synthetic training-step runner, not a dataset/checkpoint training application. `report.py` writes `tasks/T1.report.md`, preserving missing metrics as unmeasured.
