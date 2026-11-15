"""Isolated, bounded MLX sweep; stdout and --output contain JSON lines only.

Run as `python -m mlx_port.bench`. The parent never imports MLX, so even
import-time Metal failures, native crashes, OOM and timeouts become result rows.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .runtime import configure_memory, validate_memory_gb


def worker(c):
    import resource
    import numpy as np
    import mlx.core as mx
    from .model import Config, Model
    from .train_mlx import Trainer
    mx.set_default_device(mx.cpu if c["device"] == "cpu" else mx.gpu)
    mx.set_cache_limit(256*1024**2)
    # MLX working-set guidance plus a wired-memory limit; the process timeout
    # also bounds configurations that become slow under memory pressure.
    memory_limit = configure_memory(mx, c.get("memory_gb")) if c["device"] == "gpu" else None
    warmup, steps = c["warmup"], c["steps"]
    metadata = dict(mlx_version=mx.__version__, python=sys.version.split()[0],
                    pid=os.getpid(), device=c["device"], memory_limit_bytes=memory_limit,
                    effective_memory_gb=memory_limit / 1024**3 if memory_limit else None)
    if c["kind"] == "roofline":
        n = c["size"]
        dtype = mx.float32 if c["precision"] == "fp32" else mx.bfloat16
        mx.random.seed(77)
        a = mx.random.normal((n, n)).astype(dtype)
        b = mx.random.normal((n, n)).astype(dtype)
        mx.eval(a, b)
        timings = []
        for i in range(warmup+steps):
            start = time.perf_counter()
            out = a @ b
            mx.eval(out)
            if i >= warmup:
                timings.append(time.perf_counter()-start)
        seconds = statistics.mean(timings)
        return dict(c, **metadata, status="ok", seconds=seconds,
                    tflops=2*n**3/seconds/1e12, peak_memory_bytes=mx.get_peak_memory())

    smoke = c["smoke"]
    cfg = Config.tiny() if smoke else Config()
    batch, context = (4, 16) if smoke else (64, 1024)
    model = Model(cfg, c["precision"], c["checkpoint"])
    model.initialize(42)
    trainer = Trainer(model, c["micro_batch"], batch, c["compiled"],
                      c["chunk_size"], c["boundary"])
    trainer.materialize()
    rng = np.random.default_rng(42)
    ids = mx.array(rng.integers(0, cfg.vocab_size, (batch, context), dtype=np.int32))
    mx.eval(ids)
    parameters = sum(p.size for p in trainer.params.values())
    mx.reset_peak_memory()
    losses, times = [], []
    for i in range(warmup+steps):
        start = time.perf_counter()
        value, _ = trainer.step(ids)
        elapsed = time.perf_counter()-start
        scalar = float(value.item())
        if not math.isfinite(scalar):
            raise FloatingPointError(f"Nonfinite training loss at step {i}")
        if i >= warmup:
            times.append(elapsed)
            losses.append(scalar)
    train_peak = mx.get_peak_memory()
    # Measure repeated forward passes with the same graph boundary as evaluate().
    eval_rows = 8 if smoke else 1000
    mb = c["micro_batch"]
    def eval_time(rows):
        eval_ids = mx.array(rng.integers(0, cfg.vocab_size, (rows, context), dtype=np.int32))
        mx.eval(eval_ids)
        durations = []
        for i in range(warmup+steps):
            start = time.perf_counter()
            trainer.evaluate(eval_ids)
            if i >= warmup:
                durations.append(time.perf_counter()-start)
        return statistics.mean(durations)
    full_eval = eval_time(mb)
    rem = eval_rows % mb
    eval_seconds = (eval_rows//mb)*full_eval + (eval_time(rem) if rem else 0.)
    sec = statistics.mean(times)
    tokens = batch*context
    # Conventional dense-attention 6N approximation, including both embeddings.
    flops = 6*parameters*tokens + 12*cfg.num_hidden_layers*batch*context**2*cfg.hidden_size
    result = dict(c, **metadata, status="ok", model=cfg.name, global_batch=batch,
                  context=context, accumulation_steps=batch//mb, parameters=parameters,
                  seconds_per_step=sec, step_seconds_std=statistics.pstdev(times),
                  step_seconds_min=min(times), step_seconds_max=max(times),
                  tokens_per_second=tokens/sec, predicted_tokens_per_second=batch*(context-1)/sec,
                  peak_memory_bytes=train_peak, peak_memory_with_eval_bytes=mx.get_peak_memory(),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  losses=losses, eval_sequences=eval_rows, eval_seconds_per_pass=eval_seconds,
                  flops_per_step=flops, achieved_model_tflops=flops/sec/1e12,
                  projections_valid=not smoke)
    if not smoke:
        result.update(projected_train_seconds=9375*sec,
                      eval_passes=100, projected_eval_seconds=100*eval_seconds,
                      projected_total_seconds=9375*sec+100*eval_seconds)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu", "mps"], default="gpu")
    p.add_argument("--precisions", default="fp32,mp,bf16")
    p.add_argument("--micro-batches", default="1,2,4")
    p.add_argument("--compiled", default="true,false")
    p.add_argument("--chunk-sizes", default="0,1024")
    p.add_argument("--boundaries", default="micro,step")
    p.add_argument("--checkpoints", default="false")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--timeout", type=float, default=1200., help="Seconds per isolated configuration")
    p.add_argument("--memory-gb", type=float, help="Allocation/wired cap in GiB; default: 75%% of MLX recommended working set")
    p.add_argument("--matmul-size", type=int, default=4096)
    p.add_argument("--roofline-only", action="store_true")
    p.add_argument("--skip-roofline", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny 2-layer workload, 55s overall deadline; no default-run projection")
    p.add_argument("--output", type=Path, default=Path("results/mlx.jsonl"))
    p.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_json:
        config = json.loads(args.worker_json)
        try:
            record = worker(config)
        except Exception as exc:
            import traceback
            record = dict(config, status="error", error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
        print(json.dumps(record), flush=True)
        return
    if args.warmup < 5 or args.steps < 20:
        p.error("Use at least 5 warmup and 20 timed steps (also in smoke mode)")
    def csv(text, convert=str):
        return [convert(x) for x in text.split(",")]
    def boolean(x):
        if x not in ("true", "false"):
            raise ValueError("Use true,false")
        return x == "true"
    precisions = csv(args.precisions)
    micro_batches = csv(args.micro_batches, int)
    if any(x not in ("fp32", "mp", "bf16") for x in precisions):
        p.error("Precisions are fp32,mp,bf16")
    if any(x <= 0 or 64 % x for x in micro_batches):
        p.error("Micro-batches must divide 64")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.matmul_size <= 0:
        p.error("Timeout and matmul size must be positive and finite")
    try:
        validate_memory_gb(args.memory_gb)
    except ValueError as exc:
        p.error(str(exc))
    chunks = csv(args.chunk_sizes, int)
    boundaries = csv(args.boundaries)
    if any(x < 0 for x in chunks) or any(x not in ("micro", "step") for x in boundaries):
        p.error("Chunk sizes must be nonnegative; boundaries are micro,step")
    base = dict(device="gpu" if args.device == "mps" else args.device,
                warmup=args.warmup, steps=args.steps, memory_gb=args.memory_gb, smoke=args.smoke)
    jobs = []
    if not args.skip_roofline:
        for precision in ("fp32", "bf16"):
            jobs.append(dict(base, kind="roofline", precision=precision, size=512 if args.smoke else args.matmul_size))
    if not args.roofline_only:
        if args.smoke:
            grid = [("mp", 2, True, 16, "micro", False)]
        else:
            grid = itertools.product(precisions, micro_batches, csv(args.compiled, boolean),
                                     chunks, boundaries, csv(args.checkpoints, boolean))
        for precision, mb, compiled, chunk, boundary, checkpoint in grid:
            jobs.append(dict(base, kind="train", precision=precision, micro_batch=mb,
                             compiled=compiled, chunk_size=chunk, boundary=boundary, checkpoint=checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rooflines = {}
    deadline = time.monotonic()+55 if args.smoke else float("inf")
    with args.output.open("a", buffering=1) as out:
        for config in jobs:
            timeout = min(args.timeout, max(0.01, deadline-time.monotonic()))
            start = time.perf_counter()
            try:
                proc = subprocess.run([sys.executable, "-m", "mlx_port.bench", "--worker-json", json.dumps(config)],
                                      capture_output=True, text=True, timeout=timeout)
                lines = proc.stdout.strip().splitlines()
                if proc.returncode or not lines:
                    record = dict(config, status="crash", returncode=proc.returncode, stderr=proc.stderr[-4000:])
                else:
                    record = json.loads(lines[-1])
            except subprocess.TimeoutExpired:
                record = dict(config, status="timeout", timeout_seconds=timeout)
            except Exception as exc:
                record = dict(config, status="error", error=str(exc))
            record["worker_wall_seconds"] = time.perf_counter()-start
            if record["status"] == "ok":
                if config["kind"] == "roofline":
                    rooflines[config["precision"]] = record["tflops"]
                else:
                    roof = rooflines.get("fp32" if config["precision"] == "fp32" else "bf16")
                    record["matmul_roofline_tflops"] = roof
                    record["mfu_vs_matmul"] = record["achieved_model_tflops"]/roof if roof else None
            line = json.dumps(record)
            out.write(line+"\n")
            print(line, flush=True)


if __name__ == "__main__":
    main()
