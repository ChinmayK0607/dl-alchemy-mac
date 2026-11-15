"""Unattended, process-isolated A1 benchmark grid; stdout is JSONL only.

Examples:
  .venv/bin/python torch_mps/bench.py --device mps --smoke
  .venv/bin/python torch_mps/bench.py --device mps --suite recommended
  .venv/bin/python torch_mps/bench.py --device mps --precisions fp32,bf16-autocast,fp16-autocast,bf16 --micro-batches 8,16,32,64

Every configuration runs in a fresh child process with a hard timeout. GPU
failures, OOMs and compiler errors cannot poison subsequent configurations.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
from train_mps import (Options, Trainer, MODEL_CONFIGS, PRECISIONS, LOSSES, ROOT,
                       autocast, correctness, flops_per_step, reference_hashes,
                       require_device, sync, tiny_config)
import torch


class MemorySampler:
    """MPS has no public max-memory counter in this build; sample its counters.

    Report a sampled lower bound, not an exact allocator peak. Parent process
    isolation keeps maximum RSS specific to one configuration.
    """
    def __init__(self, device):
        self.device = device
        self.stop_event = threading.Event()
        self.allocated = self.driver = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        if self.device == "mps":
            self.allocated = max(self.allocated, torch.mps.current_allocated_memory())
            self.driver = max(self.driver, torch.mps.driver_allocated_memory())

    def run(self):
        while not self.stop_event.wait(0.01):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)
                break

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop_event.set()
        self.thread.join(timeout=1)

    def result(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"sampled_peak_allocated_bytes": self.allocated if self.device == "mps" else None,
                "sampled_peak_driver_bytes": self.driver if self.device == "mps" else None,
                "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "memory_method": "MPS 10ms samples (lower bound); OS process high-water RSS",
                "memory_sampling_error": self.error}


def time_calls(fn, count, device):
    sync(device)
    start = time.perf_counter()
    value = None
    for _ in range(count):
        value = fn()
    sync(device)
    return (time.perf_counter() - start) / count, value


def benchmark(payload):
    options = Options(**payload["options"])
    require_device(options.device)
    if payload["smoke"]:
        config, global_batch = tiny_config(), 8
        options = replace(options, micro_batch=max(1, options.micro_batch // 8), chunk_tokens=64)
        warmup, steps = 1, 2
        check = None
    else:
        config, global_batch = MODEL_CONFIGS["d8"], 64
        warmup, steps = payload["warmup"], payload["steps"]
        check = correctness(options)
        if check["status"] != "ok":
            return {"status": "correctness_failed", "correctness": check}
        # Release tiny verification objects before model initialization.
        import gc
        gc.collect()
        if options.device == "mps":
            torch.mps.empty_cache()
    trainer = Trainer(config, options, global_batch)
    ids = torch.randint(config.vocab_size, (global_batch, config.context_length),
                        generator=torch.Generator().manual_seed(123)).to(options.device)
    with MemorySampler(options.device) as memory:
        init_loss = sum(trainer.eval_loss(batch).item() for batch in ids.split(options.micro_batch)) / (global_batch // options.micro_batch)
        for _ in range(warmup):
            trainer.step(ids)
        start_scale = trainer.scaler.get_scale()
        sec, loss = time_calls(lambda: trainer.step(ids), steps, options.device)
        last_loss = loss.item()
        end_scale = trainer.scaler.get_scale()
        final_loss = sum(trainer.eval_loss(batch).item() for batch in ids.split(options.micro_batch)) / (global_batch // options.micro_batch)
        eval_sec = None
        if not payload["smoke"]:
            micro_ids = ids[:options.micro_batch]
            for _ in range(5):
                trainer.eval_loss(micro_ids)
            per_micro, eval_loss = time_calls(lambda: trainer.eval_loss(micro_ids), 20, options.device)
            remainder = 1000 % options.micro_batch
            remainder_sec = 0.0
            if remainder:
                remainder_sec, _ = time_calls(lambda: trainer.eval_loss(ids[:remainder]), 5, options.device)
            eval_sec = (1000 // options.micro_batch) * per_micro + remainder_sec
    finite = all(math.isfinite(value) for value in (last_loss, init_loss, final_loss))
    stable = finite and final_loss <= init_loss + 0.5
    if options.precision == "fp16-autocast":
        stable = stable and end_scale >= 256.0
    # Dense attention FLOPs convention includes masked upper triangle;
    # embedding lookup FLOPs are neglected.
    flops = flops_per_step(config, global_batch, config.context_length)
    train_seconds = 9375 * sec if not payload["smoke"] else None
    eval_seconds = 100 * eval_sec if eval_sec is not None else None
    return {"status": "ok" if stable else "nonfinite_or_unstable", "correctness": check,
            "benchmark_eligible": not payload["smoke"] and stable,
            "smoke": payload["smoke"], "model": config.to_dict(),
            "global_batch": global_batch, "micro_batch": options.micro_batch,
            "accumulation_steps": global_batch // options.micro_batch,
            "warmup_steps": warmup, "timed_steps": steps,
            "sec_per_step": sec, "tokens_per_second": global_batch * config.context_length / sec,
            "init_loss": init_loss, "last_loss": last_loss, "final_eval_loss": final_loss,
            "fp16_scale_start": start_scale, "fp16_scale_end": end_scale,
            "evaluation_seconds_per_1000_sequences": eval_sec,
            "evaluation_count": 100, "projected_train_seconds": train_seconds,
            "projected_evaluation_seconds": eval_seconds,
            "projected_total_seconds": train_seconds + eval_seconds if eval_seconds is not None else None,
            "training_tflops": flops["training_flops"] / sec / 1e12, **flops, **memory.result(),
            "reference_hashes": reference_hashes(),
            "torch_version": torch.__version__, "platform": platform.platform(),
            "mps_recommended_max_memory": torch.mps.recommended_max_memory() if options.device == "mps" else None}


def roofline(payload):
    options = Options(**payload["options"])
    require_device(options.device)
    torch.set_float32_matmul_precision(options.matmul_precision)
    n = payload["matmul_size"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[payload["dtype"]]
    torch.manual_seed(42)
    with torch.no_grad(), MemorySampler(options.device) as memory:
        a = torch.randn(n, n, device=options.device, dtype=dtype)
        b = torch.randn(n, n, device=options.device, dtype=dtype)
        out = torch.empty_like(a)
        fn = lambda: torch.mm(a, b, out=out)
        for _ in range(payload["warmup"]):
            fn()
        sec, _ = time_calls(fn, payload["steps"], options.device)
        finite = out.isfinite().all().item()
    return {"status": "ok" if finite else "nonfinite", "dtype": payload["dtype"],
            "matmul_size": n, "seconds_per_matmul": sec, "tflops": 2 * n**3 / sec / 1e12,
            "warmup_steps": payload["warmup"], "timed_steps": payload["steps"],
            "torch_version": torch.__version__, **memory.result()}


def isolated(payload, timeout):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {"status": "timeout", "timeout_seconds": timeout, "diagnostic": stderr[-6000:]}
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        result = {"status": "worker_failed", "exit_code": process.returncode, "stdout_tail": stdout[-3000:]}
    if stderr:
        result["diagnostic"] = stderr[-6000:]
    return result


def worker():
    payload = json.load(sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            result = roofline(payload) if payload["kind"] == "roofline" else benchmark(payload)
    except Exception as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() else (
            "unavailable" if "MPS unavailable" in message else "error")
        result = {"status": status, "exception": type(exc).__name__, "error": message}
        traceback.print_exc(file=sys.stderr)
    print(json.dumps(result, allow_nan=False), flush=True)


def csv_list(value):
    return value.split(",")


def configurations(args):
    fields = {"precision": args.precisions, "micro_batch": args.micro_batches,
              "attention": args.attentions, "compile": args.compiles, "optimizer": args.optimizers,
              "matmul_precision": args.matmul_precisions, "loss": args.losses, "chunk_tokens": args.chunk_tokens}
    configs = [Options(device=args.device, **dict(zip(fields, values)))
               for values in itertools.product(*fields.values())]
    if args.suite == "recommended":
        # Baseline 4 precisions x 4 microbatches, then one-factor probes. Run a
        # focused Cartesian grid around the measured winner after this pass.
        base = Options(device=args.device)
        configs += [replace(base, **{field: value}) for field, values in {
            "attention": ("mask",), "compile": ("default", "reduce-overhead"),
            "optimizer": ("single", "foreach", "fused"),
            "matmul_precision": ("high", "medium"),
            "loss": ("reference", "ce", "ce_channels"),
            "chunk_tokens": (256, 4096),
        }.items() for value in values]
    unique = {json.dumps(asdict(c), sort_keys=True): c for c in configs}
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "mps", "gpu"), default="mps")
    parser.add_argument("--suite", choices=("grid", "recommended"), default="grid")
    parser.add_argument("--precisions", type=csv_list, default=list(PRECISIONS))
    parser.add_argument("--micro-batches", type=lambda x: [int(v) for v in x.split(",")], default=[8, 16, 32, 64])
    parser.add_argument("--attentions", type=csv_list, default=["causal"])
    parser.add_argument("--compiles", type=csv_list, default=["off"])
    parser.add_argument("--optimizers", type=csv_list, default=["default"])
    parser.add_argument("--matmul-precisions", type=csv_list, default=["highest"])
    parser.add_argument("--losses", type=csv_list, default=["chunked"])
    parser.add_argument("--chunk-tokens", type=lambda x: [int(v) for v in x.split(",")], default=[1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--smoke", action="store_true", help="tiny non-measurement sanity run, hard 45s child timeout")
    parser.add_argument("--roofline", action="store_true", help="also measure 4096 cubed bf16, fp32 and fp16 matmuls")
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "results/torch_mps.jsonl")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    args.device = "mps" if args.device == "gpu" else args.device
    if args.warmup < 5 or args.steps < 20:
        parser.error("measured benchmarks require warmup >=5 and steps >=20; use --smoke for a quick check")
    choices = {"precisions": PRECISIONS, "micro_batches": (8, 16, 32, 64),
               "attentions": ("causal", "mask"), "compiles": ("off", "default", "reduce-overhead"),
               "optimizers": ("default", "single", "foreach", "fused"),
               "matmul_precisions": ("highest", "high", "medium"), "losses": LOSSES}
    for field, allowed in choices.items():
        if not all(value in allowed for value in getattr(args, field)):
            parser.error(f"invalid {field}: choices are {allowed}")
    if min(args.chunk_tokens) < 1 or args.timeout <= 0 or args.matmul_size <= 0:
        parser.error("chunk size, timeout and matmul size must be positive")
    if args.output.resolve() != ROOT and ROOT not in args.output.resolve().parents:
        parser.error("output must be inside workspace")
    configs = [Options(device=args.device)] if args.smoke else configurations(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.append else "w") as output:
        rooflines = {}
        def emit(payload):
            timeout = min(45, args.timeout) if args.smoke else args.timeout
            result = {**payload, **isolated(payload, timeout), "timestamp_unix": time.time()}
            if result["status"] == "ok" and payload["kind"] == "roofline":
                rooflines[(payload["dtype"], payload["options"]["matmul_precision"])] = result["tflops"]
            if result.get("benchmark_eligible"):
                precision = payload["options"]["precision"]
                dtype = "fp32" if precision == "fp32" else "fp16" if precision.startswith("fp16") else "bf16"
                roof = rooflines.get((dtype, payload["options"]["matmul_precision"]))
                result["matmul_roofline_tflops"] = roof
                result["mfu"] = result["training_tflops"] / roof if roof else None
            line = json.dumps(result, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)
        if args.roofline and not args.smoke:
            for dtype in ("bf16", "fp32", "fp16"):
                for precision in sorted({config.matmul_precision for config in configs}):
                    emit({"kind": "roofline", "options": asdict(Options(device=args.device, matmul_precision=precision)),
                          "dtype": dtype, "matmul_size": args.matmul_size, "warmup": args.warmup, "steps": args.steps})
        for options in configs:
            emit({"kind": "training", "options": asdict(options), "smoke": args.smoke,
                  "warmup": args.warmup, "steps": args.steps})


if __name__ == "__main__":
    main()
