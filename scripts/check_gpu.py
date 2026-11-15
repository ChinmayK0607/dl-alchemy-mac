"""Bounded GPU checks, with small artifacts under results/<hostname>/<timestamp>."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

# check.sh establishes the repository working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_port.runtime import validate_memory_gb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memory-gb", type=float, help="Override the default 75%% recommended working-set cap (GiB)")
    p.add_argument("--timeout", type=float, default=600, help="Seconds per benchmark configuration")
    a = p.parse_args()
    try:
        validate_memory_gb(a.memory_gb)
        if not 0 < a.timeout < float("inf"):
            raise ValueError("timeout must be positive and finite")
    except ValueError as exc:
        p.error(str(exc))
    host = "".join(c if c.isalnum() or c in "-_." else "_" for c in socket.gethostname())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("results") / host / stamp
    output.mkdir(parents=True)
    run_dir = Path("runs") / "machine-checks" / host / stamp
    extra = ["--memory-gb", str(a.memory_gb)] if a.memory_gb is not None else []
    commands = [
        ("device", ["-c", "import json, mlx.core as mx; mx.set_default_device(mx.gpu); "
                    "print(json.dumps(dict(mlx_version=mx.__version__, device_info=mx.device_info())))"], 30),
        # Includes separate 4096-square fp32/bf16 rooflines, then one d8 config.
        ("bench", ["-m", "mlx_port.bench", "--device", "gpu", "--precisions", "mp",
                   "--micro-batches", "8", "--compiled", "true", "--chunk-sizes", "1024",
                   "--boundaries", "micro", "--checkpoints", "false", "--warmup", "5", "--steps", "20",
                   "--timeout", str(a.timeout), "--output", str(output / "bench.jsonl"), *extra], 3*a.timeout+15),
        ("smoke", ["-m", "mlx_port.run", "--smoke", "--device", "gpu", "--wandb", "disabled",
                   "--output-dir", str(run_dir), *extra], 120),
    ]
    records = []
    for name, args, timeout in commands:
        started = time.perf_counter()
        print(f"Running {name}; results: {output}", flush=True)
        row = dict(check=name, command=[sys.executable, *args], timeout_seconds=timeout)
        try:
            proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout)
            row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr[-8000:])
            row["status"] = "ok" if proc.returncode == 0 else "error"
            if name == "bench" and proc.returncode == 0:
                results = [json.loads(line) for line in (output / "bench.jsonl").read_text().splitlines()]
                if len(results) != 3 or any(r.get("status") != "ok" for r in results):
                    row.update(status="error", error="One or more isolated GPU benchmarks failed; see bench.jsonl")
            elif name == "smoke" and proc.returncode == 0:
                summary = json.loads(proc.stdout.splitlines()[-1])
                row["summary"] = summary
                if summary.get("status") != "verification_complete":
                    row.update(status="error", error="Smoke training did not complete")
        except subprocess.TimeoutExpired:
            row.update(status="timeout", error=f"Exceeded {timeout} seconds")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds"] = time.perf_counter()-started
        (output / f"{name}.json").write_text(json.dumps(row, indent=2)+"\n")
        records.append(row)
        print(f"{name}: {row['status']}", flush=True)
    result = dict(status="ok" if all(r["status"] == "ok" for r in records) else "error",
                  hostname=host, timestamp=stamp, platform=platform.platform(), python=sys.version,
                  checks=[dict(check=r["check"], status=r["status"]) for r in records])
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(result, output=str(output))), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
