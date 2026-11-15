"""Sequential, isolated real-data runs; one JSON result per config, including errors."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import make_config, run_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", type=Path, help="JSON list of TrainConfig override objects")
    p.add_argument("--preset", choices=("local", "smoke", "reference"), default="local")
    p.add_argument("--smoke", action="store_true", help="One smoke preset, <3 minute GPU pipeline check")
    p.add_argument("--depths", default="4,5")
    p.add_argument("--learning-rates", default="0.001,0.003")
    p.add_argument("--schedules", default="linear,cos")
    p.add_argument("--dropouts", default="0,0.1")
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--memory-gb", type=float, help="Forward an explicit GPU memory cap in GiB to each run")
    p.add_argument("--data-dir", type=Path, default=Path("data/dclm"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/p2-local"))
    p.add_argument("--results", type=Path, default=Path("results/T3_sweep.jsonl"))
    p.add_argument("--timeout", type=float, default=7200, help="Seconds per config; errors/timeouts continue")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--verification-context", type=int, default=1024)
    p.add_argument("--verification-val-sequences", type=int)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error("timeout must be positive")
    preset = "smoke" if a.smoke else a.preset
    if a.configs:
        configs = json.loads(a.configs.read_text())
        if not isinstance(configs, list) or not all(isinstance(c, dict) for c in configs):
            p.error("configs must be a JSON list of objects")
    elif a.smoke:
        configs = [{}]
    else:
        configs = [dict(model_name=f"d{d}", learning_rate=float(lr), lr_schedule=s, dropout=float(drop))
                   for d, lr, s, drop in itertools.product(a.depths.split(","), a.learning_rates.split(","),
                                                          a.schedules.split(","), a.dropouts.split(","))]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = a.output_dir / "_sweep_configs"
    config_dir.mkdir(exist_ok=True)
    a.results.parent.mkdir(parents=True, exist_ok=True)
    for index, overrides in enumerate(configs):
        start = time.perf_counter()
        result = dict(index=index, config=overrides, preset=preset, device=a.device)
        try:
            c = make_config(preset, overrides)
            name = run_name(c)
            config_file = config_dir / f"{name}.json"
            config_file.write_text(json.dumps(asdict(c), indent=2) + "\n")
            log = config_dir / f"{name}.log"
            cmd = [sys.executable, "-m", "mlx_port.run", "--preset", preset, "--config", str(config_file),
                   "--device", a.device, "--data-dir", str(a.data_dir), "--output-dir", str(a.output_dir)]
            for key in ("max_steps", "verification_context", "verification_val_sequences", "memory_gb"):
                value = getattr(a, key)
                if value is not None:
                    cmd += ["--" + key.replace("_", "-"), str(value)]
            print(f"[{index+1}/{len(configs)}] {name}; log={log}", file=sys.stderr, flush=True)
            with log.open("w") as output:
                proc = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, timeout=a.timeout)
            result.update(config=asdict(c), returncode=proc.returncode, log=str(log))
            if proc.returncode:
                result.update(status="error", error=log.read_text()[-3000:])
            else:
                # Last runner line is its summary; also handles verification paths.
                summary = json.loads(log.read_text().splitlines()[-1])
                result.update(summary)
        except subprocess.TimeoutExpired:
            result.update(status="timeout", error=f"Exceeded {a.timeout} seconds", log=str(log))
        except Exception as e:
            result.update(status="error", error=f"{type(e).__name__}: {e}")
        result["worker_wall_seconds"] = time.perf_counter() - start
        row = json.dumps(result, allow_nan=False)
        with a.results.open("a") as f:
            f.write(row + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
