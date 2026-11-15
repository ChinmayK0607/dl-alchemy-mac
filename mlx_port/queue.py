"""Sequential resumable pop-quiz subprocess queue (no GPU initialization here)."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .config import make_config, run_name


def run_directory(c, output_dir, max_steps=None, context=1024, val_sequences=None):
    name = run_name(c)
    if max_steps is not None or context != 1024 or val_sequences is not None:
        name += f"-check-{max_steps}-{context}-{val_sequences}"
    return Path(output_dir) / name


def execute(a):
    plan = json.loads(a.plan.read_text())
    if not isinstance(plan, list) or not all(isinstance(c, dict) for c in plan):
        raise ValueError("Plan must be a list of run objects")
    if a.limit is not None:
        plan = plan[:a.limit]
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    durations = []
    results = []
    for index, entry in enumerate(plan):
        started = time.perf_counter()
        result = dict(index=index, config=entry, device=a.device)
        path = None
        try:
            overrides = dict(entry)
            preset = overrides.pop("preset", "local")
            group = overrides.pop("question_group")
            if group not in ("q1", "q2"):
                raise ValueError("question_group must be q1 or q2")
            if a.token_budget_sequences is not None:
                overrides["num_train_sequences"] = a.token_budget_sequences
            c = make_config(preset, overrides)
            path = run_directory(c, a.output_dir, a.max_steps, a.verification_context, a.verification_val_sequences)
            result.update(config=asdict(c), preset=preset, question_group=group, run_dir=str(path))
            cmd = [sys.executable, "-m", "mlx_port.run", "--preset", preset,
                   "--device", a.device, "--data-dir", str(a.data_dir), "--output-dir", str(a.output_dir),
                   "--wandb", a.wandb, "--question-group", group]
            for key, value in asdict(c).items():
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    cmd.append(flag if value else "--no-" + key.replace("_", "-"))
                elif value is not None:
                    cmd += [flag, str(value)]
                elif key in ("grad_norm", "data_seed", "model_seed", "eval_interval"):
                    cmd += [flag, "null"]
            for key in ("max_steps", "verification_context", "verification_val_sequences", "memory_gb"):
                if getattr(a, key) is not None:
                    cmd += ["--" + key.replace("_", "-"), str(getattr(a, key))]
            if a.dry_run:
                print(json.dumps(dict(index=index+1, total=len(plan), command=cmd, run_dir=str(path))), flush=True)
                continue
            summary_file = path / "summary.json"
            if summary_file.exists():
                result.update(json.loads(summary_file.read_text()), skipped=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
                log = path / "worker.log"
                result["log"] = str(log)
                try:
                    with log.open("w") as output:
                        proc = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, timeout=a.timeout)
                    result["returncode"] = proc.returncode
                    if summary_file.exists():
                        result.update(json.loads(summary_file.read_text()))
                    else:
                        result.update(status="error", error=log.read_text()[-3000:])
                except subprocess.TimeoutExpired:
                    result.update(status="timeout", error=f"Exceeded {a.timeout} seconds")
                if not summary_file.exists():
                    # Preserve terminal failures too, so resuming does not overwrite
                    # their logs/config or silently repeat an overnight failure.
                    result.update(final_val_loss=float("nan"), best_val_loss=float("nan"),
                                  wall_time_seconds=time.perf_counter()-started)
                    summary_file.write_text(json.dumps(result, indent=2) + "\n")
                durations.append(time.perf_counter()-started)
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", final_val_loss=float("nan"))
            durations.append(time.perf_counter()-started)
        result["worker_wall_seconds"] = time.perf_counter()-started
        eta = (len(plan)-index-1)*sum(durations)/len(durations) if durations else None
        print(f"[{index+1}/{len(plan)}] {result['status']}"
              f"{' (existing summary)' if result.get('skipped') else ''}; ETA {eta/60:.1f} min"
              if eta is not None else f"[{index+1}/{len(plan)}] {result['status']}; ETA unavailable",
              file=sys.stderr, flush=True)
        a.results.parent.mkdir(parents=True, exist_ok=True)
        with a.results.open("a") as out:
            out.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("plans/popquiz.json"))
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--memory-gb", type=float, help="Forward an explicit GPU memory cap in GiB to each run")
    p.add_argument("--data-dir", type=Path, default=Path("data/dclm"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/popquiz"))
    p.add_argument("--results", type=Path, default=Path("results/popquiz.jsonl"))
    p.add_argument("--timeout", type=float, default=3600, help="Seconds per run, including W&B startup")
    p.add_argument("--token-budget-sequences", type=int)
    p.add_argument("--limit", type=int, help="Execute only the first N plan entries")
    p.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int, help="Pipeline check; excluded from experiment summaries")
    p.add_argument("--verification-context", type=int, default=1024)
    p.add_argument("--verification-val-sequences", type=int)
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error("limit must be positive")
    try:
        execute(a)
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
