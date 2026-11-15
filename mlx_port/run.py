"""Train real DCLM: python -m mlx_port.run --preset local --device gpu."""
import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import secrets
import time

from .config import TrainConfig, PRESETS, make_config, run_name
from .data import DEFAULT_ROOT, load_datasets
from .schedules import learning_rate
from .tracking import Tracking, code_hash
from .runtime import configure_memory, validate_memory_gb


def evaluate(trainer, dataset, micro_batch, context=1024, limit=None):
    # Exactly reference NLL weighting, including the final partial val batch.
    import mlx.core as mx
    total_nll, total_tokens = 0.0, 0
    rows = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(0, rows, micro_batch):
        ids = mx.array(dataset[i:min(i+micro_batch, rows), :context].astype("int32"))
        tokens = ids.shape[0] * (ids.shape[1] - 1)
        total_nll += float(trainer.evaluate(ids).item()) * tokens
        total_tokens += tokens
    return total_nll / total_tokens


def train(c, *, device="gpu", data_dir=DEFAULT_ROOT, output_dir=Path("runs"),
          max_steps=None, verification_context=1024, verification_val_sequences=None,
          wandb="online", question_group="q1", preset="local", save_weights=False,
          memory_gb=None):
    import mlx.core as mx
    import numpy as np
    from .model import Model, depth_model_config
    from .train_mlx import Trainer
    c.validate()
    validate_memory_gb(memory_gb)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not 2 <= verification_context <= 1024:
        raise ValueError("verification_context must be in [2,1024]")
    if verification_val_sequences is not None and not 1 <= verification_val_sequences <= 1000:
        raise ValueError("verification_val_sequences must be in [1,1000]")
    if device != "cpu" and (verification_context != 1024 or verification_val_sequences is not None):
        raise ValueError("Verification shape overrides are CPU-only")
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    memory_limit = configure_memory(mx, memory_gb) if device == "gpu" else None
    train_data, val_data = load_datasets(data_dir, c.num_train_sequences, c.data_seed)
    name = run_name(c)
    is_check = max_steps is not None or verification_context != 1024 or verification_val_sequences is not None
    if is_check:
        name += f"-check-{max_steps}-{verification_context}-{verification_val_sequences}"
    path = Path(output_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    if (path / "metrics.jsonl").exists() or (path / "config.json").exists():
        raise FileExistsError(f"Run already exists: {path}; choose --run-name-suffix or --output-dir")
    started = time.perf_counter()
    model = Model(depth_model_config(int(c.model_name[1:])), c.precision, c.checkpoint, c.dropout, c.variant)
    actual_seed = c.model_seed if c.model_seed is not None else secrets.randbits(32)
    model.initialize(actual_seed)
    trainer = Trainer(model, c.micro_batch_size, c.batch_size, c.compiled, c.chunk_size,
                      c.boundary, c.total_steps, c.learning_rate, c.lr_schedule, c.warmup_percent,
                      c.weight_decay, c.beta1, c.beta2, c.grad_norm)
    trainer.materialize()
    nparams = sum(p.size for p in trainer.params.values())
    config = dict(asdict(c), run_name=name, device=device, model_config=asdict(model._config),
                  total_steps=c.total_steps, eval_steps=c.eval_steps, micro_batch_size=c.micro_batch_size,
                  parameter_count=nparams, actual_model_seed=actual_seed, verification=is_check,
                  verification_context=verification_context, verification_val_sequences=verification_val_sequences,
                  max_steps=max_steps, data_dir=str(data_dir), data_metadata=train_data.metadata,
                  mlx_version=mx.__version__, numpy_version=np.__version__,
                  preset=preset, question_group=question_group, code_hash=code_hash(),
                  wandb_requested=wandb, save_weights=save_weights,
                  memory_gb=memory_gb, memory_limit_bytes=memory_limit,
                  effective_memory_gb=memory_limit / 1024**3 if memory_limit else None)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(dict(status="started", run_dir=str(path), **config)), flush=True)
    n_batches = math.ceil(len(train_data) / c.micro_batch_size)
    run_steps = c.total_steps if max_steps is None else min(max_steps, c.total_steps)
    input_tokens, train_seconds, eval_seconds = 0, 0.0, 0.0
    final_eval, best_eval = None, float("inf")
    status = "verification_complete" if is_check else "complete"
    error, completed_steps = None, 0
    tracker = Tracking(wandb, path, config, question_group)
    mx.reset_peak_memory()
    try:
        with (path / "metrics.jsonl").open("w", buffering=1) as metrics:
            for step in range(run_steps):
                step_start = time.perf_counter()
                batches = []
                for i in range(c.num_micro_batches):
                    batch_index = (step * c.num_micro_batches + i) % n_batches
                    start = batch_index * c.micro_batch_size
                    batches.append(mx.array(np.asarray(train_data[start:start+c.micro_batch_size, :verification_context], dtype=np.int32)))
                rate = learning_rate(step, c.total_steps, c.learning_rate, c.lr_schedule, c.warmup_percent)
                value, norm = trainer.step_micro_batches(batches, rate)
                seconds = time.perf_counter() - step_start
                step_tokens = getattr(trainer, "last_step_tokens", sum(x.size for x in batches))
                train_seconds += seconds
                input_tokens += step_tokens
                completed_steps = step+1
                row = dict(step=step+1, train_loss=float(value.item()), lr=rate, grad_norm=float(norm.item()),
                           tokens=input_tokens, step_tokens=step_tokens, tokens_per_second=step_tokens/seconds,
                           seconds_per_step=seconds, wall_time_seconds=time.perf_counter()-started)
                if not all(math.isfinite(row[k]) for k in ("train_loss", "grad_norm")):
                    status, final_eval = "diverged", float("nan")
                elif (step+1) % c.eval_steps == 0 or step == run_steps-1:
                    eval_start = time.perf_counter()
                    final_eval = evaluate(trainer, val_data, c.micro_batch_size, verification_context,
                                          verification_val_sequences)
                    duration = time.perf_counter() - eval_start
                    eval_seconds += duration
                    row.update(eval_loss=final_eval, val_loss=final_eval, eval_seconds=duration)
                    if not math.isfinite(final_eval):
                        status, final_eval = "diverged", float("nan")
                    else:
                        best_eval = min(best_eval, final_eval)
                row["wall_time_seconds"] = time.perf_counter()-started
                metrics.write(json.dumps(row) + "\n")
                tracker.log(row)
                if (step+1) % c.log_every == 0 or step == 0 or "eval_loss" in row or status == "diverged":
                    print(json.dumps(row), flush=True)
                if status == "diverged":
                    break
        if save_weights and status != "diverged":
            mx.save_safetensors(str(path / "final.safetensors"), trainer.params)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        final_eval = float("nan")
    finally:
        summary = dict(status=status, run_dir=str(path), variant=c.variant, learning_rate=c.learning_rate,
                       preset=preset, question_group=question_group, code_hash=config["code_hash"],
                       steps=completed_steps, tokens=input_tokens, final_eval_loss=final_eval,
                       final_val_loss=final_eval, best_val_loss=best_eval if math.isfinite(best_eval) else float("nan"),
                       train_seconds=train_seconds, eval_seconds=eval_seconds,
                       wall_time_seconds=time.perf_counter()-started,
                       tokens_per_second=input_tokens/train_seconds if train_seconds else 0,
                       peak_memory_bytes=mx.get_peak_memory(), parameter_count=nparams, error=error,
                       verification=is_check, wandb_mode=tracker.mode)
        tracker.finish(summary)
        summary["wandb_error"] = tracker.error
        (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--question-group", choices=("q1", "q2"), default="q1")
    p.add_argument("--save-weights", action="store_true", help="Opt in to final weight saving")
    p.add_argument("--smoke", action="store_true", help="One full-context update and full validation (<60s projected on GPU)")
    p.add_argument("--preset", choices=PRESETS, default="local")
    p.add_argument("--config", type=Path, help="JSON object of TrainConfig fields")
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--memory-gb", type=float, help="Allocation/wired cap in GiB; default: 75%% of MLX recommended working set")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--depth", type=int, help="Alias for model_name=dN")
    p.add_argument("--micro-batch", type=int, help="Alias for batch_size/num_micro_batches")
    for f in fields(TrainConfig):
        if f.name in ("compiled", "checkpoint"):
            p.add_argument("--" + f.name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument("--" + f.name.replace("_", "-"), default=None)
    p.add_argument("--max-steps", type=int, help="Truncate for pipeline checks; marked verification")
    p.add_argument("--verification-context", type=int, default=1024, help="CPU-only crop for fast correctness checks")
    p.add_argument("--verification-val-sequences", type=int, help="CPU-only eval subset")
    return p


def config_from_args(a):
    overrides = json.loads(a.config.read_text()) if a.config else {}
    defaults = asdict(TrainConfig())
    for f in fields(TrainConfig):
        v = getattr(a, f.name)
        if v is not None:
            default = defaults[f.name]
            if f.name in ("grad_norm", "data_seed", "model_seed", "eval_interval") and str(v).lower() in ("none", "null"):
                v = None
            elif f.name in ("grad_norm",):
                v = float(v)
            elif f.name in ("data_seed", "model_seed", "eval_interval"):
                v = int(v)
            elif isinstance(default, bool):
                pass
            elif isinstance(default, (int, float)):
                v = type(default)(v)
            overrides[f.name] = v
    if a.depth is not None:
        overrides["model_name"] = f"d{a.depth}"
    if a.micro_batch is not None:
        batch = overrides.get("batch_size", PRESETS[a.preset].get("batch_size", TrainConfig.batch_size))
        if a.micro_batch <= 0 or batch % a.micro_batch:
            raise ValueError("micro-batch must divide batch-size")
        overrides["num_micro_batches"] = batch // a.micro_batch
    return make_config(a.preset, overrides)


def main():
    p = parser()
    a = p.parse_args()
    if a.smoke:
        a.preset, a.max_steps = "smoke", 1
    try:
        validate_memory_gb(a.memory_gb)
        c = config_from_args(a)
        train(c, device=a.device, data_dir=a.data_dir, output_dir=a.output_dir,
              max_steps=a.max_steps, verification_context=a.verification_context,
              verification_val_sequences=a.verification_val_sequences, wandb=a.wandb,
              question_group=a.question_group, preset=a.preset, save_weights=a.save_weights,
              memory_gb=a.memory_gb)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
