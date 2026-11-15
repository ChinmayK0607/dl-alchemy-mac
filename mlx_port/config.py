"""Reference TrainConfig defaults plus explicit MLX/local execution settings."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

from .schedules import learning_rate


@dataclass(frozen=True)
class TrainConfig:
    variant: str = "reference"
    model_name: str = "d8"
    learning_rate: float = 3e-3
    num_epochs: float = 1.0
    batch_size: int = 64
    num_micro_batches: int = 1
    num_evals: int = 100
    warmup_percent: float = 0.01
    weight_decay: float = 0.1
    grad_norm: float | None = 1.0
    dropout: float = 0.0
    data_seed: int | None = 42
    model_seed: int | None = 42
    beta1: float = 0.9
    beta2: float = 0.95
    lr_schedule: str = "linear"
    precision: str = "mp"
    num_train_sequences: int = 600000
    # Execution-only additions; eval_interval=None uses reference num_evals logic.
    eval_interval: int | None = None
    compiled: bool = True
    chunk_size: int = 1024
    boundary: str = "micro"
    checkpoint: bool = False
    log_every: int = 10
    run_name_suffix: str | None = None

    @property
    def micro_batch_size(self):
        return self.batch_size // self.num_micro_batches

    @property
    def total_steps(self):
        # Reference ceil micro-batches, floor accumulation groups, then epochs.
        batches = math.ceil(self.num_train_sequences / self.micro_batch_size)
        return int(self.num_epochs * (batches // self.num_micro_batches))

    @property
    def eval_steps(self):
        return self.eval_interval or (self.total_steps // self.num_evals) + 1

    def validate(self):
        # Keep config/queue dry runs independent of the MLX runtime.
        if self.variant not in ("reference", "no_prenorm", "no_residuals", "no_mlp", "no_attention", "no_rope", "mixed_residual"):
            raise ValueError("Unknown architecture variant")
        if not self.model_name.startswith("d") or not 4 <= int(self.model_name[1:]) <= 18:
            raise ValueError("model_name must be d4..d18")
        for name in ("batch_size", "num_micro_batches", "num_evals", "num_train_sequences", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size % self.num_micro_batches:
            raise ValueError("batch_size must be divisible by num_micro_batches")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be a float")
        for name in ("learning_rate", "num_epochs", "warmup_percent", "weight_decay", "dropout", "beta1", "beta2"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate < 0 or self.num_epochs <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid LR, epochs or weight decay")
        if not 0 <= self.warmup_percent <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid warmup_percent or dropout")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Betas must be in [0,1)")
        if self.grad_norm is not None and (not math.isfinite(self.grad_norm) or self.grad_norm <= 0):
            raise ValueError("grad_norm must be positive or null")
        if self.eval_interval is not None and (type(self.eval_interval) is not int or self.eval_interval <= 0):
            raise ValueError("eval_interval must be a positive integer")
        if self.total_steps < 1:
            raise ValueError("No complete optimizer steps")
        if self.precision not in ("mp", "fp32", "bf16") or self.boundary not in ("micro", "step"):
            raise ValueError("Invalid precision or boundary")
        if self.chunk_size < 0 or (self.checkpoint and self.dropout):
            raise ValueError("Invalid chunk size or dropout/checkpoint combination")
        if self.lr_schedule.startswith("wsd") and not 0 <= float(self.lr_schedule[3:]) <= 1:
            raise ValueError("WSD decay fraction must be in [0,1]")
        learning_rate(0, self.total_steps, self.learning_rate, self.lr_schedule, self.warmup_percent)
        for seed in (self.data_seed, self.model_seed):
            if seed is not None and (type(seed) is not int or seed < 0):
                raise ValueError("Seeds must be nonnegative integers or null")
        if self.run_name_suffix and any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in self.run_name_suffix):
            raise ValueError("run_name_suffix must contain only letters, digits, dash, underscore or dot")
        return self


# Smaller token budgets and fewer full validation passes are deliberate recipe
# changes. Reference defaults remain intact in TrainConfig / --preset reference.
PRESETS = {
    "reference": {},
    "local": dict(model_name="d4", num_train_sequences=20480, num_evals=4, num_micro_batches=8),
    "smoke": dict(model_name="d4", num_train_sequences=1024, num_evals=1, num_micro_batches=8, log_every=1),
}


def make_config(preset="local", overrides=None):
    return TrainConfig(**(PRESETS[preset] | (overrides or {}))).validate()


def run_name(c):
    # Readable knob names plus a digest of the entire config prevent collisions
    # even for execution knobs absent from the reference's run name.
    default = TrainConfig()
    text = f"{c.variant}-model-{c.model_name}-lr{c.learning_rate}-tok{c.num_train_sequences*1024}"
    labels = dict(batch_size="bs", num_epochs="epochs", weight_decay="wd", warmup_percent="warmup",
                  grad_norm="gn", dropout="dropout", data_seed="ds", model_seed="ms", beta1="b1",
                  beta2="b2", lr_schedule="schedule", precision="precision", num_evals="evals",
                  eval_interval="evalint", num_micro_batches="accum")
    for key, label in labels.items():
        if getattr(c, key) != getattr(default, key):
            text += f"-{label}{getattr(c, key)}"
    if c.run_name_suffix:
        text += f"-{c.run_name_suffix}"
    digest = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()[:10]
    # Leave filename room for the config digest on macOS (255 bytes/component).
    return text[:190] + "-" + digest
