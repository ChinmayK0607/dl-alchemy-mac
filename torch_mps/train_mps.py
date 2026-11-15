"""Reference A1 training semantics with MPS-selectable execution options.

No dataset downloads, logging services, or modifications to the reference repo.
Use bench.py for isolated, bounded GPU experiments.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

# Keep all optional compiler artifacts inside the authorized workspace.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlx_port.paths import reference_path
for variable, relative in {
    "TORCHINDUCTOR_CACHE_DIR": "torch_mps/.cache/inductor",
    "TRITON_CACHE_DIR": "torch_mps/.cache/triton",
    "TORCH_EXTENSIONS_DIR": "torch_mps/.cache/extensions",
}.items():
    os.environ[variable] = str(ROOT / relative)

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

REFERENCE = reference_path()
# Do not create __pycache__ in the read-only reference repository.
sys.dont_write_bytecode = True
sys.path.insert(0, str(REFERENCE))
from model_config import LMConfig, MODEL_CONFIGS
from modeling import AutoregressiveLM, LlamaRotaryEmbedding, initialize_model
from optimizers import build_masked_weight_decay_parameter_groups, ADAMW_EPSILON
from lr_schedules import build_scheduler


def reference_loss_function():
    """Load the actual loss without importing train.py's dataset/W&B dependencies."""
    source = (REFERENCE / "train.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "causal_lm_loss")
    namespace = {"F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE / "train.py"), "exec"), namespace)
    return namespace["causal_lm_loss"]


reference_loss = reference_loss_function()
PRECISIONS = ("fp32", "bf16-autocast", "fp16-autocast", "bf16")
LOSSES = ("reference", "ce", "ce_channels", "chunked")


@dataclass(frozen=True)
class Options:
    device: str = "mps"
    precision: str = "bf16-autocast"
    micro_batch: int = 8
    attention: str = "causal"
    compile: str = "off"
    optimizer: str = "default"
    matmul_precision: str = "highest"
    loss: str = "chunked"
    chunk_tokens: int = 1024
    seed: int = 42


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def require_device(device):
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; run GPU benchmarks outside the sandbox")


def autocast(options):
    if options.precision.endswith("autocast"):
        dtype = torch.bfloat16 if options.precision.startswith("bf16") else torch.float16
        return torch.autocast(options.device, dtype=dtype)
    return nullcontext()


class MPSRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    def forward(self, hidden_states, position_ids):
        # Reference forward maps mps to a CPU autocast context. Under MPS
        # autocast that leaves the RoPE matmul eligible for low precision and
        # loses high-position phase accuracy. Disable the actual device context.
        # CPU also works here, for the portable regression test.
        with torch.autocast(hidden_states.device.type, enabled=False):
            return super().forward(hidden_states, position_ids)


def make_model(config, options):
    require_device(options.device)
    torch.manual_seed(options.seed)
    model = AutoregressiveLM(config, dtype=torch.float32, qk_norm=True,
                             tie_word_embeddings=False, dropout=0.0)
    # Reference build_model reseeds before initialize_model.
    torch.manual_seed(options.seed)
    initialize_model(model)
    if options.device == "mps":
        model.model.rotary_emb = MPSRotaryEmbedding(config)
    dtype = torch.bfloat16 if options.precision == "bf16" else torch.float32
    # RoPE is positional metadata, not a low-precision trainable weight.
    # Preserve its fp32 frequencies when testing pure bf16 compute weights.
    inv_freq = model.model.rotary_emb.inv_freq
    model.to(device=options.device, dtype=dtype)
    model.model.rotary_emb.inv_freq = inv_freq.to(options.device)
    return model


def chunk_head_loss(hidden, weight, labels):
    # Explicit fp32 CE matches reference loss precision in all execution modes.
    return F.cross_entropy(F.linear(hidden, weight).float(), labels, reduction="sum")


class TrainingLoss(nn.Module):
    def __init__(self, model, options):
        super().__init__()
        self.model, self.options = model, options

    def forward(self, ids):
        opts = self.options
        mask = torch.ones_like(ids, dtype=torch.bool) if opts.attention == "mask" else None
        if opts.loss == "chunked":
            hidden = self.model.model(ids, attention_mask=mask)
            hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
            labels = ids[:, 1:].reshape(-1)
            total = hidden.new_zeros((), dtype=torch.float32)
            for start in range(0, labels.numel(), opts.chunk_tokens):
                h = hidden[start:start + opts.chunk_tokens]
                y = labels[start:start + opts.chunk_tokens]
                if torch.is_grad_enabled():
                    # Recompute one head/CE chunk at a time in backward. Simply
                    # summing CE chunks retains all softmax activations and does
                    # NOT solve the memory problem. Checkpoint preserves autocast.
                    part = checkpoint(chunk_head_loss, h, self.model.lm_head.weight, y,
                                      use_reentrant=False, preserve_rng_state=False)
                else:
                    part = chunk_head_loss(h, self.model.lm_head.weight, y)
                total = total + part
            return total / labels.numel()
        logits = self.model(ids, attention_mask=mask)
        if opts.loss == "reference":
            return reference_loss(logits, ids)
        if opts.loss == "ce_channels":
            return F.cross_entropy(logits[:, :-1].transpose(1, 2).contiguous().float(), ids[:, 1:])
        return F.cross_entropy(logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))


class Trainer:
    def __init__(self, config=MODEL_CONFIGS["d8"], options=Options(), global_batch=64):
        if options.micro_batch <= 0 or global_batch % options.micro_batch:
            raise ValueError("micro_batch must divide global_batch")
        if options.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        torch.set_float32_matmul_precision(options.matmul_precision)
        self.options, self.global_batch = options, global_batch
        self.model = make_model(config, replace(options, precision="fp32") if options.precision == "bf16" else options)
        self.params = list(self.model.parameters())
        groups = build_masked_weight_decay_parameter_groups(self.model, 0.1)
        self.masters = None
        if options.precision == "bf16":
            # BF16 compute weights + fp32 optimizer/master weights and gradients.
            self.masters = [nn.Parameter(p.detach().float().clone()) for p in self.params]
            inv_freq = self.model.model.rotary_emb.inv_freq
            self.model.to(dtype=torch.bfloat16)
            self.model.model.rotary_emb.inv_freq = inv_freq
            mapping = {id(p): master for p, master in zip(self.params, self.masters)}
            groups = [{**group, "params": [mapping[id(p)] for p in group["params"]]} for group in groups]
        kwargs = {} if options.optimizer == "default" else {
            "foreach": {"foreach": True, "single": False, "fused": False}[options.optimizer],
            "fused": options.optimizer == "fused",
        }
        self.optimizer = torch.optim.AdamW(groups, lr=3e-3, betas=(0.9, 0.95),
                                           eps=ADAMW_EPSILON, weight_decay=0.0, **kwargs)
        self.scheduler = build_scheduler(self.optimizer, "linear", warmup_steps=93, total_steps=9375)
        self.loss_module = TrainingLoss(self.model, options)
        self.loss_fn = self.loss_module
        if options.compile != "off":
            # No eager fallback: compilation failures become explicit config errors.
            self.loss_fn = torch.compile(self.loss_module, mode=options.compile, fullgraph=True)
        self.scaler = torch.amp.GradScaler(options.device, enabled=options.precision == "fp16-autocast",
                                           init_scale=256.0)
        self.model.train()

    def step(self, ids):
        if ids.shape[0] != self.global_batch:
            raise ValueError("step expects one full global batch")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        acc = self.global_batch // self.options.micro_batch
        total = torch.zeros((), device=self.options.device)
        for batch in ids.split(self.options.micro_batch):
            with autocast(self.options):
                loss = self.loss_fn(batch) / acc
            total = total + loss.detach()
            self.scaler.scale(loss).backward()
        if self.masters is not None:
            for p, master in zip(self.params, self.masters):
                master.grad = p.grad.float() if p.grad is not None else None
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.masters or self.params, 1.0, foreach=False)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.masters is not None:
            with torch.no_grad():
                for p, master in zip(self.params, self.masters):
                    p.copy_(master)
        return total

    @torch.no_grad()
    def eval_loss(self, ids):
        self.model.eval()
        with autocast(self.options):
            loss = self.loss_module(ids)
        self.model.train()
        return loss


def tiny_config():
    return LMConfig(name="correctness", vocab_size=128, context_length=24, hidden_size=64,
                    intermediate_size=224, num_hidden_layers=2, num_attention_heads=1,
                    num_key_value_heads=1, head_dim=64)


def correctness(options, steps=20):
    """CPU fp32 reference vs selected path, identical fp32 initial state and batch.

    The small model retains RoPE/QK-norm/SwiGLU and the default optimizer/schedule.
    Accumulation ratio is preserved for micro_batch 8/16/32/64.
    """
    config = tiny_config()
    baseline_opts = Options(device="cpu", precision="fp32", micro_batch=8, loss="reference", seed=options.seed)
    selected = replace(options, micro_batch=max(1, min(8, options.micro_batch // 8)))
    # Build and complete CPU work first; retain only tiny state/logits for comparison.
    base = Trainer(config, baseline_opts, global_batch=8)
    initial_state = {k: v.detach().clone() for k, v in base.model.state_dict().items()}
    gen = torch.Generator().manual_seed(123)
    ids = torch.randint(config.vocab_size, (8, config.context_length), generator=gen)
    with torch.no_grad():
        initial_logits = base.model(ids).float()
    cpu_losses = [base.eval_loss(ids).item()]
    for _ in range(steps):
        base.step(ids)
        cpu_losses.append(base.eval_loss(ids).item())
    target = Trainer(config, selected, global_batch=8)
    target.model.load_state_dict(initial_state)
    if target.masters is not None:
        # Preserve original fp32 reference masters; compute weights are quantized.
        with torch.no_grad():
            for (name, p), master in zip(target.model.named_parameters(), target.masters):
                master.copy_(initial_state[name].to(options.device))
    target_ids = ids.to(options.device)
    with torch.no_grad(), autocast(selected):
        mask = torch.ones_like(target_ids, dtype=torch.bool) if selected.attention == "mask" else None
        logits = target.model(target_ids, attention_mask=mask).float().cpu()
    target_losses = [target.eval_loss(target_ids).item()]
    for _ in range(steps):
        target.step(target_ids)
        target_losses.append(target.eval_loss(target_ids).item())
    delta = logits - initial_logits
    rms_relative = (delta.square().mean().sqrt() / initial_logits.square().mean().sqrt()).item()
    init_error = abs(cpu_losses[0] - target_losses[0])
    final_error = abs(cpu_losses[-1] - target_losses[-1])
    reduced = options.precision != "fp32"
    # Predeclared acceptance thresholds, not a bitwise equivalence claim.
    limits = {"logit_relative_rms": 0.05 if reduced else 2e-4,
              "init_loss_abs": 0.03 if reduced else 2e-4,
              "final_loss_abs": 0.15 if reduced else 2e-3}
    passed = (torch.isfinite(torch.tensor(target_losses)).all().item()
              and rms_relative <= limits["logit_relative_rms"]
              and init_error <= limits["init_loss_abs"]
              and final_error <= limits["final_loss_abs"]
              and target_losses[-1] < target_losses[0]
              and cpu_losses[-1] < cpu_losses[0])
    return {"kind": "correctness", "status": "ok" if passed else "failed", "options": asdict(options),
            "steps": steps, "shape": config.to_dict(), "thresholds": limits,
            "max_abs_logit_error": delta.abs().max().item(), "relative_rms_logit_error": rms_relative,
            "init_loss_abs_error": init_error, "final_loss_abs_error": final_error,
            "cpu_losses": cpu_losses, "target_losses": target_losses,
            "target_device": options.device, "gpu_equivalence_tested": options.device == "mps"}


def flops_per_step(config, batch=64, seq=1024):
    # Explicit formula requested: 6*N*B*T + attention forward/backward matmuls.
    d = config.hidden_size
    kv = config.num_key_value_heads * config.head_dim
    # Untied embedding/head + Q/K/V/O + three SwiGLU matrices + norms.
    n = (2 * config.vocab_size * d + config.num_hidden_layers *
         (2 * d * d + 2 * d * kv + 3 * d * config.intermediate_size +
          2 * d + 2 * config.head_dim) + d)
    attention = 12 * config.num_hidden_layers * batch * seq * seq * config.hidden_size
    return {"parameters": n, "training_flops": 6 * n * batch * seq + attention,
            "attention_flops": attention,
            "formula": "6*N*B*T + 12*L*B*T^2*d (dense attention convention)"}


def reference_hashes():
    return {name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() for name in
            ("modeling.py", "model_config.py", "train.py", "optimizers.py", "lr_schedules.py", "utils.py", "data.py")}


def add_options(parser):
    parser.add_argument("--device", choices=("cpu", "mps", "gpu"), default="mps")
    parser.add_argument("--precision", choices=PRECISIONS, default="bf16-autocast")
    parser.add_argument("--micro-batch", type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument("--attention", choices=("causal", "mask"), default="causal")
    parser.add_argument("--compile", choices=("off", "default", "reduce-overhead"), default="off")
    parser.add_argument("--optimizer", choices=("default", "single", "foreach", "fused"), default="default")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--loss", choices=LOSSES, default="chunked")
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)


def options_from_args(args):
    return Options(**{key: ("mps" if key == "device" and getattr(args, key) == "gpu" else getattr(args, key))
                      for key in Options.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_options(parser)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    options = options_from_args(args)
    if args.correctness:
        result = correctness(options, args.steps)
    else:
        trainer = Trainer(options=options)
        ids = torch.randint(4096, (64, 1024), generator=torch.Generator().manual_seed(123)).to(options.device)
        sync(options.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = trainer.step(ids)
        sync(options.device)
        result = {"kind": "training", "options": asdict(options), "steps": args.steps,
                  "seconds": time.perf_counter() - start, "loss": loss.item()}
    print(json.dumps(result, allow_nan=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
