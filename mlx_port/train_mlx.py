"""Reference training semantics with fp32 master weights and Adam state.

Compilation covers the differentiated micro-step and optimizer update. Explicit
state inputs prevent stale captured weights; accumulation never changes batch 64.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .model import Config, Model, loss


from .schedules import learning_rate


def decay_parameter(name):
    return name != "model.embed_tokens.weight" and "norm" not in name and not name.endswith("bias")


class Trainer:
    def __init__(self, model, micro_batch=1, global_batch=64, compiled=True,
                 chunk_size=0, boundary="micro", total_steps=9375,
                 learning_rate=3e-3, lr_schedule="linear", warmup_percent=0.01,
                 weight_decay=0.1, beta1=0.9, beta2=0.95, grad_norm=1.0):
        if global_batch % micro_batch or micro_batch < 1:
            raise ValueError("micro_batch must divide global_batch")
        if boundary not in ("micro", "step"):
            raise ValueError(boundary)
        self.model = model
        self.micro_batch, self.global_batch = micro_batch, global_batch
        self.boundary, self.total_steps = boundary, total_steps
        self.chunk_size = chunk_size
        self.base_lr, self.lr_schedule, self.warmup_percent = learning_rate, lr_schedule, warmup_percent
        self.params = dict(tree_flatten(model.trainable_parameters()))
        self.m = {k: mx.zeros_like(p, dtype=mx.float32) for k, p in self.params.items()}
        self.v = {k: mx.zeros_like(p, dtype=mx.float32) for k, p in self.params.items()}
        self.step_number = 0

        def objective(params, ids):
            model.update(tree_unflatten(list(params.items())))
            return loss(model, ids, chunk_size)
        grad = mx.value_and_grad(objective)

        def update(params, m, v, grads, step, lr):
            # torch clip_grad_norm_: min(1, 1/(norm+1e-6)), globally once.
            norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for g in grads.values()))
            clip = mx.array(1.) if grad_norm is None else mx.minimum(mx.array(1.), grad_norm/(norm+1e-6))
            result, new_m, new_v = {}, {}, {}
            c1, c2 = 1-beta1**step, 1-beta2**step
            for name, p in params.items():
                if model.unused_parameter(name):
                    result[name], new_m[name], new_v[name] = p, m[name], v[name]
                    continue
                g = grads[name].astype(mx.float32)*clip
                mt = beta1*m[name] + (1-beta1)*g
                vt = beta2*v[name] + (1-beta2)*g*g
                p = p*(1-lr*weight_decay) if decay_parameter(name) else p
                result[name] = p-(lr/c1)*mt/(mx.sqrt(vt)/mx.sqrt(c2)+1e-8)
                new_m[name], new_v[name] = mt, vt
            return result, new_m, new_v, norm

        # Compile training and evaluation separately: module.training is a Python
        # branch, and RNG state must be threaded through compiled dropout calls.
        def train_grad(params, ids):
            model.train()
            return grad(params, ids)
        def eval_objective(params, ids):
            model.eval()
            return objective(params, ids)
        self.grad = mx.compile(train_grad, inputs=mx.random.state, outputs=mx.random.state) if compiled else train_grad
        self.update = mx.compile(update) if compiled else update
        self.eval_loss = mx.compile(eval_objective) if compiled else eval_objective

    def materialize(self):
        mx.eval(self.params, self.m, self.v)

    def step(self, ids, lr=None):
        if ids.shape[0] != self.global_batch:
            raise ValueError("Pass the full global batch; Trainer slices micro-batches")
        return self.step_micro_batches([ids[start:start+self.micro_batch]
                                        for start in range(0, self.global_batch, self.micro_batch)], lr)

    def step_micro_batches(self, batches, lr=None):
        # Reference averages each micro-batch mean equally, including a possible
        # final partial micro-batch. Preserve this for non-divisible row budgets.
        self.model.train()
        n = self.global_batch // self.micro_batch
        if len(batches) != n:
            raise ValueError("Wrong accumulation count")
        accumulated, total_loss = None, mx.array(0., mx.float32)
        self.last_step_tokens = 0
        for batch in batches:
            value, grads = self.grad(self.params, batch)
            self.last_step_tokens += batch.size
            total_loss = total_loss + value/n
            if accumulated is None:
                accumulated = {k: g/n for k, g in grads.items()}
            else:
                accumulated = {k: accumulated[k]+g/n for k, g in grads.items()}
            if self.boundary == "micro":
                mx.eval(total_loss, accumulated)
            # Stop before applying an update or evaluating further micro-batches.
            if not bool(mx.isfinite(value).item()):
                return value, mx.array(float("nan"))
        rate = learning_rate(self.step_number, self.total_steps, self.base_lr,
                             self.lr_schedule, self.warmup_percent) if lr is None else lr
        self.params, self.m, self.v, norm = self.update(
            self.params, self.m, self.v, accumulated,
            mx.array(self.step_number+1, mx.float32), mx.array(rate, mx.float32))
        mx.eval(total_loss, self.params, self.m, self.v, norm)
        self.model.update(tree_unflatten(list(self.params.items())))
        self.step_number += 1
        return total_loss, norm

    def evaluate(self, ids):
        self.model.eval()
        value = self.eval_loss(self.params, ids)
        mx.eval(value)
        return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--precision", choices=["fp32", "mp", "bf16"], default="mp")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--boundary", choices=["micro", "step"], default="micro")
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--tokens", type=Path, required=True, help="Seed-42 pre-shuffled [>=600000,1024] integer .npy")
    p.add_argument("--val-tokens", type=Path, required=True, help="[1000,1024] integer .npy")
    p.add_argument("--save", type=Path, help="Opt in to final weights at this path")
    p.add_argument("--save-weights", action="store_true")
    args = p.parse_args()
    import numpy as np
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    train = np.load(args.tokens, mmap_mode="r")
    val = np.load(args.val_tokens, mmap_mode="r")
    if train.ndim != 2 or train.shape[0] < 600000 or train.shape[1] != 1024 or val.shape != (1000, 1024):
        p.error("Expected >=600000 train rows and exactly 1000 validation rows, context 1024")
    if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(val.dtype, np.integer):
        p.error("Token arrays must have integer dtype")
    if train[:600000].min() < 0 or train[:600000].max() >= 4096 or val.min() < 0 or val.max() >= 4096:
        p.error("Token IDs must be in [0,4096)")
    model = Model(Config(), args.precision, args.checkpoint)
    model.initialize()
    trainer = Trainer(model, args.micro_batch, compiled=args.compiled,
                      chunk_size=args.chunk_size, boundary=args.boundary)
    trainer.materialize()
    start = time.perf_counter()
    for step in range(9375):
        value, norm = trainer.step(mx.array(train[step*64:(step+1)*64].astype(np.int32)))
        record = dict(step=step+1, loss=float(value.item()), grad_norm=float(norm.item()),
                      lr=learning_rate(step), elapsed_seconds=time.perf_counter()-start)
        if (step+1) % 94 == 0 or step == 9374:
            total = 0.
            for i in range(0, 1000, args.micro_batch):
                batch = mx.array(val[i:i+args.micro_batch].astype(np.int32))
                total += float(trainer.evaluate(batch).item())*batch.shape[0]
            record["val_loss"] = total/1000
        print(json.dumps(record), flush=True)
    if args.save_weights or args.save:
        destination = args.save or Path("results/mlx_final.safetensors")
        destination.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(destination), trainer.params)


if __name__ == "__main__":
    main()
