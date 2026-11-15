"""Parity against unmodified A1 code; MLX uses its default device (or MLX_DEVICE)."""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
from mlx.utils import tree_flatten

from .model import Config, Model, loss
from .train_mlx import Trainer, decay_parameter, learning_rate

from .paths import reference_path
from .runtime import configure_test_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("results/mlx_parity.json"))
    p.add_argument("--d8-only", action="store_true")
    args = p.parse_args()
    if args.steps < 20:
        p.error("The loss-curve proof requires at least 20 optimizer steps")
    device = configure_test_device(mx)
    reference_root = reference_path()
    torch.set_num_threads(4)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference_root))
    from model_config import LMConfig
    from modeling import AutoregressiveLM, initialize_model
    from optimizers import build_optimizer, build_masked_weight_decay_parameter_groups
    from lr_schedules import build_scheduler

    # Execute the exact loss function AST without importing training's unrelated
    # checkpoint, dataset and logging dependencies or configuration I/O.
    source = ast.parse((reference_root/"train.py").read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "causal_lm_loss")
    namespace = {"F": F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "reference_causal_lm_loss", "exec"), namespace)
    ref_loss = namespace["causal_lm_loss"]
    records = []
    rng = np.random.default_rng(123)

    def pair(c, precision="fp32", checkpoint=False):
        torch.manual_seed(42)
        reference = AutoregressiveLM(LMConfig(**asdict(c)), dtype=torch.float32).cpu()
        torch.manual_seed(42)
        initialize_model(reference)
        port = Model(c, precision, checkpoint)
        port.load_reference(reference.state_dict())
        mx.eval(port.parameters())
        return reference, port

    # Real default architecture forward and backward, at a short CPU sequence.
    configs = [Config()] if args.d8_only else [Config(), Config.tiny()]
    for c in configs:
        reference, port = pair(c)
        tokens = rng.integers(0, c.vocab_size, size=(2, 16), dtype=np.int32)
        pt_ids, ids = torch.from_numpy(tokens).long(), mx.array(tokens)
        with torch.no_grad():
            expected = reference(pt_ids).numpy()
            expected_loss = ref_loss(torch.from_numpy(expected), pt_ids).item()
        actual = np.array(port(ids))
        actual_loss = float(loss(port, ids).item())
        diff = float(np.max(np.abs(expected-actual)))
        assert diff < 2e-4, (c.name, diff)
        assert abs(actual_loss-expected_loss) < 1e-4
        record = dict(test="forward", model=c.name, shape=list(tokens.shape),
                      logits_max_abs_diff=diff, torch_loss=expected_loss, mlx_loss=actual_loss,
                      parameters=sum(p.numel() for p in reference.parameters()))
        records.append(record)
        print(json.dumps(record), flush=True)
        if c.name == "d8":
            long_tokens = rng.integers(0, c.vocab_size, size=(1, 1024), dtype=np.int32)
            long_pt = torch.from_numpy(long_tokens).long()
            with torch.no_grad():
                expected_long = reference(long_pt).numpy()
                expected_long_loss = ref_loss(torch.from_numpy(expected_long), long_pt).item()
            actual_long = np.array(port(mx.array(long_tokens)))
            long_diff = float(np.max(np.abs(expected_long-actual_long)))
            actual_long_loss = float(loss(port, mx.array(long_tokens)).item())
            assert long_diff < 2e-4 and abs(actual_long_loss-expected_long_loss) < 1e-4
            rec = dict(test="full_context", model="d8", shape=[1, 1024],
                       logits_max_abs_diff=long_diff, torch_loss=expected_long_loss, mlx_loss=actual_long_loss)
            records.append(rec)
            print(json.dumps(rec), flush=True)
        # Verify mask and nontrivial position IDs as well as the fused default RoPE.
        mask_np = np.ones_like(tokens); mask_np[0, -3:] = 0; mask_np[0, 0] = 0
        positions = np.tile(np.arange(16, dtype=np.int32)+7, (2, 1))
        with torch.no_grad():
            masked_ref = reference(pt_ids, attention_mask=torch.from_numpy(mask_np),
                                   position_ids=torch.from_numpy(positions)).numpy()
        masked_mlx = np.array(port(ids, mx.array(mask_np), mx.array(positions)))
        mask_diff = float(np.max(np.abs(masked_ref-masked_mlx)))
        assert mask_diff < 2e-4, mask_diff
        records.append(dict(test="mask_position", logits_max_abs_diff=mask_diff))

        # Decay names must exactly match the reference module-class mask.
        groups = build_masked_weight_decay_parameter_groups(reference, 0.1)
        decayed_ids = {id(p) for g in groups if g["weight_decay"] for p in g["params"]}
        assert {n for n, p in reference.named_parameters() if id(p) in decayed_ids} == {
            n for n, _ in tree_flatten(port.parameters()) if decay_parameter(n)}

        opt = build_optimizer(reference, "adamw", 3e-3, 0.1, 0.9, 0.95)
        scheduler = build_scheduler(opt, "linear", 93, 9375)
        trainer = Trainer(port, micro_batch=1, global_batch=2, compiled=False)
        trainer.materialize()
        curve = []
        for step in range(args.steps):
            opt.zero_grad()
            value = ref_loss(reference(pt_ids), pt_ids)
            value.backward()
            # Initial gradients, before optimizer differences can compound.
            if step == 0:
                _, grads = trainer.grad(trainer.params, ids)
                grad_diff = max(float(np.max(np.abs(
                    p.grad.numpy()-np.array(grads[n])))) for n, p in reference.named_parameters())
                assert grad_diff < 3e-4, grad_diff
                records.append(dict(test="gradients", max_abs_diff=grad_diff))
            ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            assert abs(opt.param_groups[0]["lr"]-learning_rate(step)) < 1e-12
            opt.step(); scheduler.step()
            actual_value, actual_norm = trainer.step(ids)
            row = dict(step=step, torch_loss=value.item(), mlx_loss=actual_value.item(),
                       torch_grad_norm=ref_norm.item(), mlx_grad_norm=actual_norm.item())
            assert abs(row["torch_loss"]-row["mlx_loss"]) < 1e-4, row
            curve.append(row)
        weight_diff = max(float(np.max(np.abs(p.detach().numpy()-np.array(trainer.params[n]))))
                          for n, p in reference.named_parameters())
        records.append(dict(test="adamw_curve", model=c.name, steps=args.steps, curve=curve,
                            max_loss_diff=max(abs(x["torch_loss"]-x["mlx_loss"]) for x in curve),
                            final_weights_max_abs_diff=weight_diff))
        print(json.dumps(records[-1]), flush=True)
        if c.name == "d8":
            for precision in ("mp", "bf16"):
                _, mixed_model = pair(c, precision)
                mixed_train = Trainer(mixed_model, 1, 2, compiled=False, chunk_size=7)
                mixed_train.materialize()
                mixed_curve = []
                for _ in range(20):
                    value, _ = mixed_train.step(ids)
                    mixed_curve.append(float(value.item()))
                assert all(np.isfinite(mixed_curve)) and mixed_curve[-1] < mixed_curve[0]
                rec = dict(test="precision_stability", model="d8", precision=precision,
                           initial_loss=mixed_curve[0], final_loss=mixed_curve[-1],
                           max_loss_diff_vs_fp32=max(abs(a-b["mlx_loss"]) for a, b in zip(mixed_curve, curve)),
                           curve=mixed_curve)
                records.append(rec)
                print(json.dumps(rec), flush=True)
            continue

        # Compile, chunking, checkpointing, accumulation and low precision paths.
        baseline_curve = None
        for precision, compiled, chunk, boundary, checkpoint in [
            ("fp32", True, 0, "step", False),
            ("fp32", True, 7, "micro", True),
            ("mp", True, 7, "micro", False),
            ("bf16", True, 0, "step", False),
        ]:
            ref, model = pair(c, precision, checkpoint)
            train = Trainer(model, 1, 2, compiled, chunk, boundary)
            train.materialize()
            initial = float(train.evaluate(ids).item())
            mixed_curve = []
            for _ in range(20):
                val, _ = train.step(ids, lr=3e-4)
                mixed_curve.append(float(val.item()))
            assert all(np.isfinite(mixed_curve)) and mixed_curve[-1] < initial
            if baseline_curve is None:
                baseline_curve = mixed_curve
            curve_diff = max(abs(a-b) for a, b in zip(mixed_curve, baseline_curve))
            assert curve_diff < (1e-4 if precision == "fp32" else 0.05), curve_diff
            if precision == "fp32":
                expected_value = ref_loss(ref(pt_ids), pt_ids).item()
                assert abs(initial-expected_value) < 1e-4
            rec = dict(test="options", precision=precision, compiled=compiled, chunk_size=chunk,
                       boundary=boundary, checkpoint=checkpoint, initial_loss=initial,
                       final_loss=mixed_curve[-1], max_loss_diff_vs_fp32=curve_diff, curve=mixed_curve)
            records.append(rec)
            print(json.dumps(rec), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed", device=device, records=records), indent=2)+"\n")


if __name__ == "__main__":
    main()
