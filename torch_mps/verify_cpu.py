"""CPU-only equivalence tests; writes auditable JSONL, never touches Metal."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch
from train_mps import (Options, Trainer, TrainingLoss, make_model, tiny_config,
                       reference_loss, correctness, ROOT, reference_hashes, MPSRotaryEmbedding)


def run():
    torch.set_num_threads(4)
    config = tiny_config()
    ids = torch.randint(128, (8, 24), generator=torch.Generator().manual_seed(123))
    baseline = make_model(config, Options(device="cpu", precision="fp32"))
    expected_loss = reference_loss(baseline(ids), ids)
    expected_loss.backward()
    expected_gradients = {name: p.grad.clone() for name, p in baseline.named_parameters()}
    rows = []
    # High positions reveal bf16 rounding that a short-context test would miss.
    rotary = MPSRotaryEmbedding(config)
    positions = torch.arange(1000, 1024).unsqueeze(0).expand(8, -1)
    hidden = torch.zeros(8, 24, 64)
    expected_cos, expected_sin = rotary(hidden, positions)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual_cos, actual_sin = rotary(hidden, positions)
    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)
    rows.append({"kind": "high_position_rope_autocast", "status": "ok", "max_abs_error": 0.0,
                 "positions": [1000, 1023], "device": "cpu"})
    for loss_mode in ("reference", "ce", "ce_channels", "chunked"):
        for attention in ("causal", "mask"):
            for micro in (1, 2, 4, 8):
                options = Options(device="cpu", precision="fp32", loss=loss_mode,
                                  attention=attention, micro_batch=micro, chunk_tokens=17)
                model = make_model(config, options)
                function = TrainingLoss(model, options)
                total = torch.tensor(0.)
                for batch in ids.split(micro):
                    loss = function(batch) / (8 // micro)
                    loss.backward()
                    total += loss.detach()
                torch.testing.assert_close(total, expected_loss.detach(), rtol=1e-6, atol=1e-6)
                max_error = 0.
                for name, p in model.named_parameters():
                    torch.testing.assert_close(p.grad, expected_gradients[name], rtol=2e-4, atol=3e-6)
                    max_error = max(max_error, (p.grad - expected_gradients[name]).abs().max().item())
                rows.append({"kind": "gradient_equivalence", "status": "ok", "options": asdict(options),
                             "loss_abs_error": abs(total.item() - expected_loss.item()),
                             "max_abs_gradient_error": max_error})
    for precision in ("fp32", "bf16-autocast", "fp16-autocast", "bf16"):
        row = correctness(Options(device="cpu", precision=precision, loss="chunked", chunk_tokens=37), 20)
        if row["status"] != "ok":
            raise AssertionError(json.dumps(row))
        rows.append(row)
    for optimizer in ("single", "foreach", "fused"):
        row = correctness(Options(device="cpu", precision="fp32", optimizer=optimizer), 20)
        if row["status"] != "ok":
            raise AssertionError(json.dumps(row))
        rows.append(row)
    # Check reference masked decay semantics on the same module architecture.
    trainer = Trainer(config, Options(device="cpu", precision="fp32"), global_batch=8)
    groups = {id(p): group["weight_decay"] for group in trainer.optimizer.param_groups for p in group["params"]}
    for name, p in trainer.model.named_parameters():
        expected = 0.0 if "embed_tokens" in name or "norm" in name else 0.1
        assert groups[id(p)] == expected, name
    assert trainer.optimizer.defaults["betas"] == (0.9, 0.95)
    assert trainer.optimizer.defaults["eps"] == 1e-8
    assert trainer.optimizer.param_groups[0]["lr"] == 0.
    trainer.step(ids)
    assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.003 / 93) < 1e-12
    rows.append({"kind": "optimizer_and_scheduler", "status": "ok", "reference_hashes": reference_hashes()})
    output = ROOT / "results/torch_cpu_tests.jsonl"
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    print(json.dumps({"status": "ok", "checks": len(rows), "output": str(output),
                      "maximum_gradient_error": max(r.get("max_abs_gradient_error", 0) for r in rows)}))


if __name__ == "__main__":
    run()
