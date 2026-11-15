"""Regenerate the T1 report from measured JSONL; never invent missing metrics."""
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from train_mps import ROOT, MODEL_CONFIGS, flops_per_step


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def num(value, digits=3):
    return "unmeasured" if value is None else f"{value:,.{digits}f}"


def label(options):
    return (f'{options["precision"]}; mb={options["micro_batch"]}; {options["loss"]}'
            f'/{options["chunk_tokens"]}; {options["attention"]}; compile={options["compile"]}; '
            f'AdamW={options["optimizer"]}; matmul={options["matmul_precision"]}')


def command(options):
    names = {"micro_batch": "micro-batch", "matmul_precision": "matmul-precision", "chunk_tokens": "chunk-tokens"}
    return ".venv/bin/python torch_mps/train_mps.py " + " ".join(
        f'--{names.get(key, key)} {value}' for key, value in options.items()) + " --steps 9375"


def build():
    gpu = read_rows(ROOT / "results/torch_mps.jsonl")
    gpu_smoke = read_rows(ROOT / "results/torch_mps_smoke.jsonl")
    cpu = read_rows(ROOT / "results/torch_cpu_tests.jsonl")
    smoke = read_rows(ROOT / "results/torch_cpu_smoke.jsonl")
    eligible = [r for r in gpu if r.get("benchmark_eligible") and r["options"]["device"] == "mps"]
    best = min(eligible, key=lambda r: r["projected_total_seconds"]) if eligible else None
    roofs = {(r["dtype"], r["options"]["matmul_precision"]): r["tflops"] for r in gpu
             if r["kind"] == "roofline" and r["status"] == "ok"}
    f = flops_per_step(MODEL_CONFIGS["d8"])
    lines = ["# T1 PyTorch–MPS report", "", f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "## Result", ""]
    if best:
        lines += [f'Best measured configuration: **{label(best["options"])}**.', "",
                  f'{num(best["tokens_per_second"], 1)} input tokens/s, {num(best["sec_per_step"])} s per full 64 × 1024 optimizer step. '
                  f'Projected training: {num(best["projected_train_seconds"]/3600)} h; '
                  f'evaluation: {num(best["projected_evaluation_seconds"]/3600)} h; '
                  f'combined: **{num(best["projected_total_seconds"]/3600)} h**.', "",
                  "Final synthetic-batch training command (performance/correctness exercise, not dataset training):", "", "```sh", command(best["options"]), "```", ""]
    else:
        lines += ["**Full-recipe GPU performance is unmeasured.** This sandbox reports MPS unavailable. An external coordinator has supplied a successful tiny MPS smoke result, but no full-recipe timing results have been received. That smoke predates the final RoPE correction; the commands below rerun it before measurement. Consequently there is no defensible fastest configuration, default-run duration, "
                  "GPU peak memory, achieved GPU roofline, or MFU yet. CPU tests establish implementation correctness only; "
                  "CPU smoke speed is not a Mac GPU estimate.", "",
                  "The port and GPU experiment scripts are complete; hardware measurement and CPU-versus-MPS numerical "
                  "validation remain pending. This report does not claim the overall measured-performance objective is complete.", ""]
    lines += ["## Commands", "", "From the workspace root, run outside the sandbox:", "", "```sh",
              ".venv/bin/python torch_mps/bench.py --device mps --smoke --output results/torch_mps_smoke.jsonl",
              ".venv/bin/python torch_mps/bench.py --device mps --suite recommended --roofline --output results/torch_mps.jsonl",
              ".venv/bin/python torch_mps/report.py", "```", "",
              "The smoke test uses a tiny model, one warmup and two timed steps, with a 45-second child timeout. "
              "It is explicitly excluded from performance selection. The measured suite tries 29 unique training configurations "
              "plus nine bf16/fp32/fp16 4096³ matmul rooflines (three matmul precision settings). Each measured configuration first passes CPU-fp32 versus selected-device "
              "20-step correctness, then runs the unchanged d8 recipe with ≥5 full-step warmups and ≥20 timed full steps. "
              "The full grid has four precisions × micro-batches 8/16/32/64; additional one-factor probes cover explicit-mask "
              "versus causal SDPA, compile modes, AdamW flags, matmul precision, loss layouts and head chunk sizes.", "",
              "After the first sweep, use a focused Cartesian grid around the winner; for example (replace the values with measured winners):", "",
              "```sh", ".venv/bin/python torch_mps/bench.py --device mps --precisions bf16-autocast --micro-batches 8,16 --optimizers default,foreach,fused --compiles off,default --losses chunked,ce --matmul-precisions highest,high,medium --roofline --append",
              "```", "", "The benchmark synchronizes before/after timing, preloads a full random token batch on the device, "
              "uses equal micro-batch loss weights, clips once, and updates once per 64 sequences. Fresh subprocesses, "
              "a configurable 1,800-second hard timeout, and process-group termination isolate OOM/compiler/kernel failures. "
              "One JSON record is printed and saved per config, with errors preserved; CPU fallback is disabled. "
              "Use `--append` to preserve previous sweep results. Compile failures never silently become eager results.", "",
              "## Verified recipe", "",
              "Source code is authoritative. `model_config.py`, `modeling.py`, `train.py`, `optimizers.py`, `lr_schedules.py`, "
              "`utils.py`, `data.py`, and `MACHINE_LEARNING.md` were read in the read-only reference repository. "
              "SHA-256 fingerprints are saved in correctness and successful benchmark records.", "",
              "| Quantity | Verified value |", "|---|---|",
              "| Model | d8; 8 layers; width 512; 8 Q/K/V heads × 64; SwiGLU width 1792 |",
              "| Norm/positions | RMSNorm eps 1e-5; QK RMSNorm after RoPE; theta 500000; max positions 131072 |",
              "| Embeddings/linears | 4096 vocabulary; untied embedding/head; bias-free; dropout 0 |",
              f'| Parameters | {f["parameters"]:,} |',
              "| Batch/context | 64 × 1024; accumulation factors 8/4/2/1 for micro-batches 8/16/32/64 |",
              "| Loss | Mean next-token NLL; 1023 targets/sequence; logits at the last position excluded; fp32 loss |",
              "| AdamW | lr 0.003; betas (0.9, 0.95); eps 1e-8; weight decay 0.1 on ordinary linear weights only |",
              "| No decay | Token embeddings, normalization weights and biases |",
              "| Clipping | Global gradient norm 1.0, after accumulation and scaler unscale |",
              "| Schedule | Reference linear scheduler; 93 warmup steps; initial optimizer LR 0; decay to zero at 9375 |",
              "| Run size | 600000 sequences; 9375 updates; 614400000 input tokens (613800000 predicted targets) |",
              "| Precision | Reference mp = fp32 weights and optimizer state + CUDA bf16 autocast; MPS port explicitly enables its autocast |",
              "| Evaluation | 1000 sequences; interval floor(9375/100)+1 = 94; updates 94…9306 plus final 9375: 100 passes |", "",
              "Reference `utils.autocast_context` only enables autocast on CUDA, and reference training chooses CUDA/CPU. "
              "The port adds MPS device/autocast control while importing the reference model, initialization, optimizer grouping "
              "and scheduler. A small RoPE subclass disables autocast on the actual MPS device: the reference maps its disable context to CPU, which would otherwise leave the MPS frequency matmul under autocast. A CPU-autocast regression at positions 1000–1023 matches fp32 exactly. The exact reference loss function is loaded with AST extraction to avoid optional dataset/W&B imports. "
              "The reference repository was not modified. No packages were installed.", "",
              "## Numerical evidence", ""]
    tests = [r for r in cpu if r["kind"] == "gradient_equivalence"]
    trajectories = [r for r in cpu if r["kind"] == "correctness"]
    lines += [f'{len(cpu)} CPU checks passed. All 32 combinations of four loss modes, two attention paths, and four '
              'accumulation factors were compared against the original fp32 reference loss and every parameter gradient. '
              f'Maximum absolute gradient error: {max((r["max_abs_gradient_error"] for r in tests), default=0):.3g}. '
              'The RoPE high-position autocast regression also passes exactly. Chunk boundaries of 17 tokens cross sequence boundaries. The small model retains the reference RoPE, QK-norm, '
              'SwiGLU, initialization, masked AdamW, clipping and LR schedule.', "",
              "The following trajectories use an 8 × 24 fixed repeated batch, 2 layers, width/head dimension 64, and vocabulary 128. "
              "Target accumulation ratios preserve the full-run ratios. Tolerances were declared before testing: fp32 relative "
              "logit RMS ≤2e-4, init loss error ≤2e-4, final error ≤0.002; reduced precision ≤0.05 / 0.03 / 0.15. "
              "Both CPU-reference and target losses must decrease and remain finite. These are small-shape numerical checks, "
              "not proof of 9375-step convergence. Each GPU measured configuration reruns these checks on MPS.", "",
              "| CPU target | Optimizer | Init logit relative RMS error | Init loss | Loss after 20 updates | Reference final loss | Status |",
              "|---|---|---:|---:|---:|---:|---|"]
    for r in trajectories:
        lines += [f'| {r["options"]["precision"]} | {r["options"]["optimizer"]} | {r["relative_rms_logit_error"]:.6g} | '
                  f'{r["target_losses"][0]:.6f} | {r["target_losses"][-1]:.6f} | {r["cpu_losses"][-1]:.6f} | {r["status"]} |']
    lines += ["", "Pure bf16 uses bf16 compute weights backed by fp32 master weights/gradients and fp32 AdamW state; RoPE frequency buffers stay fp32. "
              "bf16/fp16 autocast retain fp32 parameters/state. fp16 uses dynamic GradScaler, initial scale 256; "
              "a scale reduction makes the throughput result ineligible. All losses compute fp32 log probabilities/CE. "
              "Chunked loss checkpoints output-head/CE chunks and recomputes each in backward, avoiding the full logits allocation "
              "and retention of all chunk softmax activations. The transformer is unchanged. Chunk recomputation costs additional compute.", "",
              "## Every executed benchmark configuration", "",
              "GPU records below include unavailable/error attempts. Only `benchmark_eligible=true` full d8 MPS records can win.", "",
              "| Device / kind | Configuration | Status | Input tok/s | s/full step | Projected total h | Sampled MPS allocated GiB | MFU |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in gpu + gpu_smoke + smoke:
        opts = r["options"]
        description = label(opts) if r["kind"] != "roofline" else f'{r["dtype"]} {r["matmul_size"]}³, matmul={opts["matmul_precision"]}'
        is_smoke = r.get("smoke", False)
        total = r.get("projected_total_seconds")
        peak = r.get("sampled_peak_allocated_bytes")
        step = r.get("sec_per_step") if not is_smoke else None
        tok = r.get("tokens_per_second")
        mfu = r.get("mfu")
        lines += [f'| {opts["device"]} / {"tiny smoke" if is_smoke else r["kind"]} | {description} | {r["status"]} | '
                  f'{num(tok,1)}{" (tiny)" if is_smoke and tok else ""} | {num(step)} | '
                  f'{num(total/3600 if total is not None else None)} | '
                  f'{num(peak/2**30 if peak is not None else None)} | {num(100*mfu if mfu is not None else None)}{"%" if mfu is not None else ""} |']
    lines += ["", "CPU correctness configurations (no throughput timing was performed):", "",
              "| Precision | Loss | Attention | Tiny micro-batch | Optimizer | Status | tok/s |",
              "|---|---|---|---:|---|---|---|"]
    for r in cpu:
        if "options" not in r:
            continue
        o = r["options"]
        tiny_micro = max(1, min(8, o["micro_batch"] // 8)) if r["kind"] == "correctness" else o["micro_batch"]
        lines += [f'| {o["precision"]} | {o["loss"]} | {o["attention"]} | {tiny_micro} | {o["optimizer"]} | {r["status"]} | unmeasured |']
    lines += ["", "## Projection, roofline and MFU", "",
              f'For N={f["parameters"]:,}, B=64, T=1024, L=8, d=512, the dense-attention convention is '
              f'`F_step = 6*N*B*T + 12*L*B*T^2*d = {f["training_flops"]:,}` FLOPs '
              f'({f["training_flops"]/1e12:.6f} TFLOP/update). '
              'The attention term counts forward and backward QK/AV products over the dense square. '
              'A triangular useful-work convention halves that term; neither convention includes elementwise operations or output-head recomputation. '
              '`MFU = (F_step / seconds_per_step) / (measured matching-dtype matmul TFLOP/s × 1e12)`. '
              'This is utilization relative to an achieved square-matmul roofline, not advertised hardware peak; mixed compute is approximated by its dominant matmul dtype.', "",
              "Matmuls use preallocated outputs, five warmups and at least 20 measured iterations with MPS synchronization. "
              "The harness measures bf16, fp32 and fp16 roofs, and records the float32 matmul precision setting. "
              "MFU stays unmeasured when a matching roof is absent; include `--roofline` in focused sweeps.", "",
              "| Matmul dtype | fp32 matmul setting | Achieved TFLOP/s |", "|---|---|---:|"]
    for (dtype, precision), value in roofs.items():
        lines += [f"| {dtype} | {precision} | {value:.4f} |"]
    if not roofs:
        lines += ["| bf16 / fp32 / fp16 | highest | unmeasured |"]
    lines += ["", "`projected_total_seconds = 9375*sec_per_step + 100*seconds_per_validation_pass`. "
              "Validation is measured using 20 synchronized micro-batch forwards after five warmups, with the 1000-sequence "
              "remainder timed separately. The projection includes all 100 evaluations. Throughput counts B*T input tokens, "
              "consistent with the assignment token budget, while the loss predicts B*(T−1) labels.", "",
              "## Limitations", "",
              "- No MPS access in this sandbox; an external tiny smoke test passed, but full-recipe GPU optimization choices remain implemented experiments, not measured wins. "
              "torch.compile, fused AdamW and precision/kernel support must be established by the external sweep.",
              "- MPS in this PyTorch build exposes current/driver allocation counters but no public exact peak/reset counter. "
              "The harness samples both every 10 ms and also reports process high-water RSS. MPS figures are sampled lower bounds; "
              "RSS includes CPU runtime and unified memory and must not be added to MPS allocation as if disjoint.",
              "- Synthetic batches are resident on the device and repeated. The projection excludes dataset decoding/transfers, "
              "network/W&B logging and module RMS metrics, checkpoint serialization, startup/compilation, and thermal drift over a full run. "
              "The reference saves latest checkpoints at steps 1000…9000 and at the final step, plus its completed artifact. "
              "These require separate measurements for a complete end-to-end wall-clock prediction.",
              "- `train_mps.py --steps 9375` executes the recipe on a fixed synthetic batch. It does not implement DCLM data loading, "
              "periodic checkpoint persistence or W&B; the benchmark measures periodic evaluation separately. "
              "Long-run convergence and sustained thermal performance are unverified.",
              "- Report generation is deterministic with respect to input records apart from its timestamp. Regenerate after "
              "the external coordinator replaces/appends GPU records; no nonexistent timing is inferred.", ""]
    (ROOT / "tasks/T1.report.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(ROOT / "tasks/T1.report.md"), "gpu_measurements": len(eligible),
                      "cpu_checks": len(cpu), "best": best["options"] if best else None}))


if __name__ == "__main__":
    build()
