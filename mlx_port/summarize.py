"""Refresh T2.report.md evidence after coordinator JSONL results arrive."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=Path("tasks/T2.report.md"))
    p.add_argument("--results", type=Path, nargs="*", default=[
        Path("results/mlx.jsonl"), Path("results/mlx_refine.jsonl"),
        Path("results/mlx_cpu_smoke.jsonl")])
    p.add_argument("--parity", type=Path, default=Path("results/mlx_parity.json"))
    args = p.parse_args()
    report = args.report.read_text()
    def section(name, text):
        nonlocal report
        start, end = f"<!-- {name} -->", f"<!-- END_{name} -->"
        block = start+"\n\n"+text+"\n\n"+end
        if end in report:
            report = re.sub(re.escape(start)+r".*?"+re.escape(end), lambda _: block, report, flags=re.S)
        else:
            report = report.replace(start, block)
    parity = json.loads(args.parity.read_text())
    lines = [f"Final parity status: **{parity['status']}** on CPU.", "",
             "| Check | Result |", "|---|---|"]
    for row in parity["records"]:
        test = row["test"]
        if test in ("forward", "full_context"):
            text = f"logits max abs {row['logits_max_abs_diff']:.8g}; loss abs {abs(row['torch_loss']-row['mlx_loss']):.8g}"
        elif test == "adamw_curve":
            text = f"20-step max loss abs {row['max_loss_diff']:.8g}; final weight max abs {row['final_weights_max_abs_diff']:.8g}"
        elif test in ("mask_position", "gradients"):
            text = f"max abs {row.get('logits_max_abs_diff', row.get('max_abs_diff')):.8g}"
        elif test == "precision_stability":
            test += " "+row["precision"]
            text = f"loss {row['initial_loss']:.7f} → {row['final_loss']:.7f}; curve diff vs fp32 {row['max_loss_diff_vs_fp32']:.7g}"
        else:
            test += f" {row['precision']}, compile={row['compiled']}, CE={row['chunk_size']}, {row['boundary']}, checkpoint={row['checkpoint']}"
            text = f"loss {row['initial_loss']:.7f} → {row['final_loss']:.7f}; curve diff vs fp32 {row.get('max_loss_diff_vs_fp32', 0):.7g}"
        lines.append(f"| {row.get('model', '')} {test} | {text} |")
    section("PARITY_RESULTS", "\n".join(lines))

    rows = []
    lines = ["| File:line | Device/workload | Configuration | Status | Input tok/s | s/step | Peak MLX MiB | GEMM TFLOP/s |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for path in args.results:
        if not path.exists():
            continue
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            r = json.loads(raw)
            rows.append(r)
            if r["kind"] == "roofline":
                config = f"{r['precision']} GEMM {r['size']}³"
            else:
                config = (f"{r['precision']}, mb={r['micro_batch']}, compile={r['compiled']}, "
                          f"CE={r['chunk_size']}, {r['boundary']}, checkpoint={r['checkpoint']}")
            def fmt(key, digits=3):
                return f"{r[key]:.{digits}f}" if key in r else "—"
            peak = f"{r['peak_memory_bytes']/1024**2:.2f}" if "peak_memory_bytes" in r else "—"
            workload = r.get("model", "smoke" if r.get("smoke") else "d8/GEMM")
            status = r["status"]
            if r.get("error_type"):
                status += ": "+r["error_type"]
            lines.append(f"| {path.name}:{number} | {r['device']} / {workload} | {config} | {status} | {fmt('tokens_per_second', 1)} | {fmt('seconds_per_step', 6)} | {peak} | {fmt('tflops')} |")
    section("BENCHMARK_TABLE", "\n".join(lines))
    valid = [r for r in rows if r.get("kind") == "train" and r.get("status") == "ok"
             and r.get("device") == "gpu" and r.get("projections_valid")
             and r.get("global_batch") == 64 and r.get("context") == 1024]
    if valid:
        best = min(valid, key=lambda r:r["projected_total_seconds"])
        flags = (f"--precisions {best['precision']} --micro-batches {best['micro_batch']} "
                 f"--compiled {str(best['compiled']).lower()} --chunk-sizes {best['chunk_size']} "
                 f"--boundaries {best['boundary']} --checkpoints {str(best['checkpoint']).lower()}")
        roof = best.get("matmul_roofline_tflops")
        if roof is None:
            precision = "fp32" if best["precision"] == "fp32" else "bf16"
            matches = [r for r in rows if r.get("kind") == "roofline" and r.get("status") == "ok"
                       and r.get("device") == "gpu" and not r.get("smoke") and r.get("size") == 4096
                       and r.get("precision") == precision]
            if matches:
                roof = matches[-1]["tflops"]
        utilization = f"{100*best['achieved_model_tflops']/roof:.2f}%" if roof else "unavailable (run matching roofline)"
        summary = (f"Measured best among {len(valid)} successful full-recipe GPU trials: **{best['precision']}, "
                   f"micro-batch {best['micro_batch']}, compiled={best['compiled']}, CE chunk {best['chunk_size']}, "
                   f"boundary {best['boundary']}, checkpoint={best['checkpoint']}**.\n\n"
                   f"Throughput **{best['tokens_per_second']:.1f} input tok/s**, "
                   f"**{best['seconds_per_step']:.6f} s/update**; projected train "
                   f"**{best['projected_train_seconds']/3600:.3f} h**, evaluation "
                   f"**{best['projected_eval_seconds']/3600:.3f} h**, total "
                   f"**{best['projected_total_seconds']/3600:.3f} h**. MFU: **{utilization}**.\n\n"
                   f"Final reproduction command:\n\n```sh\n.venv/bin/python -m mlx_port.bench --device gpu {flags} --output results/mlx.jsonl\n```\n\n"
                   "The measured ranking is limited to successful configurations in the supplied result files.")
    else:
        summary = ("**Measured fastest GPU configuration: unavailable. Projected 9375-step GPU time: "
                   "unavailable. GPU matmul roofline and MFU: unavailable.**\n\n"
                   "The installed Metal-backed MLX import reports `No Metal device available`. "
                   "All successful timings so far are CPU smoke checks. No CPU-to-GPU extrapolation is used. "
                   "A conservative, unranked candidate is mp, micro-batch 2 (32 accumulation steps), "
                   "compiled, CE chunk 1024, micro boundary, no activation checkpointing. "
                   "It must be compared with the coordinator sweep before being called fastest.\n\n"
                   "After GPU results arrive, refresh the tables and selection with "
                   "`.venv/bin/python -m mlx_port.summarize`.")
    section("GPU_SUMMARY", summary)
    args.report.write_text(report)


if __name__ == "__main__":
    main()
