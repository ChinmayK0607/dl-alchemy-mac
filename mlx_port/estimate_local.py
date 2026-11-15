"""Conservative GPU projections grounded in coordinator measurements, not CPU timings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import make_config


def architecture(depth):
    d = 64 * depth
    # Two vocabulary matrices; 4 attention + 3 SwiGLU matrices per layer;
    # two residual RMSNorm scales, two head_dim QK scales, final norm.
    params = 2*4096*d + depth*(4*d*d+3*d*int(3.5*d)+2*d+2*64)+d
    flops = 6*params*64*1024 + 12*depth*64*1024**2*d
    return params, flops


def projection(c, model_tflops):
    depth = int(c.model_name[1:])
    params, flops = architecture(depth)
    # Forward ~one third of train FLOPs per sequence; 1000 eval rows.
    eval_flops = flops/3/64*1000
    eval_passes = len([s for s in range(1,c.total_steps+1) if s % c.eval_steps == 0 or s == c.total_steps])
    training = c.total_steps*flops/model_tflops/1e12
    evaluation = eval_passes*eval_flops/model_tflops/1e12
    return dict(parameters=params, flops_per_update=flops, assumed_model_tflops=model_tflops,
                assumed_tokens_per_second=64*1024/(flops/model_tflops/1e12),
                tokens_per_parameter=c.num_train_sequences*1024/params, eval_passes=eval_passes,
                projected_train_seconds=training, projected_eval_seconds=evaluation,
                startup_io_allowance_seconds=20, projected_total_seconds=training+evaluation+20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, default=Path("results/mlx_sweep.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/T3_projections.json"))
    a=p.parse_args()
    measured=[json.loads(l) for l in a.measurements.read_text().splitlines() if l.strip()]
    baseline=[r for r in measured if r.get("kind")=="train" and r.get("status")=="ok" and r.get("micro_batch")==8]
    roofline=next(r["tflops"] for r in measured if r.get("kind")=="roofline" and r["precision"]=="bf16" and r["status"]=="ok")
    # Deliberately lower achieved throughput for small models: ~29-39% of the
    # measured bf16 GEMM roofline vs d8's measured ~60-62%.
    rates=(.9,1.2)
    result=dict(formula="6*N*B*S + 12*L*B*S*S*d", measured_d8_micro8=baseline,
                bf16_matmul_roofline_tflops=roofline,
                projected_small_model_mfu_range=[r/roofline for r in rates], presets={})
    for preset in ("local","smoke"):
        c=make_config(preset)
        result["presets"][preset]=dict(config=asdict(c),
            conservative=projection(c,rates[0]), optimistic=projection(c,rates[1]))
    # 16-run P2 grid: d4,d5 x 2 LR x 2 schedule x p=0,.1. Nonzero
    # dropout uses explicit attention; allow 1.5-2.5x runtime for those runs.
    total_low=total_high=0.
    for depth in (4,5):
        c=make_config("local",dict(model_name=f"d{depth}"))
        low=projection(c,rates[1])["projected_total_seconds"]
        high=projection(c,rates[0])["projected_total_seconds"]
        total_low += 4*low + 4*low*1.5
        total_high += 4*high + 4*high*2.5
    result["p2_16_run_projection_seconds"]=[total_low,total_high]
    result["caveat"]="Small-model rates and dropout penalties are assumptions; no small-model GPU measurement or convergence claim."
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
