"""Summarize observed pop-quiz results; never infer missing experiments."""
import argparse
import json
import math
from pathlib import Path

Q1 = ("reference", "no_prenorm", "no_residuals", "no_mlp", "no_attention", "no_rope")
VARIANTS = Q1 + ("mixed_residual",)


def summarize(source, output_dir, token_budget_sequences=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in Path(source).read_text().splitlines() if Path(source).exists() else []:
        if line.strip():
            row = json.loads(line)
            if row.get("verification") or row.get("status") == "verification_complete":
                continue
            if token_budget_sequences is not None and row.get("config", {}).get("num_train_sequences") != token_budget_sequences:
                continue
            rows.append(row)
    # Resume appends duplicate summaries; each run contributes exactly once.
    rows = list({r.get("run_dir", f"row-{i}"): r for i, r in enumerate(rows)}.values())
    signatures = set()
    for r in rows:
        c = r.get("config", {})
        signature = {k:v for k,v in c.items() if k not in ("variant", "learning_rate", "run_name_suffix")}
        signatures.add(json.dumps(signature, sort_keys=True))
    if len(signatures) > 1:
        raise ValueError("Incompatible recipe settings in results; use a separate results file or --token-budget-sequences")

    def variant(r):
        return r.get("config", {}).get("variant", r.get("variant"))
    def lr(r):
        return r.get("config", {}).get("learning_rate", r.get("learning_rate"))
    def value(r):
        x = r.get("final_val_loss", r.get("final_eval_loss"))
        return float(x) if x is not None else float("nan")
    def good(r):
        return r.get("status") == "complete" and math.isfinite(value(r))

    best = {}
    for v in VARIANTS:
        options = [r for r in rows if variant(r) == v and good(r)]
        if options:
            best[v] = min(options, key=value)
    # Q1's LR grid excludes the extra q2 reference at 1e-4.
    q1_best = {}
    for v in Q1:
        options = [r for r in rows if variant(r) == v and lr(r) != .0001 and good(r)]
        if options:
            q1_best[v] = min(options, key=value)
    ranking = sorted(q1_best, key=lambda v: value(q1_best[v]))
    lines = ["# Pop-quiz measured results", "", f"Unique recorded runs: {len(rows)} / 37. Verification runs excluded.",
             "Missing or failed runs do not supply a validation loss. Rankings are provisional until the grid completes.", "",
             "## Q1", "", "| Rank | Variant | Best final validation loss | LR | Finite / recorded |",
             "|---:|---|---:|---:|---:|"]
    for v in ranking + [v for v in Q1 if v not in ranking]:
        observed = [r for r in rows if variant(r) == v and lr(r) != .0001]
        finite = sum(good(r) for r in observed)
        if v in q1_best:
            b = q1_best[v]
            lines.append(f"| {ranking.index(v)+1} | {v} | {value(b):.6f} | {lr(b):g} | {finite} / {len(observed)} |")
        else:
            lines.append(f"| — | {v} | unavailable | — | {finite} / {len(observed)} |")
    lines += ["", "## Q2", ""]
    if all(v in best for v in ("mixed_residual", "reference")):
        delta = value(best["mixed_residual"]) - value(best["reference"])
        lines.append(f"ΔL = min_lr L_mixed − min_lr L_reference = **{delta:+.6f}** nats (observed finite runs).")
        lines.append(f"Best mixed LR: {lr(best['mixed_residual']):g}; best reference LR: {lr(best['reference']):g}.")
    else:
        delta = None
        lines.append("ΔL unavailable: finite completed runs for both mixed_residual and reference are required.")
    lines += ["", "## Run grid", "", "| Variant | LR | Status | Final validation loss |", "|---|---:|---|---:|"]
    for r in sorted(rows, key=lambda r: (str(variant(r)), lr(r) or 0)):
        label = f"{value(r):.6f}" if good(r) else "non-finite" if r.get("status") == "diverged" else "unavailable"
        lines.append(f"| {variant(r)} | {lr(r):g} | {r.get('status')} | {label} |")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, v in zip(axes.flat, VARIANTS):
        points = sorted([r for r in rows if variant(r) == v and good(r)], key=lr)
        ax.plot([lr(r) for r in points], [value(r) for r in points], "o-")
        ax.set(xscale="log", xlabel="Learning rate", ylabel="Final validation NLL", title=v)
        diverged = [r for r in rows if variant(r) == v and r.get("status") == "diverged"]
        if diverged:
            ax.text(.02,.98,"Non-finite at LR: " + ", ".join(f"{lr(r):g}" for r in diverged),
                    transform=ax.transAxes, va="top", fontsize=8)
        if not points:
            ax.text(.5,.5,"No finite completed runs",transform=ax.transAxes,ha="center")
    for ax in list(axes.flat)[len(VARIANTS):]:
        ax.set_visible(False)
    fig.savefig(output_dir / "popquiz_loss_vs_lr.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
    curves = 0
    for v,b in best.items():
        metrics = Path(b["run_dir"]) / "metrics.jsonl"
        if not metrics.exists():
            continue
        points = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        points = [p for p in points if math.isfinite(p.get("val_loss", p.get("eval_loss", float("nan"))))]
        if points:
            ax.plot([p.get("tokens", p["step"]) for p in points],
                    [p.get("val_loss",p.get("eval_loss")) for p in points],"o-",label=f"{v}, LR {lr(b):g}")
            curves += 1
    ax.set(xlabel="Input tokens",ylabel="Validation NLL",title="Validation curves at best observed LR")
    if curves:
        ax.legend()
    else:
        ax.text(.5,.5,"No completed validation curves",transform=ax.transAxes,ha="center")
    fig.savefig(output_dir / "popquiz_best_curves.png", dpi=160)
    plt.close(fig)
    lines += ["", "![Final validation loss versus LR](popquiz_loss_vs_lr.png)", "",
              "![Validation curves at best LR](popquiz_best_curves.png)", ""]
    (output_dir / "popquiz_summary.md").write_text("\n".join(lines))
    return dict(runs=len(rows), q1_ranking=ranking, delta_loss=delta, best={v:lr(b) for v,b in best.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/popquiz.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--token-budget-sequences", type=int)
    a = p.parse_args()
    print(json.dumps(summarize(a.results,a.output_dir,a.token_budget_sequences)))


if __name__ == "__main__":
    main()
