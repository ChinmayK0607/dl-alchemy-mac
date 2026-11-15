"""CSV and final eval-loss comparison from completed runs (or --results sweep JSONL)."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--results", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/T3_comparison.png"))
    p.add_argument("--include-verification", action="store_true")
    a = p.parse_args()
    if a.results:
        summaries = [json.loads(line) for line in a.results.read_text().splitlines() if line.strip()]
    else:
        summaries = [json.loads(f.read_text()) for f in sorted(a.runs_dir.rglob("summary.json"))]
    rows = []
    for s in summaries:
        if s.get("status") != "complete" and not (a.include_verification and s.get("status") == "verification_complete"):
            continue
        c = json.loads((Path(s["run_dir"]) / "config.json").read_text())
        rows.append(dict(run_dir=s["run_dir"], model=c["model_name"], learning_rate=c["learning_rate"],
                         schedule=c["lr_schedule"], dropout=c["dropout"], seed=c["model_seed"],
                         num_train_sequences=c["num_train_sequences"], precision=c["precision"], data_seed=c["data_seed"],
                         final_eval_loss=s["final_eval_loss"], tokens=s["tokens"],
                         wall_time_seconds=s["wall_time_seconds"], verification=c["verification"]))
    if not rows:
        p.error("No completed runs (verification runs are excluded by default)")
    rows.sort(key=lambda r: r["final_eval_loss"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep font discovery/cache writes within the workspace on restricted Macs.
    cache=Path(".build/cache").resolve()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME",str(cache))
    os.environ.setdefault("MPLCONFIGDIR",str(cache/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['model']} N={r['num_train_sequences']} lr={r['learning_rate']} {r['schedule']} p={r['dropout']} seed={r['seed']}"
              for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(3, .4*len(rows)+1)))
    ax.barh(range(len(rows)), [r["final_eval_loss"] for r in rows], color="#3478b9")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final validation NLL (nats per next token; lower is better)")
    ax.set_title("CPU pipeline verification" if all(r["verification"] for r in rows) else "DCLM local experiments")
    fig.tight_layout()
    fig.savefig(a.output, dpi=160)
    plt.close(fig)
    print(json.dumps(dict(output=str(a.output), csv=str(a.output.with_suffix('.csv')), rows=len(rows))))


if __name__ == "__main__":
    main()
