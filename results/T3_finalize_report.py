from pathlib import Path
import json
import re

root=Path.cwd()
report=root/'tasks/T3.report.md'
s=report.read_text()
cache=root/'data/dclm'
m=json.loads((cache/'train_ds42/metadata.json').read_text())
files={str(p.relative_to(cache)):p.stat().st_size for p in cache.rglob('*') if p.is_file()}
corpus_payload=m['download_payload_bytes']+2048000
materialized=sum(files.values())
body=f'''Download completed: **100,000 train sequences + exactly 1,000 validation
sequences**. Main token transfer payload: **{corpus_payload:,} bytes**
({m['download_payload_bytes']:,} train range bytes + 2,048,000 validation bytes),
plus 2,104 source manifest/metadata bytes. The independent 32-row network
cross-check downloaded another 65,536 token bytes; metadata/probe traffic and
HTTP headers are additional small overhead. Final `data/dclm/` file sizes total
**{materialized:,} bytes** (well below 2 GB); train tokens themselves are
204,800,000 bytes. Coalescing fetched 135,661 source row slots to obtain the
100,000 requested rows without transferring the full source file.

| Integrity record | SHA256 |
|---|---|
| Downloaded training prefix | `{m['tokens_sha256']}` |
| Original source row IDs | `{m['source_rows_sha256']}` |
| Complete validation tokens (publisher digest verified) | `ab657da123009fdd93fc972cc0a8e97308143355ee42720ca843fc8540eea353` |

The range plan has 90,895 requests; the one-time transfer took about 21 minutes.
This preparation time is separate from the per-experiment GPU projections.
See `results/T3_download.log` and the source/cache metadata under `data/dclm/`.
'''
s=re.sub(r'<!-- DATA_RESULTS -->.*?<!-- END_DATA_RESULTS -->',lambda _: '<!-- DATA_RESULTS -->\n'+body+'<!-- END_DATA_RESULTS -->',s,flags=re.S)
tests=json.loads((root/'results/T3_tests.json').read_text())
assert tests['status']=='passed' and tests['real_data_checked']
by={r['test']:r for r in tests['records']}
assert 'independent_remote_source_rows' in by
parity=json.loads((root/'results/mlx_parity.json').read_text())
assert parity['status']=='passed'
summaries=[]
for directory in ('runs/T3-cpu','runs/T3-cpu-sweep'):
 for f in sorted((root/directory).rglob('summary.json')):
  summary=json.loads(f.read_text())
  config=json.loads((f.parent/'config.json').read_text())
  summaries.append((summary,config))
assert len(summaries)==4
sweep=[json.loads(l) for l in (root/'results/T3_cpu_sweep.jsonl').read_text().splitlines()]
assert [r['status'] for r in sweep]==['error','verification_complete','verification_complete']
baseline=json.loads((root/'results/T3_unigram.json').read_text())
body=f'''**All tests passed on CPU.** Evidence: `results/T3_tests.json`,
`results/mlx_parity.json`, `results/T3_d4_full_context.json`,
`results/T3_pipeline.log`, `results/T3_cpu_sweep.jsonl`.

- Existing parity suite: {len(parity['records'])} checks, including d8/full-context
  logits, masks/positions, gradients, 20-step optimizer trajectories and
  fp32/mp/bf16, compilation/chunking/checkpoint options.
- Real d4, 1×1024 DCLM forward with reference weights: logits max absolute
  difference 4.3273e-5.
- Schedules: {by['schedules_vs_reference']['points']:,} points against reference,
  all schedules/aliases, multiple budgets/warmup fractions; maximum absolute
  LR difference {by['schedules_vs_reference']['max_abs_diff']:.3g}.
- Global shuffle semantics: 240 token-exact synthetic batch comparisons for
  seeds None/0/42/123 and prefixes 23/128/257, including partial batches.
- Real cache: {by['real_dclm_first_batches_token_exact']['batch_comparisons']}
  token-exact first-batch comparisons with the unmodified reference loader,
  train prefixes 17/1024/20480/100000 and the validation split; row indices
  also match the reference's full-source permutation. Independently re-fetched
  32 original remote rows and compared every token against the cached prefix.
- Shared-mask dropout: six sites on the two-layer fixture; logits difference
  1.85e-6, gradients 2.54e-7, dropout-disabled eval difference 2.03e-6.
  Compiled RNG advances, reseeding reproduces the curve, and repeated eval is
  deterministic. Nonzero dropout/mp also runs through the real-data CPU sweep.
- Non-default AdamW settings (LR .002, decay .23, betas .8/.97, clip .3,
  cosine/warmup) and partial accumulation: seven updates; loss difference
  2.39e-7, final weights difference 1.46e-5.
- Partial validation weighting; preset/default intervals; reference defaults
  and d4..d9 architecture equality; unique run names; CLI aliases/nullable
  values, precedence and invalid-config rejection.
- Synthetic runner integration: two updates, partial eval batch, all required
  files, 5,901,056 fp32 checkpoint parameters, reloaded evaluation difference
  **0**, and overwrite protection.
- Both real-data presets run three CPU updates and save metrics/final weights.
  A sequential CPU sweep records an intentional invalid-dropout error then
  completes two valid trials. CSV/PNG summary generation passes.

CPU pipeline configurations below use the actual d4 model, vocabulary 4096,
global batch 64, micro-batch 8 and mp, but **crop context to 32 and evaluate
16 validation rows** to keep sandbox verification small. Their schedule/token
budgets stay those of the selected preset. They are explicitly tagged
`verification_complete`, excluded from default plots, and are **not** timing
benchmarks or GPU/local convergence results. Reported tok/s includes first
compilation and is only the observed pipeline rate.

| CPU configuration | Updates | Input tokens | Pipeline tok/s | Wall s | Final cropped eval NLL | Peak MLX MiB |
|---|---:|---:|---:|---:|---:|---:|
'''
for summary,c in summaries:
 label=('smoke' if c['num_train_sequences']==1024 else 'local')
 if c.get('run_name_suffix'):
  label+=f" LR {c['learning_rate']}, {c['lr_schedule']}, dropout {c['dropout']}"
 body+=f"| {label} | {summary['steps']} | {summary['tokens']} | {summary['tokens_per_second']:.1f} | {summary['wall_time_seconds']:.2f} | {summary['final_eval_loss']:.6f} | {summary['peak_memory_bytes']/1024**2:.1f} |\n"
body+='''| Intentional invalid dropout=1 | 0 | 0 | — | — | expected validation error | — |
| Synthetic integration fixture, context 16, eval 9 | 2 | 2048 | 414.0 (first observed run) | 8.25 | 6.008547 | 182.2 |

Actual CPU commands (the final test command additionally checks remote rows):

```sh
MLX_DEVICE=cpu \\
  .venv/bin/python -m mlx_port.parity_check
MLX_DEVICE=cpu \\
  .venv/bin/python -m mlx_port.test_t3 --verify-source
# Repeat with --preset local; use a fresh output directory to rerun.
MLX_DEVICE=cpu \\
  .venv/bin/python -m mlx_port.run --preset smoke --device cpu --max-steps 3 \\
  --verification-context 32 --verification-val-sequences 16 --output-dir runs/T3-cpu
MLX_DEVICE=cpu \\
  .venv/bin/python -m mlx_port.sweep --preset local --device cpu \\
  --configs results/T3_cpu_grid.json --max-steps 2 --verification-context 32 \\
  --verification-val-sequences 16 --output-dir runs/T3-cpu-sweep \\
  --results results/T3_cpu_sweep.jsonl --timeout 120
.venv/bin/python -m mlx_port.baseline
```

'''
body+=f"Training-unigram held-out baseline (20,480 train rows, add-one smoothing): **{baseline['train_unigram_val_nll_nats']:.6f} nats**; empirical validation unigram entropy: **{baseline['empirical_val_unigram_entropy_nats']:.6f} nats**. Uniform loss is **{baseline['uniform_nll_nats']:.6f} nats**. See `results/T3_unigram.json`.\n\n"
body+='''CPU plot and CSV: `results/T3_cpu_verification.png` and
`results/T3_cpu_verification.csv`. They are pipeline diagnostics only.
'''
s=re.sub(r'<!-- TEST_RESULTS -->.*?<!-- END_TEST_RESULTS -->',lambda _: '<!-- TEST_RESULTS -->\n'+body+'<!-- END_TEST_RESULTS -->',s,flags=re.S)
report.write_text(s)
print('Final report written:',report)
