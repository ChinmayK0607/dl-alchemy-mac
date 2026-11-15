# T5: make dl-alchemy-mac a portable git repo (for a bigger Mac)

Read tasks/COMMON.md (sandbox rules unchanged: CPU only; no git commits/pushes — the coordinator commits).
The coordinator will `git init` this directory and push it to a private GitHub repo. Another Mac (likely M5 Pro) will
clone it fresh. Make that work with no edits.

1. No absolute paths anywhere in code, tests, plans or READMEs (grep for /Users/). The course reference repo
   (https://github.com/deep-learning-alchemy/assignments, branch a1_release, commit
   baafc3f2d364e5ddbba75f0fe8d1cb29326920e5) becomes a git submodule at `reference/` — you can't add it (no git), so:
   resolve the reference path as env `DL_ALCHEMY_REFERENCE` > `<repo>/reference` > `<repo>/../dl-alchemy-a1`, in ONE helper
   (e.g. mlx_port/paths.py) used by every module (mlx_port tests, parity_check, torch_mps). Give a clear error if missing.
   Training (run.py/queue.py) must not need the reference at all — verify.
2. pyproject.toml for `uv sync` (python 3.12; pin what .venv has: mlx 0.32.4, torch 2.15.0, wandb 0.31.0, numpy,
   matplotlib, safetensors/sentencepiece only if imported). Mark torch as needed only for parity/torch_mps if feasible
   (optional extra "parity"), but keep it simple and correct.
3. MLX memory limit: never hardcode 18/14 GB. Default = a safe fraction of mx.device_info()['max_recommended_working_set_size'];
   `--memory-gb` still overrides. (On this M4 24 GB, 18 GB failed; 14 GB worked.) Apply to bench.py and run.py.
4. scripts/setup.sh: idempotent — `git submodule update --init`, `uv sync`, download data
   (`python -m mlx_port.data --num-train-sequences 100000 --data-seed 42`; flag to choose N, note 600000 = full A1 run),
   print MLX device info. scripts/check.sh: CPU-free quick GPU checks the user runs on the new Mac:
   roofline matmul, bench d8 mp micro-batch 8, smoke preset — writing results to results/<hostname>/.
5. .gitignore: .venv, data/, runs/, wandb/, __pycache__, .build/, *.safetensors, logs/*.log (keep logs dir out entirely).
   Keep results/ (small JSON/MD/PNG) and tasks/ (they are the project history).
6. Remove the CPU-only MLX build path from tests/docs if it depends on the deleted .build/mlx-cpu (keep tests runnable on GPU
   machines: tests should use the normal mlx and default device; document `MLX_DEVICE=cpu` style if needed).
7. Run whatever tests you can on CPU (note: normal pip mlx may not work in your sandbox — if so, say which tests you
   could not run). Report: tasks/T5.report.md with a file list of what changed.
