# Deep Learning Alchemy on Apple Silicon

MLX training and measured MLX/PyTorch MPS benchmarks for Stanford CS 312 A1.
Targets Apple Silicon macOS with Metal access and Python 3.12. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Git first.

## Fresh Mac

Clone this repository, then run from its root:

```sh
scripts/setup.sh
scripts/check.sh
```

Setup initializes the pinned course submodule, runs `uv sync --locked`, downloads
100,000 seed-42 globally shuffled DCLM sequences plus all 1,000 validation
sequences, and prints MLX device information. Repeating the command reuses the
completed cache; interrupted downloads resume with the same options. The default
download is about 207 MB of token payload with a zero merge gap. No model weights,
data, virtual environment, or run logs belong in git.

`scripts/check.sh` runs GPU-only 4096-square fp32/bf16 matmul rooflines, the full
d8 / mp / micro-batch 8 benchmark (global batch 64 × context 1024, 5 warmup and
20 measured updates), and one real-data smoke update plus validation. Expect
several minutes for the d8 check. Each check has a timeout; failed workers produce
records and the script exits nonzero if any check fails. Small JSON artifacts go
to `results/<hostname>/<UTC timestamp>/`; smoke run artifacts go to `runs/`.
No W&B account or PyTorch installation is required for these checks.

```sh
# Optional parity/PyTorch MPS tools:
scripts/setup.sh --parity
# Or install just the extra after setup:
uv sync --locked --extra parity

# Full A1 data on a fresh checkout (600000 sequences = 614.4M input tokens):
scripts/setup.sh --num-train-sequences 600000
# If a smaller completed cache already exists, choose a new directory:
scripts/setup.sh --num-train-sequences 600000 --data-dir data/a1-full
```

An existing completed prefix is immutable; it cannot be extended in place.
Pass the matching `--data-dir` to training if you chose a different directory.
The smaller local preset needs only 20,480 sequences; downloading 100,000 does
not change its token budget.

## Training and correctness

```sh
.venv/bin/python -m mlx_port.run --preset local --device gpu --wandb disabled
# Full default A1 recipe, accumulated in micro-batches of 8:
.venv/bin/python -m mlx_port.run --preset reference --micro-batch 8 \
  --device gpu --data-dir data/a1-full --wandb disabled
.venv/bin/python -m mlx_port.queue --dry-run

# Parity extra required; tests use normal MLX and its default device:
.venv/bin/python -m mlx_port.parity_check
.venv/bin/python -m mlx_port.test_t3
.venv/bin/python -m mlx_port.test_variants
.venv/bin/python -m mlx_port.test_t4
# Import-free portability tests run on CPU without Metal or the parity extra:
.venv/bin/python -m unittest mlx_port.test_portable -v
```

`run` and `queue` train independently of the course checkout and PyTorch.
Parity tests and all `torch_mps` tools resolve the course source in one place,
`mlx_port/paths.py`: `DL_ALCHEMY_REFERENCE`, then `reference/` in this repository,
then the legacy sibling `../dl-alchemy-a1`. A bad explicit override is an error.
Use `DL_ALCHEMY_REFERENCE=path/to/checkout` to select another complete checkout.

MLX allocation and wired limits default to **75% of the device's recommended
working set**, adapting to the Mac's memory. `--memory-gb N` overrides the cap in
GiB for `run`, `bench`, `queue`, `sweep`, and `scripts/check.sh`. Effective limits
are recorded in training configs and benchmark rows. They are execution settings,
not model-recipe changes. CPU runs do not configure Metal memory limits.

To request CPU MLX on a Mac with an accessible Metal runtime, prefix a test with
`MLX_DEVICE=cpu`. This variable is implemented by the test harness; training uses
`--device cpu`. Normal MLX may initialize Metal during import even in CPU mode,
so this does not make sandboxed/headless sessions with no Metal device work.
Some runner lifecycle tests deliberately use short CPU-only verification shapes.
There is no custom MLX build or library-path override. PyTorch CPU checks can run
without Metal via `.venv/bin/python torch_mps/verify_cpu.py`.

See [MLX usage](mlx_port/README.md), [MPS benchmarks](torch_mps/README.md), and
[task reports](tasks/) for numerical checks, measurements and limitations. Prior
M4 measurements remain historical; measure the new Mac before projecting time.

## Coordinator: register the reference before the first push

This task does not run git commands or create the submodule. After initializing
this repository, the coordinator must run:

```sh
git submodule add -b a1_release https://github.com/deep-learning-alchemy/assignments reference
git -C reference checkout baafc3f2d364e5ddbba75f0fe8d1cb29326920e5
```

Include the generated `.gitmodules` and `reference` gitlink in the coordinator's
commit. The gitlink pins this exact commit; `a1_release` documents the source
branch. Fresh clones then need no path edits. Setup uses `git submodule update
--init --recursive`, which follows the pinned gitlink rather than a moving branch.
`results/` and `tasks/` are intentionally retained as project history.
