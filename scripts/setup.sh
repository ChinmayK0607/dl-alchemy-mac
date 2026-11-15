#!/bin/sh
# Run from any directory. The coordinator must register reference/ as a submodule.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

sequences=100000
data_dir=data/dclm
parity=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --num-train-sequences|--data-dir)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --num-train-sequences) sequences=$2 ;;
                --data-dir) data_dir=$2 ;;
            esac
            shift 2 ;;
        --parity) parity=true; shift ;;
        --help|-h)
            echo 'Usage: scripts/setup.sh [--num-train-sequences N] [--data-dir DIR] [--parity]'
            echo 'Default: 100000 seed-42 sequences. Full A1: 600000 (use a new data directory to enlarge an existing cache).'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$sequences" in
    ''|*[!0-9]*|0) echo 'N must be a positive integer' >&2; exit 2 ;;
esac
command -v uv >/dev/null 2>&1 || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
mkdir -p .build/uv-cache
export UV_CACHE_DIR="$project_root/.build/uv-cache"
git submodule update --init --recursive
if [ "$parity" = true ]; then
    uv sync --locked --python 3.12 --extra parity
else
    uv sync --locked --python 3.12
fi
.venv/bin/python - <<'PY'
from mlx_port.paths import reference_path
print(f"Course reference: {reference_path()}")
PY
# A zero merge gap keeps even the full 600k prefix below the download cap.
.venv/bin/python -m mlx_port.data --num-train-sequences "$sequences" --data-seed 42 --merge-gap 0 --data-dir "$data_dir"
.venv/bin/python - <<'PY'
import json
import mlx.core as mx
from mlx_port.runtime import MEMORY_FRACTION
mx.set_default_device(mx.gpu)
print(json.dumps(dict(mlx_version=mx.__version__, device_info=mx.device_info(),
                      default_memory_fraction=MEMORY_FRACTION), indent=2))
PY
echo 'Setup complete. Run scripts/check.sh for bounded GPU checks.'
