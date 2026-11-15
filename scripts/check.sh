#!/bin/sh
# GPU-only checks: no parity suite or CPU training gate.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
if [ ! -x .venv/bin/python ]; then
    echo 'Run scripts/setup.sh first.' >&2
    exit 1
fi
exec .venv/bin/python scripts/check_gpu.py "$@"
