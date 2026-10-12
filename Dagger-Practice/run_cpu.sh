#!/usr/bin/env bash
# No installs, downloads, GPU processes, sockets, or hardware launchers.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ABC=${ABC:-"$ROOT/vendor/abc"}
PYTHON=${PYTHON:-"$ROOT/envs/abc/bin/python"}
REV=d0832d12651d1b260a652861a14648dc5f3660c7
test "$(git -C "$ABC" rev-parse HEAD)" = "$REV" || {
    echo "ABC revision mismatch; expected $REV" >&2
    exit 1
}
if test -n "$(git -C "$ABC" status --porcelain --untracked-files=no)"; then
    echo "ABC tracked source has local modifications; refusing unpinned run" >&2
    exit 1
fi
export PYTHONPATH="$ABC:$ROOT"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$PYTHON" -c 'from torchcodec.decoders import VideoDecoder; import torch, torchcodec; print("native decoder:", torch.__version__, torchcodec.__version__)'
"$PYTHON" -m pytest -q -o "cache_dir=$ROOT/cache/pytest-abc" "$ROOT/Dagger-Practice" "$@"
