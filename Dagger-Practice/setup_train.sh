#!/usr/bin/env bash
# Explicit setup on the school GPU server. Never invoked by data preparation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV="$ROOT/envs/train"
if (($#)); then
  echo '[参数错误] setup 不接受额外参数。' >&2
  exit 2
fi
command -v uv >/dev/null || { echo '[缺少依赖] 请先在训练服务器安装 uv。' >&2; exit 1; }
command -v git >/dev/null
if [[ -e "$ENV" && ! -x "$ENV/bin/python" ]]; then
  echo "[拒绝覆盖] $ENV 已存在但不是可用环境，请选择新的解压目录。" >&2
  exit 1
fi
if [[ ! -x "$ENV/bin/python" ]]; then uv venv --python 3.12 "$ENV"; fi
uv pip install --python "$ENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r "$ROOT/Dagger-Practice/requirements-train.txt"
export PYTHONPATH="$ROOT"
OPENPI_REV=$("$ENV/bin/python" -c 'from yam_openpi import OPENPI_REVISION; print(OPENPI_REVISION)')
clone_pinned() {
  local url=$1 target=$2 revision=$3
  if [[ ! -e "$target" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$url" "$target"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$target" checkout --detach "$revision"
  fi
  if [[ $(git -C "$target" rev-parse HEAD) != "$revision" ]] || \
      [[ -n $(git -C "$target" status --porcelain --untracked-files=no) ]]; then
    echo "[拒绝修改] $target 不是干净的固定版本，请使用新的解压目录。" >&2
    exit 1
  fi
}
mkdir -p "$ROOT/vendor"
clone_pinned https://github.com/Physical-Intelligence/openpi.git "$ROOT/vendor/openpi" "$OPENPI_REV"
LEROBOT_REV=$("$ENV/bin/python" -c \
  'import sys, tomllib; print(tomllib.loads(open(sys.argv[1]).read())["tool"]["uv"]["sources"]["lerobot"]["rev"])' \
  "$ROOT/vendor/openpi/pyproject.toml")
clone_pinned https://github.com/huggingface/lerobot.git "$ROOT/vendor/lerobot" "$LEROBOT_REV"
echo '[环境已准备] 尚未启动训练、未占用模型显存；接下来运行 train.sh check。'
