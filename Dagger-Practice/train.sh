#!/usr/bin/env bash
# Portable Pi0.5 HG-RLT training entrypoint (HG-DAgger rounds + RL token).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command=${1:-help}
if (($#)); then shift; fi
case "$command" in
  help)
    echo 'HG-RLT · GPU 服务器训练入口'
    echo '  setup                    显式创建独立环境并拉取固定版本源码；需要联网'
    echo '  check [训练参数]         CPU 检查父模型、数据和真实 OpenPI batch，不训练'
    echo '  train [训练参数]         从现有 baseline / 上轮 checkpoint 全量微调'
    echo '  seal --checkpoint DIR --output ROUND.json --model-round N'
    echo '  pack-model --checkpoint DIR --output hg-r1-model.tar --model-round 1'
    echo '  upload-model --archive hg-r1-model.tar [--config 本机OSS配置]'
    echo
    echo '完整数据包解压后：'
    echo '  sha256sum -c SHA256SUMS'
    echo '  bash Dagger-Practice/train.sh setup'
    echo '  bash Dagger-Practice/train.sh check --exp-name r1'
    echo '  bash Dagger-Practice/train.sh train --exp-name r1'
    echo
    echo '默认 2000 steps / batch 8；可传 --steps / --batch-size / --fsdp-devices。'
    echo '只恢复同一次训练时加 --resume，总 steps 和数据必须不变。'
    echo '无父模型包须加 --parent-checkpoint /path/to/CHECKPOINT（含 params/ 和 assets/）。'
    echo '已有兼容环境可设置 HG_TRAIN_PYTHON，源码可设置 HG_OPENPI / HG_LEROBOT。'
    echo '全量微调使用 H100/H200 等合适 GPU；本入口不会改成 LoRA，也不会启动机器人。'
    echo '训练后 pack-model + upload-model 回传 OSS；实验室 download-model，不需要两端互通。'
    echo '返程包仅含推理/下轮父模型所需文件；同一训练的续跑仍使用学校端完整 checkpoint。'
    echo 'OSS 工具只使用本机配置；默认 bucket=ayixiayi-model-relay，不复制或打包凭据。'
    exit 0
    ;;
  setup) exec bash "$ROOT/Dagger-Practice/setup_train.sh" "$@" ;;
  upload-model)
    exec "${HG_TRANSFER_PYTHON:-python3}" "$ROOT/dagger_offline/oss_transfer.py" upload --kind model --cache "$ROOT/cache/oss" "$@"
    ;;
  check|train|seal|pack-model) ;;
  *) echo "[参数错误] 未知命令：$command；请运行 help。" >&2; exit 2 ;;
esac
PYTHON=${HG_TRAIN_PYTHON:-"$ROOT/envs/train/bin/python"}
if [[ ! -x "$PYTHON" ]]; then
  echo '[缺少环境] 请先运行 train.sh setup，或设置 HG_TRAIN_PYTHON。' >&2
  exit 1
fi
OPENPI=${HG_OPENPI:-"$ROOT/vendor/openpi"}
LEROBOT=${HG_LEROBOT:-"$ROOT/vendor/lerobot"}
export PYTHONPATH="$ROOT:$OPENPI:$OPENPI/src:$OPENPI/packages/openpi-client/src:$LEROBOT"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$ROOT/cache/openpi"}
export HF_HOME=${HF_HOME:-"$ROOT/cache/huggingface"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "$command" != train ]]; then
  export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
else
  unset JAX_PLATFORMS
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
fi
case "$command" in
  check) exec "$PYTHON" -m dagger_offline.workflow train-check --openpi "$OPENPI" "$@" ;;
  train) exec "$PYTHON" -m dagger_offline.workflow train --openpi "$OPENPI" "$@" ;;
  seal) exec "$PYTHON" -m dagger_offline.deployment seal "$@" ;;
  pack-model) exec "$PYTHON" -m dagger_offline.workflow pack-model "$@" ;;
esac
