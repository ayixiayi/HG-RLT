#!/usr/bin/env bash
# Installed HG-RLT entrypoint. Only daemon-start/-restart enable hardware, with onsite confirmation.
set -euo pipefail
H=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
R=${HG_RHOS_ROOT:-"$H/vendor/rhos"}
ABC=${HG_ABC:-"$H/vendor/abc"}
OPENPI=${HG_OPENPI:-"$H/vendor/openpi"}
LEROBOT=${HG_LEROBOT:-"$H/vendor/lerobot"}
LIVE=${HG_LIVE_PYTHON:-"$H/envs/live/bin/python"}
DATA=${HG_DATA_PYTHON:-"$H/envs/data/bin/python"}
MODEL=${HG_MODEL_PYTHON:-"$H/envs/model/bin/python"}
export PYTHONPATH="$ABC:$R:$R/XPolicyLab:$R/Client/arms/YAM/gello_software:$R/Client/arms/YAM/i2rt:$H"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPYCACHEPREFIX="$H/cache/pycache"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$H/cache/openpi"}
command=${1:-help}
if (($#)); then shift; fi
case "$command" in
  paths)
    if (($#)); then echo '[参数错误] paths 不接受额外参数。' >&2; exit 2; fi
    printf '[项目] %s\n[入口] %s/hgrlt\n' "$H" "$H"
    printf '[运行副本] %s\n[ABC] %s\n[OpenPI] %s\n[LeRobot] %s\n' "$R" "$ABC" "$OPENPI" "$LEROBOT"
    printf '[采集 Python] %s\n[数据 Python] %s\n[推理 Python] %s\n' "$LIVE" "$DATA" "$MODEL"
    printf '[历史数据] %s/data/baseline\n[原始记录] %s/data/recordings\n[轮次数据] %s/data/rounds\n' "$H" "$H" "$H"
    printf '[模型] %s/models\n[导出包] %s/exports\n[日志] %s/logs\n' "$H" "$H" "$H"
    ;;
  upload-training)
    exec "${HG_TRANSFER_PYTHON:-python3}" "$H/dagger_offline/oss_transfer.py" upload --kind training --cache "$H/cache/oss" "$@"
    ;;
  download-model)
    # Only verified files are installed. Never select/launch the downloaded model.
    exec "${HG_TRANSFER_PYTHON:-python3}" "$H/dagger_offline/oss_transfer.py" download --kind model --cache "$H/cache/oss" "$@"
    ;;
  prepare-data|pack-training|train-check)
    # CPU file operations use the project-local data environment.
    if [[ ! -x "$DATA" ]]; then
      echo "[缺少环境] $DATA；可用 HG_DATA_PYTHON 指定兼容的数据处理环境。" >&2
      exit 1
    fi
    export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1
    export PYTHONPATH="$H:$OPENPI:$OPENPI/src:$OPENPI/packages/openpi-client/src:$LEROBOT:$PYTHONPATH"
    exec "$DATA" -m dagger_offline.workflow "$command" --rhos-root "$R" --openpi "$OPENPI" "$@"
    ;;
  collect)
    exec "$LIVE" "$H/Dagger-Practice/collect.py" --rhos-root "$R" --abc "$ABC" "$@"
    ;;
  rlt-learn)
    # Small actor-critic on CPU (data env): the GPU stays with the model server.
    export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
    exec "$DATA" -m dagger_offline.rlt_learn "$@"
    ;;
  collect-compat)
    # Leader aligns only after h (USB hubs that reset under sustained servo load).
    exec "$LIVE" "$H/Dagger-Practice/collect.py" --rhos-root "$R" --abc "$ABC" \
      --leader-on-demand "$@"
    ;;
  daemon-check|daemon-start|daemon-stop|daemon-restart)
    if (($#)); then
      echo "[参数错误] $command 不接受额外参数。" >&2
      exit 2
    fi
    options=()
    if [[ "$command" != daemon-check ]]; then options+=("--${command#daemon-}"); fi
    exec "$LIVE" "$H/Dagger-Practice/daemon.py" --rhos-root "$R" "${options[@]}"
    ;;
  preflight|prepare)
    if (($#)) && { (($# != 2)) || [[ "$1" != --deployment ]]; }; then
      echo '[参数错误] preflight / prepare 只接受 --deployment ROUND.json；采集请用 record。' >&2
      exit 2
    fi
    options=()
    output="$H/logs/preflight-only.h5"
    if [[ "$command" == prepare ]]; then
      options+=(--prepare-start)
      output="$H/logs/prepare-$(date +%Y%m%d-%H%M%S)-$$.h5"
    fi
    exec "$LIVE" "$H/Dagger-Practice/run_robot.py" --rhos-root "$R" --abc "$ABC" \
      --output "$output" "${options[@]}" "$@"
    ;;
  record)
    # Interactive onsite TTY and strict daemon required; h/p authorizes control.
    exec "$LIVE" "$H/Dagger-Practice/run_robot.py" --rhos-root "$R" --abc "$ABC" \
      --execute "$@"
    ;;
  serve)
    export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS
    exec "$MODEL" "$R/scripts/serve_pi05_heavier.py" "$@"
    ;;
  warmup)
    # Model requests only. This never opens cameras, leaders or the robot RPC.
    exec "$LIVE" "$R/scripts/check_pi05_heavier.py" "$@"
    ;;
  serve-round)
    export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS
    # Pinned OpenPI imports lerobot.common: the vendored LeRobot it pins comes
    # before the model env's site-packages copy.
    export PYTHONPATH="$H:$OPENPI:$OPENPI/src:$OPENPI/packages/openpi-client/src:$LEROBOT:$PYTHONPATH"
    exec "$MODEL" -m dagger_offline.deployment serve --rhos-root "$R" \
      --openpi "$OPENPI" "$@"
    ;;
  ui)
    # Web console (foreground). Children are the ./hgrlt commands above.
    exec "$LIVE" -m dagger_offline.webui "$@"
    ;;
  console)
    # The web console in the background: start | stop | status | url | log | attach.
    exec bash "$H/Dagger-Practice/console.sh" "$@"
    ;;
  *)
    heading='' reset=''
    if [[ -t 1 && ! ${NO_COLOR+x} && ${TERM:-} != dumb ]]; then
      heading=$'\033[1;36m' reset=$'\033[0m'
    fi
    printf '%sHG-RLT · 实验室入口%s\n' "$heading" "$reset"
    echo '用法：./hgrlt <命令>；相对路径以项目根目录为准。'
    echo
    echo '  环境检查'
    echo '    paths                     显示当前安装的实际路径'
    echo '    preflight                 离线检查配置、模型资产和依赖'
    echo '    daemon-check              只读检查硬件进程'
    echo '    warmup                    预热 baseline 模型推理'
    echo
    echo '  现场操作'
    echo '    collect --output-dir data/recordings/r1 [--episodes 20]'
    echo '                              连续采集、自动编号；逐条确认归零并标记结果'
    echo '    collect-compat ...        同上，兼容模式：主臂只在按 h 后对齐，策略执行时不出力'
    echo '    collect ... --deployment R.json --rlt data/rlt/r1   RLT：r 交给 RL actor'
    echo '    rlt-learn --recordings data/recordings/X --run data/rlt/r1   另开终端，在线训练并发布 actor'
    echo '    k 结束本条 → 1 成功 / 2 失败 / 3 弃用；下一条提示输入 q 结束采集。'
    echo '    prepare                   确认后限速回双臂零位，夹爪张开'
    echo '    record --output NEW.h5    新建记录，先 HOLD；按 h / p 才控制'
    echo '    h 停住并准备接管（两个扳机都扣到底再松开）· p 策略 · s 保持 · k 结束并标结果 · q 弃用 · ? 帮助'
    echo
    echo '  采集后处理（CPU）'
    echo '    prepare-data --records data/recordings/r1/records.json --output data/rounds/r1'
    echo '    train-check --round data/rounds/r1'
    echo '    pack-training --round data/rounds/r1 --output exports/hg-r1.tar'
    echo '    upload-training --archive exports/hg-r1.tar [--config OSS配置]'
    echo '    第一轮合并原 BC train/val；后续轮次加 --previous-round 和 --parent-checkpoint。'
    echo '    默认按完整 episode 随机分 train/val，验证比例 20%、种子 42；同组始终在同一集合。'
    echo '    --val-fraction / --split-seed 调整划分；旧 --train / --val 手工入口仍可使用。'
    echo '    训练包包含父模型、数据和脚本；上传后打印接收端下载命令。'
    echo '    换模型/历史数据：--parent-checkpoint / --previous-train / --previous-val。'
    echo
    echo '  模型回传'
    echo '    download-model --source oss://bucket/完整地址 --destination models/r1'
    echo '    下载后使用 models/r1/deployment.json 选择模型。'
    echo '    OSS 默认 bucket=ayixiayi-model-relay；用 --destination / --endpoint / --config 更换。'
    echo
    echo '  网页控制台'
    echo '    console start [--port 8810]  后台启动，打印带令牌的网址；SSH 断开不影响采集'
    echo '    console status | url | log | attach | stop'
    echo '    ui [--host 127.0.0.1] [--port 8810]   前台运行同一控制台'
    echo '    其他电脑：ssh -N -L 8810:127.0.0.1:8810 -p 2222 ziyu@<实验室地址>，再打开网址'
    echo
    echo '  服务管理（现场确认设备与端口空闲后使用）'
    echo '    daemon-restart            托住双臂：下电退出旧进程，再使能新进程（FAULT 恢复用这个）'
    echo '    daemon-stop               托住双臂：关闭电机并退出硬件进程'
    echo '    daemon-start              使能电机；旧进程须已停止'
    echo '    serve                     启动旧 baseline 模型服务'
    echo '    serve-round --deployment ROUND.json   启动已封存的新轮次模型'
    echo
    echo 'preflight / prepare / record 可附加 --deployment ROUND.json。'
    echo '记录保存在指定的新文件；完整保存后可进入数据准备。'
    echo '操作指南：docs/OPERATOR_GUIDE.md；本次项目：docs/THIS_PROJECT.md。'
    echo '设置 NO_COLOR=1 可关闭颜色；重定向输出时自动使用纯文本。'
    if [[ "$command" != help ]]; then exit 2; fi
    ;;
esac
