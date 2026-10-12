# YAM / Heavier 执行手册

项目在实验室电脑 `~/Project/HG-RLT`（实际路径 `/mnt/sda/ziyu/Project/HG-RLT`）。
入口、环境、baseline、历史数据都在这个目录里。下面的命令按段整段复制到实验室终端执行；GPU 训练部分在训练服务器执行。

## 1. 当前配置

| 项目 | 内容 |
| --- | --- |
| 任务 | `choose_the_heavier_one`：从天平上拿起较重的物体 |
| baseline | `models/baseline/`（`pi05_yam_heavier`，checkpoint `pi05_yam_20k_20260924_023553/39999`），端口 8767 |
| 历史 BC | `data/baseline/local/balance_0922_train`（52 条）、`balance_0922_val`（5 条） |
| 本轮录制 | `data/recordings/heavier-r1/` |
| 本轮数据 / 训练包 / 模型 | `data/rounds/heavier-r1/`、`exports/heavier-r1-training.tar`、`models/heavier-r1/` |
| OSS | `oss://ayixiayi-model-relay/relay/hg-dagger`（上海），用本机已有 ossutil 配置 |
| 起点 | 双臂零位、夹爪张开 |
| 主臂 | 两只被动 GELLO，关节 XL330-M288、扳机 XL330-M077，独立 5 V 供电 |

## 2. 上机准备

终端 A，启动 baseline 模型服务并保持窗口（8767 已在监听就跳过）：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt serve
```

终端 B，等 A 显示开始监听后执行：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt preflight
./hgrlt warmup
./hgrlt daemon-check
```

`warmup` 首次请求约 9–12 s，之后约 80 ms。`daemon-check` 显示 `safe_hold · HOLD 确认 True` 和 `CONTRACT OK` 即可采集。

硬件进程没启动、状态是 `fault`，或电脑重启过：托住双臂后执行，按提示输入 `HG-DAEMON-ON-SITE`：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt daemon-restart
./hgrlt daemon-check
```

`daemon-restart` 先关闭电机（双臂失去保持力），等旧进程退出后重新使能，不回零。只需停机用 `./hgrlt daemon-stop`。

## 3. 采集

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt collect --output-dir data/recordings/heavier-r1 --episodes 20
```

USB 集线器容易在主臂持续出力时复位，这种情况用兼容入口，参数相同：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt collect-compat --output-dir data/recordings/heavier-r1 --episodes 20
```

| 入口 | 主臂行为 |
| --- | --- |
| `collect` | 策略执行和 HOLD 时，主臂自动跟随机械臂 |
| `collect-compat` | 策略执行时主臂不出力；按 `h` 后主臂才对齐到机械臂姿态 |
| 任一入口加 `--passive-leaders` | 主臂始终被动；`h` 立即按当前主臂姿态接管 |

每一条的流程：

1. 重置场景，按回车开始；确认路径无障碍后再按回车，双臂回零位、夹爪张开。
2. 录制从 HOLD 开始。按 `p` 交给策略。
3. 出现失败趋势时按 `h`：机械臂立即停住，终端提示接管准备。双手握住两只主臂，**两个扳机都扣到底再松开**，进入人工接管。
4. 纠正到能继续任务的状态后按 `p` 交还策略，主臂重新跟随机械臂。
5. 按 `k` 结束本条，等 HOLD 和控制权释放后按 `1` 成功、`2` 失败、`3` 弃用。选 `2` 后再选失败类型：`1` 选错、`2` 抓取失败、`3` 掉落、`4` 卡住、`5` 超时、`6` 其他。
6. 下一条提示处按回车继续，输入 `q` 结束本次采集。

| 按键 | 作用 |
| --- | --- |
| `p` | 交给策略 |
| `h` | 停住并准备接管；双扳机扣下松开后正式接管 |
| `s` | HOLD；接管准备时取消 |
| `c` / `x` | 设检查点 / 丢弃检查点后的片段并 HOLD |
| `k` | 结束本条，进入结果标记 |
| `q` | 录制中：整条弃用；下一条提示处：结束采集 |
| `?` | 按键帮助 |

文件自动编号保存在 `data/recordings/heavier-r1/episodes/`，重复同一命令接着编号。
`records.json` 自动更新，只收录完整、可用且标为成功或失败的条目。
弃用和未完成的条目（连同 `.prepare.jsonl`）自动移到 `data/recordings/heavier-r1/trash/`，`episodes/` 里只留正常标记的录制；编号不复用。
人工接管和策略执行各尽量保持超过 1 秒（30 帧）的连续片段，训练窗口不跨 HOLD 和接管边界。

实验结束按现场流程停机、断电；HOLD 是带电保持。

## 4. 采集时的提示和处理

| 终端提示 | 含义 | 处理 |
| --- | --- | --- |
| `跟随卡住：joint L4 stalled …; check for contact` | 某关节被顶着 3 s 不动，通常是接触或主臂没握住 | 机械臂已 HOLD，本条保留；按 `h` 接管纠正或 `p` 重试 |
| `策略拒绝：… prediction jump … exceeds 0.200000` | 相邻两次预测跳变过大，未发送 | 已 HOLD；按 `h` 纠正或 `p` 重试 |
| `主臂串口断开或已重连（USB 断开？）` | 主臂 USB 复位 | 本条中止；检查 USB 线 / 集线器，后续改用 `collect-compat` |
| `主臂 … 扭矩未能关闭，请给主臂断电` | USB 复位后舵机仍带扭矩 | 给主臂断电再上电 |
| `set_pu(...) failed … Device or resource busy` | 相机还被上一次进程占用 | 等几秒重新运行；连续出现时拔插相机 |
| `Confirmed SAFE_HOLD required` | 硬件进程不在 HOLD 或已 FAULT | 托住双臂执行 `./hgrlt daemon-restart` |
| `Policy request failed` | 模型服务没就绪或断开 | 确认终端 A 的服务在监听，`./hgrlt warmup` 通过后再采 |

## 5. 准备数据、打包并上传

采集结束后在实验室终端整段执行：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt prepare-data \
  --records data/recordings/heavier-r1/records.json \
  --output data/rounds/heavier-r1 &&
./hgrlt train-check --round data/rounds/heavier-r1 &&
./hgrlt pack-training \
  --round data/rounds/heavier-r1 --output exports/heavier-r1-training.tar &&
./hgrlt upload-training --archive exports/heavier-r1-training.tar
```

- `prepare-data` 按完整 episode 随机分 train/val（20%，种子 42），筛选片段、转换视频、合并历史 BC，最后做一次 CPU 训练 batch 检查。每条录制约 2 分钟，主要是视频编码。
- 检查输出 train/val 三个池（历史 / 人工 / 策略）的窗口数和动作 batch `(8, 30, 32)`。任一集合缺少人工或策略窗口时按日志补采，轮次目录换新名称重新准备。
- 训练包约 13 GB，其中父模型参数约 12 GB。训练服务器已有同一 baseline 时，`pack-training` 加 `--without-parent`。
- 上传结束后保存打印的完整 OSS 地址和三条接收命令。

## 6. 训练服务器下载与训练

平台环境需要 `python3`、`ossutil 1.7.19`、`git`、`uv` 和自己的 OSS 配置。首次配置：

```bash
umask 077
ossutil config -c "$HOME/.ossutilconfig" -L CH
chmod 600 "$HOME/.ossutilconfig"
```

整段复制上传结果中的三条接收命令，把最后的 `--destination` 改成 `"$HOME/hg-training"`。下载完成后，在分配了 GPU 的终端执行：

```bash
cd "$HOME/hg-training"
sha256sum -c SHA256SUMS &&
bash Dagger-Practice/train.sh setup &&
bash Dagger-Practice/train.sh check \
  --store "$HOME/hg-runs" --exp-name heavier-r1 \
  --steps 2000 --batch-size 8 --save-interval 500 --rl-token --rl-alpha 1 &&
bash Dagger-Practice/train.sh train \
  --store "$HOME/hg-runs" --exp-name heavier-r1 \
  --steps 2000 --batch-size 8 --save-interval 500 --rl-token --rl-alpha 1
```

`--rl-token --rl-alpha 1` 在同一次训练里同时训出 VLA 和 RLT 用的 RL token；`check` 输出会多一行 `RL token`。不需要 RLT 时去掉这两个参数。
`$HOME` 不是持久化存储时，把 `$HOME/hg-training`、`$HOME/hg-runs` 换成平台的持久化路径。
checkpoint 保存在 `$HOME/hg-runs/checkpoints/pi05_yam_hgdagger/heavier-r1/<step>/`。任务被中断后，在原训练命令末尾加 `--resume` 续跑。

## 7. 回传模型

训练服务器上，提示出现时输入要部署的已保存 step：

```bash
cd "$HOME/hg-training"
printf '输入要部署的已保存 step：'; IFS= read -r STEP
bash Dagger-Practice/train.sh pack-model \
  --checkpoint "$HOME/hg-runs/checkpoints/pi05_yam_hgdagger/heavier-r1/$STEP" \
  --output exports/heavier-r1-model.tar --model-round 1 --port 8769 &&
bash Dagger-Practice/train.sh upload-model --archive exports/heavier-r1-model.tar
```

实验室终端，提示出现时粘贴模型的完整 OSS 地址：

```bash
cd "$HOME/Project/HG-RLT"
printf '粘贴第一轮模型完整 OSS 地址：'; IFS= read -r MODEL_SOURCE
./hgrlt download-model --source "$MODEL_SOURCE" --destination models/heavier-r1 &&
./hgrlt preflight --deployment models/heavier-r1/deployment.json
```

## 8. 评估和第二轮

终端 A 启动第一轮模型（端口 8769，baseline 服务可以停掉以释放显存）：

```bash
"$HOME/Project/HG-RLT/hgrlt" serve-round --deployment models/heavier-r1/deployment.json
```

终端 B 用新模型采第二轮：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt daemon-check &&
./hgrlt collect --deployment models/heavier-r1/deployment.json \
  --output-dir data/recordings/heavier-r2 --episodes 20
```

评估另外安排测试场景，记录成功率、人工接管次数和失败类型，与 baseline 比较；需要人工救回的回合不算自主成功。

第二轮采集结束后准备数据：

```bash
cd "$HOME/Project/HG-RLT"
PARENT_R1="$(python3 - <<'PY'
import json
from pathlib import Path
manifest = Path("models/heavier-r1/deployment.json").resolve()
print((manifest.parent / json.loads(manifest.read_text())["checkpoint"]).resolve())
PY
)"
./hgrlt prepare-data \
  --records data/recordings/heavier-r2/records.json \
  --previous-round data/rounds/heavier-r1 \
  --parent-checkpoint "$PARENT_R1" \
  --output data/rounds/heavier-r2 &&
./hgrlt pack-training \
  --round data/rounds/heavier-r2 --output exports/heavier-r2-training.tar &&
./hgrlt upload-training --archive exports/heavier-r2-training.tar
```

之后重复第 6、7 节，实验名换成 `heavier-r2`，`pack-model` 用 `--model-round 2`。
参数含义、checkpoint 和 OSS 更换见 [操作指南](OPERATOR_GUIDE.md)。

## 9. RLT 在线强化（可选）

RLT 冻结 VLA，在实验室在线训练一个小 actor，精修关键阶段的动作；需要第 6 节加了 `--rl-token` 训练、第 7 节下载封存的模型。

终端 A，启动带 RL token 的模型：

```bash
"$HOME/Project/HG-RLT/hgrlt" serve-round --deployment models/heavier-r1/deployment.json
```

终端 C，启动 learner 并保持窗口（CPU 运行，不占显卡）：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt rlt-learn --recordings data/recordings/heavier-rlt1 --run data/rlt/heavier-rlt1
```

终端 B，采集：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt daemon-check &&
./hgrlt collect --deployment models/heavier-r1/deployment.json \
  --output-dir data/recordings/heavier-rlt1 --rlt data/rlt/heavier-rlt1 --episodes 20
```

每条：`p` 让 VLA 完成简单阶段，到关键阶段按 `r` 交给 RL actor，关键阶段结束可按 `p` 交回 VLA 收尾；危险时 `h` 接管；`k` 结束后如实标 `1` 成功 / `2` 失败，这个标注就是奖励。
任务成功且 RL 那一段之后没有人工接管，这段 RL 得 +1；之后有人工接管或失败得 0。

HG 与 RLT 一起用：每轮 HG 训练加 `--rl-token`，回传的模型同时用于 HG 采集和 RLT；同一次 `collect` 里 `p` / `r` / `h` 可以混用，人工接管照常进入下一轮 HG 数据。
每换一个模型（HG 新一轮），RLT 用新的 `--rlt` 目录和录制目录重新开始。
learner 每收到一条标好的录制就更新并发布 actor，采集在动作块边界自动换上新版本；replay 累计 64 条决策后开始训练，之前 actor 等同 VLA。
actor 输出离 VLA 参考动作最多 0.15 rad，并照常经过全部安全检查。本版 replay 只收录 RL 执行段，人工接管段继续用于 HG 训练。
