# HG-RLT 操作指南

HG-RLT 包含两种方法。HG-DAgger 用“策略执行 → 人工接管纠偏 → 聚合数据 → 更新策略”改进已有模型；
RLT 在同一采集流程里在线训练小 actor 精修关键阶段（第 2.6 节）。
本项目适配双臂 YAM 和 Pi0.5：三路图像、14 维状态/动作、30 帧动作窗口。
训练从 baseline 或上轮权重开始全量微调，按历史数据 / 本轮人工 / 本轮策略 = 80% / 10% / 10% 采样，沿用父模型归一化统计量。

原论文 [HG-DAgger](https://ar5iv.labs.arxiv.org/html/1810.02890) 的 Algorithm 1 将人工接管标签加入初始示教数据后继续监督训练。
这里的 80:10:10 混合采样和 80:20 train/val 划分是本项目的训练配置；论文没有规定分别采集 train 和 val。

采集、数据准备、OSS 收发和推理使用完整的实验室安装；GPU 训练使用从 OSS 下载的独立训练包。
本次已安装项目的直接操作步骤见 [YAM / Heavier 执行手册](THIS_PROJECT.md)。

## 1. 项目入口与目录

在项目根目录运行：

```bash
./hgrlt paths
./hgrlt help
./hgrlt preflight
```

`hg` 自动定位自己的安装目录，加载项目内的 Python 和源码依赖；相对参数路径始终以项目根目录为准。
也可以从任意工作目录使用入口的绝对路径。`preflight` 检查文件、配置、MuJoCo 模型和导入依赖。

| 目录 | 内容 |
| --- | --- |
| `Dagger-Practice/`、`dagger_offline/` | 采集、数据处理、训练与传输源码 |
| `configs/` | 机器人配置和 baseline 部署配置 |
| `vendor/` | 固定版本 ABC、OpenPI、LeRobot、RhOS 运行副本及机器人模型 |
| `envs/`、`runtime/` | 项目 Python 环境和解释器 |
| `data/baseline/` | 历史 BC 数据，含源数据及 train/val 划分 |
| `data/recordings/` | 原始 HDF5 |
| `data/rounds/` | 各轮处理后的训练数据 |
| `models/` | baseline 和下载后的 HG 模型 |
| `exports/` | 训练包、模型包及校验文件 |
| `cache/`、`logs/` | 模型资产/传输缓存和运行日志 |
| `docs/` | 当前操作文档 |

完整安装使用独立的采集、CPU 数据和模型推理环境；已有兼容环境可通过
`HG_LIVE_PYTHON`、`HG_DATA_PYTHON`、`HG_MODEL_PYTHON` 指定。
源码位置可通过 `HG_RHOS_ROOT`、`HG_ABC`、`HG_OPENPI`、`HG_LEROBOT` 指定。
硬件协议要求见 [补丁说明](../Dagger-Practice/patches/README.md)。

## 2. 硬件进程、模型服务与采集

### 2.1 模型服务

在单独终端启动 baseline 并保持窗口；封存的新轮次模型用 `serve-round`：

```bash
./hgrlt serve
./hgrlt serve-round --deployment models/r1/deployment.json
```

baseline 监听 8767，新轮次默认 8769，均为 GPU 0。另一个终端执行 `./hgrlt warmup` 预热 baseline，首次请求约 9–12 s，之后约 80 ms。

### 2.2 硬件进程

| 命令 | 作用 |
| --- | --- |
| `./hgrlt daemon-check` | 只读检查：状态、HOLD、会话协议版本和 CAN 反馈时间戳 |
| `./hgrlt daemon-start` | 使能电机并启动硬件进程，不回零；6001/6002 端口须空闲 |
| `./hgrlt daemon-stop` | 关闭电机并退出硬件进程；先托住双臂 |
| `./hgrlt daemon-restart` | 依次执行 stop 和 start，一次确认；FAULT 恢复和电脑重启后使用 |

start / stop / restart 只接受现场终端，需输入确认口令。采集要求 `daemon-check` 显示 `safe_hold`、`HOLD 确认 True`、`CONTRACT OK`。
YAM 没有物理急停，现场操作者须能随时断电。

### 2.3 连续采集

```bash
./hgrlt collect --output-dir data/recordings/r1 --episodes 20
./hgrlt collect-compat --output-dir data/recordings/r1 --episodes 20
```

每条流程：重置场景 → 回车开始 → 确认路径后自动回零位、夹爪张开 → HOLD → `p` 策略 / `h` 接管 → `k` 结束 → 标结果。
重复同一命令从已有最大编号继续，文件保存为 `episodes/episode-000001.h5` 等。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--output-dir DIR` | 必填 | 本轮采集目录，保存原始 HDF5 和自动生成的 `records.json`；每轮使用新目录 |
| `--episodes N` | `0` | 本次最多尝试次数，含弃用条目；`0` 不限，在下一条提示处输入 `q` 结束 |
| `--deployment FILE` | baseline | 用封存的 HG 模型采集，填写其 `deployment.json` |
| `--group NAME` | 每条独立 | 本次保存的条目绑定为同一来源组，整体进入 train 或 val |
| `--passive-leaders` | 关闭 | 主臂始终被动；`h` 立即按当前主臂姿态接管 |
| `--leader-on-demand` | 关闭 | 主臂只在按 `h` 后对齐机械臂；`collect-compat` 即 `collect --leader-on-demand` |
| `--index-only` | 关闭 | 只从 HDF5 刷新清单和统计，可在无硬件环境使用 |

单条录制用 `./hgrlt prepare`（回零）加 `./hgrlt record --output NEW.h5`，接受同样的主臂参数。

### 2.4 主臂与接管

主臂关节和扳机舵机（扳机已拆弹簧）在启动时切换为电流位置模式（模式 5，关节限流 1200 mA、扳机 300 mA，受各舵机 Current Limit 封顶）；跟随时扳机对齐机械臂的夹爪开度，目标留在校准范围两端以内 0.05，不顶死限位。

| 模式 | 策略执行 / HOLD 时主臂 | 按 `h` 后 |
| --- | --- | --- |
| 跟随（`collect`） | 跟随机械臂当前姿态和夹爪 | 机械臂 HOLD，主臂保持在机械臂姿态，扳机断电确认后接管 |
| 按需（`collect-compat`） | 不出力 | 机械臂 HOLD，主臂限速对齐到机械臂姿态，扳机断电确认后接管 |
| 被动（`--passive-leaders`） | 不出力 | 立即接管 |

跟随和按需模式下，按 `h` 后机械臂 HOLD，主臂关节保持对齐，**两个扳机断电**：每个扳机先完全松开（校准值 ≥0.8）再扣到底（≤0.2），两侧都完成即确认双手握在主臂上；随后扳机上电回到机械臂夹爪开度，两侧误差都在 0.05 以内时进入人工接管，主臂和扳机关扭矩，手臂以当前姿态为锚点做相对映射。回位超过 2 秒未到位时终端提示两侧数值。`s` / `p` 随时取消。
接管时夹爪保持机械臂当前开度，扳机与夹爪之间的差记为偏移；扳机每动一点，偏移就消掉同样多，消完后夹爪与扳机位置一致。夹爪不跳变，两个方向都立即响应，全开、全闭都能到达；主动模式下扳机已回位，偏移最多约 0.1。
两个扳机都要确认，是因为没人握住的主臂会下垂，把对应机械臂带向够不到的目标。`s` / `p` 取消接管准备。
结束接管（`p` / `s`）时主臂在原位开扭矩，再限速回到跟随。

| 按键 | 操作 |
| --- | --- |
| `p` | 交给策略 |
| `h` | HOLD 并准备接管：扳机断电，完全松开再扣到底确认，回位后接管（被动模式下立即接管） |
| `s` | HOLD；接管准备时取消 |
| `c` / `x` | 设置检查点 / 丢弃检查点后的片段并 HOLD |
| `k` | HOLD、释放控制权，进入结果标记 |
| `1` / `2` / `3` | 成功 / 失败 / 弃用；失败后再选类型 `1`–`6`：选错 / 抓取失败 / 掉落 / 卡住 / 超时 / 其他 |
| `q` | 录制中：本条弃用；下一条提示处：结束连续采集 |
| `?` | 按键帮助 |

成功和失败都纳入清单，失败 rollout 中的人工纠偏同样可用于学习。
弃用、中断、未完成的条目在每次刷新清单时移到采集目录的 `trash/`（同名冲突时加序号），`episodes/` 只保留正常标记的录制；新编号跳过 `trash/` 里已用的号。
任务结果保存在 HDF5 的 `outcome` 属性，来源组保存在 `metadata_json.collection.group`，`records.json` 可随时重建。

### 2.5 策略执行与控制上限

策略每次推理 30 步动作。剩 8 步时用最新观测预取下一块；新块到达后跳过观测之后已执行的步数，块与块之间不停顿；切换时前 8 步从旧块剩余动作线性过渡到新块，两次推理之间的差异分摊到这几步里。
预取结果晚于 12 步或超时，机械臂 HOLD 并用新观测重新推理。

| 项目 | 人工 | 策略 |
| --- | --- | --- |
| 关节速度上限 | 1.5 rad/s（每帧 0.05 rad） | 1.5 rad/s（每帧 0.05 rad） |
| 跟随误差上限 J1–J3、J5–J6 | 0.15 rad | 0.08 rad |
| 跟随误差上限 J4（腕俯仰） | 0.25 rad | 0.25 rad |
| 相邻预测跳变 | 0.30 rad | 0.30 rad，跨块同样检查（过渡之后） |
| 行程预算 | 不计 | 每块 3 rad、每段 30 rad、整条 60 rad |
| 卡滞 | 被顶住 90 帧（3 s）基本不动 → HOLD | 同左 |

速度上限覆盖示教数据（balance_0922 每帧关节变化 p95 0.049 rad）。跟随误差上限决定接触时的推力：力矩约为 kp × 误差，J1–J3 kp 80，J4–J6 kp 10。
J4 在正常遥操作中就有较大跟随误差（p95 左 0.195、右 0.125 rad），因此单独放宽。

结束后按现场流程停机、断电；HOLD 为带电保持。

### 2.6 RLT 在线强化

| 组成 | 说明 |
| --- | --- |
| 模型 | 训练时加 `--rl-token`（`--rl-alpha 1` 与 VLA 同训，`0` 只训 token），封存后 `serve-round` 每次推理同时返回动作和 RL token |
| `collect --rlt DIR` | 启用 `r` 键；必须配合该模型的 `--deployment` |
| `r` | 交给 RL actor：每个 10 步动作块由 actor 根据（RL token、关节状态、已提交动作、VLA 参考块）给出。块与块之间不停顿：当前块剩 4 步（`lead`）时用新观测决策下一块，这几步在观测时已经确定，作为“已提交动作”进入状态；新块按观测时刻对齐，推理期间已执行的步数由已提交动作占用 |
| `p` / `h` / `s` | 回到 VLA / 准备接管 / HOLD，含义不变 |
| `rlt-learn --recordings DIR --run DIR` | 另开终端；读取已标注录制生成 replay，TD3 训练后先保存完整训练状态 `DIR/learner.pkl`，再原子发布 `DIR/actor/actor.npz`；重启后从该状态继续，版本号接续，actor 缺失或落后时按状态补发；同一 `--run` 只能运行一个 learner（排他锁），且只对应一个录制目录，换录制目录需要新的 `--run` |
| `DIR/config.json` | 采集端和 learner 共用的 RLT 配置，第一次使用时生成；`rlt-learn --config` 只能修改训练超参数（学习率、β 等），网络结构、动作块长度等需要新目录 |
| 奖励 | 每段 RL 执行的最后一个决策：本条标成功、且这段之后没有人工接管 → +1；之后有人工接管或标失败 → 0。用 `p` 交回 VLA 收尾、`s` 暂停都保留这份奖励 |
| 丢弃 | `c` / `x` 丢弃的区间不进入 replay，跨入该区间的 RL 执行段在区间前结束且不得奖励；整条弃用的录制不进入 replay |
| 时序 | 每条转移覆盖从本次观测到下一决策观测的时间，前后首尾相接；动作是按观测时刻对齐的块，前几步是实际执行的已提交动作，之后是 actor 的输出；状态和下一状态都包含各自的已提交动作，折扣按两次观测之间的控制步数计算。自己的块一步都没执行的决策（例如刚接受就按 `k`，或回复晚于整块）不进入 replay，结果记在前一个执行过的决策上。没有已提交动作的旧版（同步）RLT 录制会被拒绝，需要新的 `--rlt` 运行目录 |
| 运行目录与模型 | `--rlt DIR` 第一次使用时绑定当前封存模型（`DIR/model.json`）；learner 在绑定之前只等待、不训练；actor 和训练状态都记录所属模型，加载时核对；已有未绑定训练结果的目录不能再绑定。换模型（例如 HG 新一轮）用新目录，旧 replay 的 RL token 与新模型不通用 |
| 安全 | actor 输出与参考块偏差 ≤0.15 rad（夹爪 0.30），之后照常经过第 2.5 节全部检查 |

RL 决策（token、关节状态、参考块、actor 输出、actor 版本）保存在录制的 `rl_decisions` 组；每次 VLA 推理的完整预测块保存在 `policy_chunks` 组。
RL 执行段在事件里仍记为 `POLICY_RUNNING`，HG 数据准备把它当作策略片段。

### 2.7 网页控制台

一个网页代替模型服务、硬件进程和采集三个终端。页面里的命令就是本节的 `./hgrlt` 命令，由控制台作为子进程启动；采集和硬件进程重启 / 关闭在伪终端里运行，按键和确认口令由操作者在页面输入。

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt console start
```

`console start` 在后台（tmux 服务 `hgrlt`）运行控制台，打印带访问令牌的网址；SSH 断开、关闭网页都不影响采集。其他电脑先建隧道，再用浏览器打开该网址：

```bash
ssh -N -L 8810:127.0.0.1:8810 -p 2222 ziyu@<实验室地址>
```

| 命令 | 作用 |
| --- | --- |
| `./hgrlt console start [--port 8810] [--new-token]` | 后台启动；已在运行时只打印网址 |
| `./hgrlt console status` / `url` | 是否在运行、访问网址 |
| `./hgrlt console log` | 跟踪 `logs/console.log` |
| `./hgrlt console attach` | 进入控制台的 tmux 窗口（`Ctrl-b d` 离开） |
| `./hgrlt console stop [--force]` | 关闭控制台，过程见下文 |
| `./hgrlt ui [--host 127.0.0.1] [--port 8810]` | 前台运行同一控制台，`Ctrl-C` 关闭 |

| 页面 | 内容 |
| --- | --- |
| 采集 | 选类型（评估写 `data/eval/<名称>`，默认 `MMDD-模型`；采集写 `data/recordings/<名称>`）、模型、条数、`collect` / `collect-compat`、可选 `--rlt data/rlt/<名称>`，确认后运行 `./hgrlt collect …`。大字显示 HOLD / POLICY_RUNNING / INTERVENTION / FAULT、接管准备时两侧扳机的确认进度、本条阶段和累计统计；按当前阶段给出大按钮，下方是终端输出 |
| 模型 | baseline（8767）和每个 `models/*/deployment.json`（端口、checkpoint、policy_id 前 12 位）；启动 / 停止一个由控制台运行的模型服务，显示其日志 |
| 硬件 | 硬件进程归属（本项目 / 其他项目 / 未运行）；`daemon-check`；重启、关闭硬件进程 |
| 状态 | GPU 显存和计算进程（用户、命令）、模型端口和硬件端口、其他终端里的采集进程、安装版本 |
| 结果 | 每个 `data/eval`、`data/recordings` 目录的成功 / 失败 / 弃用 / 未完成、失败类型、人工接管次数；同一天（`MMDD-` 开头）的评估目录列成对比表 |

采集页的按键与终端按键一一对应：`p` `h` `s` `c` `x` `k` `q` `r` `?`、结果 `1` / `2` / `3`、失败类型 `1`–`6`、回车。勾选“我在机械臂旁”后按钮和键盘快捷键生效；`s`（HOLD）始终可用。
开始提示处“开始本条”发送回车，“结束本次采集”发送 `q` 回车；回零确认处“确认回零”发送回车。页面顶部的模式标签在其他页面也可见。

安全规则：

- 开始采集、重启或关闭硬件进程会让机械臂运动或断电，需在页面输入 `我在机械臂旁`。同一时间只运行其中一个；控制台的采集进行中、或其他终端已有 `collect.py` / `run_robot.py` 进程时拒绝。
- 硬件进程命令仍会提示自己的确认，在“硬件”页终端输入 `HG-DAEMON-ON-SITE`（关闭时输入 `off`）发送；控制台不代答。
- 模型服务一次只启动一个；端口已被占用或 GPU 0 空闲显存不足 10 GiB 时拒绝。其他人启动的模型服务和 GPU 进程标为“其他人在用”，控制台只停止自己启动的进程；正在采集的会话所用的模型不能停止。
- 开始采集前，所选模型的端口须已在监听。
- 只监听 `127.0.0.1`；访问令牌保存在 `~/.config/hgrlt/console-token`（仅本人可读），首次打开网址后改用 HttpOnly Cookie，重启控制台网址不变，`--new-token` 更换。`--host` 监听其他地址时，持有令牌的人都能操作机械臂。
- 命令只由页面表单的校验字段组成：模型须在 `models/` 内，输出目录须在 `data/` 内，名称只含字母、数字、`.`、`_`、`-`。

结束与关闭：

| 操作 | 行为 |
| --- | --- |
| 页面“结束采集” | 按当前阶段发送按键：录制中先 `s` 再 `q`（本条弃用，文件移入 `trash/`）；结果标记时 `3` 弃用；选失败类型时 Ctrl-C（本条记为未完成，移入 `trash/`）；回零确认时取消（不运动）；开始提示处 `q` 回车。回零运动和落盘过程等待完成 |
| 关闭网页 | 采集继续；重新打开后显示完整状态和最近输出 |
| 无人操作 10 分钟 | 任何阶段 10 分钟无人按键（人工接管除外）：保持并保存本条（不标结果；选失败类型时同样不标），不等确认自动回零，然后结束采集；终端里的 `collect` 同样适用 |
| `console stop`、页面“关闭控制台”、前台 `Ctrl-C` | 先按上一行结束采集，停止控制台启动的模型服务，再退出；最多等待约 200 秒 |
| `console stop --force` | 只在控制台卡住时使用：直接结束控制台，采集进程失去终端随之退出，正在录制的一条不保存，硬件进程租约过期后保持 |
| 硬件进程命令进行中关闭 | 仍在等待确认口令的命令收到空行后放弃，不改变状态；已确认的命令运行到结束 |

日志：`logs/console.log` 记录启动、每个操作请求和子进程退出码；`logs/console-jobs/` 保存每个子进程的完整终端输出；`console start` 的输出（含网址）在 `logs/console.out`。三者都仅本人可读。
RLT 的 `rlt-learn` 仍在单独终端运行（第 2.6 节）。

## 3. 数据准备、打包和上传

### 3.1 一轮数据的处理顺序

先完成第 2 节的连续采集，再整段执行：

```bash
./hgrlt prepare-data \
  --records data/recordings/r1/records.json \
  --output data/rounds/r1 &&
./hgrlt train-check --round data/rounds/r1 &&
./hgrlt pack-training --round data/rounds/r1 --output exports/r1-training.tar &&
./hgrlt upload-training --archive exports/r1-training.tar
```

`&&` 使前一步成功后才进入下一步。`r1` 是自定义的轮次名称，各命令引用同一个目录即可。
命令行中的相对路径以项目根目录为准。轮次输出目录和导出包均使用新名称。

| 命令 | 输入 | 产物或结果 |
| --- | --- | --- |
| `prepare-data` | 本轮 HDF5、父模型、历史数据 | 筛选片段、转换格式、聚合历史数据，生成轮次目录，最后执行 CPU 数据检查 |
| `train-check` | `--round` 指定的已准备轮次 | 再次检查数据与训练配置，构造真实 OpenPI batch |
| `pack-training` | `--round` 指定的轮次及其父模型 | `--output` 指定的 `.tar` / `.tar.gz` 训练包 |
| `upload-training` | `--archive` 指定的训练包 | OSS 内容地址、校验信息和接收端下载命令 |

`prepare-data` 已自动执行一次 CPU 检查，单独的 `train-check` 适合在打包前复查已有轮次。
训练使用第 5 节的 `train.sh train`。

耗时与体积参考：`prepare-data` 每条录制约 2 分钟（主要是视频编码）；含 baseline 父模型的训练包约 13 GB，其中参数约 12 GB。

### 3.2 自动划分 train / val

采集时统一保存 episode，`prepare-data` 读取清单后随机划分，默认约 80% train、20% val，种子 `42`。

| 集合或参数 | 含义 | 行为 |
| --- | --- | --- |
| `train` | 训练集 | 从其中的历史 / 人工 / 策略三个池抽取窗口、计算梯度并更新参数 |
| `val` | 留出的验证集 | 与 train 隔离；当前入口检查其数据和窗口完整性，训练循环尚未定期计算 validation loss |
| `group` | 完整 rollout 或相互关联的来源组 | 同组全部文件和所有帧进入同一个集合；默认每条 episode 一组 |
| `--val-fraction 0.2` | 留给 val 的组数比例 | 范围大于 0、小于 1；向上取整，并保证两边至少各有一组 |
| `--split-seed 42` | 划分随机种子 | 同一份来源组清单与种子产生相同划分，改变 JSON 条目顺序不影响结果 |

划分按完整来源组进行，再切分 30 帧训练窗口，避免同一次操作的相邻帧同时进入 train 和 val。
至少需要两组有效录制；20 个独立组默认分为 16 个 train、4 个 val，帧数和窗口数的比例可以不同。
同一 rollout 拆出的多个文件保持相同 `group`；需按场景隔离时，将同场景相关条目标为同组。

需要修改比例时，例如预留 10%：

```bash
./hgrlt prepare-data \
  --records data/recordings/r1/records.json \
  --val-fraction 0.1 --split-seed 42 \
  --output data/rounds/r1
```

轮次清单保存已确定的划分和 `split_config`；训练、打包、续训都沿用它。
历史 BC 及上一轮数据继续保留原划分，本次只划分新采集数据。
train 和 val 分别汇总后，都需要有效的人工接管窗口与策略执行窗口，每个窗口包含至少连续 30 帧。
这一要求按每个集合汇总检查，单个文件可以只贡献其中一种片段。HOLD、丢帧和时间断点会切断窗口。
每次接管前 15 帧（0.5 秒，从按 `h` 或进入人工控制算起）的策略帧不进策略池：这是操作员出手要阻止的动作（做法同 [I3L](https://arxiv.org/abs/2610.17748)）。人工接管段不受影响；`prepare-data` 会打印本轮去掉的帧数，`round.json` 的 `pre_intervention` 记录设置。
实机成功率评估使用另外安排的测试场景，结果单独记录。

### 3.3 录制 JSON 的具体定义

`collect` 自动维护采集目录中的 `records.json`，通常直接将它传给 `--records`。
手工选择文件时可另建清单：顶层是数组，每个对象描述一个 HDF5 文件。

| 字段 | 类型与取值 | 含义 |
| --- | --- | --- |
| `path` | 非空字符串；相对路径或绝对路径 | 本条 HDF5 文件。相对路径以 **JSON 文件所在目录** 为基准 |
| `group` | 非空字符串，建议避免首尾空白 | 来源组，规则与 `GROUP=FILE` 中的 `GROUP` 相同 |
| `role` | 此入口固定为字符串 `"current"` | 本轮采集数据；历史数据通过下节的历史数据参数加入 |
| `outcome` | 自动清单写入 `"success"` / `"fail"`；手工清单可省略 | 任务结果摘要；筛选有效片段仍读取 HDF5 的完整性、可用性及控制事件 |
| `split` | 可选；字符串 `"train"` 或 `"val"` | 默认全部省略以自动划分；手动固定时每个对象都填写，同一 `group` 保持一致 |

自动清单示例，文件位于 `data/recordings/r1/records.json`：

```json
[
  {
    "path": "episodes/episode-000001.h5",
    "group": "hg:example-episode-1",
    "role": "current",
    "outcome": "success"
  },
  {
    "path": "episodes/episode-000002.h5",
    "group": "hg:example-episode-2",
    "role": "current",
    "outcome": "fail"
  }
]
```

实际组名由程序生成。仓库的 [records.example.json](records.example.json) 展示从 `docs/` 引用相同录制的写法。
手工清单与自动清单分开保存：自动清单会在继续采集或 `--index-only` 时从 HDF5 重建。
JSON 使用 UTF-8、双引号，最后一个字段和数组元素末尾均省略逗号；说明文字写在 JSON 之外。

已有明确 train/val 的实验仍可使用全部带 `split` 的 JSON，或 `--train GROUP=FILE` / `--val GROUP=FILE`：

```bash
./hgrlt prepare-data \
  --train scene-a=data/recordings/manual/take-01.h5 \
  --val scene-b=data/recordings/manual/take-02.h5 \
  --output data/rounds/manual-r1
```

等号左边是非空来源组，右边是实际文件；参数可重复。两种清单入口各选其一，显式划分会原样保留。

### 3.4 父模型、历史数据与可选参数

第一轮默认读取部署配置中的 baseline，以及
`data/baseline/local/balance_0922_train` 和 `data/baseline/local/balance_0922_val`。
父模型是本轮将继续微调的已有模型，第一轮使用 baseline，后续轮次使用上轮训练结果。

| `prepare-data` 参数 | 默认值 / 是否必填 | 含义与边界 |
| --- | --- | --- |
| `--output DIR` | 必填，新目录 | 本轮处理结果的根目录，例如 `data/rounds/r1` |
| `--records FILE.json` | 常规入口 | `collect` 生成的录制清单，或第 3.3 节格式的手工清单 |
| `--val-fraction FLOAT` | `0.2` | 自动划分时的验证组比例，范围 `(0, 1)` |
| `--split-seed N` | `42` | 自动划分种子，非负整数；与训练窗口采样的 `--seed` 独立 |
| `--train GROUP=FILE` | 高级手工入口，与 `--val` 同时提供 | 显式添加训练录制，可重复，与 `--records` 二选一 |
| `--val GROUP=FILE` | 高级手工入口，与 `--train` 同时提供 | 显式添加验证录制，可重复 |
| `--parent-checkpoint DIR` | 第一轮默认 baseline；后续轮次显式提供 | 包含 `params/` 和 `assets/` 的 checkpoint 根目录 |
| `--baseline-data DIR` | `data/baseline` | 默认历史 BC 根目录；其下含 `local/balance_0922_train` 和 `local/balance_0922_val` |
| `--previous-train DIR` | 可选，与 `--previous-val` 同时提供 | 显式指定已有的 LeRobot BC 训练目录，覆盖默认历史数据选择 |
| `--previous-val DIR` | 可选，与 `--previous-train` 同时提供 | 显式指定对应的 LeRobot BC 验证目录；两份数据均含 `meta/split.json` 来源信息 |
| `--previous-round DIR` | 后续轮次使用 | 选择上一轮完整数据目录或其 `lerobot/combined.json`；同时显式指定父模型，与上述 BC 覆盖参数二选一 |
| `--norm-stats FILE` | 默认沿用父模型的唯一统计量；后续轮次沿用上一轮统计量 | 显式选择匹配父模型的归一化统计文件，适用于父模型存在多份统计量的情况 |

训练采样概率固定为历史数据 / 本轮人工 / 本轮策略 = 80% / 10% / 10%。
这些数值控制窗口采样概率；采集数量按任务需要安排，单个 batch 的数量由采样结果决定。
下一轮聚合时，上轮所有数据进入历史池，本轮新录制分别进入人工池和策略池。

`prepare-data` 和 `train-check` 还接受下列配置检查参数。按准备采用的训练设置检查即可；
这些命令行设置只用于本次检查，正式训练时仍在 `train.sh check` 和 `train.sh train` 中填写相同设置。

| 配置检查参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--batch-size N` | `8` | 检查真实训练 batch；示例输出动作形状 `(8, 30, 32)`，32 是 OpenPI 模型的填充维度，YAM 原始动作为 14 维 |
| `--steps N` | `2000` | 用于构造训练配置的总步数 |
| `--save-interval N` | `500` | 训练配置中的 checkpoint 保存间隔 |
| `--seed N` | `42` | 窗口采样等随机过程的种子 |
| `--fsdp-devices N` | `1` | 训练配置中的 FSDP 分组设备数；正式训练时按分配的 GPU 数设置 |
| `--store DIR` | `training` | 训练配置的输出根目录；正式训练建议显式选择持久化存储 |
| `--name NAME` | `pi05_yam_hgdagger` | 训练配置名称，参与 checkpoint 路径组成 |
| `--exp-name NAME` | `first` | 实验名称，正式训练时每轮使用新名称 |
| `--resume` | 关闭 | 构造续跑配置；实际恢复优化器和训练进度由 `train.sh train --resume` 执行 |
| `--rhos-root DIR` | 入口的 `HG_RHOS_ROOT`，默认 `vendor/rhos` | 指定兼容的 RhOS 运行副本；第一轮默认父模型从该副本的部署配置读取 |
| `--openpi DIR` | 入口的 `HG_OPENPI`，默认 `vendor/openpi` | 指定固定版本、源码未修改的 OpenPI checkout |

### 3.5 检查与训练包参数

| 命令与参数 | 默认值 / 是否必填 | 含义 |
| --- | --- | --- |
| `train-check --round DIR` | 默认 `round`；实验室操作显式填写 | 已准备的轮次目录，也可直接填写 `lerobot/combined.json` 的路径 |
| `train-check --parent-checkpoint DIR` | 默认读取该轮清单 | 覆盖检查使用的父模型；训练端处理无父模型包时需要显式填写 |
| `pack-training --round DIR` | 默认 `round`；实验室操作显式填写 | 要打包的轮次目录或训练清单，与 `prepare-data --output` 对应 |
| `pack-training --output FILE` | 必填，新文件 | `.tar` 或 `.tar.gz` 训练包的保存路径 |
| `pack-training --parent-checkpoint DIR` | 默认读取该轮清单 | 指定随包发送的父模型，需与该轮数据的模型/统计量约定匹配 |
| `pack-training --without-parent` | 关闭，默认包含父模型 | 接收端已有同一父模型时省去参数传输；接收端的 `check`、`train` 都显式传入父模型目录 |

`pack-training` 的公共解析器也接受 `--rhos-root` 和 `--openpi`，打包分支的文件选择由轮次清单和父模型参数决定。

| 轮次产物 | 内容 |
| --- | --- |
| `abc/` | 筛选后的连续片段、ABC 格式中间结果和跳过片段的记录 |
| `lerobot/` | 转换后的 LeRobot 数据和归一化统计量 |
| `lerobot/combined.json` | 自动生成的训练清单，关联历史 / 人工 / 策略三个池；后续检查、打包和训练读取此文件 |

输入录制清单使用第 3.3 节的对象数组；生成的 `combined.json` 由程序维护。
CPU 检查输出 train/val 三个池的窗口数和真实 OpenPI batch 形状。

训练包包含聚合后的训练数据、父模型、统计量、训练脚本、操作指南和 SHA256 清单。
公开 tokenizer 缓存存在时随包携带；原始 HDF5 与 ABC 图像中间结果保留在原录制目录和轮次目录。

### 3.6 上传参数

| `upload-training` 参数 | 默认值 / 是否必填 | 含义 |
| --- | --- | --- |
| `--archive FILE` | 必填 | 上一步生成的训练包，例如 `exports/r1-training.tar` |
| `--destination OSS_PREFIX` | `oss://ayixiayi-model-relay/relay/hg-dagger/training` | 目标 bucket 下的完整授权前缀，工具会生成绑定内容哈希的发布地址 |
| `--endpoint URL` | `https://oss-cn-shanghai.aliyuncs.com` | bucket 所在区域的 OSS endpoint |
| `--config FILE` | ossutil 自己的默认配置 | 选择本机凭据配置文件；凭据由 ossutil 使用 |
| `--ossutil FILE` | `PATH` 中的 `ossutil` | 指定兼容 ossutil 1.7.19 的可执行文件 |
| `--cache DIR` | `cache/oss` | 本项目的传输缓存；中断后重复同一命令继续传输 |

入口已选择训练包类型 `training`。上传结束后使用输出的完整 OSS 地址和接收命令；更换 bucket、配置和接收路径见第 4 节。

## 4. 更换 OSS

默认根地址为 `oss://ayixiayi-model-relay/relay/hg-dagger`，endpoint 为 `https://oss-cn-shanghai.aliyuncs.com`。
传输调用本机 `ossutil 1.7.19` 及其默认配置。已有可用配置可以直接使用。
首次配置在终端交互填写：

```bash
ossutil config -c "$HOME/.ossutilconfig" -L CH
chmod 600 "$HOME/.ossutilconfig"
```

更换 bucket、授权前缀、区域或配置文件时，替换下面三项；命令中的示例值需改成自己的值：

```bash
./hgrlt upload-training --archive exports/r1-training.tar \
  --destination oss://YOUR_BUCKET/YOUR_AUTHORIZED_PREFIX/hg-dagger/training \
  --endpoint https://oss-cn-shanghai.aliyuncs.com \
  --config "$HOME/.ossutilconfig"
```

账号只获授权某个前缀时，`--destination` 包含完整授权前缀。
自定义 ossutil 路径用 `--ossutil /absolute/path/to/ossutil`。
凭据保存在各环境自己的 OSS 配置中。

上传完成后打印完整 OSS 地址和三条接收命令：下载接收脚本、验证脚本 SHA256、下载解包。
在训练服务器整段复制这三条命令，将最后的 `--destination` 设为持久化目录，例如 `"$HOME/hg-training"`。
接收端使用自定义配置时，第一条 ossutil 命令加 `-c 配置路径`，最后一条 Python 命令加 `--config 配置路径`。
传输中断后重复原命令可使用断点缓存；下载目标须为新目录。

## 5. GPU 训练

以下命令在训练包解压目录执行。首次 `setup` 要求 `uv`、`git` 和网络，建立 Python 3.12 / CUDA 12 JAX 环境及固定版本源码。
使用平台提供的持久化目录保存训练产物：

```bash
cd "$HOME/hg-training"
bash Dagger-Practice/train.sh setup
bash Dagger-Practice/train.sh check --store "$HOME/hg-runs" --exp-name r1
bash Dagger-Practice/train.sh train --store "$HOME/hg-runs" --exp-name r1
```

`check` 使用 CPU；`train` 使用平台分配的可见 GPU。已有兼容环境可通过 `HG_TRAIN_PYTHON` 指定。

| 参数 | 默认值 |
| --- | --- |
| `--steps` / `--batch-size` | 2000 / 8 |
| `--save-interval` / `--seed` | 500 / 42 |
| `--fsdp-devices` | 1；可见 GPU 数须为其倍数，batch-size 须为可见 GPU 数的倍数 |
| `--name` / `--exp-name` | `pi05_yam_hgdagger` / `first`；示例显式使用 `r1` |
| `--rl-token` | 关闭；开启后同时训练 RLT 的 RL token（`L_ro + alpha·L_vla`），checkpoint 带 token 模块 |
| `--rl-alpha` | 1.0；VLA 损失权重，`0` 只训 RL token、VLA 保持父模型不变 |
| `--rl-width` / `--rl-stride` | 512 / 1；RL token 编解码器宽度、相机 token 抽样步长 |
| `--round` | 包内 `round/`，也可指定轮次目录或 `combined.json` |
| `--store` | 包内 `training/`；上面显式使用 `$HOME/hg-runs` |

checkpoint 位于 `STORE/checkpoints/NAME/EXP/<实际step>/`，相邻保存 `EXP.dagger.json`。
同一次运行续跑在原命令后加 `--resume`，数据、父模型和训练设置保持一致。
下一轮使用新实验名，从上轮参数初始化并建立新的优化器状态。

## 6. 更换 checkpoint

### 训练初始化

数据准备时选择父模型和历史 BC；替换以下示例路径后，作为第 3 节 `prepare-data` 的替代命令执行：

```bash
./hgrlt prepare-data \
  --records data/recordings/r1/records.json \
  --parent-checkpoint /absolute/path/to/checkpoint \
  --previous-train /absolute/path/to/bc-train \
  --previous-val /absolute/path/to/bc-val \
  --output data/rounds/r1
```

父模型目录包含 `params/` 和 `assets/`，需匹配 Pi0.5 架构、YAM 数据和动作语义。
多份统计量时用 `--norm-stats` 指定。BC 数据应含 `meta/split.json` 来源与划分信息。
父模型选择保存在轮次清单中，后续打包沿用；使用无父模型包时，训练端的 `check` 和 `train` 都加 `--parent-checkpoint`。

### 推理模型

训练服务器选取实际完成保存的 step，替换 `ACTUAL_STEP` 后执行：

```bash
cd "$HOME/hg-training"
bash Dagger-Practice/train.sh pack-model \
  --checkpoint "$HOME/hg-runs/checkpoints/pi05_yam_hgdagger/r1/ACTUAL_STEP" \
  --output exports/r1-model.tar --model-round 1 --port 8769
bash Dagger-Practice/train.sh upload-model --archive exports/r1-model.tar
```

自定义 OSS 时为 `upload-model` 添加第 4 节参数，并将前缀末尾的 `training` 改为 `model`。
模型包包含推理/下一轮初始化所需参数、统计量、部署清单和训练来源；完整续训 checkpoint 留在训练存储中。

实验室终端输入上传结果的完整 OSS 地址后下载：

```bash
printf '粘贴模型完整 OSS 地址：'; IFS= read -r MODEL_SOURCE
./hgrlt download-model --source "$MODEL_SOURCE" --destination models/r1
./hgrlt preflight --deployment models/r1/deployment.json
```

在独立模型终端执行 `./hgrlt serve-round --deployment models/r1/deployment.json`。
采集时使用同一清单：

```bash
./hgrlt collect --deployment models/r1/deployment.json \
  --output-dir data/recordings/r2 --episodes 20
```

`--deployment` 选择本项目封存的 HG 模型。外部 baseline 的推理适配需要同时核对部署配置、预处理、统计量和推理协议。
部署清单绑定参数内容、轮次和端口；更换端口通过 `pack-model --port` 重新打包。

## 7. 下一轮

在新的采集目录录制本轮 episode 后，`--records` 选择其自动清单，`--previous-round` 指定上轮完整数据，`--parent-checkpoint` 指定上轮模型包内的 checkpoint。
上轮所有数据进入历史池，新一轮保持 80:10:10 采样及原归一化统计量。
本次项目手册提供直接从部署清单读取父模型的完整命令。

长期保留原始 HDF5、每轮数据及历史引用、完整训练 checkpoint 与运行记录、下载后的模型目录和 OSS 地址。
源码与指南可放 Git；数据、模型、日志按实验存储管理。

## 8. 运行边界与常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `跟随卡住：joint X stalled for 90 ticks` | 关节被顶住，通常是接触或主臂没握住；已 HOLD，`h` 纠正或 `p` 重试，本条保留 |
| `策略拒绝：… prediction jump …` | 相邻预测跳变超过 0.30 rad，未发送；已 HOLD，`h` / `p` 恢复 |
| `主臂串口断开或已重连（USB 断开？）` | 主臂 USB 复位，常见于集线器；本条中止，改直连主机 USB 或用 `collect-compat` |
| `主臂 … 扭矩未能关闭，请给主臂断电` | USB 复位后重开串口仍关不掉扭矩；给主臂断电 |
| `A leader is stale: 左/右主臂…` | 该主臂采样超过 0.2 s 未更新；检查它的 USB 线、集线器和供电 |
| `Leader … failed (-1000)` | 主臂 USB 刚断开，串口处于忙状态；同上，并看 `/dev/serial/by-id/` 是否重新出现 |
| `Leader servo … model … cannot follow` | 舵机型号不支持模式 5；加 `--passive-leaders` |
| `set_pu(...) failed … Device or resource busy` | 相机还没从上一条释放；启动时自动重试 3 次（间隔 1 秒），仍失败时重新运行，持续出现时拔插相机 |
| `Confirmed SAFE_HOLD required` / `fault` | 硬件进程不在 HOLD；托住双臂 `./hgrlt daemon-restart` |
| `Daemon requires exclusive session_api_version=1` | 运行的是旧硬件进程；`./hgrlt daemon-restart` 换成本项目的硬件进程 |
| `Policy request failed` | 模型服务未就绪或断开；确认服务监听后 `./hgrlt warmup` |
| 输出已存在 | 选用新的录制文件、轮次目录或导出包名 |
| OSS 403 | 检查 bucket、区域、授权前缀和对象读写权限；ossutil v1 的 `stat` 另需 ACL 权限 |
| 校验失败 | 核对完整 OSS 地址，重新下载或打包 |
| 封存模型预热 | `warmup` 面向 baseline；封存模型用 `SealedPolicyClient` 做推理与延迟检查 |

软件检查、真实纠偏采集、GPU 训练、独立场景实机评估分别记录结果。
