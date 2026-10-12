# RhOSPolicy 接入与独立运行环境

HG-RLT 是独立的公开仓库 [`ayixiayi/HG-RLT`](https://github.com/ayixiayi/HG-RLT)，以子模块形式挂在 RhOSPolicy 根目录的 `HG-RLT/`。
RhOSPolicy 只记录子模块的提交和一个转发入口；HG-RLT 的源码、测试、依赖定义、协议补丁和文档在自己的仓库里提交和审阅。
HG-RLT 的 Python、运行依赖、模型和数据由自身管理，RhOSPolicy 的主环境继续按自己的配置运行。

## 获取与入口

```bash
git clone --recurse-submodules git@github.com:yuz1wan/RhOSPolicy.git
# 已有 RhOSPolicy checkout：
git submodule update --init HG-RLT
```

在 RhOSPolicy 根目录执行：

```bash
./scripts/hgrlt help
./scripts/hgrlt paths
```

`scripts/hgrlt` 将参数原样交给子模块的 `hgrlt` 入口，由它选择环境和目录；也可直接运行 `./HG-RLT/hgrlt`。
`help` 和 `paths` 在纯源码 checkout 中即可运行；`preflight` 检查已准备好的完整运行环境。

已有独立完整安装时，通过 `HG_RLT_ROOT` 选择它。实验室安装在 `~/Project/HG-RLT`：

```bash
HG_RLT_ROOT="$HOME/Project/HG-RLT" ./scripts/hgrlt paths
HG_RLT_ROOT="$HOME/Project/HG-RLT" ./scripts/hgrlt preflight
```

相对文件参数始终以选中的安装根目录为准，例如 `collect --output-dir data/recordings/r1` 保存到该安装的 `data/recordings/`。
实验室安装本身就是 `ayixiayi/HG-RLT` 的工作副本：更新源码用 `git pull`，`vendor/`、`envs/`、`runtime/`、配置、数据、模型、日志和缓存由 `.gitignore` 排除，不受影响。

## HG-RLT 自己管理的环境

HG-RLT 使用 Python 3.12，各环境按用途分别维护：

| 用途 | 默认解释器 | 依赖定义或准备入口 | 覆盖变量 |
| --- | --- | --- | --- |
| 采集、现场控制、离线预检 | `envs/live/bin/python` | `Dagger-Practice/requirements-live.txt` 与匹配的 YAM 运行依赖 | `HG_LIVE_PYTHON` |
| 数据转换、训练 batch 检查 | `envs/data/bin/python` | `Dagger-Practice/requirements-data-cpu.txt`，CPU PyTorch/JAX | `HG_DATA_PYTHON` |
| Pi0.5 模型推理 | `envs/model/bin/python` | 固定版本 OpenPI 的兼容 CUDA 推理环境 | `HG_MODEL_PYTHON` |
| GPU 训练 | `envs/train/bin/python` | 训练包中的 `bash Dagger-Practice/train.sh setup`；依赖 `requirements-train.txt` | `HG_TRAIN_PYTHON` |
| ABC 原生接口测试 | `envs/abc/bin/python` | `Dagger-Practice/requirements-cpu.txt`，与数据转换环境分开 | `run_cpu.sh` 的 `PYTHON` |
| OSS 文件传输 | 系统 `python3` | Python 标准库与本机 ossutil 配置 | `HG_TRANSFER_PYTHON` |

RhOSPolicy 与 HG-RLT 的环境更新各自进行。启动 HG-RLT 时使用上表中的解释器，
即使终端已经激活 RhOSPolicy 的环境，HG-RLT 的默认解释器选择仍由自身入口确定。
自定义解释器使用绝对路径，并满足对应依赖定义。

完整安装还需准备以下内容，再执行 `hg preflight`：

| 安装内容 | 默认位置 | 自定义入口 |
| --- | --- | --- |
| 兼容 RhOS / YAM / XPolicyLab 运行副本及机器人配置 | `vendor/rhos/` | `HG_RHOS_ROOT` |
| 固定版本 ABC | `vendor/abc/` | `HG_ABC` |
| 固定版本 OpenPI | `vendor/openpi/` | `HG_OPENPI` |
| 匹配的 LeRobot v2.1 源码 | `vendor/lerobot/` | `HG_LEROBOT` |
| baseline 或本轮 checkpoint | `models/` | 操作指南的 checkpoint / deployment 参数 |
| 历史 BC 数据及后续录制 | `data/` | 操作指南的历史数据和录制清单参数 |

硬件时间戳、控制会话和传输协议的匹配要求见 [协议补丁说明](../Dagger-Practice/patches/README.md)。
YAM 和 XPolicyLab 的协议改动按各自上游仓库维护，HG 运行副本按已验证的组合部署。
机器人配置、模型和数据准备好后，采集、数据处理及训练包流程见 [操作指南](OPERATOR_GUIDE.md)。

## 源码与实验产物的边界

`ayixiayi/HG-RLT` 保存源码、依赖定义、测试、补丁和文档；运行依赖、Python 环境、安装配置、原始记录、模型、训练输出、传输包及凭据按安装或实验存储管理，由 `.gitignore` 排除。
源码中的 [录制清单示例](records.example.json) 用于说明输入格式。

功能修改在 HG-RLT 仓库提交；需要 RhOSPolicy 使用新版本时，在 RhOSPolicy 里更新子模块指针并提交 PR。
YAM、i2rt 和 XPolicyLab 的协议改动按各自上游仓库维护，HG-RLT 的 `Dagger-Practice/patches/` 保存已验证组合的补丁和校验值。

现场控制按操作指南确认设备、操作者和断电条件；机械臂操作和 GPU 训练由对应命令显式启动。
