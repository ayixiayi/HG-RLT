# HG-RLT · YAM / Pi0.5

在双臂 YAM 上用两种人在环方法改进 Pi0.5 策略：

- **HG-DAgger**：策略执行时人工接管纠偏，聚合数据后在训练服务器全量微调 VLA，再回实验室采下一轮。
- **RLT（RL Token）**：训练时顺带得到 RL token；实验室在线训练一个小 actor，在关键阶段精修 VLA 的动作，成功/失败标注就是奖励。

两者共用同一套采集、安全检查、录制、数据准备和 OSS 传输：每轮训练加 `--rl-token`，同一个模型既用于 HG 采集，也用于 RLT。

已安装项目从实验室终端启动：

```bash
cd "$HOME/Project/HG-RLT"
./hgrlt paths
./hgrlt preflight
```

- [本次 YAM / Heavier 执行手册](docs/THIS_PROJECT.md)：上机准备、采集、数据准备、训练与模型回传，按顺序复制命令。
- [通用操作指南](docs/OPERATOR_GUIDE.md)：运行目录、checkpoint/OSS 更换和操作边界。
- [RhOSPolicy 接入与独立环境](docs/RHOS_INTEGRATION.md)：作为 RhOSPolicy 子模块的接入方式、独立安装入口、环境与依赖的归属。
- `./hgrlt help`：查看采集、数据准备、上传下载及服务命令。

采集入口：`./hgrlt collect`（主臂全程跟随机械臂）、`./hgrlt collect-compat`（主臂只在按 `h` 后对齐，适合 USB 集线器不稳时）。
接管：按 `h` 停住机械臂，双手握住两只主臂，两个扳机同时扣住 0.5 秒。
RLT：采集加 `--rlt DIR`，关键阶段按 `r` 交给 RL actor；另开终端运行 `./hgrlt rlt-learn`（执行手册第 9 节）。

源码位于 `Dagger-Practice/`、`dagger_offline/`；运行依赖位于 `vendor/`、`envs/`、`runtime/`。
原始记录与轮次数据放 `data/`，模型放 `models/`，传输包放 `exports/`，日志与缓存分别放 `logs/`、`cache/`。

现场控制要求操作者在场；软件检查、真实数据采集、GPU 训练和实机评估各自记录结果。
