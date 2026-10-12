# HG-RLT 运行入口与 ABC 实践

日常 YAM / Pi0.5 操作使用项目根目录的 `./hgrlt`，完整步骤见
[本次项目手册](../docs/THIS_PROJECT.md) 和 [通用操作指南](../docs/OPERATOR_GUIDE.md)。

| 文件 | 用途 |
| --- | --- |
| `lab.sh` | 实验室命令分发，由根目录 `hgrlt` 调用 |
| `run_robot.py` | YAM 策略执行、人工接管、交还和异步 HDF5 录制 |
| `daemon.py` | 硬件进程只读检查与现场确认启动 |
| `train.sh`、`setup_train.sh` | 独立训练包的安装、CPU 检查、Pi0.5 训练与模型导出 |
| `requirements-live.txt`、`requirements-data-cpu.txt`、`requirements-train.txt` | 分离的采集、数据和训练依赖 |
| `patches/` | 时间戳、会话和传输协议补丁与验证 |
| `bundle.py` | 白名单源码分发包；数据与模型由 `hgrlt pack-training` / `train.sh pack-model` 导出 |

## ABC 原生实践

`prepare.py`、`launch.py` 和 `run_cpu.sh` 用于
[amazon-far/abc](https://github.com/amazon-far/abc) 原生模型的数据与训练接口实践。
固定版本为 [d0832d1](https://github.com/amazon-far/abc/commit/d0832d12651d1b260a652861a14648dc5f3660c7)。
ABC 原生训练使用 `.pt` 权重；YAM / Pi0.5 训练使用上面的 `train.sh`。

完整实验室安装含 `vendor/abc` 和 `envs/abc`，从项目根目录运行原生 CPU 测试：

```bash
bash Dagger-Practice/run_cpu.sh
```

另一个兼容环境可通过 `ABC` 和 `PYTHON` 指定。
测试覆盖 HDF5 导出、30 帧窗口、来源组划分、80:10:10 采样、归一化和训练运行记录。
测试产物使用合成数据。
