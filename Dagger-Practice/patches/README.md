# HG-DAgger 部署补丁（不启动服务或硬件）

## 独占会话、follower 接收时间戳、有界 websocket

`hg-protocol.patch` 基于 2026-10-09 从实际 RhOSPolicy 读取的源码制作；不是基于干净上游，不能按 Git HEAD 猜测可应用。
12 个已有文件的原始 SHA256 在 `hg-protocol-base.sha256`，14 个补丁后文件（含两份新测试）在 `hg-protocol-patched.sha256`。
整包未应用到正在使用的共享 RhOSPolicy，也不包含修改30/60Hz配置、启动/重启daemon的动作。

- `session_api_version=1` 为选择性严格合同：原子 acquire，任何活跃租约或初始宽限期阻止抢占，命令携带 owner token；确认 HOLD 后 release。旧 begin 路径仍兼容，但不能抢占活跃的严格会话。token 是进程间所有权标识，不是网络认证。
- 同步检查租约和14个关节的原始 CAN 接收时间；过期/未来/倒退反馈锁定 FAULT，不能把未知反馈重新包装成 confirmed HOLD。拒绝旧 token 不触发通用 RPC 错误 HOLD，避免打断新 owner。
- `received_monotonic_s` 是同一主机的 **CAN 帧解码时间**，不是电机编码器采样时间，也不测量解码前队列延迟。缓存读取不刷新它；runner必须和daemon同主机。
- HG 设置每次 websocket connect+HELLO/reset/infer/close 的 `operation_timeout_s=2`；异常终止client，禁止再发新请求。清理额外最多等待0.5s（加系统调度误差），不是硬实时。Python无法终止永久阻塞线程；控制线程的2s过期锁定不依赖worker返回，晚结果无执行权。
- 新录制为 `observation_command_pair_v2`，保留每关节接收时间；导出再次验证100ms至ACK和非倒退。旧v1/原生记录仍可读取，但没有新增反馈新鲜度证据，不能追溯声称有。

只在**新建隔离副本**演练；先读目标AGENTS，保存dirty状态。下面命令中的路径必须换为隔离源码目录和本补丁目录，不能照抄成现用项目路径：

```bash
cd /path/to/isolated-rhos-copy
P=/absolute/path/to/Dagger-Practice/patches
# 若此检查全通过，已应用，不再打补丁：
sha256sum -c "$P/hg-protocol-patched.sha256"
# 未应用时，只有下面原始hash全匹配且check成功，才可继续；任一步失败即停。
sha256sum -c "$P/hg-protocol-base.sha256" &&
  git apply --check "$P/hg-protocol.patch" &&
  git apply "$P/hg-protocol.patch" &&
  sha256sum -c "$P/hg-protocol-patched.sha256"
```

部分已改/不匹配时人工审查，不强制覆盖，不用模糊上下文替代hash。将来改共享源码及重启daemon需要单独的空闲窗口和授权；文件替换不会更新现有Python进程。

CPU协议测试使用 `requirements-live.txt` 的独立环境。设 `R` 为补丁后RhOSPolicy副本、`H` 为HG包根目录、`ABC` 为干净固定ABC，**从H执行**：

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
export PYTHONPATH="$ABC:$R:$R/XPolicyLab:$R/Client/arms/YAM/gello_software:$R/Client/arms/YAM/i2rt:$H"
/path/to/live-env/bin/python -m pytest -q \
  "$R/tests/test_ws_model_client_operation_timeout.py" \
  "$R/tests/test_yam_pi05_inference.py" \
  "$R/tests/test_follower_receive_timestamps.py" \
  "$R/Client/arms/YAM/gello_software/tests/test_safety_controller.py" \
  "$R/Client/arms/YAM/gello_software/tests/test_safety_rpc.py" \
  "$R/Client/arms/YAM/gello_software/tests/test_safety_daemon_client.py" \
  "$R/Client/arms/YAM/gello_software/tests/test_robot_lifecycle.py" \
  "$R/Client/arms/YAM/gello_software/tests/test_parallel_yam_init.py" \
  Dagger-Practice/patches/hg_protocol_integration.py
```

集成测试启动的是临时 localhost/inproc RPC + 假电机，不是硬件daemon；验证两次录制、释放/重获、旧owner拒绝以及反馈在发送前过期。不得拿它代替现场急停、失联或接管验收。

## GELLO leader 采样时间戳补丁

前序时间戳补丁已在ziyu部署；2026-10-09实际相机/leader联读暴露了串口事务持锁超过10ms，
导致正常读取也报 `Dynamixel joint sample lock is busy`。新版用独立sample锁发布完整样本，
不等待在途USB事务，不刷新旧数据的时间戳，10ms锁等待和调用方新鲜度限制不变。
只更新HG隔离副本，未修改共享RhOSPolicy驱动，也不需重启follower daemon。
`dynamixel_fresh_sample.py` 现有14项无硬件测试；新回归在旧驱动实际失败、新驱动通过。
随后两臂leader与三相机300次真实联读通过，最大leader样本年龄32.04ms；有一次串口通信失败，
完整样本保留且未伪造新时间。该采样检查不含follower运动或标定验收。

目标：`Client/arms/YAM/gello_software/gello/dynamixel/driver.py`。
SHA256：

- 补丁前：`b2c4e0cd294ff463d4c55fe06247fec2ef4c1122f2ef2ce45fd0160639616cf8`
- 旧时间戳版本（共享目录仍为此版本）：`e612824863ec030856669e3d7ac2ba045f978de1329733480fc6ac925e1e9cdf`
- 新补丁后：`8685269cbcb3f25b91f5621f095196dafa481bd8734ef29ab04da2b893c2e10b`
- 测试：`ba2ce1483d024d39fef6112acfce9f757d2f14a1d0dda34c0fa59473f05d4ad0`

迁移前先读取目标仓库的 AGENTS 和 dirty 状态，再校验目标 hash。
已经匹配新补丁后hash则不要重复应用。本patch从补丁前版本生成，不可直接叠加到旧时间戳版本；
旧版本须保留备份后核对差异升级。与已知hash均不匹配时先核对差异，不能强制覆盖。
补丁相对于 RhOSPolicy 根目录，先用 `git apply --check` 检查。
确认有部署授权后才应用补丁；本 README 不授权远程修改或硬件操作。

纯软件检查（环境须已具备 numpy、pytest、dynamixel_sdk）：

```bash
PYTHONPATH=Client/arms/YAM/gello_software python -m pytest \
  /path/to/Dagger-Practice/patches/dynamixel_fresh_sample.py -q
```

`get_joint_sample()` 返回完整硬件采样的角度与同一锁内发布的 monotonic 时间。
读取失败、不完整、fake、锁忙不会生成“新”时间；调用者仍负责新鲜度和左右臂倒退检查。
这些 mock 测试不证明真实采样周期、标定、USB/串口稳定性或双臂实机安全。

## Follower 无锁采样补丁

`follower-lockfree-sample.patch` 在 `hg-protocol.patch` 之后应用（前置 hash 即 `hg-protocol-patched.sha256` 中这两个文件，后置 hash 见 `follower-lockfree-sample.sha256`）。

2026-10-12 现场：daemon 健康检查每 0.5s 调 `get_joint_sample`，而控制循环在整个 CAN 收发期间持有 `_state_lock`；10ms 等锁偶发超时即被判为反馈失效，daemon 锁存 FAULT（`RIGHT follower control path unhealthy`）。补丁改为直接读取每周期整体替换发布的 `JointStates`，不等控制锁；新鲜度仍由每关节 CAN 接收时间和 100ms 上限判定。

驱动只在进程启动时加载，部署后需按现场流程停止旧 daemon 再 `./hgrlt daemon-start`。
