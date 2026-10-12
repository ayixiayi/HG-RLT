"""Small, line-oriented Chinese CLI output; no terminal dependency or animation."""

import os
import sys
import time


def message(label, text, *, level="info", stream=None):
    stream = sys.stdout if stream is None else stream
    colors = {"info": "36", "success": "32", "warning": "33", "error": "31"}
    prefix = f"[{time.strftime('%H:%M:%S')}] [{label}]"
    if stream.isatty() and "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb":
        prefix = f"\033[1;{colors[level]}m{prefix}\033[0m"
    print(f"{prefix} {text}", file=stream, flush=True)


def show_mode(state, *, waiting_for_policy=False):
    text, level = {
        "HOLD": (
            "HOLD · 保持｜等待模型结果；h 接管 / s 停止"
            if waiting_for_policy
            else "HOLD · 保持｜h 人工接管 / p 策略交还",
            "warning",
        ),
        "INTERVENTION": ("INTERVENTION · 人工跟随｜p 交还策略 / s 停止", "success"),
        "POLICY_RUNNING": ("POLICY_RUNNING · 策略控制｜h 立即接管 / s 停止", "info"),
        "FAULT": ("FAULT · 故障锁定｜本条记录不可用于训练，请检查停止原因", "error"),
    }[state]
    message("模式", text, level=level)


def show_controls():
    message("操作", "按键直接生效，无需回车；启动后保持，不会自动运动")
    print(
        "  h  人工接管（主臂跟随时：先 HOLD，两个扳机都扣到底再松开才接管）\n"
        "  p  交还策略      r  交给 RL actor（--rlt 时）      s  停止 / HOLD\n"
        "  c  标记检查点    x  丢弃检查点后的片段并保持\n"
        "  k  结束本条并标结果    q  整条弃用    ?  查看按键\n"
        "  结束后：1 成功 / 2 失败 / 3 弃用；标记前已 HOLD 并释放控制权",
        flush=True,
    )
