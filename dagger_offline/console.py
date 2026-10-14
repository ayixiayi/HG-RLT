"""Small, line-oriented Chinese CLI output; no terminal dependency or animation."""

import contextlib
import os
import re
import sys
import threading
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
        "  h  人工接管（主臂跟随时：先 HOLD，扳机断电；两扳机完全松开再扣到底，回位后接管）\n"
        "  p  交还策略      r  交给 RL actor（--rlt 时）      s  停止 / HOLD\n"
        "  c  标记检查点    x  丢弃检查点后的片段并保持\n"
        "  k  结束本条并标结果    q  整条弃用    ?  查看按键\n"
        "  结束后：1 成功 / 2 失败 / 3 弃用；标记前已 HOLD 并释放控制权",
        flush=True,
    )


class DriverOutput:
    """Line filter for the GELLO Dynamixel driver's own prints.

    Connection attempts and its reader thread's ``warning, comm failed: N``
    lines (one -3001/-3002 pair on every start is normal; the reader retries)
    are dropped; a retried connection attempt is reported in this format.
    Everything else passes through unchanged.
    """

    ATTEMPT = re.compile(r"Attempting to initialize Dynamixel driver \(attempt (\d+)/(\d+)\)")
    HIDDEN = (
        re.compile(r"Successfully initialized Dynamixel driver on .*"),
        re.compile(r"warning, comm failed: -?\d+"),
    )

    PREFIXES = ("Attempting to initialize", "Successfully initialized", "warning, comm")

    def __init__(self, stream):
        self.stream, self._buffer, self._lock = stream, "", threading.Lock()

    def write(self, text):
        with self._lock:
            pending = ""
            if (
                self._buffer
                and text.startswith(self.PREFIXES)
                and not self._buffer.startswith(self.PREFIXES)
            ):
                # A driver line starts after an unterminated prompt.
                pending, self._buffer = self._buffer, ""
            self._buffer += text
            *lines, self._buffer = self._buffer.split("\n")
        if pending:
            self.stream.write(pending)
        for line in lines:
            self._line(line)
        return len(text)

    def _line(self, line):
        bare = line.strip()
        attempt = self.ATTEMPT.fullmatch(bare)
        if attempt:
            if attempt.group(1) != "1":
                message(
                    "主臂",
                    f"串口连接重试 {attempt.group(1)}/{attempt.group(2)}",
                    level="warning",
                    stream=self.stream,
                )
            return
        if any(pattern.fullmatch(bare) for pattern in self.HIDDEN):
            return
        self.stream.write(line + "\n")

    def flush(self):
        # A prompt without newline goes out as is; a driver line still being
        # written (print sends text and newline separately) stays buffered.
        with self._lock:
            rest = self._buffer
            if rest and not rest.startswith(self.PREFIXES):
                self._buffer = ""
                self.stream.write(rest)
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def driver_output():
    """Route stdout through DriverOutput while hardware drivers run."""
    previous = sys.stdout
    sys.stdout = DriverOutput(previous)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = previous
