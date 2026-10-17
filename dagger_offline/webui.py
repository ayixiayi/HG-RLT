"""Web operator console: `./hgrlt ui`, then open the printed URL.

One page replaces the model / daemon / collection terminals. Children are the
same `./hgrlt` commands an operator types: collection and daemon restart/stop
run in a pseudo-terminal whose prompts the operator answers from the page; the
console never answers an onsite confirmation itself.

Security: a per-install token (0600 file) passed once in the URL and then kept
in an HttpOnly SameSite=Strict cookie guards every request; state changes are
POST-only with the X-HGRLT header; the server binds 127.0.0.1 unless --host
says otherwise; argv is built from validated fields only (deployments inside
models/, output directories inside data/); anything that moves or powers the
arms needs the typed phrase and runs alone.
"""

import argparse
import codecs
import collections
import fcntl
import hmac
import http.server
import json
import os
import re
import secrets
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "webui_static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MOTION_PHRASE = "我在机械臂旁"
COOKIE = "hgrlt_ui"
HEADER = "X-HGRLT"
BASELINE_PORT = 8767
DAEMON_PORTS = (6001, 6002)  # safety / hardware RPC of the YAM profile
DEPLOYMENT_SCHEMA = "hg-pi05-sealed-v1"
# A Pi0.5 server needs roughly this much free GPU memory to load.
MIN_GPU_FREE_MIB = 10240
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_EPISODES = 500
MAX_BODY = 64 * 1024
OUTPUT_LIMIT = 256 * 1024  # characters of output kept per job for a new page
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
# Interpreter flags may precede the script (python -u .../collect.py).
COLLECT = re.compile(r"^\S*python\S*(?: \S+)*? (?:\S*/)?(?:collect|run_robot)\.py(?: |$)")
# Keys written as is to a collection; 'enter' and 'ctrl-c' are named.
SESSION_KEYS = {k: k for k in "phscxkqr?123456"} | {"enter": "\r", "ctrl-c": "\x03"}
DAEMON_KEYS = {"enter": "\r", "ctrl-c": "\x03"}
# Make the pty slave the child's controlling terminal (Ctrl-C, job control).
CTTY_SHIM = (
    "import fcntl,os,sys,termios;"
    "fcntl.ioctl(0,termios.TIOCSCTTY,0);"
    "os.execvp(sys.argv[1],sys.argv[1:])"
)


class Refused(Exception):
    """A request the console declines; code is the HTTP status."""

    def __init__(self, message, code=409):
        super().__init__(message)
        self.code = code


# ---- token and log -------------------------------------------------------------------


def console_token(fresh=False, file=None):
    """HGRLT_UI_TOKEN, else ~/.config/hgrlt/console-token (0600, kept across starts)."""
    if os.environ.get("HGRLT_UI_TOKEN"):
        return os.environ["HGRLT_UI_TOKEN"]
    file = Path(file or Path.home() / ".config" / "hgrlt" / "console-token")
    if not fresh and file.is_file():
        saved = file.read_text().strip()
        if len(saved) >= 32:
            os.chmod(file, 0o600)
            return saved
    token = secrets.token_urlsafe(32)
    file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(token + "\n")
    os.replace(temporary, file)
    return token


def private_file(path):
    """Open for append, owner-only (it can hold paths and operator input)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8", buffering=1)


# ---- collection output -> state ------------------------------------------------------

LINE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] \[([^\]]+)\] ?(.*)$")
MODES = ("HOLD", "POLICY_RUNNING", "INTERVENTION", "FAULT")
PROMPTS = {
    "回车开始本条；q 结束本次连续采集：": "await_start",
    "按回车确认；输入其它内容或 Ctrl-C 取消：": "confirm_homing",
    "确认重启：": "await_phrase",
    "确认下电：": "await_phrase",
    "确认启动并使能新 daemon：": "await_phrase",
}
NOTICES = {"跟随卡住", "策略拒绝", "故障", "采集停止", "采集结束", "检查点", "丢弃", "主臂", "无人操作"}
STATS = re.compile(
    r"累计 (\d+) 条 · 成功 (\d+) · 失败 (\d+) · 弃用 (\d+) · 未完成 (\d+) · "
    r"未标结果 (\d+) · 纳入清单 (\d+) 条"
)


class SessionParser:
    """Mode and stage of `./hgrlt collect` (or a daemon command) from its output.

    Reads the console.message lines (`[HH:MM:SS] [标签] 文本`) and the input()
    prompts, which end without a newline. Display only: control stays with the
    child process and the operator's keys.
    """

    def __init__(self):
        self.stage = "starting"
        self.mode = None
        self.takeover = None
        self.rl = False
        self.episode = None
        self.stats = None
        self.notice = None
        self._partial = ""
        self._typed = ""

    def state(self):
        return {
            "stage": self.stage,
            "mode": self.mode,
            "takeover": None if self.takeover is None else dict(self.takeover),
            "rl": self.rl,
            "episode": self.episode,
            "stats": self.stats,
            "notice": self.notice,
        }

    def feed(self, text):
        """Returns True when the state changed."""
        before = self.state()
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            # A carriage return rewrites the line: the last segment is visible.
            self._line(line.rstrip("\r").split("\r")[-1])
        tail = self._partial.split("\r")[-1].rstrip()
        for prompt, stage in PROMPTS.items():
            if tail.endswith(prompt):
                self.stage = stage
        return self.state() != before

    def typed(self, data):
        """What the operator sent; an answered prompt is no longer waiting."""
        before = self.state()
        for char in data:
            if char != "\r":
                self._typed = (self._typed + char)[-64:]
                continue
            line, self._typed = self._typed.strip().lower(), ""
            if self.stage == "await_start":
                self.stage = "ending" if line == "q" else "starting" if not line else self.stage
            elif self.stage == "confirm_homing":
                self.stage = "homing" if not line else "starting"
            elif self.stage == "await_phrase":
                self.stage = "running"
        return self.state() != before

    def _line(self, line):
        match = LINE.match(line)
        if not match:
            return
        label, text = match.groups()
        if label == "模式":
            mode = text.split(" · ")[0].strip()
            if mode in MODES:
                self.mode = mode
                self.stage = "fault" if mode == "FAULT" else "recording"
                self.takeover = None
                if mode != "POLICY_RUNNING":
                    self.rl = False
        elif label == "操作":
            self.stage = "recording"
        elif label == "接管准备":
            if "扳机已断电" in text:
                self.takeover = {"phase": "confirm", "left": False, "right": False}
            elif self.takeover is not None:
                if "左扳机已确认" in text:
                    self.takeover["left"] = True
                if "右扳机已确认" in text:
                    self.takeover["right"] = True
                if "确认完成" in text:
                    self.takeover["phase"] = "return"
                if "扳机未回到夹爪开度" in text:
                    self.takeover["warning"] = text
        elif label == "接管":
            self.takeover = None
        elif label == "RL" and "RL actor 接管" in text:
            self.rl = True
        elif label == "结果标记":
            self.stage, self.takeover = "review", None
        elif label == "失败类型":
            self.stage = "failure_type"
        elif label in ("保存", "丢弃") and text.startswith("已确认 HOLD 并释放控制权"):
            # The end of an episode; `x` also reports under 丢弃 while recording.
            done = "记录已保存" in text or "整条记录" in text
            self.stage = "saved" if done else "saving"
        elif label == "采集统计":
            found = STATS.search(text)
            if found:
                keys = ("total", "success", "fail", "discard", "incomplete", "unlabelled", "listed")
                self.stats = dict(zip(keys, map(int, found.groups()), strict=True))
        elif label == "下一条":
            self.stage, self.mode, self.takeover, self.rl = "next", None, None, False
            self.episode = text.split("；")[0].strip()
        elif label == "归零确认" and "自动回零" in text:
            self.stage = "homing"  # unattended: no prompt, nothing to answer
        elif label == "无人操作":
            self.stage = "ending"
        elif label in ("归零完成", "现场检查"):
            self.stage = "starting"
        elif label == "故障":
            self.stage, self.mode = "fault", "FAULT"
        elif label in ("采集停止", "采集结束"):
            self.stage = "ended"
        elif label == "录制清单":
            self.stage = "finished"
        if label in NOTICES:
            self.notice = {"label": label, "text": text, "time": time.time()}


# ---- children ------------------------------------------------------------------------


class Job:
    """One child of the console: a collection, a daemon command or a model server."""

    def __init__(self, kind, label, argv, *, meta=None, pty=False, transcript=None):
        self.id = secrets.token_hex(6)
        self.kind, self.label, self.argv = kind, label, list(argv)
        self.meta = dict(meta or {})
        self.started, self.ended, self.code = time.time(), None, None
        self.parser = SessionParser() if pty else None
        self.master = None
        self.proc = None
        self.stopping = False
        self._output = collections.deque()
        self._size = 0
        self._lock = threading.Lock()
        self._write = threading.Lock()
        self._transcript = transcript

    @property
    def running(self):
        return self.ended is None

    def output(self):
        with self._lock:
            return "".join(self._output)

    def append(self, text):
        with self._lock:
            self._output.append(text)
            self._size += len(text)
            while self._size > OUTPUT_LIMIT and len(self._output) > 1:
                self._size -= len(self._output.popleft())
        if self._transcript is not None:
            try:
                self._transcript.write(text)
            except (OSError, ValueError):
                pass

    def write(self, data, timeout=2.0):
        if self.master is None or not self.running:
            raise Refused("这个进程已经结束")
        data = data.encode()
        with self._write:
            # Nonblocking: a child that stops reading cannot hang the console.
            deadline = time.monotonic() + timeout
            while data:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([], [self.master], [], left)[1]:
                    raise Refused("进程没有读取输入（终端输入已满）")
                try:
                    data = data[os.write(self.master, data):]
                except BlockingIOError:
                    continue

    def signal(self, signum):
        if self.proc is not None and self.running:
            try:
                os.killpg(self.proc.pid, signum)
            except ProcessLookupError:
                pass

    def snapshot(self, output=False):
        view = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "argv": self.argv,
            "pid": None if self.proc is None else self.proc.pid,
            "started": self.started,
            "ended": self.ended,
            "code": self.code,
            "running": self.running,
            "stopping": self.stopping,
            "meta": self.meta,
            "state": None if self.parser is None else self.parser.state(),
        }
        if output:
            view["output"] = self.output()
        return view


# ---- machine probes (replaced in tests) ----------------------------------------------


class System:
    """Read-only views of the machine: GPU, processes, listening ports, git."""

    def __init__(self, root):
        self.root = Path(root)

    def _run(self, argv, timeout=10):
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    def processes(self):
        """{pid: (user, args)} of every process."""
        text = self._run(["ps", "-eo", "pid=,user:32=,args="]) or ""
        table = {}
        for line in text.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                table[int(parts[0])] = (parts[1], redact(parts[2] if len(parts) == 3 else ""))
        return table

    def gpu(self):
        query = "index,name,memory.used,memory.total"
        text = self._run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
        if text is None:
            return None
        gpus = []
        for line in text.strip().splitlines():
            index, name, used, total = (part.strip() for part in line.split(","))
            gpus.append(
                {"index": int(index), "name": name, "used_mib": int(used), "total_mib": int(total)}
            )
        apps = self._run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        )
        compute = []
        for line in (apps or "").strip().splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3 and parts[0].isdigit():
                used = int(parts[1]) if parts[1].isdigit() else None
                compute.append({"pid": int(parts[0]), "used_mib": used, "name": parts[2]})
        return {"gpus": gpus, "compute": compute}

    def listening(self):
        """{port: inode} of TCP sockets in LISTEN state (IPv4 and IPv6)."""
        ports = {}
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                rows = Path(table).read_text().splitlines()[1:]
            except OSError:
                continue
            for row in rows:
                fields = row.split()
                if len(fields) > 9 and fields[3] == "0A":
                    ports.setdefault(int(fields[1].rsplit(":", 1)[1], 16), int(fields[9]))
        return ports

    def socket_owners(self, inodes):
        """{inode: pid} for sockets of processes this user can inspect."""
        wanted = {f"socket:[{inode}]": inode for inode in inodes}
        owners = {}
        if not wanted:
            return owners
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                for fd in (proc / "fd").iterdir():
                    inode = wanted.get(os.readlink(fd))
                    if inode is not None:
                        owners[inode] = int(proc.name)
            except OSError:
                continue
        return owners

    def git(self):
        describe = self._run(["git", "-C", str(self.root), "describe", "--always", "--dirty"], 5)
        subject = self._run(["git", "-C", str(self.root), "log", "-1", "--format=%h %cs %s"], 5)
        return {
            "describe": (describe or "").strip() or None,
            "commit": (subject or "").strip() or None,
        }


SECRET_ARG = re.compile(
    r"((?:--?|\b)[\w-]*(?:token|secret|password|passwd|key)(?![a-z])[\w-]*)(=|\s+)(\S+)",
    re.IGNORECASE,
)


def redact(args):
    """Command lines are shown on the page: hide values of token/secret arguments."""
    return None if args is None else SECRET_ARG.sub(r"\1\2<隐藏>", args)


def collect_processes(processes, exclude=()):
    """External `collect.py` / `run_robot.py` runs (as cache/hg-deploy/lab-pull.sh)."""
    return [
        {"pid": pid, "user": user, "args": args}
        for pid, (user, args) in sorted(processes.items())
        if pid not in exclude and COLLECT.match(args)
    ]


def classify_daemon(args, root):
    """Whose hardware daemon owns the safety port: this install, another, unknown."""
    if args is None:
        return "unknown"
    if "yam_safety_daemon" not in args:
        return "other"
    return "hgrlt" if f"{Path(root).resolve()}/vendor/rhos/" in args else "foreign"


# ---- results (read-only) --------------------------------------------------------------


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_episode(path, policies):
    """One recording's outcome and takeovers from its HDF5 attrs and events."""
    import h5py

    row = {"name": path.name}
    try:
        with h5py.File(path, "r") as recording:
            attrs = recording.attrs
            row.update(
                outcome=_text(attrs.get("outcome", "unlabelled")),
                complete=bool(attrs.get("complete", False)),
                usable=bool(attrs.get("usable", False)) and not bool(attrs.get("unusable", True)),
                failure_type=_text(attrs["failure_type"]) if "failure_type" in attrs else None,
            )
            states, reasons, stamps = [], [], []
            if "dagger_events" in recording:
                events = recording["dagger_events"]
                states = [_text(s) for s in events["controller_state"][:]]
                reasons = [_text(s) for s in events["reason"][:]]
                stamps = [int(t) for t in events["timestamp_ns"][:]]
            try:
                metadata = json.loads(_text(attrs["metadata_json"]))
            except (KeyError, ValueError):
                metadata = {}
    except OSError as exc:
        # Unreadable usually means a collection still writes it.
        return {**row, "outcome": "unreadable", "error": str(exc)[:200]}
    row["takeovers"] = sum(
        state == "INTERVENTION" and (i == 0 or states[i - 1] != "INTERVENTION")
        for i, state in enumerate(states)
    )
    row["rl_segments"] = reasons.count("rl selected")
    row["duration_s"] = round((stamps[-1] - stamps[0]) / 1e9, 1) if len(stamps) > 1 else None
    policy_id = metadata.get("sealed_policy_id")
    if policy_id:
        row["model"] = policies.get(policy_id, f"未知模型 {policy_id[:12]}")
    elif "model_round" in metadata:
        row["model"] = "baseline" if metadata["model_round"] == 0 else f"round {metadata['model_round']}"
    else:
        row["model"] = None
    return row


def summarize(rows):
    kept = [r for r in rows if r["folder"] == "episodes"]
    labelled = [r for r in kept if r.get("outcome") in ("success", "fail") and r.get("usable")]
    success = [r for r in labelled if r["outcome"] == "success"]
    failure_types = collections.Counter(
        r.get("failure_type") or "unspecified" for r in labelled if r["outcome"] == "fail"
    )
    takeovers = sum(r.get("takeovers", 0) for r in labelled)
    autonomous = [r for r in success if r.get("takeovers", 0) == 0]
    trash = [r for r in rows if r["folder"] == "trash"]
    return {
        "attempts": len(rows),
        "labelled": len(labelled),
        "success": len(success),
        "fail": len(labelled) - len(success),
        "success_rate": len(success) / len(labelled) if labelled else None,
        "autonomous_success": len(autonomous),
        "autonomous_rate": len(autonomous) / len(labelled) if labelled else None,
        "takeovers": takeovers,
        "takeovers_per_episode": takeovers / len(labelled) if labelled else None,
        "episodes_with_takeover": sum(r.get("takeovers", 0) > 0 for r in labelled),
        "failure_types": dict(failure_types),
        "discarded": sum(r.get("outcome") == "discard" for r in trash),
        "incomplete": sum(r.get("outcome") != "discard" for r in trash),
        "unlabelled": sum(r.get("outcome") not in ("success", "fail") for r in kept),
        "models": sorted({r["model"] for r in rows if r.get("model")}),
    }


# ---- the console ---------------------------------------------------------------------


class Console:
    def __init__(self, root=ROOT, *, token, system=None, hgrlt=None, logs=None, on_shutdown=None):
        self.root = Path(root).resolve()
        self.token = token
        self.system = system or System(self.root)
        self.hgrlt = str(hgrlt or self.root / "hgrlt")
        self.logs = Path(logs or self.root / "logs")
        self.on_shutdown = on_shutdown
        self.jobs = {}
        self.lock = threading.RLock()
        self.clients = set()
        self.closing = False
        self._check = threading.Lock()
        self._cache = {}
        self._log = private_file(self.logs / "console.log")

    # -- log and events

    def log(self, text):
        try:
            self._log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")
        except (OSError, ValueError):
            pass

    def broadcast(self, event):
        data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            client.put(data)

    def _job_event(self, job):
        self.broadcast({"type": "job", "job": job.snapshot()})

    # -- paths and fields

    @property
    def models_dir(self):
        return self.root / "models"

    @property
    def data_dir(self):
        return self.root / "data"

    def _name(self, value, what):
        if not isinstance(value, str) or not NAME.fullmatch(value):
            raise Refused(f"{what}只能用字母、数字、点、下划线和连字符（最长 64，首字符为字母或数字）", 400)
        return value

    def _inside(self, path, base, what):
        resolved, base = path.resolve(), base.resolve()
        if not resolved.is_relative_to(base) or resolved == base:
            raise Refused(f"{what}必须在 {base} 内", 400)
        return resolved

    def deployment(self, name):
        """baseline, or models/NAME/deployment.json; returns (path or None, port, info)."""
        if name == "baseline":
            return None, BASELINE_PORT, {"name": "baseline", "port": BASELINE_PORT}
        self._name(name, "模型名")
        path = self._inside(self.models_dir / name / "deployment.json", self.models_dir, "模型")
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise Refused(f"读不出 models/{name}/deployment.json：{exc}", 400) from exc
        port = manifest.get("port")
        if manifest.get("schema") != DEPLOYMENT_SCHEMA or type(port) is not int or not 1024 <= port <= 65535:
            raise Refused(f"models/{name}/deployment.json 不是封存的部署清单", 400)
        policy_id = manifest.get("policy_id")
        inference = (manifest.get("run") or {}).get("inference") or {}
        return path, port, {
            "name": name,
            "port": port,
            "policy_id": policy_id if isinstance(policy_id, str) else None,
            "model_round": manifest.get("model_round"),
            "checkpoint": manifest.get("checkpoint"),
            "rl_token": inference.get("rl_token") is not None,
        }

    def models(self):
        found = [self.deployment("baseline")[2] | {"checkpoint": "configs/deployments/pi05_yam_heavier.yaml"}]
        for path in sorted(self.models_dir.glob("*/deployment.json")):
            try:
                found.append(self.deployment(path.parent.name)[2])
            except Refused as exc:
                found.append({"name": path.parent.name, "error": str(exc)})
        return found

    def policies(self):
        return {m["policy_id"]: m["name"] for m in self.models() if m.get("policy_id")}

    def session_plan(self, p):
        """argv of `./hgrlt collect` from validated fields only."""
        if not isinstance(p, dict):
            raise Refused("参数格式错误", 400)
        command = p.get("command", "collect")
        if command not in ("collect", "collect-compat"):
            raise Refused("入口只能是 collect 或 collect-compat", 400)
        kind = p.get("kind", "eval")
        if kind not in ("eval", "recordings"):
            raise Refused("类型只能是 eval（评估）或 recordings（采集）", 400)
        if p.get("rlt"):
            kind = "recordings"  # rlt-learn reads data/recordings/<目录名>
        model = p.get("deployment", "baseline")
        deployment, port, _ = self.deployment(model if isinstance(model, str) else "")
        name = self._name(p.get("name"), "目录名")
        output = self._inside(self.data_dir / kind / name, self.data_dir, "输出目录")
        episodes = p.get("episodes", 20)
        if type(episodes) is not int or not 0 <= episodes <= MAX_EPISODES:
            raise Refused(f"条数为 0–{MAX_EPISODES} 的整数（0 不限）", 400)
        argv = [self.hgrlt, command, "--output-dir", str(output), "--episodes", str(episodes)]
        if deployment is not None:
            argv += ["--deployment", str(deployment)]
        rlt = p.get("rlt") or None
        if rlt is not None:
            if deployment is None:
                raise Refused("RLT 需要带 RL token 的封存模型，baseline 不能用 --rlt", 400)
            rlt_dir = self._inside(self.data_dir / "rlt" / self._name(rlt, "RLT 目录名"), self.data_dir, "RLT 目录")
            argv += ["--rlt", str(rlt_dir)]
        return {"argv": argv, "output": str(output), "port": port, "deployment": model, "kind": kind}

    # -- state

    def _running(self, *kinds):
        with self.lock:
            return [j for j in self.jobs.values() if j.running and (not kinds or j.kind in kinds)]

    def _session(self):
        running = self._running("session")
        return running[0] if running else None

    def _external_collect(self):
        mine = {j.proc.pid for j in self._running() if j.proc is not None}
        return collect_processes(self.system.processes(), exclude=mine)

    def _motion_gate(self, params, what):
        if self.closing:
            raise Refused("控制台正在关闭")
        if not isinstance(params, dict) or params.get("confirm") != MOTION_PHRASE:
            raise Refused(f"{what}会让机械臂运动或断电：输入“{MOTION_PHRASE}”确认", 403)
        busy = self._running("session", "daemon")
        if busy:
            raise Refused(f"{busy[0].label} 仍在运行；结束后再操作")
        external = self._external_collect()
        if external:
            raise Refused(f"已有采集进程在运行（PID {external[0]['pid']}，{external[0]['user']}）")

    def port_owners(self, ports):
        listening = self.system.listening()
        inodes = {port: listening[port] for port in ports if port in listening}
        owners = self.system.socket_owners(inodes.values())
        processes = self.system.processes()
        mine = {j.proc.pid: j for j in self._running() if j.proc is not None}
        view = {}
        for port in ports:
            if port not in inodes:
                view[port] = {"listening": False}
                continue
            pid = owners.get(inodes[port])
            user, args = processes.get(pid, (None, None))
            job = mine.get(pid)
            view[port] = {
                "listening": True,
                "pid": pid,
                "user": user,
                "args": args,
                "console_job": None if job is None else job.id,
            }
        return view

    def status(self):
        processes = self.system.processes()
        mine = {j.proc.pid for j in self._running() if j.proc is not None}
        gpu = self.system.gpu()
        if gpu is not None:
            for app in gpu["compute"]:
                user, args = processes.get(app["pid"], (None, None))
                app.update(user=user, args=args, console=app["pid"] in mine)
        models = self.models()
        ports = sorted({BASELINE_PORT, *(m["port"] for m in models if "port" in m)})
        owners = self.port_owners([*DAEMON_PORTS, *ports])
        daemon = owners[DAEMON_PORTS[0]]
        daemon["kind"] = (
            classify_daemon(daemon.get("args"), self.root) if daemon["listening"] else "none"
        )
        return {
            "time": time.time(),
            "gpu": gpu,
            "ports": [
                {"port": port, "models": [m["name"] for m in models if m.get("port") == port], **owners[port]}
                for port in ports
            ],
            "daemon": {**daemon, "ports": {str(p): owners[p]["listening"] for p in DAEMON_PORTS}},
            "collect": collect_processes(processes, exclude=mine),
            "git": self.system.git(),
            "root": str(self.root),
        }

    def meta(self):
        try:
            from dagger_offline.live_recording import FAILURE_TYPES
        except ImportError:  # labels only; the collection itself owns them
            FAILURE_TYPES = {}
        return {
            "phrase": MOTION_PHRASE,
            "failure_types": list(FAILURE_TYPES.items()),
            "today": f"{datetime.now():%m%d}",
            "root": str(self.root),
            "max_episodes": MAX_EPISODES,
        }

    def snapshot(self):
        with self.lock:
            jobs = [j.snapshot(output=True) for j in self.jobs.values()]
        return {"type": "hello", "jobs": jobs, "meta": self.meta(), "closing": self.closing}

    # -- children

    def _spawn(self, kind, label, argv, *, meta=None, pty=False):
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        job = Job(kind, label, argv, meta=meta, pty=pty)
        job._transcript = private_file(self.logs / "console-jobs" / f"{stamp}-{kind}-{job.id}.log")
        job._transcript.write(f"$ {' '.join(argv)}\n")
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "PYTHONUNBUFFERED": "1"}
        if pty:
            master, slave = os.openpty()
            try:
                attrs = termios.tcgetattr(slave)
                attrs[1] &= ~termios.ONLCR  # keep "\n": the page renders lines
                termios.tcsetattr(slave, termios.TCSANOW, attrs)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
                job.proc = subprocess.Popen(
                    [sys.executable, "-I", "-c", CTTY_SHIM, *argv],
                    stdin=slave, stdout=slave, stderr=slave,
                    cwd=self.root, env=env, start_new_session=True,
                )
            except BaseException:
                os.close(master)
                raise
            finally:
                os.close(slave)
            os.set_blocking(master, False)
            job.master, source = master, master
        else:
            job.proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=self.root, env=env, start_new_session=True,
            )
            source = job.proc.stdout.fileno()
        with self.lock:
            # Only the latest ended job of a kind stays visible.
            for old in [j for j in self.jobs.values() if j.kind == kind and not j.running]:
                del self.jobs[old.id]
            self.jobs[job.id] = job
        self.log(f"start {kind} pid {job.proc.pid}: {' '.join(argv)}")
        threading.Thread(target=self._pump, args=(job, source), daemon=True, name=f"job-{job.id}").start()
        self._job_event(job)
        return job

    def _pump(self, job, fd):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            if not select.select([fd], [], [], 0.5)[0]:
                # A detached grandchild may keep the pty open: the job ends
                # with its own process once no output is pending.
                if job.proc.poll() is not None:
                    break
                continue
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                continue
            except OSError:  # EIO: every child side of the pty closed
                data = b""
            if not data:
                break
            text = ANSI.sub("", decoder.decode(data)).replace("\r\n", "\n")
            job.append(text)
            self.broadcast({"type": "output", "id": job.id, "text": text})
            if job.parser is not None and job.parser.feed(text):
                self.broadcast({"type": "state", "id": job.id, "state": job.parser.state()})
        job.code = job.proc.wait()
        job.ended = time.time()
        if job.master is not None:
            os.close(job.master)
            job.master = None
        elif job.proc.stdout is not None:
            job.proc.stdout.close()
        try:
            job._transcript.write(f"\n[退出码 {job.code}]\n")
            job._transcript.close()
        except (OSError, ValueError):
            pass
        self.log(f"end {job.kind} pid {job.proc.pid}: code {job.code}")
        self._job_event(job)

    def _job(self, params, kinds):
        job_id = params.get("job") if isinstance(params, dict) else None
        with self.lock:
            job = self.jobs.get(job_id) if isinstance(job_id, str) else None
        if job is None or job.kind not in kinds:
            raise Refused("没有这个由控制台启动的进程", 404)
        return job

    # -- operations

    def start_session(self, params):
        plan = self.session_plan(params)
        with self.lock:
            self._motion_gate(params, "采集")
            listening = self.system.listening()
            if plan["port"] not in listening:
                raise Refused(f"端口 {plan['port']} 没有模型服务在监听；先在“模型”页启动并等它开始监听")
            label = f"{params.get('command', 'collect')} → {Path(plan['output']).relative_to(self.root)}"
            meta = {k: plan[k] for k in ("output", "port", "deployment", "kind")}
            job = self._spawn("session", label, plan["argv"], meta=meta, pty=True)
        return job.snapshot()

    def start_daemon(self, action, params):
        if action not in ("restart", "stop"):
            raise Refused("未知的硬件操作", 404)
        with self.lock:
            self._motion_gate(params, "硬件进程重启 / 关闭")
            job = self._spawn("daemon", f"daemon-{action}", [self.hgrlt, f"daemon-{action}"], pty=True)
        return job.snapshot()

    def daemon_check(self):
        if self._running("daemon"):
            raise Refused("硬件进程正在重启或关闭，结束后再检查")
        if not self._check.acquire(blocking=False):
            raise Refused("检查正在进行")
        try:
            env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
            try:
                done = subprocess.run(
                    [self.hgrlt, "daemon-check"], cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=60,
                )
                output, code = ANSI.sub("", done.stdout + done.stderr), done.returncode
            except subprocess.TimeoutExpired:
                output, code = "daemon-check 60 秒内没有结束", None
        finally:
            self._check.release()
        if code == 0 and "CONTRACT OK" in output:
            verdict = "ok"
        elif "session_api_version" in output:
            verdict = "foreign"
        elif "SAFE_HOLD" in output:
            verdict = "not_hold"
        else:
            verdict = "error"
        self.log(f"daemon-check: code {code}, {verdict}")
        return {"code": code, "verdict": verdict, "output": output[-20000:]}

    def start_model(self, params):
        name = params.get("name") if isinstance(params, dict) else None
        deployment, port, _ = self.deployment(name if isinstance(name, str) else "")
        with self.lock:
            if self.closing:
                raise Refused("控制台正在关闭")
            running = self._running("model")
            if running:
                raise Refused(f"控制台已在运行模型 {running[0].meta['name']}；一次只启动一个，先停止它")
            owner = self.port_owners([port])[port]
            if owner["listening"]:
                who = f"PID {owner['pid']}（{owner['user']}）" if owner.get("pid") else "其他用户的进程"
                raise Refused(f"端口 {port} 已被 {who} 占用")
            gpu = self.system.gpu()
            if gpu is None or not gpu["gpus"]:
                raise Refused("读不到显存（nvidia-smi）；不启动模型")
            card = gpu["gpus"][0]
            free = card["total_mib"] - card["used_mib"]
            if free < MIN_GPU_FREE_MIB:
                processes = self.system.processes()
                users = ", ".join(
                    f"PID {a['pid']} {processes.get(a['pid'], ('?', ''))[0]} {a['used_mib']} MiB"
                    for a in gpu["compute"]
                ) or "看不到进程"
                raise Refused(
                    f"GPU 0 只剩 {free} MiB（需要 {MIN_GPU_FREE_MIB} MiB）；占用：{users}"
                )
            argv = [self.hgrlt, "serve"] if deployment is None else [
                self.hgrlt, "serve-round", "--deployment", str(deployment)
            ]
            job = self._spawn("model", f"模型 {name}", argv, meta={"name": name, "port": port, "listening": False})
        threading.Thread(target=self._watch_model, args=(job,), daemon=True).start()
        return job.snapshot()

    def _watch_model(self, job):
        while job.running:
            listening = job.meta["port"] in self.system.listening()
            if listening != job.meta["listening"]:
                job.meta["listening"] = listening
                self._job_event(job)
            time.sleep(1)

    def stop_model(self, params):
        job = self._job(params, ("model",))
        if not job.running:
            raise Refused("这个模型服务已经停止")
        session = self._session()
        if session is not None and session.meta.get("port") == job.meta["port"] and not job.stopping:
            raise Refused("正在采集的会话使用这个模型；先结束采集")
        job.stopping = True
        self._job_event(job)
        threading.Thread(target=self._terminate, args=(job,), daemon=True).start()
        return job.snapshot()

    def _terminate(self, job, waits=((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5))):
        for signum, seconds in waits:
            if not job.running:
                return
            self.log(f"stop {job.kind} pid {job.proc.pid}: {signal.Signals(signum).name}")
            job.signal(signum)
            deadline = time.monotonic() + seconds
            while job.running and time.monotonic() < deadline:
                time.sleep(0.1)

    def key(self, params):
        job = self._job(params, ("session", "daemon"))
        keys = SESSION_KEYS if job.kind == "session" else DAEMON_KEYS
        name = params.get("key")
        if not isinstance(name, str) or name not in keys:
            raise Refused("不支持的按键", 400)
        self._send(job, keys[name])
        return {"ok": True}

    def line(self, params):
        job = self._job(params, ("session", "daemon"))
        text = params.get("text")
        if not isinstance(text, str) or len(text) > 200 or not all(c.isprintable() for c in text):
            raise Refused("只能发送一行可见文字（最多 200 字）", 400)
        self._send(job, text + "\r")
        return {"ok": True}

    def _send(self, job, data):
        job.write(data)
        if job.parser is not None and job.parser.typed(data):
            self.broadcast({"type": "state", "id": job.id, "state": job.parser.state()})

    def stop_session(self, why="页面结束采集"):
        job = self._session()
        if job is None:
            raise Refused("没有正在进行的采集")
        with self.lock:
            if job.stopping:
                return job.snapshot()
            job.stopping = True
        self.log(f"stop session ({why})")
        job.append(f"\n[控制台] {why}：按当前阶段发送 s / q 结束采集\n")
        self._job_event(job)
        threading.Thread(target=self._wind_down, args=(job,), daemon=True).start()
        return job.snapshot()

    # Keys that end a collection from each stage; other stages finish on their own.
    WIND_DOWN = {
        "await_start": ["q\r"],  # next-episode prompt: end the collection
        "confirm_homing": ["n\r"],  # decline homing: no motion is sent
        "recording": ["s", "q"],  # HOLD, then discard this episode (kept in trash/)
        "review": ["3"],  # unlabelled result: discard
        "failure_type": ["\x03"],  # interrupt: recorded as incomplete, kept in trash/
    }

    wind_down_s = 90.0

    def _wind_down(self, job):
        """End a collection the way an operator would; signals only as a fallback."""
        deadline = time.monotonic() + self.wind_down_s
        last = (None, 0.0)
        while job.running and time.monotonic() < deadline:
            stage = job.parser.stage
            keys = self.WIND_DOWN.get(stage)
            # Repeat a stage's keys only if the stage has not moved for 5 s.
            if keys and (last[0] != stage or time.monotonic() - last[1] > 5):
                for data in keys:
                    try:
                        self._send(job, data)
                    except (Refused, OSError):
                        break
                    time.sleep(0.3)
                last = (stage, time.monotonic())
            time.sleep(0.2)
        if job.running:
            self.log("session did not end from keys; interrupting")
            job.append("\n[控制台] 按键未能结束采集，发送中断信号\n")
            self._terminate(job, waits=((signal.SIGINT, 20), (signal.SIGTERM, 20)))
        if job.running:
            self.log(f"session pid {job.proc.pid} still running after SIGTERM")

    # -- results

    def _rows(self, directory):
        rows, policies = [], None
        data = self.data_dir.resolve()
        for folder in ("episodes", "trash"):
            for path in sorted((directory / folder).glob("episode-*.h5")):
                # Only files that really live under data/ (no symlink out).
                if not path.resolve().is_relative_to(data):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                key = (str(path), stat.st_mtime_ns, stat.st_size)
                row = self._cache.get(key)
                if row is None:
                    if policies is None:
                        policies = self.policies()
                    row = read_episode(path, policies) | {"folder": folder}
                    if row["outcome"] != "unreadable":
                        self._cache[key] = row
                rows.append(row)
        return rows

    def result_dirs(self):
        found = []
        for kind in ("eval", "recordings"):
            base = self.data_dir / kind
            if not base.is_dir():
                continue
            for directory in sorted(base.iterdir()):
                inside = directory.resolve().is_relative_to(self.data_dir.resolve())
                if inside and directory.is_dir() and (directory / "episodes").is_dir():
                    found.append((kind, directory))
        return found

    def results(self):
        session = self._session()
        active = session.meta.get("output") if session else None
        listing = []
        for kind, directory in self.result_dirs():
            summary = summarize(self._rows(directory))
            listing.append({
                "kind": kind,
                "name": directory.name,
                "active": active is not None and Path(active) == directory.resolve(),
                **summary,
            })
        compare = collections.defaultdict(list)
        for item in listing:
            date = re.match(r"^(\d{4})-", item["name"])
            if item["kind"] == "eval" and date:
                compare[date.group(1)].append(item["name"])
        return {"dirs": listing, "compare": dict(sorted(compare.items(), reverse=True))}

    def result_detail(self, kind, name):
        if kind not in ("eval", "recordings"):
            raise Refused("类型只能是 eval 或 recordings", 400)
        directory = self._inside(self.data_dir / kind / self._name(name, "目录名"), self.data_dir, "目录")
        if not (directory / "episodes").is_dir():
            raise Refused("没有这个采集目录", 404)
        rows = self._rows(directory)
        return {"kind": kind, "name": name, "summary": summarize(rows), "episodes": rows}

    # -- shutdown

    def close(self):
        """Stop children cleanly: a collection by its own keys, servers by signal."""
        with self.lock:
            self.closing = True
        self.log("console closing")
        session = self._session()
        if session is not None:
            try:
                self.stop_session("控制台关闭")
            except Refused:
                pass
            # The model stays up until the collection has ended: a stop that
            # waits on inference must not turn into a failed request.
            deadline = time.monotonic() + 150
            while session.running and time.monotonic() < deadline:
                time.sleep(0.2)
        for job in self._running("daemon"):
            # A daemon command still at its confirmation prompt is declined.
            if job.parser.stage == "await_phrase":
                try:
                    self._send(job, "\r")
                except (Refused, OSError):
                    pass
        for job in self._running("model"):
            job.stopping = True
            threading.Thread(target=self._terminate, args=(job,), daemon=True).start()
        deadline = time.monotonic() + 60
        while self._running() and time.monotonic() < deadline:
            time.sleep(0.2)
        for job in self._running():
            self.log(f"{job.kind} pid {job.proc.pid} still running at exit")
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            client.put(None)
        self.log("console closed")

    # -- HTTP

    def make_server(self, host="127.0.0.1", port=8810):
        console = self

        class Handler(RequestHandler):
            pass

        Handler.console = console
        server = http.server.ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        return server


class RequestHandler(http.server.BaseHTTPRequestHandler):
    console = None
    server_version = "HG-RLT"
    sys_version = ""

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        pass

    def _headers(self, code, content_type, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        for name, value in extra:
            self.send_header(name, value)

    def _send(self, code, body, content_type="application/json; charset=utf-8", extra=()):
        data = (
            body if isinstance(body, bytes)
            else json.dumps(body, ensure_ascii=False).encode() if not isinstance(body, str)
            else body.encode()
        )
        self._headers(code, content_type, extra)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _token_ok(self, value):
        return isinstance(value, str) and hmac.compare_digest(value.encode(), self.console.token.encode())

    def _cookie(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                return value
        return None

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle("GET")

    def do_POST(self):  # noqa: N802
        self._handle("POST")

    def do_PUT(self):  # noqa: N802
        self._handle("PUT")

    def do_DELETE(self):  # noqa: N802
        self._handle("DELETE")

    def do_PATCH(self):  # noqa: N802
        self._handle("PATCH")

    def do_OPTIONS(self):  # noqa: N802
        self._handle("OPTIONS")

    def do_HEAD(self):  # noqa: N802
        self._handle("HEAD")

    def _handle(self, method):
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        console = self.console
        try:
            if "token" in query:
                if not self._token_ok(query["token"][0]):
                    console.log(f"wrong token from {self.client_address[0]}")
                    return self._send(401, "访问令牌不对。", "text/plain; charset=utf-8")
                cookie = f"{COOKIE}={console.token}; HttpOnly; SameSite=Strict; Path=/"
                return self._send(302, b"", "text/plain", [("Set-Cookie", cookie), ("Location", url.path or "/")])
            if not self._token_ok(self._cookie()):
                return self._send(
                    401, "用 ./hgrlt ui 打印的网址打开（含访问令牌）。", "text/plain; charset=utf-8"
                )
            route = ROUTES.get(url.path)
            static = STATIC_FILES.get(url.path)
            if route is None and static is None:
                return self._send(404, {"error": "没有这个地址"})
            allowed = "GET" if static is not None else route[0]
            if method != allowed:
                return self._send(405, {"error": f"只接受 {allowed}"}, extra=[("Allow", allowed)])
            if method == "POST":
                if self.headers.get(HEADER) != "1":
                    return self._send(403, {"error": f"缺少 {HEADER} 请求头"})
                fetch_site = self.headers.get("Sec-Fetch-Site")
                if fetch_site not in (None, "same-origin", "none"):
                    return self._send(403, {"error": "只接受本页发出的请求"})
            if static is not None:
                name, content_type = static
                return self._send(200, (STATIC / name).read_bytes(), content_type)
            _, handler = route
            if handler is None:
                return self._events()
            params = self._body() if method == "POST" else {k: v[0] for k, v in query.items()}
            if method == "POST":
                console.log(f"POST {url.path} {json.dumps(_loggable(params), ensure_ascii=False)}")
            return self._send(200, handler(console, params))
        except Refused as exc:
            if method == "POST":
                console.log(f"refused {url.path}: {exc}")
            return self._send(exc.code, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:  # noqa: BLE001 - one request fails, the console stays up
            console.log(f"error {url.path}: {type(exc).__name__}: {exc}")
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise Refused("Content-Length 无效", 400) from exc
        if length > MAX_BODY:
            raise Refused("请求太大", 413)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw or b"{}")
        except ValueError as exc:
            raise Refused("请求体不是 JSON", 400) from exc
        if not isinstance(params, dict):
            raise Refused("请求体应为 JSON 对象", 400)
        return params

    def _events(self):
        import queue

        console = self.console
        inbox = queue.Queue()
        self._headers(200, "text/event-stream; charset=utf-8", [("Connection", "keep-alive")])
        self.end_headers()
        with console.lock:
            hello = console.snapshot()
            console.clients.add(inbox)
        try:
            self.wfile.write(f"data: {json.dumps(hello, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    data = inbox.get(timeout=15)
                except queue.Empty:
                    data = b": ping\n\n"
                if data is None:
                    return
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            with console.lock:
                console.clients.discard(inbox)


def _loggable(params):
    # The motion phrase is not a secret, but typed lines can be anything.
    return {k: ("…" if k == "text" else v) for k, v in params.items()}


def _shutdown(console, _params):
    if console.on_shutdown is None:
        raise Refused("这个控制台不能从页面关闭", 404)
    threading.Timer(0.2, console.on_shutdown).start()
    return {"ok": True}


ROUTES = {
    "/api/events": ("GET", None),
    "/api/state": ("GET", lambda c, p: c.snapshot()),
    "/api/status": ("GET", lambda c, p: c.status()),
    "/api/models": ("GET", lambda c, p: c.models()),
    "/api/results": ("GET", lambda c, p: c.results()),
    "/api/results/detail": ("GET", lambda c, p: c.result_detail(p.get("kind"), p.get("name"))),
    "/api/session/plan": ("POST", lambda c, p: c.session_plan(p)),
    "/api/session/start": ("POST", lambda c, p: c.start_session(p)),
    "/api/session/stop": ("POST", lambda c, p: c.stop_session()),
    "/api/pty/key": ("POST", lambda c, p: c.key(p)),
    "/api/pty/line": ("POST", lambda c, p: c.line(p)),
    "/api/daemon/check": ("POST", lambda c, p: c.daemon_check()),
    "/api/daemon/restart": ("POST", lambda c, p: c.start_daemon("restart", p)),
    "/api/daemon/stop": ("POST", lambda c, p: c.start_daemon("stop", p)),
    "/api/model/start": ("POST", lambda c, p: c.start_model(p)),
    "/api/model/stop": ("POST", lambda c, p: c.stop_model(p)),
    "/api/shutdown": ("POST", _shutdown),
}


def main():
    parser = argparse.ArgumentParser(description="HG-RLT 网页控制台", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；默认只接受本机（SSH 隧道）")
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--new-token", action="store_true", help="换一个新的访问令牌")
    parser.add_argument("--pidfile", type=Path, help="启动后立即写入 PID（console.sh 使用）")
    args = parser.parse_args()
    if args.pidfile:
        args.pidfile.write_text(f"{os.getpid()}\n")
    stop = threading.Event()
    console = Console(ROOT, token=console_token(args.new_token), on_shutdown=stop.set)
    try:
        server = console.make_server(args.host, args.port)
    except OSError as exc:
        print(f"[控制台] 无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    console.log(f"listening on {args.host}:{args.port} (pid {os.getpid()})")
    print(f"HG-RLT 控制台：http://{args.host}:{args.port}/?token={console.token}", flush=True)
    if args.host in ("127.0.0.1", "localhost", "::1"):
        print(
            f"其他电脑：ssh -N -L {args.port}:127.0.0.1:{args.port} -p 2222 ziyu@<实验室地址>，再打开上面的网址",
            flush=True,
        )
    else:
        print("[控制台] 监听非本机地址：能访问这个地址并持有令牌的人都能操作机械臂", flush=True)
    print(f"日志：{console.logs / 'console.log'}；Ctrl-C 关闭（会先结束采集）", flush=True)
    while not stop.wait(0.5):
        pass
    print("[控制台] 正在关闭：先结束采集和控制台启动的模型服务", flush=True)
    console.close()
    server.shutdown()
    server.server_close()
    print("[控制台] 已关闭", flush=True)


if __name__ == "__main__":
    main()
