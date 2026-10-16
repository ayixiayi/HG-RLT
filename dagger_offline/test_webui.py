import contextlib
import http.client
import io
import json
import os
import socket
import stat
import sys
import textwrap
import threading
import time
from pathlib import Path

import h5py
import pytest

from dagger_offline import webui
from dagger_offline.collection import run_collection
from dagger_offline.console import message, show_mode
from dagger_offline.test_live_runner import Harness

TOKEN = "t" * 43
POLICY_ID = "ab" * 32

# Stands in for ./hgrlt: the same prompts and console lines, keys logged to FAKE_LOG.
FAKE = textwrap.dedent(
    r'''
    import os, select, signal, socket, sys, termios, time, tty

    def msg(label, text):
        print(f"[{time.strftime('%H:%M:%S')}] [{label}] {text}", flush=True)

    log = open(os.environ["FAKE_LOG"], "a")

    def note(text):
        log.write(text + "\n")
        log.flush()

    def keys():
        while True:
            select.select([0], [], [])
            for key in os.read(0, 64).decode():
                note(f"KEY {key!r}")
                yield key

    command = sys.argv[1]
    if os.environ.get("FAKE_REAL") and command in ("collect", "collect-compat"):
        os.execv(sys.executable, [sys.executable, os.environ["FAKE_REAL"], *sys.argv[1:]])
    if command in ("collect", "collect-compat"):
        note("ARGV " + " ".join(sys.argv[1:]))
        note(f"TTY {sys.stdin.isatty()}")
        number = 0
        while True:
            number += 1
            msg("下一条", f"episode-{number:06d}.h5；重置场景后再开始，归零仍需现场确认")
            answer = input("回车开始本条；q 结束本次连续采集：").strip().lower()
            note(f"LINE {answer!r}")
            if answer == "q":
                break
            msg("归零确认", "双臂回零、夹爪张开")
            if input("按回车确认；输入其它内容或 Ctrl-C 取消："):
                msg("采集停止", "已取消归零，未发送运动指令")
                sys.exit(1)
            msg("归零完成", "反馈已到位，已确认 HOLD；可启动新记录。")
            saved = termios.tcgetattr(0)
            tty.setcbreak(0)
            note(f"CBREAK {not termios.tcgetattr(0)[3] & termios.ICANON}")
            msg("操作", "按键直接生效，无需回车；启动后保持，不会自动运动")
            msg("模式", "HOLD · 保持｜h 人工接管 / p 策略交还")
            outcome, stream = None, keys()
            try:
                for key in stream:
                    if key == "p":
                        msg("模式", "POLICY_RUNNING · 策略控制｜h 立即接管 / s 停止")
                    elif key == "s":
                        msg("模式", "HOLD · 保持｜h 人工接管 / p 策略交还")
                    elif key == "h":
                        msg("模式", "HOLD · 保持｜等待模型结果；h 接管 / s 停止")
                        msg("接管准备", "已保持，扳机已断电。双手握住两只主臂，两个扳机各自完全松开再扣到底确认；s / p 取消")
                        msg("接管准备", "左扳机已确认")
                        msg("接管准备", "右扳机已确认")
                        msg("接管准备", "确认完成，扳机上电回到夹爪开度")
                    elif key == "c":  # here: the triggers are back at the gripper
                        msg("模式", "INTERVENTION · 人工跟随｜p 交还策略 / s 停止")
                        msg("接管", "人工接管：主臂和扳机已放松，按主臂控制")
                    elif key == "q":
                        outcome = "discard"
                    elif key == "k":
                        msg("模式", "HOLD · 保持｜h 人工接管 / p 策略交还")
                        msg("结果标记", "1 成功 · 2 失败 · 3 弃用；直接按键，无需回车")
                        choice = next(k for k in stream if k in "123")
                        outcome = {"1": "success", "2": "fail", "3": "discard"}[choice]
                        if outcome == "fail":
                            msg("失败类型", "1 选错 · 2 抓取失败 · 3 掉落 · 4 卡住 · 5 超时 · 6 其他")
                            next(k for k in stream if k in "123456")
                    if outcome is not None:
                        break
            finally:
                termios.tcsetattr(0, termios.TCSADRAIN, saved)
            keep = outcome != "discard"
            msg("保存" if keep else "丢弃", "已确认 HOLD 并释放控制权；正在等待落盘，请勿中断。")
            msg("保存" if keep else "丢弃", "已确认 HOLD 并释放控制权；记录已保存。没有自动归位。" if keep else "已确认 HOLD 并释放控制权；整条记录标记不可用。没有删除原文件。")
            msg("采集统计", f"累计 {number} 条 · 成功 {number - 1} · 失败 1 · 弃用 0 · 未完成 0 · 未标结果 0 · 纳入清单 {number} 条 · trash 0 条")
        msg("录制清单", "records.json")
    elif command in ("daemon-restart", "daemon-stop"):
        msg("重启警告", "先关闭双臂电机")
        answer = input("托住双臂后输入 HG-DAEMON-ON-SITE 确认重启：")
        note(f"PHRASE {answer!r}")
        if answer != "HG-DAEMON-ON-SITE":
            print("Daemon restart not authorized", flush=True)
            sys.exit(1)
        msg("硬件进程", "已退出，端口释放")
    elif command == "daemon-check":
        msg("协议", "检查通过（CONTRACT OK）。仅只读检查，不是实机验收。")
    elif command in ("serve", "serve-round"):
        note("SERVE " + " ".join(sys.argv[1:]))
        server = socket.socket()
        server.bind(("127.0.0.1", int(os.environ["FAKE_PORT"])))
        server.listen()
        print("listening", flush=True)
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            time.sleep(0.1)
    '''
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeSystem(webui.System):
    """Real /proc ports; GPU and process table under test control."""

    def __init__(self, root):
        super().__init__(root)
        self.extra_ports = set()
        self.extra_processes = {}
        self.free_mib = 20000

    def processes(self):
        # Never the machine's other processes (a real collection may be running).
        mine = {os.getpid(), *(j.proc.pid for j in self.console.jobs.values() if j.proc)}
        table = {pid: row for pid, row in super().processes().items() if pid in mine}
        return {**table, **self.extra_processes}

    def gpu(self):
        total = 24564
        return {"gpus": [{"index": 0, "name": "Fake", "used_mib": total - self.free_mib, "total_mib": total}],
                "compute": [{"pid": 4242, "used_mib": total - self.free_mib, "name": "python"}]}

    def listening(self):
        ports = super().listening()
        return {**ports, **{port: -1 for port in self.extra_ports}}

    def git(self):
        return {"describe": "test", "commit": "test"}


def deployment(root, name, port, policy_id=POLICY_ID):
    folder = root / "models" / name
    folder.mkdir(parents=True)
    (folder / "deployment.json").write_text(json.dumps({
        "schema": webui.DEPLOYMENT_SCHEMA, "model_round": 1, "port": port, "policy_id": policy_id,
        "checkpoint": "checkpoint/x/1999", "files": {}, "run": {"inference": {"rl_token": {"width": 512}}},
    }))
    return folder / "deployment.json"


@pytest.fixture
def lab(tmp_path, monkeypatch):
    root = tmp_path / "HG-RLT"
    (root / "data").mkdir(parents=True)
    port = free_port()
    deployment(root, "r1", port)
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    hgrlt = tmp_path / "hgrlt"
    hgrlt.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    hgrlt.chmod(0o755)
    log = tmp_path / "fake.log"
    log.touch()
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_PORT", str(port))
    system = FakeSystem(root)
    console = webui.Console(root, token=TOKEN, system=system, hgrlt=hgrlt)
    system.console = console
    server = console.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Lab:
        pass

    lab = Lab()
    lab.root, lab.console, lab.system, lab.log, lab.port, lab.hgrlt = root, console, system, log, port, hgrlt
    lab.address = server.server_address

    def request(method, path, body=None, *, cookie=TOKEN, header=True, extra=None, raw=None):
        connection = http.client.HTTPConnection(*lab.address, timeout=10)
        headers = dict(extra or {})
        if cookie is not None:
            headers["Cookie"] = f"{webui.COOKIE}={cookie}"
        if method == "POST" and header:
            headers["X-HGRLT"] = "1"
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = payload.decode()
        return response.status, dict(response.getheaders()), payload

    lab.request = request
    lab.post = lambda path, body=None, **kw: request("POST", path, {} if body is None else body, **kw)
    lab.get = lambda path, **kw: request("GET", path, **kw)
    yield lab
    console.close()
    server.shutdown()
    server.server_close()


def wait(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def session_state(lab):
    job = lab.console._session()
    return job.parser.state() if job else None


def fake_log(lab):
    return lab.log.read_text().splitlines()


SESSION = {"kind": "eval", "deployment": "r1", "name": "1016-r1", "episodes": 2, "command": "collect"}


# ---- auth and request rules ----------------------------------------------------------


def test_every_request_needs_the_token_and_state_changes_need_post_and_header(lab):
    assert lab.get("/api/state", cookie=None)[0] == 401
    assert lab.get("/", cookie=None)[0] == 401
    assert lab.get("/api/state", cookie="x" * 43)[0] == 401
    assert lab.request("PUT", "/api/state", cookie=None)[0] == 401
    assert lab.get(f"/?token={'x' * 43}", cookie=None)[0] == 401

    status, headers, _ = lab.get(f"/?token={TOKEN}", cookie=None)
    assert status == 302 and headers["Location"] == "/"
    cookie = headers["Set-Cookie"]
    assert cookie.startswith(f"{webui.COOKIE}={TOKEN};")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie

    status, headers, page = lab.get("/")
    assert status == 200 and "HG-RLT" in page
    assert headers["X-Frame-Options"] == "DENY" and headers["Cache-Control"] == "no-store"
    assert lab.get("/api/state")[0] == 200

    assert lab.post("/api/session/plan", SESSION, header=False)[0] == 403
    assert lab.post("/api/session/plan", SESSION)[0] == 200
    assert lab.post("/api/session/plan", SESSION, extra={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert lab.get("/api/session/plan")[0] == 405
    assert lab.get("/api/session/start")[0] == 405
    assert lab.request("PUT", "/api/state")[0] == 405
    assert lab.request("DELETE", "/api/session/stop")[0] == 405
    assert lab.post("/api/state")[0] == 405
    assert lab.post("/api/session/plan", raw=b"not json")[0] == 400
    assert lab.post("/api/session/plan", raw=b"[1, 2]")[0] == 400
    assert lab.post("/api/session/plan", raw=b"{}", extra={"Content-Length": str(webui.MAX_BODY + 1)})[0] == 413


def test_static_and_result_paths_cannot_leave_their_roots(lab):
    for path in ("/../webui.py", "/webui_static/../webui.py", "/%2e%2e/%2e%2e/etc/passwd",
                 "/../../../../etc/passwd", "/app.js/../../webui.py", "/api/../webui.py"):
        assert lab.get(path)[0] == 404, path
    assert lab.get("/app.js")[0] == 200
    for query in ("kind=eval&name=../../etc", "kind=../..&name=x", "kind=eval&name=..",
                  "kind=eval&name=%2e%2e%2fmodels", "kind=eval&name=a/b", "kind=eval&name="):
        status, _, body = lab.get(f"/api/results/detail?{query}")
        assert status == 400, (query, body)
    assert lab.get("/api/results/detail?kind=eval&name=missing")[0] == 404


def test_token_file_is_owner_only_and_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("HGRLT_UI_TOKEN", raising=False)
    file = tmp_path / "config" / "hgrlt" / "console-token"
    first = webui.console_token(file=file)
    assert len(first) >= 32 and webui.console_token(file=file) == first
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert stat.S_IMODE(file.parent.stat().st_mode) == 0o700
    assert webui.console_token(fresh=True, file=file) != first
    log = webui.private_file(tmp_path / "logs" / "console.log")
    log.close()
    assert stat.S_IMODE((tmp_path / "logs" / "console.log").stat().st_mode) == 0o600


def test_console_writes_an_owner_only_log(lab):
    lab.post("/api/session/plan", SESSION)
    path = lab.root / "logs" / "console.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "POST /api/session/plan" in path.read_text()
    assert TOKEN not in path.read_text()


# ---- argv from validated fields -------------------------------------------------------


def test_session_argv_is_built_only_from_validated_fields(lab, tmp_path):
    console, root = lab.console, lab.root
    plan = console.session_plan(SESSION | {"episodes": 20})
    assert plan["argv"] == [
        str(lab.hgrlt), "collect", "--output-dir", str(root / "data/eval/1016-r1"), "--episodes", "20",
        "--deployment", str(root / "models/r1/deployment.json"),
    ]
    assert plan["port"] == lab.port
    baseline = console.session_plan({"kind": "recordings", "deployment": "baseline", "name": "heavier-r2",
                                     "episodes": 0, "command": "collect-compat"})
    assert baseline["argv"] == [str(lab.hgrlt), "collect-compat", "--output-dir",
                                str(root / "data/recordings/heavier-r2"), "--episodes", "0"]
    assert baseline["port"] == webui.BASELINE_PORT
    rlt = console.session_plan(SESSION | {"rlt": "rlt1"})
    assert rlt["argv"][-2:] == ["--rlt", str(root / "data/rlt/rlt1")]
    # RLT episodes always land where rlt-learn reads them, whatever the kind.
    assert rlt["kind"] == "recordings" and rlt["argv"][2:4] == [
        "--output-dir", str(root / "data/recordings/1016-r1")]

    outside = tmp_path / "outside"
    deployment(outside.parent, "outside-model", 9100)
    (root / "models/evil").symlink_to(outside.parent / "models" / "outside-model")
    (root / "data/eval").mkdir()
    (root / "data/eval/link").symlink_to(tmp_path)
    bad = [
        {"name": "../x"}, {"name": "x/y"}, {"name": "a;rm -rf ~"}, {"name": "a b"}, {"name": "$(id)"},
        {"name": ""}, {"name": ".hidden"}, {"name": "-rf"}, {"name": "x" * 65}, {"name": 5}, {"name": "link"},
        {"name": "a\nb"}, {"deployment": "../models/r1"}, {"deployment": "/etc"}, {"deployment": "r1/../r1"},
        {"deployment": "missing"}, {"deployment": "evil"}, {"deployment": None}, {"deployment": "r1;ls"},
        {"episodes": -1}, {"episodes": 501}, {"episodes": "3"}, {"episodes": 2.5}, {"episodes": True},
        {"episodes": "3; ls"}, {"command": "rm"}, {"command": "collect --execute"}, {"kind": "../x"},
        {"kind": "rounds"}, {"rlt": "../x"}, {"rlt": "a|b"}, {"deployment": "baseline", "rlt": "rlt1"},
    ]
    for change in bad:
        with pytest.raises(webui.Refused) as caught:
            console.session_plan(SESSION | change)
        assert caught.value.code == 400, change
    status, _, body = lab.post("/api/session/plan", SESSION | {"name": "../../etc"})
    assert status == 400 and "目录名" in body["error"]


def test_models_list_baseline_and_sealed_deployments(lab):
    status, _, models = lab.get("/api/models")
    assert status == 200
    assert [m["name"] for m in models] == ["baseline", "r1"]
    assert models[0]["port"] == webui.BASELINE_PORT
    assert models[1]["policy_id"] == POLICY_ID and models[1]["rl_token"]


# ---- phrase gating and mutual exclusion -------------------------------------------------


def test_motion_actions_need_the_phrase_and_run_alone(lab):
    lab.system.extra_ports.add(lab.port)
    status, _, body = lab.post("/api/session/start", SESSION)
    assert status == 403 and webui.MOTION_PHRASE in body["error"]
    assert lab.post("/api/session/start", SESSION | {"confirm": "我在"})[0] == 403
    assert lab.post("/api/daemon/restart", {"confirm": "yes"})[0] == 403
    assert lab.post("/api/daemon/stop", {})[0] == 403

    lab.system.extra_ports.discard(lab.port)
    status, _, body = lab.post("/api/session/start", SESSION | {"confirm": webui.MOTION_PHRASE})
    assert status == 409 and "没有模型服务" in body["error"]
    lab.system.extra_ports.add(lab.port)

    lab.system.extra_processes = {
        999999: ("ziyu", "/x/envs/live/bin/python /x/Dagger-Practice/collect.py --rhos-root /x"),
    }
    for path, body in (("/api/session/start", SESSION), ("/api/daemon/restart", {})):
        status, _, reply = lab.post(path, body | {"confirm": webui.MOTION_PHRASE})
        assert status == 409 and "999999" in reply["error"], path
    lab.system.extra_processes = {}

    status, _, job = lab.post("/api/session/start", SESSION | {"confirm": webui.MOTION_PHRASE})
    assert status == 200 and job["kind"] == "session"
    for path, body in (("/api/session/start", SESSION), ("/api/daemon/restart", {}), ("/api/daemon/stop", {})):
        status, _, reply = lab.post(path, body | {"confirm": webui.MOTION_PHRASE})
        assert status == 409 and "仍在运行" in reply["error"], path
    # The console's own collection is not an external one.
    assert lab.console._external_collect() == []
    wait(lambda: session_state(lab)["stage"] == "await_start")
    assert lab.post("/api/pty/line", {"job": job["id"], "text": "q"})[0] == 200
    wait(lambda: not lab.console._session())


def test_collect_detection_matches_lab_pull():
    table = {
        1: ("ziyu", "/h/envs/live/bin/python /h/Dagger-Practice/collect.py --rhos-root /h/vendor/rhos"),
        2: ("ziyu", "/h/envs/live/bin/python3.12 /h/Dagger-Practice/run_robot.py --execute"),
        3: ("ziyu", "grep -E collect.py"),
        4: ("ziyu", "vim /h/Dagger-Practice/collect.py"),
        5: ("ziyu", "/h/envs/live/bin/python -m dagger_offline.webui"),
        6: ("ziyu", "python3 collect.py"),
        7: ("ziyu", "/h/envs/live/bin/python -u /h/Dagger-Practice/collect.py --output-dir x"),
        8: ("ziyu", "python3 -B -X dev Dagger-Practice/run_robot.py --execute"),
        9: ("ziyu", "python -m pytest dagger_offline/test_collect.py"),
        10: ("ziyu", "/h/envs/data/bin/python -m pytest dagger_offline/test_collection.py"),
    }
    assert [p["pid"] for p in webui.collect_processes(table)] == [1, 2, 6, 7, 8]
    assert [p["pid"] for p in webui.collect_processes(table, exclude={1, 6})] == [2, 7, 8]


def test_process_arguments_hide_secrets():
    args = ("python yam_safety_daemon.py --hardware-port 6001 --bootstrap-token 6ef1 "
            "--api-key=abc OPENAI_API_KEY=sk-1 --keyboard-layout us http://x/?token=zz")
    shown = webui.redact(args)
    for secret in ("6ef1", "abc", "sk-1", "zz"):
        assert secret not in shown
    assert "--hardware-port 6001" in shown and "--bootstrap-token <隐藏>" in shown
    assert "--keyboard-layout us" in shown
    assert webui.redact(None) is None


def test_daemon_owner_is_classified(tmp_path):
    root = tmp_path / "HG-RLT"
    root.mkdir()
    ours = f"{root}/envs/live/bin/python {root}/vendor/rhos/Client/arms/YAM/gello_software/experiments/yam_safety_daemon.py"
    foreign = "/home/ziyu/RhOSPolicy/.venv/bin/python Client/arms/YAM/gello_software/experiments/yam_safety_daemon.py"
    assert webui.classify_daemon(ours, root) == "hgrlt"
    assert webui.classify_daemon(foreign, root) == "foreign"
    assert webui.classify_daemon("nginx: worker", root) == "other"
    assert webui.classify_daemon(None, root) == "unknown"


# ---- pty sessions ---------------------------------------------------------------------


def start_session(lab, **change):
    lab.system.extra_ports.update((lab.port, webui.BASELINE_PORT))
    status, _, job = lab.post("/api/session/start", SESSION | change | {"confirm": webui.MOTION_PHRASE})
    assert status == 200, job
    return job["id"]


def to_recording(lab, job):
    wait(lambda: session_state(lab)["stage"] == "await_start")
    assert lab.post("/api/pty/line", {"job": job, "text": ""})[0] == 200
    wait(lambda: session_state(lab)["stage"] == "confirm_homing")
    assert lab.post("/api/pty/key", {"job": job, "key": "enter"})[0] == 200
    wait(lambda: session_state(lab)["mode"] == "HOLD")


def test_pty_session_forwards_keys_and_follows_modes(lab):
    job = start_session(lab)
    to_recording(lab, job)
    key = lambda k: lab.post("/api/pty/key", {"job": job, "key": k})  # noqa: E731
    assert key("p")[0] == 200
    wait(lambda: session_state(lab)["mode"] == "POLICY_RUNNING")
    assert key("h")[0] == 200
    state = wait(lambda: (s := session_state(lab))["takeover"] and s["takeover"]["phase"] == "return" and s)
    assert state["mode"] == "HOLD" and state["takeover"]["left"] and state["takeover"]["right"]
    key("c")
    state = wait(lambda: (s := session_state(lab))["mode"] == "INTERVENTION" and s)
    assert state["takeover"] is None
    key("?")
    key("k")
    wait(lambda: session_state(lab)["stage"] == "review")
    key("2")
    wait(lambda: session_state(lab)["stage"] == "failure_type")
    key("4")
    state = wait(lambda: (s := session_state(lab))["stage"] == "await_start" and s)
    assert state["episode"] == "episode-000002.h5" and state["stats"]["total"] == 1
    for bad in ({"key": "a"}, {"key": "\x1b"}, {"key": "pp"}, {"key": None}):
        assert lab.post("/api/pty/key", {"job": job} | bad)[0] == 400, bad
    for text in ("\x03", "a\nb", "a\rb", "x" * 201, 5):
        assert lab.post("/api/pty/line", {"job": job, "text": text})[0] == 400, text
    assert lab.post("/api/pty/key", {"job": "nope", "key": "s"})[0] == 404
    lab.post("/api/pty/line", {"job": job, "text": "q"})
    wait(lambda: not lab.console._session())
    ended = lab.console.jobs[job]
    assert ended.code == 0 and ended.parser.stage == "finished"
    log = fake_log(lab)
    assert "TTY True" in log and "CBREAK True" in log
    argv = next(line for line in log if line.startswith("ARGV "))
    assert argv == (f"ARGV collect --output-dir {lab.root / 'data/eval/1016-r1'} --episodes 2 "
                    f"--deployment {lab.root / 'models/r1/deployment.json'}")
    keys = [line for line in log if line.startswith("KEY ")]
    assert keys == ["KEY 'p'", "KEY 'h'", "KEY 'c'", "KEY '?'", "KEY 'k'", "KEY '2'", "KEY '4'"]
    assert [line for line in log if line.startswith("LINE ")] == ["LINE ''", "LINE 'q'"]
    # Output survives a page reload: the snapshot carries it.
    _, _, hello = lab.get("/api/state")
    output = next(j for j in hello["jobs"] if j["id"] == job)["output"]
    assert "[结果标记]" in output and "\x1b" not in output
    transcript = next((lab.root / "logs/console-jobs").glob(f"*-session-{job}.log"))
    assert stat.S_IMODE(transcript.stat().st_mode) == 0o600 and "[失败类型]" in transcript.read_text()


# The real collection loop (run_collection, run_loop, review_outcome) on the
# fake robot of test_live_runner, driven over the pty like the page does.
REAL_COLLECT = textwrap.dedent(
    """
    import argparse, time
    from pathlib import Path

    from dagger_offline.collection import run_collection
    from dagger_offline.test_live_runner import Clock, Harness, runner

    class Paced(Clock):
        def sleep(self, seconds):
            self.value += seconds
            self.tick += 1
            time.sleep(0.004)

    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()

    def record(path, group):
        h = Harness(path, keys={})
        h.clock = Paced()
        with runner.keyboard() as poll:
            h.poll = poll
            runner.show_controls()
            h.run(review=lambda: runner.review_outcome(poll),
                  metadata={"collection": {"group": group}, "model_round": 0})

    run_collection(args.output_dir, record, episodes=args.episodes)
    """
)


def test_real_collection_loop_over_the_pty(lab, tmp_path, monkeypatch):
    real = tmp_path / "real_collect.py"
    real.write_text(REAL_COLLECT)
    monkeypatch.setenv("FAKE_REAL", str(real))
    job = start_session(lab, deployment="baseline", name="1016-baseline")
    key = lambda k: lab.post("/api/pty/key", {"job": job, "key": k})  # noqa: E731
    line = lambda text: lab.post("/api/pty/line", {"job": job, "text": text})  # noqa: E731
    mode = lambda m: wait(lambda: session_state(lab)["mode"] == m, timeout=30)  # noqa: E731
    stage = lambda s: wait(lambda: session_state(lab)["stage"] == s, timeout=30)  # noqa: E731

    stage("await_start")
    line("")
    mode("HOLD")
    key("p")
    mode("POLICY_RUNNING")
    key("h")  # passive leaders: immediate takeover
    mode("INTERVENTION")
    key("p")
    mode("POLICY_RUNNING")
    key("k")
    stage("review")
    key("2")
    stage("failure_type")
    key("3")
    stage("await_start")
    state = session_state(lab)
    assert state["episode"] == "episode-000002.h5"
    assert (state["stats"]["total"], state["stats"]["fail"]) == (1, 1)
    line("")
    mode("HOLD")
    key("p")
    mode("POLICY_RUNNING")
    key("k")
    stage("review")
    key("1")
    wait(lambda: not lab.console._session(), timeout=60)
    ended = lab.console.jobs[job]
    assert ended.code == 0, ended.output()[-2000:]
    assert ended.parser.stage == "finished"

    _, _, results = lab.get("/api/results")
    summary = next(d for d in results["dirs"] if d["name"] == "1016-baseline")
    assert (summary["labelled"], summary["success"], summary["fail"]) == (2, 1, 1)
    assert (summary["takeovers"], summary["autonomous_success"]) == (1, 1)
    assert summary["failure_types"] == {"drop": 1} and summary["models"] == ["baseline"]


def test_parsed_prompts_and_labels_exist_in_the_programs():
    root = Path(__file__).resolve().parents[1]
    sources = "".join(
        (root / name).read_text()
        for name in ("Dagger-Practice/run_robot.py", "Dagger-Practice/daemon.py",
                     "Dagger-Practice/collect.py", "dagger_offline/collection.py", "dagger_offline/console.py")
    )
    for prompt in webui.PROMPTS:
        assert prompt in sources, prompt
    for text in ("扳机已断电", "扳机已确认", "确认完成，扳机上电回到夹爪开度", "扳机未回到夹爪开度",
                 "RL actor 接管", "已确认 HOLD 并释放控制权", "记录已保存", "整条记录标记不可用",
                 '"结果标记"', '"失败类型"', '"下一条"', '"归零确认"', '"归零完成"', '"录制清单"',
                 '"采集统计"', '"故障"', '"采集停止"', '"采集结束"', '"接管准备"', '"接管"', '"模式"'):
        assert text in sources, text


def test_events_stream_starts_with_state_and_streams_output(lab):
    connection = http.client.HTTPConnection(*lab.address, timeout=10)
    connection.request("GET", "/api/events", headers={"Cookie": f"{webui.COOKIE}={TOKEN}"})
    response = connection.getresponse()
    assert response.status == 200 and response.getheader("Content-Type").startswith("text/event-stream")
    first = json.loads(response.readline().decode().removeprefix("data: "))
    assert first["type"] == "hello" and first["meta"]["phrase"] == webui.MOTION_PHRASE
    job = start_session(lab)
    seen = []
    while not any(e.get("type") == "state" and e["state"]["stage"] == "await_start" for e in seen):
        line = response.readline().decode()
        if line.startswith("data: "):
            seen.append(json.loads(line.removeprefix("data: ")))
    assert any(e["type"] == "job" and e["job"]["id"] == job for e in seen)
    assert any(e["type"] == "output" and "[下一条]" in e["text"] for e in seen)
    connection.close()
    lab.post("/api/pty/line", {"job": job, "text": "q"})
    wait(lambda: not lab.console._session())


def test_stop_holds_discards_the_open_episode_and_ends(lab):
    job = start_session(lab)
    to_recording(lab, job)
    lab.post("/api/pty/key", {"job": job, "key": "p"})
    wait(lambda: session_state(lab)["mode"] == "POLICY_RUNNING")
    assert lab.post("/api/session/stop")[0] == 200
    wait(lambda: not lab.console._session(), timeout=30)
    assert lab.console.jobs[job].code == 0
    log = fake_log(lab)
    assert log[log.index("KEY 'p'") + 1:] == ["KEY 's'", "KEY 'q'", "LINE 'q'"]
    assert lab.post("/api/session/stop")[0] == 409


def test_closing_keeps_the_model_until_the_collection_ended(lab):
    status, _, model = lab.post("/api/model/start", {"name": "r1"})
    assert status == 200
    server = lab.console.jobs[model["id"]]
    wait(lambda: server.meta["listening"])
    job = start_session(lab)
    to_recording(lab, job)
    lab.console.close()
    session = lab.console.jobs[job]
    assert session.code == 0 and not server.running
    assert server.ended >= session.ended


def test_closing_the_console_labels_nothing_and_ends_the_collection(lab):
    job = start_session(lab)
    to_recording(lab, job)
    lab.post("/api/pty/key", {"job": job, "key": "k"})
    wait(lambda: session_state(lab)["stage"] == "review")
    lab.console.close()
    assert not lab.console._session() and lab.console.jobs[job].code == 0
    log = fake_log(lab)
    # At the result prompt the open episode is discarded, then the collection ends.
    assert log[log.index("KEY 'k'") + 1:] == ["KEY '3'", "LINE 'q'"]
    assert lab.post("/api/session/start", SESSION | {"confirm": webui.MOTION_PHRASE})[0] == 409


def test_a_child_that_stops_reading_cannot_hang_input_or_shutdown(lab, tmp_path):
    stuck = tmp_path / "stuck.py"
    stuck.write_text("import time\nprint('回车开始本条；q 结束本次连续采集：', end='', flush=True)\ntime.sleep(60)\n")
    job = lab.console._spawn("session", "stuck", [sys.executable, str(stuck)], pty=True)
    wait(lambda: job.parser.stage == "await_start")
    started = time.monotonic()
    with pytest.raises(webui.Refused):
        for _ in range(10_000):  # the pty input queue fills: refused, not blocked
            job.write("x" * 100 + "\r", timeout=0.5)
    assert time.monotonic() - started < 10
    lab.console.wind_down_s = 2
    lab.console.stop_session()
    wait(lambda: not job.running, timeout=40)  # keys refused, then SIGINT
    assert job.code != 0


def test_daemon_command_waits_for_the_operator_phrase(lab):
    status, _, job = lab.post("/api/daemon/restart", {"confirm": webui.MOTION_PHRASE})
    assert status == 200
    running = lab.console.jobs[job["id"]]
    wait(lambda: running.parser.stage == "await_phrase")
    time.sleep(0.5)
    assert not any(line.startswith("PHRASE") for line in fake_log(lab))  # never auto-answered
    assert lab.post("/api/pty/key", {"job": job["id"], "key": "p"})[0] == 400
    assert lab.post("/api/daemon/check")[0] == 409
    assert lab.post("/api/pty/line", {"job": job["id"], "text": "HG-DAEMON-ON-SITE"})[0] == 200
    wait(lambda: not running.running)
    assert running.code == 0 and "PHRASE 'HG-DAEMON-ON-SITE'" in fake_log(lab)
    status, _, check = lab.post("/api/daemon/check")
    assert status == 200 and check["verdict"] == "ok" and "CONTRACT OK" in check["output"]


# ---- models ---------------------------------------------------------------------------


def test_model_server_refusals_and_only_own_children_stop(lab):
    lab.system.free_mib = 4000
    status, _, body = lab.post("/api/model/start", {"name": "r1"})
    assert status == 409 and "4000" in body["error"]
    lab.system.free_mib = 20000

    foreign = socket.socket()
    foreign.bind(("127.0.0.1", lab.port))
    foreign.listen()
    try:
        status, _, body = lab.post("/api/model/start", {"name": "r1"})
        assert status == 409 and str(os.getpid()) in body["error"]
        _, _, status_view = lab.get("/api/status")
        port = next(p for p in status_view["ports"] if p["port"] == lab.port)
        assert port["listening"] and port["pid"] == os.getpid() and port["console_job"] is None
    finally:
        foreign.close()
    for name in ("../r1", "r1;ls", None, "missing"):
        assert lab.post("/api/model/start", {"name": name})[0] == 400, name

    status, _, job = lab.post("/api/model/start", {"name": "r1"})
    assert status == 200 and job["argv"] == [
        str(lab.hgrlt), "serve-round", "--deployment", str(lab.root / "models/r1/deployment.json")]
    running = lab.console.jobs[job["id"]]
    wait(lambda: running.meta["listening"])
    assert lab.post("/api/model/start", {"name": "baseline"})[0] == 409  # one at a time
    _, _, status_view = lab.get("/api/status")
    port = next(p for p in status_view["ports"] if p["port"] == lab.port)
    assert port["console_job"] == job["id"]

    session = start_session(lab)
    status, _, body = lab.post("/api/model/stop", {"job": job["id"]})
    assert status == 409 and "采集" in body["error"]
    lab.post("/api/session/stop")
    wait(lambda: not lab.console._session(), timeout=30)
    assert lab.console.jobs[session].code == 0

    assert lab.post("/api/model/stop", {"job": "someone-else"})[0] == 404
    assert lab.post("/api/model/stop", {"job": session})[0] == 404  # not a model job
    assert lab.post("/api/model/stop", {"job": job["id"]})[0] == 200
    wait(lambda: not running.running)
    assert f"SERVE serve-round --deployment {lab.root / 'models/r1/deployment.json'}" in fake_log(lab)


# ---- output parsing -------------------------------------------------------------------


def console_text(*calls):
    stream = io.StringIO()
    for call in calls:
        if call[0] == "mode":
            with contextlib.redirect_stdout(stream):
                show_mode(call[1])
        else:
            message(*call, stream=stream)
    return stream.getvalue()


@pytest.mark.parametrize("chunk", [1, 7, 10_000])
def test_parser_follows_the_real_console_lines(chunk):
    text = (
        console_text(("下一条", "episode-000003.h5；重置场景后再开始，归零仍需现场确认"))
        + "回车开始本条；q 结束本次连续采集："
    )
    parser = webui.SessionParser()
    states = []

    def feed(text):
        for i in range(0, len(text), chunk):
            parser.feed(text[i:i + chunk])
        states.append(parser.state())

    feed(text)
    assert states[-1]["stage"] == "await_start" and states[-1]["episode"] == "episode-000003.h5"
    parser.typed("\r")
    assert parser.stage == "starting"
    feed("\n" + console_text(("归零确认", "双臂回零、夹爪张开")) + "按回车确认；输入其它内容或 Ctrl-C 取消：")
    assert parser.stage == "confirm_homing"
    parser.typed("\r")
    assert parser.stage == "homing"
    feed("\n" + console_text(("归零完成", "反馈已到位"), ("操作", "按键直接生效"), ("mode", "HOLD")))
    assert (parser.stage, parser.mode) == ("recording", "HOLD")
    feed(console_text(("mode", "POLICY_RUNNING"), ("RL", "RL actor 接管策略执行；h 接管 / p 回到 VLA / s 停止")))
    assert parser.mode == "POLICY_RUNNING" and parser.rl
    feed(console_text(("mode", "HOLD"), ("接管准备", "已保持，扳机已断电。双手握住两只主臂")))
    assert parser.takeover == {"phase": "confirm", "left": False, "right": False} and not parser.rl
    feed(console_text(("接管准备", "右扳机已确认")))
    assert parser.takeover["right"] and not parser.takeover["left"]
    feed(console_text(("接管准备", "左扳机已确认"), ("接管准备", "确认完成，扳机上电回到夹爪开度")))
    assert parser.takeover == {"phase": "return", "left": True, "right": True}
    feed(console_text(("接管准备", "扳机未回到夹爪开度（左 0.50→0.30）；松开手指或按 s 取消")))
    assert "0.50" in parser.takeover["warning"]
    feed(console_text(("mode", "INTERVENTION"), ("接管", "人工接管：主臂和扳机已放松")))
    assert parser.mode == "INTERVENTION" and parser.takeover is None
    feed(console_text(("跟随卡住", "已保持，不会自动重试。")))
    assert parser.notice["label"] == "跟随卡住"
    feed(console_text(("mode", "HOLD"), ("结果标记", "1 成功 · 2 失败 · 3 弃用")))
    assert parser.stage == "review"
    feed(console_text(("失败类型", "1 选错 · 2 抓取失败")))
    assert parser.stage == "failure_type"
    feed(console_text(("保存", "已确认 HOLD 并释放控制权；正在等待落盘，请勿中断。")))
    assert parser.stage == "saving"
    feed(console_text(("保存", "已确认 HOLD 并释放控制权；记录已保存。没有自动归位。")))
    assert parser.stage == "saved"
    feed(console_text(("采集统计", "累计 3 条 · 成功 1 · 失败 1 · 弃用 1 · 未完成 0 · 未标结果 0 · 纳入清单 2 条 · trash 1 条（本次移入 1）")))
    assert parser.stats == {"total": 3, "success": 1, "fail": 1, "discard": 1, "incomplete": 0,
                            "unlabelled": 0, "listed": 2}
    feed(console_text(("mode", "FAULT"), ("故障", "采集异常退出")))
    assert (parser.stage, parser.mode) == ("fault", "FAULT")
    feed(console_text(("采集停止", "Control/recording loop stalled")))
    assert parser.stage == "ended" and parser.notice["label"] == "采集停止"
    feed(console_text(("录制清单", "/x/records.json")))
    assert parser.stage == "finished"


def test_parser_ignores_other_lines_and_rewritten_text():
    parser = webui.SessionParser()
    assert not parser.feed("[模式] POLICY_RUNNING without time\nPOLICY_RUNNING\n")
    assert parser.mode is None
    parser.feed("progress 10%\rprogress 90%\r[12:00:00] [模式] HOLD · 保持\n")
    assert parser.mode == "HOLD"
    parser.feed("[12:00:01] [下一条] episode-000001.h5；x\n回车开始本条；q 结束本次连续采集：")
    parser.typed("q")
    assert parser.stage == "await_start"
    parser.typed("\r")
    assert parser.stage == "ending"
    daemon = webui.SessionParser()
    daemon.feed("托住双臂后输入 HG-DAEMON-ON-SITE 确认重启：")
    assert daemon.stage == "await_phrase"
    daemon.typed("HG-DAEMON-ON-SITE\r")
    assert daemon.stage == "running"


# ---- results --------------------------------------------------------------------------


def synthetic(path, *, outcome, complete=True, usable=True, failure_type=None, states=(), metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as recording:
        recording.attrs.update(complete=complete, usable=usable, unusable=not usable, outcome=outcome,
                               metadata_json=json.dumps(metadata or {}))
        if failure_type:
            recording.attrs["failure_type"] = failure_type
        events = recording.create_group("dagger_events")
        events["timestamp_ns"] = [i * 1_000_000_000 for i in range(len(states))]
        events["controller_state"] = list(states)
        events["reason"] = ["x"] * len(states)


def test_results_summarize_recordings_and_compare_one_day(lab):
    root = lab.root
    plan = iter((
        ({0: ["p"], 40: ["k"]}, "success"),  # autonomous
        (None, "success"),  # p h p k: one takeover
        (None, ("fail", "grasp")),
        (None, "discard"),
    ))

    def record(path, group):
        keys, outcome = next(plan)
        Harness(path, keys=keys).run(
            review=lambda: outcome, metadata={"collection": {"group": group}, "model_round": 0}
        )

    run_collection(root / "data/eval/1016-baseline", record, episodes=4, prompt=lambda _: "")
    r1 = root / "data/eval/1016-r1"
    synthetic(r1 / "episodes/episode-000001.h5", outcome="fail", failure_type="drop",
              states=("HOLD", "POLICY_RUNNING", "HOLD"), metadata={"sealed_policy_id": POLICY_ID, "model_round": 1})
    synthetic(r1 / "episodes/episode-000002.h5", outcome="success",
              states=("HOLD", "POLICY_RUNNING", "INTERVENTION", "INTERVENTION", "POLICY_RUNNING", "INTERVENTION", "HOLD"),
              metadata={"sealed_policy_id": POLICY_ID})
    synthetic(r1 / "trash/episode-000003.h5", outcome="in_progress", complete=False, usable=False)
    synthetic(root / "data/eval/1015-baseline/episodes/episode-000001.h5", outcome="success", metadata={"model_round": 0})
    synthetic(root / "data/recordings/heavier-r1/episodes/episode-000001.h5", outcome="success")
    (root / "data/eval/notes").mkdir()  # no episodes/: not listed
    outside = lab.root.parent / "outside"
    synthetic(outside / "episodes/episode-000001.h5", outcome="success")
    (root / "data/eval/1016-link").symlink_to(outside)  # a run directory outside data/
    (r1 / "episodes/episode-000009.h5").symlink_to(outside / "episodes/episode-000001.h5")

    status, _, results = lab.get("/api/results")
    assert status == 200
    dirs = {(d["kind"], d["name"]): d for d in results["dirs"]}
    assert set(dirs) == {("eval", "1015-baseline"), ("eval", "1016-baseline"), ("eval", "1016-r1"),
                         ("recordings", "heavier-r1")}
    base = dirs["eval", "1016-baseline"]
    assert (base["labelled"], base["success"], base["fail"]) == (3, 2, 1)
    assert (base["autonomous_success"], base["takeovers"], base["episodes_with_takeover"]) == (1, 2, 2)
    assert base["failure_types"] == {"grasp": 1} and base["discarded"] == 1 and base["incomplete"] == 0
    assert base["models"] == ["baseline"] and base["success_rate"] == pytest.approx(2 / 3)
    r1_summary = dirs["eval", "1016-r1"]
    assert r1_summary["models"] == ["r1"] and r1_summary["failure_types"] == {"drop": 1}
    assert (r1_summary["takeovers"], r1_summary["autonomous_success"], r1_summary["incomplete"]) == (2, 0, 1)
    assert results["compare"] == {"1016": ["1016-baseline", "1016-r1"], "1015": ["1015-baseline"]}

    status, _, detail = lab.get("/api/results/detail?kind=eval&name=1016-baseline")
    assert status == 200 and [r["folder"] for r in detail["episodes"]] == ["episodes"] * 3 + ["trash"]
    assert [r["takeovers"] for r in detail["episodes"][:3]] == [0, 1, 1]
    assert detail["episodes"][2]["failure_type"] == "grasp"
    assert all(r["duration_s"] is not None for r in detail["episodes"])
