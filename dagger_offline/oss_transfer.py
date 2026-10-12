"""实验室 ↔ OSS ↔ 学校：可单文件启动的传输工具，仅使用 Python 标准库。

认证交给各机器本地的 ossutil 配置；不读取、复制或接受 AccessKey。
发布地址绑定内容哈希，RELEASE.json 最后上传；下载不启动训练或推理。
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

SCHEMA = "hg-oss-release-v1"
ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
DEFAULT_PREFIX = "oss://ayixiayi-model-relay/relay/hg-dagger"


def message(label, text):
    # This file is also downloaded alone, before the project or environment exists.
    prefix = f"[{label}]"
    if (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.getenv("TERM") != "dumb"
    ):
        prefix = f"\033[1;36m{prefix}\033[0m"
    print(f"{prefix} {text}", flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
        or name == "."
        or "\\" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError(f"不安全的包内路径：{name!r}")
    return name


def oss_prefix(uri):
    uri = uri.rstrip("/")
    match = re.fullmatch(r"oss://([a-z0-9][a-z0-9-]{1,61}[a-z0-9])/(.+)", uri)
    if not match:
        raise ValueError("必须指定 oss://BUCKET/专用前缀，不能使用 bucket 根目录")
    safe_name(match[2])
    return uri


@contextlib.contextmanager
def locked_cache(cache):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"同一传输正在运行，请勿重复启动：{cache}") from exc
        yield


class OSS:
    """ossutil 1.7.19-compatible transport; the CLI alone handles credentials."""

    def __init__(self, executable="ossutil", config=None, endpoint=ENDPOINT):
        if not re.fullmatch(r"https://[a-z0-9.-]+\.aliyuncs\.com", endpoint):
            raise ValueError("endpoint 必须是阿里云 OSS 的 HTTPS 地址")
        binary = shutil.which(str(executable))
        if binary is None:
            raise FileNotFoundError(
                "未找到 ossutil；请先安装，或用 --ossutil 指定已有可执行文件"
            )
        self.executable = str(Path(binary).resolve())
        self.endpoint = endpoint
        self.config = (
            Path(config).expanduser().resolve() if config is not None else None
        )

    def _run(self, arguments, cache, *, allow_missing=False):
        command = [self.executable, "-e", self.endpoint]
        if self.config is not None:
            command += ["-c", str(self.config)]
        result = subprocess.run(
            [*command, *map(str, arguments)],
            cwd=cache,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode == 0:
            return True
        # An authorization/network error must never be mistaken for absence.
        if (
            allow_missing
            and re.search(r"\bStatusCode=404\b", result.stdout)
            and re.search(r"\bErrorCode=NoSuchKey\b", result.stdout)
        ):
            return False
        # Do not echo raw provider diagnostics, which can contain account details.
        raise RuntimeError(
            f"ossutil {arguments[0]} 失败（退出码 {result.returncode}）；"
            "请在本机检查 ossutil 配置、权限和网络；认证文件不应放入训练包"
        )

    def copy(self, source, destination, cache, *, allow_missing=False):
        checkpoint = cache / "checkpoints" / identity([str(source), str(destination)])
        return self._run(
            [
                "cp",
                source,
                destination,
                "-f",
                "--parallel",
                "2",
                "--checkpoint-dir",
                checkpoint,
            ],
            cache,
            allow_missing=allow_missing,
        )


def archive_inventory(archive, kind):
    members = {}
    for member in archive.getmembers():
        name = safe_name(member.name)
        if not member.isfile() or name in members:
            raise ValueError(f"训练/模型包仅允许不重复的普通文件：{name}")
        members[name] = member
    required = {
        "training": {
            "Dagger-Practice/train.sh",
            "round/lerobot/combined.json",
            "round/lerobot/norm_stats.json",
        },
        "model": {"deployment.json"},
    }[kind]
    if not required.issubset(members) or "SHA256SUMS" not in members:
        raise ValueError(f"不是完整的 {kind} 包；请使用 pack-training / pack-model")
    if members["SHA256SUMS"].size > 64 * 1024 * 1024:
        raise ValueError("包内校验清单过大")
    with archive.extractfile(members["SHA256SUMS"]) as stream:
        lines = stream.read().decode("utf-8").splitlines()
    sums = {}
    for line in lines:
        value, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("包内 SHA256SUMS 格式错误")
        safe_name(name)
        if name in sums:
            raise ValueError("包内校验清单有重复文件")
        sums[name] = value
    if set(sums) != set(members) - {"SHA256SUMS"}:
        raise ValueError("包内文件与 SHA256SUMS 不一致")
    return members, sums


def extract_verified(archive_path, destination, kind):
    """Validate every byte in a private staging dir before publishing the folder."""
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"不覆盖已有目录：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members, sums = archive_inventory(archive, kind)
        if (
            sum(m.size for m in members.values())
            > shutil.disk_usage(destination.parent).free
        ):
            raise OSError("目标磁盘空间不足；需要同时存放下载缓存和解压文件")
        with tempfile.TemporaryDirectory(
            prefix=".hg-unpack-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary) / "ready"
            stage.mkdir(mode=0o700)
            for name, member in members.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if name in sums and digest(target) != sums[name]:
                    raise ValueError(f"包内 SHA256 校验失败：{name}")
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            if os.path.lexists(destination):
                raise FileExistsError(f"不覆盖已有目录：{destination}")
            stage.rename(destination)
    return destination


def upload(archive_path, destination, kind, oss, *, cache=None):
    archive_path = Path(archive_path).expanduser().resolve(strict=True)
    destination = oss_prefix(destination)
    if not archive_path.is_file() or not archive_path.name.endswith(
        (".tar", ".tar.gz")
    ):
        raise ValueError("请先打包为 .tar / .tar.gz，再上传")
    before = archive_path.stat()
    with tarfile.open(archive_path) as archive:
        archive_inventory(archive, kind)
    message("1/3 · 校验", f"{archive_path}（{before.st_size / 1024**3:.2f} GiB）")
    bootstrap = Path(__file__).read_bytes()
    release = {
        "schema": SCHEMA,
        "kind": kind,
        "archive": {
            "name": "bundle.tar.gz"
            if archive_path.name.endswith(".gz")
            else "bundle.tar",
            "bytes": before.st_size,
            "sha256": digest(archive_path),
        },
        "bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
    }
    uri = f"{destination}/{identity(release)}"
    cache = Path(
        cache or archive_path.parent / ".hg-oss-cache"
    ).expanduser().resolve() / identity(uri)
    with locked_cache(cache):
        marker = cache / "RELEASE.json"
        # v1 stat queries ACLs first, requiring more than object read/write access.
        if oss.copy(f"{uri}/RELEASE.json", marker, cache, allow_missing=True):
            if json.loads(marker.read_text()) != release:
                raise ValueError("已发布地址的内容不一致；拒绝覆盖")
            message("已发布", "相同内容已存在，不重复上传")
        else:
            marker.write_text(json.dumps(release, indent=2) + "\n")
            helper = cache / "oss_transfer.py"
            helper.write_bytes(bootstrap)
            message("2/3 · 上传", f"{uri}；中断后重复原命令即可续传")
            oss.copy(archive_path, f"{uri}/{release['archive']['name']}", cache)
            oss.copy(helper, f"{uri}/oss_transfer.py", cache)
            after = archive_path.stat()
            if any(
                getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            ):
                raise ValueError("上传期间本地训练/模型包被修改；未发布完成标记")
            oss.copy(marker, f"{uri}/RELEASE.json", cache)
    message("3/3 · OSS 地址", uri)
    message(
        "接收端", "先在接收机器配置 ossutil，再执行以下命令（目标目录必须不存在）："
    )
    helper_name = f"hg_oss_{identity(release)[:12]}.py"
    print(
        shlex.join(
            ["ossutil", "-e", oss.endpoint, "cp", f"{uri}/oss_transfer.py", helper_name]
        )
    )
    print(
        f"printf '%s\\n' '{release['bootstrap_sha256']}  {helper_name}' | sha256sum -c -"
    )
    print(
        shlex.join(
            [
                "python3",
                helper_name,
                "download",
                "--kind",
                kind,
                "--source",
                uri,
                "--endpoint",
                oss.endpoint,
                "--destination",
                "./hg-training" if kind == "training" else "./hg-model",
            ]
        )
    )
    return uri


def download(source, destination, kind, oss, *, cache=None):
    source = oss_prefix(source)
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"不覆盖已有目录：{destination}")
    cache = Path(
        cache or destination.parent / ".hg-oss-cache"
    ).expanduser().resolve() / identity(source)
    with locked_cache(cache):
        marker = cache / "RELEASE.json"
        message("1/3 · 发布清单", source)
        oss.copy(f"{source}/RELEASE.json", marker, cache)
        release = json.loads(marker.read_text())
        if (
            not isinstance(release, dict)
            or release.get("schema") != SCHEMA
            or release.get("kind") != kind
            or identity(release) != source.rsplit("/", 1)[-1]
        ):
            raise ValueError("发布类型、版本或内容地址不匹配；拒绝下载")
        info = release.get("archive", {})
        if (
            not isinstance(info, dict)
            or info.get("name") not in {"bundle.tar", "bundle.tar.gz"}
            or type(info.get("bytes")) is not int
            or info["bytes"] <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", str(info.get("sha256")))
        ):
            raise ValueError("发布清单的文件名、长度或哈希无效")
        archive_path = cache / info["name"]
        message("2/3 · 下载", "使用独立断点缓存；中断后重复原命令即可续传")
        if (
            not archive_path.is_file()
            or archive_path.stat().st_size != info["bytes"]
            or digest(archive_path) != info["sha256"]
        ):
            oss.copy(f"{source}/{info['name']}", archive_path, cache)
        if (
            archive_path.stat().st_size != info["bytes"]
            or digest(archive_path) != info["sha256"]
        ):
            raise ValueError("OSS 下载包的长度或 SHA256 不一致；未生成目标目录")
        extract_verified(archive_path, destination, kind)
    message("3/3 · 校验完成", str(destination))
    if kind == "training":
        entry = destination / "Dagger-Practice/train.sh"
        message(
            "学校端",
            f"依次运行 bash {shlex.quote(str(entry))} setup / check / train；未自动启动训练",
        )
    else:
        message(
            "实验室端",
            f"部署清单：{destination / 'deployment.json'}；未切换模型，未启动推理或机器人",
        )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("upload", "download"):
        sub = commands.add_parser(command)
        sub.add_argument("--kind", choices=("training", "model"), required=True)
        sub.add_argument(
            "--ossutil", default="ossutil", help="已有 ossutil 1.7.19 可执行文件"
        )
        sub.add_argument(
            "--config", type=Path, help="本机 ossutil 配置路径；不读取或复制"
        )
        sub.add_argument("--endpoint", default=ENDPOINT)
        sub.add_argument("--cache", type=Path, help="保留断点缓存；默认在包/目标目录旁")
        if command == "upload":
            sub.add_argument("--archive", type=Path, required=True)
            sub.add_argument(
                "--destination", help=f"OSS 专用前缀；默认 {DEFAULT_PREFIX}/包类型"
            )
        else:
            sub.add_argument(
                "--source", required=True, help="上传命令打印的完整 OSS 内容地址"
            )
            sub.add_argument(
                "--destination", type=Path, required=True, help="接收机器自己的新目录"
            )
    args = parser.parse_args(argv)
    oss = OSS(args.ossutil, args.config, args.endpoint)
    if args.command == "upload":
        return upload(
            args.archive,
            args.destination or f"{DEFAULT_PREFIX}/{args.kind}",
            args.kind,
            oss,
            cache=args.cache,
        )
    return download(args.source, args.destination, args.kind, oss, cache=args.cache)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        message("失败", str(exc))
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        message("已中断", "未自动启动训练/推理；重复原命令可以续传")
        raise SystemExit(130) from None
