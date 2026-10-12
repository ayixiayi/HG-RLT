"""Allowlisted code-only transfer archive; no credentials, envs, weights or data."""

import argparse
import hashlib
import subprocess
import tarfile
import tempfile
from pathlib import Path

REVISION = "d0832d12651d1b260a652861a14648dc5f3660c7"
FILES = (
    "README.md",
    "requirements-cpu.txt",
    "run_cpu.sh",
    "prepare.py",
    "launch.py",
    "bundle.py",
    "test_recorder_export.py",
    "test_native_data.py",
    "test_launch.py",
)
HG_FILES = (
    ".gitignore",
    "README.md",
    "AGENTS.md",
    "hgrlt",
    "docs/OPERATOR_GUIDE.md",
    "docs/THIS_PROJECT.md",
    "docs/RHOS_INTEGRATION.md",
    "docs/records.example.json",
    "requirements.txt",
    "yam_openpi.py",
    "convert_mcap.py",
    "Dagger-Practice/requirements-data-cpu.txt",
    "Dagger-Practice/requirements-live.txt",
    "Dagger-Practice/lab.sh",
    "Dagger-Practice/train.sh",
    "Dagger-Practice/setup_train.sh",
    "Dagger-Practice/requirements-train.txt",
    "Dagger-Practice/daemon.py",
    "Dagger-Practice/run_robot.py",
    "Dagger-Practice/collect.py",
    "Dagger-Practice/patches/README.md",
    "Dagger-Practice/patches/dynamixel-fresh-sample.patch",
    "Dagger-Practice/patches/dynamixel_fresh_sample.py",
    "Dagger-Practice/patches/hg-protocol.patch",
    "Dagger-Practice/patches/hg-protocol-base.sha256",
    "Dagger-Practice/patches/hg-protocol-patched.sha256",
    "Dagger-Practice/patches/hg_protocol_integration.py",
    "dagger_offline/requirements.txt",
    "dagger_offline/arbitration.py",
    "dagger_offline/round_data.py",
    "dagger_offline/lerobot_bridge.py",
    "dagger_offline/training.py",
    "dagger_offline/workflow.py",
    "dagger_offline/training_package.py",
    "dagger_offline/oss_transfer.py",
    "dagger_offline/deployment.py",
    "dagger_offline/evaluation.py",
    "dagger_offline/console.py",
    "dagger_offline/collection.py",
    "dagger_offline/live_control.py",
    "dagger_offline/live_robot.py",
    "dagger_offline/live_recording.py",
    "dagger_offline/test_collection.py",
    "dagger_offline/test_arbitration.py",
    "dagger_offline/test_round_data.py",
    "dagger_offline/test_lerobot_bridge.py",
    "dagger_offline/test_training.py",
    "dagger_offline/test_oss_transfer.py",
    "dagger_offline/test_deployment.py",
    "dagger_offline/test_evaluation.py",
    "dagger_offline/test_official_export.py",
    "dagger_offline/test_official_control.py",
    "dagger_offline/test_live_control.py",
    "dagger_offline/test_live_robot.py",
    "dagger_offline/test_live_recording.py",
    "dagger_offline/test_live_runner.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hg-dagger",
        action="store_true",
        help="Include the Pi0.5 live adapter, tests, and driver patch",
    )
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.abc), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.abc), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != REVISION or dirty:
        raise ValueError("requires clean pinned ABC checkout")
    checksum = args.output.with_name(args.output.name + ".sha256")
    if args.output.exists() or checksum.exists():
        raise FileExistsError(args.output)
    local = Path(__file__).resolve().parent
    entries = [(local / name, f"Dagger-Practice/{name}") for name in FILES]
    if args.hg_dagger:
        entries.extend((local.parent / name, name) for name in HG_FILES)
    for path, name in entries:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/nonregular source: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dagger-transfer-") as temporary:
        stage = Path(temporary)
        official = stage / "abc.bundle"
        subprocess.run(
            ["git", "-C", str(args.abc), "bundle", "create", str(official), "HEAD"],
            check=True,
        )
        entries.append((official, "abc.bundle"))
        manifest = stage / "SHA256SUMS"
        manifest.write_text(
            "".join(f"{sha256(path)}  {name}\n" for path, name in entries)
        )
        with tarfile.open(args.output, "x:gz") as archive:
            for path, name in entries:
                archive.add(path, arcname=name, recursive=False)
            archive.add(manifest, arcname="SHA256SUMS", recursive=False)
    with checksum.open("x") as stream:
        stream.write(f"{sha256(args.output)}  {args.output.name}\n")
    print(f"Code-only bundle: {args.output}\nChecksum: {checksum}")


if __name__ == "__main__":
    main()
