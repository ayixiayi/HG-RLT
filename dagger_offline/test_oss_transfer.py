"""Real CLI/file round trips against a fake ossutil, never OSS credentials or GPU."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from dagger_offline.deployment import load_deployment, seal
from dagger_offline.oss_transfer import OSS, download, extract_verified, main, upload
from dagger_offline.training_package import pack_model

pytest_plugins = ["dagger_offline.test_deployment"]


def make_fake_oss(directory):
    """Model the documented ossutil file interface, not application internals."""
    directory.mkdir()
    store = directory / "objects"
    store.mkdir()
    executable = directory / "ossutil"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-e', required=True)
parser.add_argument('-c')
commands = parser.add_subparsers(dest='command', required=True)
stat = commands.add_parser('stat')
stat.add_argument('source')
cp = commands.add_parser('cp')
cp.add_argument('source')
cp.add_argument('destination')
cp.add_argument('-f', action='store_true')
cp.add_argument('--parallel')
cp.add_argument('--checkpoint-dir')
args = parser.parse_args()
store = Path(__file__).parent / 'objects'
if (store / 'deny').exists():
    print('StatusCode=403, ErrorCode=AccessDenied, synthetic-provider-private-detail')
    raise SystemExit(1)
if args.command == 'stat' and (store / 'deny-acl').exists():
    print('StatusCode=403, ErrorCode=AccessDenied, bucket acl')
    raise SystemExit(1)

def local(value):
    return store / value[6:] if value.startswith('oss://') else Path(value)

source = local(args.source)
if not source.is_file():
    print('StatusCode=404, ErrorCode=NoSuchKey')
    raise SystemExit(1)
if args.command == 'stat':
    raise SystemExit(0)
destination = local(args.destination)
destination.parent.mkdir(parents=True, exist_ok=True)
if (store / 'fail-payload-once').exists() and source.name.startswith('bundle.'):
    (store / 'fail-payload-once').unlink()
    destination.write_bytes(source.read_bytes()[:17])
    print('synthetic interrupted download')
    raise SystemExit(1)
if (store / 'fail-publish-once').exists() and args.destination.startswith('oss://') and destination.name == 'RELEASE.json':
    (store / 'fail-publish-once').unlink()
    raise SystemExit(1)
shutil.copyfile(source, destination)
"""
    )
    executable.chmod(0o755)
    return executable, store


@pytest.fixture
def transport(tmp_path):
    executable, store = make_fake_oss(tmp_path / "fake-oss")
    return OSS(executable), store


def write_archive(path, entries, *, damage=None):
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    if damage == "checksum":
        hashes["deployment.json"] = "0" * 64
    entries = dict(entries)
    entries["SHA256SUMS"] = "".join(
        f"{value}  {name}\n" for name, value in hashes.items()
    ).encode()
    with tarfile.open(path, "w") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if damage in {"traversal", "symlink", "hardlink", "duplicate", "unlisted"}:
            member = tarfile.TarInfo(
                {"traversal": "../escaped", "duplicate": "deployment.json"}.get(
                    damage, "extra"
                )
            )
            if damage in {"symlink", "hardlink"}:
                member.type = (
                    tarfile.SYMTYPE if damage == "symlink" else tarfile.LNKTYPE
                )
                member.linkname = "../../escaped"
            archive.addfile(member)


def test_model_oss_roundtrip_preserves_identity_and_excludes_optimizer(
    candidate, transport, tmp_path
):
    checkpoint, _, manifest = candidate
    original = seal(checkpoint, manifest, model_round=2)
    (checkpoint / "train_state").mkdir()
    (checkpoint / "train_state/optimizer").write_bytes(b"not-for-inference")
    # A unrelated file must never be pulled into the allowlisted model export.
    (checkpoint / "unrelated.txt").write_text("do not export")
    archive = pack_model(checkpoint, tmp_path / "round-model.tar.gz", model_round=2)
    oss, store = transport
    # Object read/write permission is sufficient; querying ACLs is not required.
    (store / "deny-acl").touch()
    arguments = [
        "upload",
        "--kind",
        "model",
        "--archive",
        str(archive),
        "--ossutil",
        oss.executable,
    ]
    uri = main(arguments)
    assert uri.startswith("oss://ayixiayi-model-relay/relay/hg-dagger/model/")
    assert main(arguments) == uri
    assert len(list(store.rglob("RELEASE.json"))) == 1
    # The school tree need not remain reachable from the laboratory.
    checkpoint.parent.rename(checkpoint.parent.with_name("school-only"))
    target = download(uri, tmp_path / "实验室 模型", "model", oss)
    relocated = load_deployment(target / "deployment.json")
    assert relocated["policy_id"] == original["policy_id"]
    assert relocated["model_round"] == 2
    installed = Path(relocated["checkpoint"])
    assert installed.is_relative_to(target)
    assert (installed / "params/weights").read_bytes() == b"synthetic-not-model-weights"
    assert not (installed / "train_state").exists()
    assert not (installed / "unrelated.txt").exists()
    assert (
        json.loads(Path(str(installed.parent) + ".dagger.json").read_text())
        == original["run"]
    )
    with pytest.raises(FileExistsError):
        download(uri, target, "model", oss)
    assert (
        load_deployment(target / "deployment.json")["policy_id"]
        == original["policy_id"]
    )
    with pytest.raises(ValueError, match="不匹配"):
        download(uri, tmp_path / "wrong-kind", "training", oss)
    assert not (tmp_path / "wrong-kind").exists()


def test_failed_publish_and_interrupted_download_can_be_retried(transport, tmp_path):
    oss, store = transport
    archive = tmp_path / "round.tar"
    write_archive(
        archive, {"deployment.json": b"synthetic", "checkpoint/weights": b"payload"}
    )
    (store / "fail-publish-once").touch()
    with pytest.raises(RuntimeError):
        upload(archive, "oss://test-bucket/round", "model", oss)
    assert not list(store.rglob("RELEASE.json"))
    uri = upload(archive, "oss://test-bucket/round", "model", oss)
    target = tmp_path / "lab"
    (store / "fail-payload-once").touch()
    with pytest.raises(RuntimeError):
        download(uri, target, "model", oss)
    assert not target.exists()
    download(uri, target, "model", oss)
    assert (target / "checkpoint/weights").read_bytes() == b"payload"


@pytest.mark.parametrize(
    "damage", ["traversal", "symlink", "hardlink", "duplicate", "checksum", "unlisted"]
)
def test_bad_archive_never_publishes_destination(tmp_path, damage):
    archive = tmp_path / "bad.tar"
    write_archive(archive, {"deployment.json": b"synthetic"}, damage=damage)
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        extract_verified(archive, target, "model")
    assert not target.exists()
    assert not (tmp_path / "escaped").exists()


def test_oss_corruption_and_auth_errors_fail_closed(transport, tmp_path, capsys):
    oss, store = transport
    archive = tmp_path / "round.tar"
    write_archive(archive, {"deployment.json": b"synthetic"})
    (store / "deny").touch()
    with pytest.raises(RuntimeError, match="配置") as error:
        upload(archive, "oss://test-bucket/round", "model", oss)
    assert (
        "synthetic-provider-private-detail"
        not in str(error.value) + capsys.readouterr().out
    )
    assert not list(store.rglob("RELEASE.json"))
    (store / "deny").unlink()
    uri = upload(archive, "oss://test-bucket/round", "model", oss)
    payload = next(store.rglob("bundle.tar"))
    with payload.open("r+b") as stream:
        stream.seek(1024)
        stream.write(b"corruption")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="SHA256"):
        download(uri, target, "model", oss)
    assert not target.exists()
    marker = next(store.rglob("RELEASE.json"))
    metadata = json.loads(marker.read_text())
    metadata["archive"]["name"] = "../elsewhere.tar"
    marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="不匹配"):
        download(uri, target, "model", oss)
    assert not target.exists()
