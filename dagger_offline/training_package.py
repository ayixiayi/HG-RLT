"""Allowlisted, relocatable Pi0.5 training/model archives; no robot code or credentials."""

import json
import os
import tarfile
import tempfile
from pathlib import Path

from dagger_offline.oss_transfer import safe_name
from dagger_offline.training import WindowDataset, digest, validate_parent

CODE_FILES = (
    "docs/OPERATOR_GUIDE.md",
    "docs/THIS_PROJECT.md",
    "docs/records.example.json",
    "Dagger-Practice/patches/README.md",
    "yam_openpi.py",
    "Dagger-Practice/train.sh",
    "Dagger-Practice/setup_train.sh",
    "Dagger-Practice/requirements-train.txt",
    "Dagger-Practice/requirements-data-cpu.txt",
    "dagger_offline/console.py",
    "dagger_offline/training.py",
    "dagger_offline/rl_token.py",
    "dagger_offline/training_package.py",
    "dagger_offline/oss_transfer.py",
    "dagger_offline/workflow.py",
    "dagger_offline/deployment.py",
)


def tree_files(root, destination):
    """Materialize file symlinks (BC videos), never silently skip linked dirs."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        if not path.is_file():
            raise ValueError(f"非普通文件或目录链接，不能打包：{path}")
        yield path, f"{destination}/{path.relative_to(root).as_posix()}"


def _output_path(output):
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output) or os.path.lexists(str(output) + ".sha256"):
        raise FileExistsError(output)
    if not output.name.endswith((".tar", ".tar.gz")):
        raise ValueError("训练/模型包使用 .tar 或 .tar.gz 后缀")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_archive(files, output, *, expected=None):
    output = _output_path(output)
    with tempfile.TemporaryDirectory(
        prefix="hg-package-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        sums = stage / "SHA256SUMS"
        archive_path = stage / "bundle.tar"
        mode = "x:gz" if output.name.endswith(".gz") else "x"
        names = set()
        with (
            sums.open("x") as inventory,
            tarfile.open(archive_path, mode, dereference=True) as archive,
        ):
            for path, name in files:
                safe_name(name)
                if name in names or name == "SHA256SUMS":
                    raise ValueError(f"包内路径重复：{name}")
                names.add(name)
                before = path.stat()
                value = digest(path)
                if (
                    expected is not None
                    and name in expected
                    and value != expected[name]
                ):
                    raise ValueError(f"封存后模型内容被修改：{path}")
                archive.add(path, arcname=name, recursive=False)
                after = path.stat()
                if any(
                    getattr(before, field) != getattr(after, field)
                    for field in (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                ):
                    raise ValueError(f"打包过程中输入被修改：{path}")
                inventory.write(f"{value}  {name}\n")
            inventory.flush()
            archive.add(sums, arcname="SHA256SUMS", recursive=False)
        # Publish only a completed archive. link() refuses to replace any file.
        os.link(archive_path, output)
    with output.with_name(output.name + ".sha256").open("x") as stream:
        stream.write(f"{digest(output)}  {output.name}\n")
    return output


def pack_model(checkpoint, output, *, model_round, port=8769):
    """Seal a committed round and export inference/next-round parent files only.

    Keep training optimizer state on the school server for same-run resume.
    Relocation changes neither the policy identity nor normalization.
    """
    from dagger_offline.deployment import seal

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = _output_path(output)
    with tempfile.TemporaryDirectory(
        prefix="hg-model-", dir=output.parent
    ) as temporary:
        manifest = Path(temporary) / "deployment.json"
        deployment = seal(checkpoint, manifest, model_round=model_round, port=port)
        prefix = f"checkpoint/{checkpoint.parent.parent.name}/{checkpoint.parent.name}/{checkpoint.name}"
        deployment["checkpoint"] = prefix
        manifest.write_text(json.dumps(deployment, indent=2) + "\n")
        files = [(manifest, "deployment.json")]
        files.extend(
            (checkpoint / name, f"{prefix}/{name}") for name in deployment["files"]
        )
        # Preserve exactly the sidecar used by seal(), even if the source run changes.
        sidecar = Path(temporary) / "run.dagger.json"
        sidecar.write_text(json.dumps(deployment["run"], indent=2) + "\n")
        files.append((sidecar, f"{Path(prefix).parent.as_posix()}.dagger.json"))
        expected = {
            f"{prefix}/{name}": value for name, value in deployment["files"].items()
        }
        return _write_archive(files, output, expected=expected)


def pack_training(manifest_path, parent, output, *, include_parent=True):
    """Copy the selected mixture and optionally its parent, without lab paths.

    Historical source filenames inside provenance remain historical references;
    only the operational dataset/stat paths are rewritten. No raw HDF5 or ABC
    image exports are needed by OpenPI. Existing dataset files are never edited.
    """
    manifest_path, parent = (
        Path(path).expanduser().resolve() for path in (manifest_path, parent)
    )
    output = _output_path(output)
    stats = manifest_path.parent / "norm_stats.json"
    validate_parent(parent / "params", stats)
    for split in ("train", "val"):
        WindowDataset(manifest_path, split)
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_checkpoint"] = "../../parent" if include_parent else None
    project = Path(__file__).resolve().parents[1]
    files = [(project / name, name) for name in CODE_FILES]
    files.append((stats, "round/lerobot/norm_stats.json"))
    for index, entry in enumerate(manifest["datasets"]):
        source = (manifest_path.parent / entry["path"]).resolve()
        entry["path"] = f"datasets/{index:03d}-{entry['component']}-{entry['split']}"
        for folder in ("meta", "data", "videos", "images"):
            if (source / folder).is_dir():
                files.extend(
                    tree_files(
                        source / folder, f"round/lerobot/{entry['path']}/{folder}"
                    )
                )
    if include_parent:
        for folder in ("params", "assets"):
            files.extend(tree_files(parent / folder, f"parent/{folder}"))
    # Cache only this public model asset, never the user's whole HOME/cache.
    cache = Path(os.environ.get("OPENPI_DATA_HOME", Path.home() / ".cache/openpi"))
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    if tokenizer.is_file():
        files.append((tokenizer, "cache/openpi/big_vision/paligemma_tokenizer.model"))
    with tempfile.TemporaryDirectory(
        prefix="hg-training-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        portable = stage / "combined.json"
        portable.write_text(json.dumps(manifest, indent=2) + "\n")
        files.append((portable, "round/lerobot/combined.json"))
        original = manifest_path.parent / "lerobot_round.json"
        if original.is_file():
            files.append((original, "provenance/lerobot_round.json"))
        return _write_archive(files, output)
