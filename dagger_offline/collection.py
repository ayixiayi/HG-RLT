"""Repeated episodes and a derived records list; HDF5 owns outcomes and groups."""

import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py

from dagger_offline.console import message


@contextmanager
def collection_lock(directory):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有采集进程使用目录：{directory}") from exc
        lock.seek(0)
        marker = lock.read()
        if not marker:
            if (directory / "records.json").exists():
                raise FileExistsError("目录已有其他录制清单，请选择新的采集目录")
            lock.write("hg-collection-v1\n")
            lock.flush()
        elif marker != "hg-collection-v1\n":
            raise ValueError("采集目录版本不匹配")
        (directory / "episodes").mkdir(exist_ok=True)
        yield directory


def _move_to_trash(directory, path):
    """Keep discarded/incomplete raw files, apart from labelled episodes."""
    trash = directory / "trash"
    trash.mkdir(exist_ok=True)
    for source in (path, path.with_suffix(".prepare.jsonl")):
        if source.exists():
            target = trash / source.name
            suffix = 1
            while target.exists():
                target = trash / f"{source.stem}.{suffix}{source.suffix}"
                suffix += 1
            os.replace(source, target)


def episode_numbers(directory):
    """Episode numbers in use, including those moved to trash (never reused)."""
    directory = Path(directory)
    return [
        int(path.name.split(".")[0].removeprefix("episode-"))
        for folder in ("episodes", "trash")
        for path in (directory / folder).glob("episode-*.h5")
    ]


def refresh_records(directory):
    """Rebuild the index under collection_lock.

    Discarded and incomplete recordings move to ``trash/`` in the same
    collection directory; ``episodes/`` keeps only labelled, usable ones.
    """
    directory = Path(directory)
    records = []
    counts = dict(total=0, success=0, fail=0, discard=0, incomplete=0, unlabelled=0)
    trashed = 0
    for path in sorted((directory / "episodes").glob("episode-*.h5")):
        counts["total"] += 1
        unwanted = False
        try:
            with h5py.File(path, "r") as recording:
                outcome = recording.attrs.get("outcome", "unlabelled")
                if outcome in ("success", "fail"):
                    counts[outcome] += 1
                if not recording.attrs.get("complete", False):
                    counts["incomplete"] += 1
                    unwanted = True
                elif not recording.attrs.get("usable", False) or recording.attrs.get(
                    "unusable", True
                ):
                    counts["discard"] += 1
                    unwanted = True
            if unwanted:
                _move_to_trash(directory, path)
                trashed += 1
                continue
            with h5py.File(path, "r") as recording:
                outcome = recording.attrs.get("outcome", "unlabelled")
                if outcome not in ("success", "fail"):
                    counts["unlabelled"] += 1
                    continue
                metadata = json.loads(recording.attrs["metadata_json"])
                group = metadata.get("collection", {}).get("group")
                if not isinstance(group, str) or not group.strip():
                    raise ValueError(f"录制缺少来源组：{path}")
                row = {
                    "path": str(path.relative_to(directory)),
                    "group": group,
                    "role": "current",
                    "outcome": outcome,
                }
                if "failure_type" in recording.attrs:
                    row["failure_type"] = str(recording.attrs["failure_type"])
                records.append(row)
        except OSError as exc:
            counts["incomplete"] += 1
            # Unreadable may only mean another reader holds it: never move.
            message("记录待检查", f"{path.name}：{exc}", level="warning")
    # This is a replaceable view of HDF5 metadata, not a second source of labels.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".records-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(records, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, directory / "records.json")
        finally:
            temporary.unlink(missing_ok=True)
    message(
        "采集统计",
        f"累计 {counts['total']} 条 · 成功 {counts['success']} · 失败 {counts['fail']} · "
        f"弃用 {counts['discard']} · 未完成 {counts['incomplete']} · "
        f"未标结果 {counts['unlabelled']} · 纳入清单 {len(records)} 条 · "
        f"trash {len(list((directory / 'trash').glob('episode-*.h5')))} 条"
        + (f"（本次移入 {trashed}）" if trashed else ""),
    )
    return records, counts


def run_collection(directory, record_episode, *, episodes=0, group=None, prompt=input):
    """One manual start per attempt; the callback owns zeroing and safe recording."""
    if type(episodes) is not int or episodes < 0:
        raise ValueError("--episodes 为本次最多采集的次数，0 表示不限次数")
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("--group 需要非空来源组名")
    with collection_lock(directory) as root:
        refresh_records(root)
        attempted = 0
        while episodes == 0 or attempted < episodes:
            index = 1 + max(episode_numbers(root), default=0)
            path = root / "episodes" / f"episode-{index:06d}.h5"
            message("下一条", f"{path.name}；重置场景后再开始，归零仍需现场确认")
            answer = prompt("回车开始本条；q 结束本次连续采集：").strip().lower()
            if answer == "q":
                break
            if answer:
                message("等待", "请按回车开始，或输入 q 结束")
                continue
            try:
                record_episode(path, group or f"hg:{uuid.uuid4().hex}")
            except BaseException:
                try:
                    refresh_records(root)
                except Exception as exc:
                    message("清单待刷新", str(exc), level="error")
                raise
            attempted += 1
            refresh_records(root)
        message("录制清单", str(root / "records.json"), level="success")
        message(
            "数据准备",
            "prepare-data --records 此清单 --output 新轮次目录；自动划分 train/val",
        )
        return root / "records.json"
