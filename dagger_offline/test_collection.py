import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from dagger_offline.collection import collection_lock, run_collection
from dagger_offline.test_live_runner import Harness
from dagger_offline.workflow import assign_splits


def test_repeated_collection_labels_resume_and_grouped_training_input(tmp_path):
    outcomes = iter(("success", ("fail", "timeout"), "discard", "success"))

    def record(path, group):
        outcome = next(outcomes)
        Harness(path).run(
            review=lambda: outcome, metadata={"collection": {"group": group}}
        )

    root = tmp_path / "recordings"
    manifest = run_collection(root, record, episodes=3, prompt=lambda _: "")
    first = root / "episodes/episode-000001.h5"
    original = first.read_bytes()
    rows = json.loads(manifest.read_text())
    assert [row["outcome"] for row in rows] == ["success", "fail"]
    assert "failure_type" not in rows[0] and rows[1]["failure_type"] == "timeout"
    assert len({row["group"] for row in rows}) == 2
    assert all("split" not in row for row in rows)
    # Discarded episodes leave episodes/ but stay in the same directory.
    assert not (root / "episodes/episode-000003.h5").exists()
    with h5py.File(root / "trash/episode-000003.h5") as recording:
        assert recording.attrs["outcome"] == "discard"
        assert not recording.attrs["usable"]

    run_collection(root, record, episodes=1, prompt=lambda _: "")
    assert first.read_bytes() == original
    assert (root / "episodes/episode-000004.h5").is_file()  # 3 is not reused
    assert sorted(p.name for p in (root / "episodes").glob("*.h5")) == [
        "episode-000001.h5",
        "episode-000002.h5",
        "episode-000004.h5",
    ]
    rows = json.loads(manifest.read_text())
    assert len(rows) == 3
    assert {row["split"] for row in assign_splits(rows)} == {"train", "val"}
    assert [row["path"] for row in rows] == [
        "episodes/episode-000001.h5",
        "episodes/episode-000002.h5",
        "episodes/episode-000004.h5",
    ]


def test_interrupted_collection_preserves_previous_index_and_never_reuses_number(
    tmp_path,
):
    root = tmp_path / "recordings"

    def record(path, group):
        harness = Harness(path)
        if path.name == "episode-000002.h5":
            harness.safety.release_error = "before"
        harness.run(review=lambda: "success", metadata={"collection": {"group": group}})

    with pytest.raises((TimeoutError, BaseExceptionGroup)):
        run_collection(root, record, episodes=3, prompt=lambda _: "")
    assert [row["path"] for row in json.loads((root / "records.json").read_text())] == [
        "episodes/episode-000001.h5"
    ]
    with h5py.File(root / "trash/episode-000002.h5") as recording:
        assert not recording.attrs["complete"]
    run_collection(root, record, episodes=1, prompt=lambda _: "")
    assert (root / "episodes/episode-000003.h5").is_file()
    assert not (root / "episodes/episode-000002.h5").exists()


def test_collection_requires_manual_start_and_exclusive_directory(tmp_path):
    def record(*_):
        raise AssertionError("operator did not start an episode")

    root = tmp_path / "recordings"
    manifest = run_collection(root, record, prompt=lambda _: "q")
    assert json.loads(manifest.read_text()) == []
    assert not list((root / "episodes").iterdir())
    with collection_lock(root), pytest.raises(RuntimeError, match="已有采集进程"):
        run_collection(root, record, prompt=lambda _: "q")


def test_collect_cli_index_only_needs_no_devices_and_recording_needs_tty(tmp_path):
    script = Path(__file__).resolve().parents[1] / "Dagger-Practice/collect.py"
    args = [
        sys.executable,
        str(script),
        "--rhos-root",
        str(tmp_path / "no-rhos"),
        "--abc",
        str(tmp_path / "no-abc"),
        "--output-dir",
        str(tmp_path / "recordings"),
    ]
    denied = subprocess.run(args, input="", text=True, capture_output=True)
    assert denied.returncode == 1 and "交互终端" in denied.stdout
    assert not (tmp_path / "recordings").exists()
    indexed = subprocess.run(
        [*args, "--index-only"], input="", text=True, capture_output=True
    )
    assert indexed.returncode == 0, indexed.stdout + indexed.stderr
    assert json.loads((tmp_path / "recordings/records.json").read_text()) == []


def test_trash_moves_preparation_log_and_never_overwrites(tmp_path):
    from dagger_offline.collection import _move_to_trash, episode_numbers

    (tmp_path / "episodes").mkdir()
    for content in ("first", "second"):
        (tmp_path / "episodes/episode-000007.h5").write_text(content)
        (tmp_path / "episodes/episode-000007.prepare.jsonl").write_text(content)
        _move_to_trash(tmp_path, tmp_path / "episodes/episode-000007.h5")
    names = sorted(p.name for p in (tmp_path / "trash").iterdir())
    assert names == [
        "episode-000007.1.h5",
        "episode-000007.h5",
        "episode-000007.prepare.1.jsonl",
        "episode-000007.prepare.jsonl",
    ]
    assert (tmp_path / "trash/episode-000007.h5").read_text() == "first"
    assert not list((tmp_path / "episodes").iterdir())
    assert max(episode_numbers(tmp_path)) == 7
