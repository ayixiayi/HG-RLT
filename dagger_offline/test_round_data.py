import json
import shutil

import h5py
import numpy as np
import pytest

pytest_plugins = ["dagger_offline.test_official_export"]


def test_automatic_split_keeps_unequal_source_groups_and_is_order_independent():
    from dagger_offline.workflow import assign_splits

    records = [
        {"path": f"{group}-{part}.h5", "group": group, "role": "current"}
        for group, count in (("a", 5), ("b", 2), ("c", 1), ("d", 3), ("e", 1))
        for part in range(count)
    ]
    assigned = assign_splits(records, val_fraction=0.4, seed=42)
    mapping = {row["path"]: row["split"] for row in assigned}
    assert mapping == {
        row["path"]: row["split"]
        for row in assign_splits(list(reversed(records)), val_fraction=0.4, seed=42)
    }
    groups = {split: set() for split in ("train", "val")}
    for row in assigned:
        groups[row["split"]].add(row["group"])
    assert len(groups["val"]) == 2 and len(groups["train"]) == 3
    assert groups["train"].isdisjoint(groups["val"])
    assert all("split" not in row for row in records)
    assert mapping != {
        row["path"]: row["split"]
        for row in assign_splits(records, val_fraction=0.4, seed=43)
    }
    # Explicit saved assignments survive a different seed/fraction.
    assert assign_splits(assigned, val_fraction=0.8, seed=17) == assigned
    with pytest.raises(ValueError, match="全部省略"):
        assign_splits([*records[:-1], assigned[-1]])
    with pytest.raises(ValueError, match="两个独立"):
        assign_splits(records[:5])


@pytest.fixture
def round_inputs(recording, tmp_path):
    source = recording[0]
    records = []
    for role in ("previous", "current"):
        for split in ("train", "val"):
            path = tmp_path / f"{role}-{split}.h5"
            shutil.copyfile(source, path)
            with h5py.File(path, "r+") as f:
                f.attrs["fixture_id"] = path.stem
                if split == "val":
                    f["data/q_des_right"][:] += 100
            records.append(
                {"path": str(path), "group": path.stem, "role": role, "split": split}
            )
    stats = tmp_path / "parent_norm.json"
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {"mean": [0.0] * 14, "std": [1.0] * 14}
                    for key in ("state", "actions")
                }
            }
        )
    )
    return records, stats


def test_prepare_exports_pure_segments_and_samples_80_10_10(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round, sample_window

    records, stats = round_inputs
    output = tmp_path / "round"
    manifest = prepare_round(records, output, stats, horizon=3, image_size=(16, 16))
    assert manifest["complete"]
    assert (output / "norm_stats.json").read_bytes() == stats.read_bytes()
    assert [c["weight"] for c in manifest["mixture"]] == [0.8, 0.1, 0.1]
    for episode in manifest["episodes"]:
        rows = np.fromfile(
            output / episode["path"] / "states_actions.bin", dtype=np.float64
        ).reshape(-1, 28)
        start, end = episode["frame_range"]
        with h5py.File(episode["source"]) as f:
            expected_actions = np.concatenate(
                [f["data/q_des_left"][start:end], f["data/q_des_right"][start:end]],
                axis=1,
            )
        np.testing.assert_array_equal(rows[:, 14:], expected_actions)
    expected = {
        "intervention": [(20, 65), (80, 111)],
        "autonomous": [(0, 20), (65, 80)],
    }
    for component, ranges in expected.items():
        episodes = [
            e
            for e in manifest["episodes"]
            if e["component"] == component and e["split"] == "train"
        ]
        assert [tuple(e["frame_range"]) for e in episodes] == ranges
        assert sum(e["windows"] for e in episodes) == sum(
            end - start - 2 for start, end in ranges
        )
    counts = {"previous": 0, "intervention": 0, "autonomous": 0}
    for index in range(10000):
        sample = sample_window(manifest, index, seed=42)
        episode = sample["episode"]
        assert episode["split"] == "train"
        assert 0 <= sample["offset"] <= episode["frames"] - 3
        counts[episode["component"]] += 1
    for key, probability in zip(counts, (0.8, 0.1, 0.1), strict=True):
        assert abs(counts[key] / 10000 - probability) < 0.015
    assert sample_window(manifest, 971, seed=42) == sample_window(
        manifest, 971, seed=42
    )
    assert {
        e["group"] for e in manifest["episodes"] if e["split"] == "train"
    }.isdisjoint({e["group"] for e in manifest["episodes"] if e["split"] == "val"})
    with pytest.raises(FileExistsError):
        prepare_round(records, output, stats, horizon=3)


@pytest.mark.parametrize("violation", ["group", "duplicate"])
def test_leakage_is_rejected_before_output(round_inputs, tmp_path, violation):
    from dagger_offline.round_data import prepare_round

    records, stats = round_inputs
    if violation == "group":
        records[1]["group"] = records[0]["group"]
    else:
        shutil.copyfile(records[0]["path"], records[1]["path"])
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="group|duplicate"):
        prepare_round(records, output, stats, horizon=3)
    assert not output.exists()


def test_nonfinite_and_stale_samples_break_windows(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round

    records, stats = round_inputs
    # Current training source: invalidate a state sample in an intervention.
    path = records[2]["path"]
    with h5py.File(path, "r+") as f:
        f["data/q_right"][40, 0] = np.nan
        # Stream stops updating after frame 89: after 100ms, samples must be excluded.
        values = f["timestamps/q_des_left"][:]
        del f["timestamps/q_des_left"]
        f["timestamps"].create_dataset("q_des_left", data=values[:90])
        values = f["data/q_des_left"][:90]
        del f["data/q_des_left"]
        f["data"].create_dataset("q_des_left", data=values)
    manifest = prepare_round(
        records, tmp_path / "filtered", stats, horizon=3, image_size=(16, 16)
    )
    ranges = [
        e["frame_range"]
        for e in manifest["episodes"]
        if e["component"] == "intervention" and e["split"] == "train"
    ]
    assert ranges == [[20, 40], [41, 65], [80, 93]]


def test_current_only_preparation_for_existing_lerobot_bc(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round, sample_window

    records, stats = round_inputs
    manifest = prepare_round(
        records[2:],
        tmp_path / "current",
        stats,
        horizon=3,
        image_size=(16, 16),
        current_only=True,
    )
    assert manifest["current_only"]
    assert {e["component"] for e in manifest["episodes"]} == {
        "intervention",
        "autonomous",
    }
    with pytest.raises(ValueError, match="previous"):
        sample_window(manifest, 0)


def test_missing_component_refuses_to_renormalize_weights(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round

    records, stats = round_inputs
    # All autonomous runs have <30 frames. Do not silently convert 80:10:10 to 8:1.
    output = tmp_path / "short"
    with pytest.raises(ValueError, match="autonomous"):
        prepare_round(records, output, stats, horizon=30, image_size=(16, 16))
    assert not output.exists()


def test_hold_and_camera_gap_cannot_be_bridged(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round

    records, stats = round_inputs
    with h5py.File(records[2]["path"], "r+") as f:
        times = f["timestamps/top"][:]
        del f["dagger_events"]
        events = f.create_group("dagger_events")
        events.create_dataset("timestamp_ns", data=times[[0, 20, 35, 40, 65, 80]])
        events.create_dataset(
            "controller_state",
            data=[
                "policy",
                "intervention",
                "hold",
                "intervention",
                "policy",
                "intervention",
            ],
            dtype=h5py.string_dtype(),
        )
        # All streams resume after a 1-second outage; no action chunk may span it.
        for stream in f["timestamps"].values():
            stream[90:] += 1_000_000_000
    manifest = prepare_round(
        records, tmp_path / "gapped", stats, horizon=3, image_size=(16, 16)
    )
    ranges = [
        e["frame_range"]
        for e in manifest["episodes"]
        if e["component"] == "intervention" and e["split"] == "train"
    ]
    assert ranges == [[20, 35], [40, 65], [80, 90], [91, 120]]


def test_wrong_camera_rate_is_not_relabelled_as_30hz(round_inputs, tmp_path):
    from dagger_offline.round_data import prepare_round

    records, stats = round_inputs
    with h5py.File(records[2]["path"], "r+") as f:
        for stream in f["timestamps"].values():
            stream[:] = 1_000_000_000 + np.arange(120) * 16_666_667
    with pytest.raises(ValueError, match="30 Hz"):
        prepare_round(records, tmp_path / "wrong_rate", stats, horizon=3)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_pre_transition_actions_are_not_human_labels(recording, arm):
    from dagger_offline.round_data import read_source

    path, times, _, _ = recording
    with h5py.File(path, "r+") as f:
        f[f"timestamps/q_des_{arm}"][20] = times[20] - 1_000_000
        _, _, _, labels, valid, _ = read_source(f, "current", 100)
    assert labels[20] == "INTERVENTION"
    assert not valid[20]
    assert valid[21]


@pytest.mark.parametrize("boundary", ["hold", "discard"])
def test_subframe_boundaries_break_windows(recording, boundary):
    from dagger_offline.round_data import read_source

    path, times, _, _ = recording
    with h5py.File(path, "r+") as f:
        start, end = times[40] + 5_000_000, times[40] + 15_000_000
        if boundary == "hold":
            del f["dagger_events"]
            events = f.create_group("dagger_events")
            events.create_dataset(
                "timestamp_ns",
                data=[times[0], times[20], start, end, times[65], times[80]],
            )
            events.create_dataset(
                "controller_state",
                data=[
                    "policy",
                    "intervention",
                    "hold",
                    "intervention",
                    "policy",
                    "intervention",
                ],
                dtype=h5py.string_dtype(),
            )
        else:
            f["discard_segments/start_ns"][:] = [start]
            f["discard_segments/end_ns"][:] = [end]
        _, _, _, labels, valid, _ = read_source(f, "current", 100)
    assert labels[40] == labels[41] == "INTERVENTION"
    assert valid[40] and valid[42]
    assert not valid[41]  # Conservative boundary tick removal prevents bridging.


def test_sampler_is_uniform_over_windows_not_episode_count():
    from dagger_offline.round_data import COMPONENTS, sample_window

    episodes = [
        {"split": "train", "component": c, "windows": n, "id": f"{c}-{n}"}
        for c in COMPONENTS
        for n in (1, 9)
    ]
    manifest = {"complete": True, "episodes": episodes}
    short = sum(
        sample_window(manifest, i)["episode"]["windows"] == 1 for i in range(10000)
    )
    assert 800 < short < 1200


def test_cli_default_30_step_chunks(round_inputs, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    records, stats = round_inputs
    for record in records:
        with h5py.File(record["path"], "r+") as f:
            times = f["timestamps/top"][:]
            f["dagger_events/timestamp_ns"][:] = times[[0, 35, 75, 110]]
        record["path"] = Path(record["path"]).name
    inputs = tmp_path / "records.json"
    inputs.write_text(json.dumps(records))
    output = tmp_path / "cli_round"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dagger_offline.round_data",
            "--records",
            str(inputs),
            "--output",
            str(output),
            "--norm-stats",
            str(stats),
        ],
        check=True,
    )
    manifest = json.loads((output / "round.json").read_text())
    assert manifest["complete"] and manifest["horizon"] == 30
    assert {
        name: sum(
            e["windows"]
            for e in manifest["episodes"]
            if e["component"] == name and e["split"] == "train"
        )
        for name in ("previous", "intervention", "autonomous")
    } == {"previous": 23, "intervention": 11, "autonomous": 12}
