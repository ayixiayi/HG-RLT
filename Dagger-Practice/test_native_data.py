"""Official export and native ABC/TorchCodec data path, without substitutions."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from abc_minimal import dataloader as dl
from abc_minimal.config import DiTConfig, MixtureComponent, TrainConfig
from abc_minimal.preprocess import load_norm_stats
from prepare import prepare

pytest_plugins = ["test_recorder_export"]


@pytest.fixture
def inputs(recorded, tmp_path):
    node, _, states, actions, colors = recorded
    records = []
    for role in ("previous", "current"):
        for split in ("train", "val"):
            path = tmp_path / f"{role}-[{split}].h5"
            shutil.copyfile(node.current_file_name, path)
            with h5py.File(path, "r+") as f:
                f.attrs["recording_id"] = path.stem
                if split == "val":
                    f["data/q_des_right"][:] += 10
            records.append(
                {"path": str(path), "group": path.stem, "role": role, "split": split}
            )
    stats = tmp_path / "parent.json"
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {"mean": [0.05] * 14, "std": [2.0] * 14}
                    for key in ("state", "actions")
                }
            }
        )
    )
    return records, stats, states, actions, colors


@pytest.fixture
def round_data(inputs, tmp_path):
    records, stats, *_ = inputs
    output = tmp_path / "round"
    manifest = prepare(records, output, stats)
    return output, manifest


def dataset(root, split="train", component="intervention", cls=dl.EpisodeDataset):
    return cls(
        root / component / split,
        load_norm_stats(root / "norm_stats.json"),
        train=False,
        default_task_name="wrong_fallback",
        mask_state_ratio=0,
        model_config=DiTConfig(),
        norm_preset=None,
    )


def test_export_to_native_windows_and_batch(round_data, inputs):
    root, manifest = round_data
    _, stats, states, actions, colors = inputs
    assert (root / "norm_stats.json").read_bytes() == stats.read_bytes()
    assert [row["weight"] for row in manifest["mixture"]] == [0.8, 0.1, 0.1]
    human = dataset(root)
    autonomous = dataset(root, component="autonomous")
    assert len(human) == 31  # 60-frame intervention, 30-frame windows
    assert len(autonomous) == 32  # 40 + 50 frames, independently windowed
    for offset in (0, 30):
        sample = human[offset]
        np.testing.assert_allclose(
            sample["state"], (states[40 + offset] - 0.05) / 2.000001, rtol=1e-5
        )
        np.testing.assert_allclose(
            sample["actions"],
            (actions[40 + offset : 70 + offset] - 0.05) / 2.000001,
            rtol=1e-5,
        )
        assert sample["prompt"] == "synthetic offline only"
        for name, color in zip(("top", "left", "right"), colors, strict=True):
            assert sample["images"][name].shape == (3, 224, 224)
            # Official export is 168x224; preprocessing centers it vertically
            # with 28 black rows above and below rather than stretching.
            image = sample["images"][name]
            assert torch.count_nonzero(image[:, :28]) == 0
            assert torch.count_nonzero(image[:, 196:]) == 0
            np.testing.assert_allclose(
                image[:, 28:196].mean(dim=(1, 2)),
                np.array(color) / 255,
                atol=0.05,
            )
    # The first window after the autonomous segment boundary begins at frame100,
    # never at frame11 spanning into human control.
    np.testing.assert_allclose(
        autonomous[11]["actions"], (actions[100:130] - 0.05) / 2.000001, rtol=1e-5
    )
    held_out = dataset(root, split="val")[0]["actions"]
    np.testing.assert_allclose(
        held_out[:, 7:] - human[0]["actions"][:, 7:], 10 / 2.000001, rtol=1e-5
    )
    for item in manifest["mixture"]:
        assert not (root / item["val_dir"]).is_symlink()
    config = TrainConfig(
        cache_root=str(root), batch_size=2, train_steps=2, num_workers=0
    )
    config.flow.mask_state_ratio = 0
    components = [
        MixtureComponent(
            train_dir=m["train_dir"],
            val_dir=m["val_dir"],
            weight=m["weight"],
            task_name="wrong_fallback",
        )
        for m in manifest["mixture"]
    ]
    config.mixture = components
    loader, _ = dl.build_train_loader(
        config,
        config.resolve_mixture(),
        load_norm_stats(root / "norm_stats.json"),
        dl.DataParallelScope(0, 1, "shared", True),
        0,
        config.model,
    )
    batch = next(iter(loader))
    assert batch["actions"].shape == (2, 30, 14)
    assert batch["state"].shape == (2, 14)
    assert batch["prompt"] == ["synthetic offline only"] * 2
    assert all(torch.isfinite(v).all() for v in batch["images"].values())


def test_native_mixture_probabilities_and_resume_indices(round_data):
    root, _ = round_data

    # Attach provenance to real decoded samples; do not replace the data path.
    class IndexedEpisode(dl.EpisodeDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            assert sample["actions"].shape == (30, 14)
            assert all(
                torch.isfinite(image).all() for image in sample["images"].values()
            )
            return self.label, index

    components = []
    for name in ("previous", "intervention", "autonomous"):
        source = dataset(root, component=name, cls=IndexedEpisode)
        source.label = name
        components.append(source)
    mixture = dl.MixtureDataset(components, [0.8, 0.1, 0.1], 2000, seed=17)
    draws = [mixture[i] for i in range(2000)]
    for name, expected in zip(
        ("previous", "intervention", "autonomous"), (0.8, 0.1, 0.1), strict=True
    ):
        assert abs(sum(label == name for label, _ in draws) / 2000 - expected) < 0.03
    # Different-length policy episodes should receive 11:21 valid windows,
    # not a 50:50 episode-level probability.
    policy = [index for name, index in draws if name == "autonomous"]
    assert abs(sum(index < 11 for index in policy) / len(policy) - 11 / 32) < 0.10
    full = list(dl.GlobalStepSampler(0, 8, 3, 0, 1))
    resumed = list(dl.GlobalStepSampler(3, 8, 3, 0, 1))
    assert resumed == full[9:]
    assert [mixture[i] for i in resumed] == [mixture[i] for i in full][9:]


@pytest.mark.parametrize("violation", ["group", "duplicate", "missing"])
def test_split_and_empty_pool_fail_without_publishing(inputs, tmp_path, violation):
    records, stats, *_ = inputs
    if violation == "group":
        records[1]["group"] = records[0]["group"]
    elif violation == "duplicate":
        shutil.copyfile(records[0]["path"], records[1]["path"])
    else:
        records = records[:2]
    output = tmp_path / "bad-round"
    with pytest.raises(ValueError, match="group|duplicate|empty component"):
        prepare(records, output, stats)
    assert not output.exists()


def test_prepare_cli_relative_paths_and_existing_output(inputs, tmp_path):
    records, stats, *_ = inputs
    for row in records:
        row["path"] = Path(row["path"]).name
    record_file = tmp_path / "records.json"
    record_file.write_text(json.dumps(records))
    output = tmp_path / "cli-round"
    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare.py")),
        "--records",
        str(record_file),
        "--output",
        str(output),
        "--norm-stats",
        str(stats),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    before = (output / "round.json").read_bytes()
    manifest = json.loads(before)
    assert len(manifest["exports"]) == 8
    assert len(manifest["episodes"]) == 12
    assert all(not row["summary"]["skipped_files"] for row in manifest["exports"])
    rerun = subprocess.run(command, check=False, capture_output=True, timeout=60)
    assert rerun.returncode != 0
    assert b"FileExistsError" in rerun.stderr
    assert (output / "round.json").read_bytes() == before
