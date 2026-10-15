"""CPU contracts for the real LeRobot/OpenPI training adapter."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")


@pytest.fixture
def mixture(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    entries = []
    for component in ("previous", "intervention", "autonomous"):
        for split in ("train", "val"):
            root = tmp_path / component / split
            ds = LeRobotDataset.create(
                repo_id=f"local/{component}_{split}",
                root=root,
                fps=30,
                robot_type="yam_dual",
                use_videos=False,
                features={
                    **{
                        key: {"dtype": "float32", "shape": (14,), "names": None}
                        for key in ("observation.state", "action")
                    },
                    **{
                        f"observation.images.{cam}": {
                            "dtype": "image",
                            "shape": (16, 16, 3),
                            "names": ["height", "width", "channel"],
                        }
                        for cam in ("main", "left", "right")
                    },
                },
            )
            episodes = []
            for ep, length in enumerate((3, 11)):
                for frame in range(length):
                    ds.add_frame(
                        {
                            "observation.state": np.full(14, ep, np.float32),
                            "action": np.arange(14, dtype=np.float32)
                            + frame
                            + ep * 100,
                            "task": "choose_the_heavier_one",
                            **{
                                f"observation.images.{cam}": np.full(
                                    (16, 16, 3), color, np.uint8
                                )
                                for cam, color in (
                                    ("main", 25),
                                    ("left", 85),
                                    ("right", 190),
                                )
                            },
                        }
                    )
                ds.save_episode()
                episodes.append(
                    {
                        "episode_index": ep,
                        "frames": length,
                        "group": f"{component}-{split}-{ep}",
                    }
                )
            entries.append(
                {
                    "path": str(root.relative_to(tmp_path)),
                    "repo_id": ds.repo_id,
                    "component": component,
                    "split": split,
                    "episodes": episodes,
                }
            )
    norm = {
        k: {
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-2.0] * 14,
            "q99": [2.0] * 14,
        }
        for k in ("state", "actions")
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": norm}))
    import hashlib

    manifest = {
        "complete": True,
        "horizon": 3,
        "datasets": entries,
        "norm_sha256": hashlib.sha256(
            (tmp_path / "norm_stats.json").read_bytes()
        ).hexdigest(),
    }
    path = tmp_path / "lerobot_round.json"
    path.write_text(json.dumps(manifest))
    return path


def test_real_windows_no_padding_and_reproducible_resume(mixture):
    from dagger_offline.training import MixtureSampler, WindowDataset

    ds = WindowDataset(mixture, "train")
    assert len(ds) == 30  # each component: (3-3+1)+(11-3+1)=10
    np.testing.assert_array_equal(ds[0]["action"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(ds[9]["action"][:, 0], [108, 109, 110])
    assert ds[9]["prompt"] == "choose_the_heavier_one"
    full = list(MixtureSampler(ds, seed=71, start=0, stop=10000))
    resumed = list(MixtureSampler(ds, seed=71, start=450, stop=10000))
    assert full[450:] == resumed
    counts = np.bincount(np.array(full) // 10, minlength=3) / len(full)
    np.testing.assert_allclose(counts, [0.8, 0.1, 0.1], atol=0.015)
    # One short window versus nine long windows per component, not 1:1 episodes.
    assert 0.08 < np.mean(np.array(full) % 10 == 0) < 0.12
    assert not any(e["split"] == "val" for e in ds.entries)


def test_bad_group_and_normalization_fail_closed(mixture):
    from dagger_offline.training import WindowDataset

    payload = json.loads(mixture.read_text())
    payload["datasets"][1]["episodes"][0]["group"] = payload["datasets"][0]["episodes"][
        0
    ]["group"]
    mixture.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="group"):
        WindowDataset(mixture, "train")
    payload["datasets"][1]["episodes"][0]["group"] = "val-only"
    mixture.write_text(json.dumps(payload))
    (mixture.parent / "norm_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="normalization"):
        WindowDataset(mixture, "train")


def test_upstream_torch_loader_uses_mixture(mixture):
    pytest.importorskip("openpi")
    from openpi.training.data_loader import TorchDataLoader

    from dagger_offline.training import MixtureSampler, WindowDataset

    ds = WindowDataset(mixture, "train")

    # Native collator cannot stack prompt strings; test the numerical view here.
    class Numeric:
        def __len__(self):
            return len(ds)

        def __getitem__(self, index):
            return {"actions": ds[index]["action"].numpy()}

    loader = TorchDataLoader(
        Numeric(),
        local_batch_size=4,
        sampler=MixtureSampler(ds, seed=9, start=0, stop=8),
        num_batches=2,
        framework="pytorch",
    )
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["actions"].shape == (4, 3, 14)
    assert np.all(np.diff(batches[0]["actions"].numpy()[:, :, 0], axis=1) == 1)


def test_run_contract_distinguishes_resume_and_new_round(tmp_path):
    from dagger_offline.training import record_run

    contract = {
        "manifest_sha256": "abc",
        "parent_params": "/parent/params",
        "batch_size": 8,
    }
    record_run(tmp_path / "run.json", contract, resume=False)
    with pytest.raises(FileExistsError):
        record_run(tmp_path / "run.json", contract, resume=False)
    record_run(tmp_path / "run.json", contract, resume=True)
    with pytest.raises(ValueError, match="contract"):
        record_run(tmp_path / "run.json", {**contract, "batch_size": 16}, resume=True)
    with pytest.raises(FileNotFoundError):
        record_run(tmp_path / "absent.json", contract, resume=True)


def test_real_openpi_transforms_and_batch(mixture, tmp_path):
    pytest.importorskip("openpi")
    from dagger_offline.training import (
        MixtureSampler,
        WindowDataset,
        build_config,
        create_loader,
    )

    config = build_config(
        mixture,
        tmp_path,
        tmp_path / "parent/params",
        name="dagger_r1",
        exp_name="cpu",
        steps=200,
        batch_size=2,
    )
    assert config.save_interval == 500 and not config.overwrite
    assert config.weight_loader.params_path == str(tmp_path / "parent/params")
    loader = create_loader(config, mixture, start_step=0, num_batches=1)
    observation, actions = next(iter(loader))
    assert actions.shape == (2, 3, 32)
    assert observation.state.shape == (2, 32)
    assert observation.tokenized_prompt.shape == (2, 200)
    raw = WindowDataset(mixture, "train")
    index = next(iter(MixtureSampler(raw, seed=config.seed, start=0, stop=2)))
    row = raw[index]
    expected = row["action"].numpy().copy()
    mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    expected[:, mask] -= row["observation.state"].numpy()[mask]
    np.testing.assert_allclose(np.asarray(actions[0, :, :14]), expected / 2, rtol=1e-5)
    assert all(np.asarray(v).all() for v in observation.image_masks.values())


def test_lr_decay_follows_steps_and_periodic_checkpoints_are_optional(
    mixture, tmp_path
):
    pytest.importorskip("openpi")
    from dagger_offline.training import build_config

    def config(**kwargs):
        return build_config(
            mixture,
            tmp_path,
            tmp_path / "parent/params",
            name="dagger_r1",
            exp_name="cpu",
            batch_size=2,
            **kwargs,
        )

    long = config(steps=20000)
    assert long.lr_schedule.decay_steps == 20000 and long.keep_period is None
    assert config().lr_schedule.decay_steps == 2000  # the earlier default
    tuned = config(steps=20000, decay_steps=15000, keep_period=5000)
    assert tuned.lr_schedule.decay_steps == 15000 and tuned.keep_period == 5000
    assert config(steps=50).lr_schedule.decay_steps == 2000  # smoke runs
    with pytest.raises(ValueError, match="warmup"):
        config(steps=2000, decay_steps=50)
    with pytest.raises(ValueError, match="keep period"):
        config(steps=2000, keep_period=0)


def test_resume_step_uses_committed_orbax_checkpoint(tmp_path):
    ocp = pytest.importorskip("orbax.checkpoint")
    from dagger_offline.training import resume_step

    with pytest.raises(ValueError, match="checkpoint"):
        resume_step(tmp_path / "missing")
    path = tmp_path / "checkpoints"
    with ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions()) as manager:
        manager.save(5, args=ocp.args.StandardSave({"step": np.array(6)}))
        manager.wait_until_finished()
    assert resume_step(path) == 6


def test_attach_existing_bc_preserves_source_split(mixture):
    from dagger_offline.training import WindowDataset, attach_previous

    payload = json.loads(mixture.read_text())
    for split, ids in (("train", [0, 2]), ("val", [1, 3])):
        path = mixture.parent / "previous" / split / "meta/split.json"
        path.write_text(
            json.dumps(
                {
                    "complete": True,
                    "source_root_relative": "../../../old_source",
                    "source_episode_indices": ids,
                }
            )
        )
    payload["datasets"] = [
        e for e in payload["datasets"] if e["component"] != "previous"
    ]
    payload["current_only"] = True
    mixture.write_text(json.dumps(payload))
    output = mixture.parent / "combined.json"
    attach_previous(
        mixture,
        output,
        mixture.parent / "previous/train",
        mixture.parent / "previous/val",
    )
    assert len(WindowDataset(output, "train")) == 30
    bad = mixture.parent / "previous/val/meta/split.json"
    content = json.loads(bad.read_text())
    content["source_episode_indices"] = [0, 3]
    bad.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="group"):
        attach_previous(
            mixture,
            mixture.parent / "bad.json",
            mixture.parent / "previous/train",
            mixture.parent / "previous/val",
        )
    assert not (mixture.parent / "bad.json").exists()


def test_readiness_requires_matching_parent_without_loading_weights(mixture, tmp_path):
    from dagger_offline.training import build_config, check_training

    parent = tmp_path / "parent"
    config = build_config(
        mixture, tmp_path, parent / "params", name="hg", exp_name="check", batch_size=2
    )
    with pytest.raises(FileNotFoundError):
        check_training(config, mixture)
    (parent / "params").mkdir(parents=True)
    with pytest.raises(ValueError, match="normalization"):
        check_training(config, mixture)
    stats = parent / "assets/baseline/norm_stats.json"
    stats.parent.mkdir(parents=True)
    payload = json.loads((mixture.parent / "norm_stats.json").read_text())
    stats.write_text(json.dumps(payload, indent=4))
    assert check_training(config, mixture) == {
        split: {"previous": 10, "intervention": 10, "autonomous": 10}
        for split in ("train", "val")
    }
    payload["norm_stats"]["actions"]["q99"][7] += 0.5
    stats.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="normalization"):
        check_training(config, mixture)


def test_training_entrypoint_restores_upstream_hook(mixture, tmp_path, monkeypatch):
    pytest.importorskip("openpi")
    from openpi.training import data_loader
    from scripts import train

    from dagger_offline.training import build_config, run_training

    parent = tmp_path / "parent"
    (parent / "params").mkdir(parents=True)
    (parent / "assets/yam").mkdir(parents=True)
    parent_stats = parent / "assets/yam/norm_stats.json"
    parent_stats.write_text(
        json.dumps(
            json.loads((mixture.parent / "norm_stats.json").read_text()), indent=2
        )
    )
    config = build_config(
        mixture,
        tmp_path,
        parent / "params",
        name="dagger",
        exp_name="test",
        batch_size=2,
        steps=200,
    )
    original = data_loader.create_data_loader
    called = []

    def fake_train(cfg):
        called.append(cfg)
        assert data_loader.create_data_loader is not original
        loader = data_loader.create_data_loader(cfg, shuffle=True)
        assert next(iter(loader))[1].shape == (2, 3, 32)

    monkeypatch.setattr(train, "main", fake_train)
    run_training(config, mixture)
    assert called == [config]
    assert data_loader.create_data_loader is original
    with pytest.raises(FileExistsError):
        run_training(config, mixture)
    import dataclasses

    from dagger_offline import training

    monkeypatch.setattr(training, "resume_step", lambda _: 6)
    resumed = dataclasses.replace(config, resume=True)
    changed = dataclasses.replace(
        resumed, optimizer=dataclasses.replace(config.optimizer, b1=0.8)
    )
    with pytest.raises(ValueError, match="contract"):
        run_training(changed, mixture)
    assert called == [config]
    run_training(resumed, mixture)
    assert called == [config, resumed]
    wrong_stats = json.loads(parent_stats.read_text())
    wrong_stats["norm_stats"]["actions"]["q99"][7] = 3.0
    parent_stats.write_text(json.dumps(wrong_stats))
    with pytest.raises(ValueError, match="normalization"):
        run_training(resumed, mixture)


def test_next_round_aggregates_all_previous_components(mixture, tmp_path):
    import shutil

    from dagger_offline.training import WindowDataset, attach_previous_round

    old = tmp_path / "old_round"
    old.mkdir()
    payload = json.loads(mixture.read_text())
    for entry in payload["datasets"]:
        shutil.copytree(mixture.parent / entry["path"], old / entry["path"])
    (old / "lerobot_round.json").write_text(json.dumps(payload))
    (old / "norm_stats.json").write_bytes(
        (mixture.parent / "norm_stats.json").read_bytes()
    )
    current = json.loads(mixture.read_text())
    current["current_only"] = True
    current["datasets"] = [
        e for e in current["datasets"] if e["component"] != "previous"
    ]
    for entry in current["datasets"]:
        for ep in entry["episodes"]:
            ep["group"] = "new-" + ep["group"]
    mixture.write_text(json.dumps(current))
    output = mixture.parent / "round2.json"
    attach_previous_round(mixture, output, old / "lerobot_round.json")
    ds = WindowDataset(output, "train")
    assert ds.component_ranges == [(0, 30), (30, 40), (40, 50)]
    # Same HDF5 can be re-exported under different roots and group names.
    next(e for e in payload["datasets"] if e["split"] == "val")["episodes"][0][
        "source_sha256"
    ] = "same-recording"
    next(e for e in current["datasets"] if e["split"] == "train")["episodes"][0][
        "source_sha256"
    ] = "same-recording"
    (old / "lerobot_round.json").write_text(json.dumps(payload))
    mixture.write_text(json.dumps(current))
    with pytest.raises(ValueError, match="hash"):
        attach_previous_round(
            mixture, mixture.parent / "leaked.json", old / "lerobot_round.json"
        )
    assert not (mixture.parent / "leaked.json").exists()


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_rl_token_training_config_contract_and_real_size_shapes(
    mixture, tmp_path, monkeypatch, alpha
):
    pytest.importorskip("openpi")
    from flax import nnx
    from scripts import train

    from dagger_offline.rl_token import RLTCheckpointWeightLoader, RLTPi0Config
    from dagger_offline.training import build_config, rl_token_shapes, run_training

    parent = tmp_path / "parent"
    (parent / "params").mkdir(parents=True)
    (parent / "assets/yam").mkdir(parents=True)
    (parent / "assets/yam/norm_stats.json").write_text(
        (mixture.parent / "norm_stats.json").read_text()
    )
    config = build_config(
        mixture,
        tmp_path,
        parent / "params",
        name="dagger",
        exp_name=f"rlt-{alpha}",
        batch_size=2,
        steps=10,
        rl_token={"alpha": alpha},
    )
    assert isinstance(config.model, RLTPi0Config) and config.model.rl_alpha == alpha
    assert isinstance(config.weight_loader, RLTCheckpointWeightLoader)
    assert (config.freeze_filter == nnx.Nothing) == (alpha != 0)
    shapes = rl_token_shapes(config)
    assert shapes["token_width"] == 2048 and 1e6 < shapes["params"] < 1e8
    seen = []
    monkeypatch.setattr(train, "main", seen.append)
    run_training(config, mixture)
    contract = json.loads(Path(str(config.checkpoint_dir) + ".dagger.json").read_text())
    assert contract["inference"]["rl_token"]["alpha"] == alpha
    assert seen == [config]
