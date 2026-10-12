import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from abc_minimal.checkpointing import restore_training_state, save_checkpoint
from abc_minimal.preprocess import load_norm_stats
from launch import initialize, load_safe, validate_run

pytest_plugins = ["test_native_data"]


@pytest.fixture
def run_inputs(round_data, tmp_path):
    root, _ = round_data
    parent = tmp_path / "parent.pt"
    stats = load_norm_stats(root / "norm_stats.json")
    torch.save({"model": {"weight": torch.ones(1)}, "norm_stats": stats}, parent)
    clip = tmp_path / "clip"
    clip.mkdir()
    # Asset bytes exercise identity only, not CLIP model/tokenizer loading.
    (clip / "ViT-B-32.pt").write_bytes(b"synthetic identity fixture")
    (clip / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"synthetic tokens fixture")
    output = tmp_path / "run"
    settings = {
        "steps": 10,
        "batch_size": 2,
        "world_size": 1,
        "save_interval": 2,
        "seed": 17,
    }
    initialize(root, parent, clip, output, settings)
    return root, parent, clip, output


def test_portable_contract_and_no_overwrite(run_inputs, tmp_path):
    root, parent, clip, output = run_inputs
    original = validate_run(output / "run.json", root, parent, clip)
    moved = tmp_path / "other-server"
    moved.mkdir()
    for source, name in ((root, "data"), (clip, "assets"), (output, "experiment")):
        shutil.copytree(source, moved / name)
    shutil.copyfile(parent, moved / "weights.pt")
    migrated = validate_run(
        moved / "experiment/run.json",
        moved / "data",
        moved / "weights.pt",
        moved / "assets",
    )
    assert migrated.batch_size == original.batch_size
    assert migrated.pretrained_ckpt_name == str(moved / "weights.pt")
    with pytest.raises(FileExistsError):
        initialize(
            root,
            parent,
            clip,
            output,
            json.loads((output / "run.json").read_text())["settings"],
        )


@pytest.mark.parametrize("change", ["data", "parent", "asset", "world"])
def test_drift_is_rejected(run_inputs, monkeypatch, change):
    root, parent, clip, output = run_inputs
    if change == "data":
        with next(root.glob("previous/train/*/states_actions.bin")).open("ab") as f:
            f.write(b"changed")
    elif change == "parent":
        ckpt = load_safe(parent)
        ckpt["model"]["weight"] += 1
        temporary = parent.with_suffix(".tmp")
        torch.save(ckpt, temporary)
        temporary.replace(parent)
    elif change == "asset":
        (clip / "ViT-B-32.pt").write_bytes(b"changed")
    else:
        monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="contract changed|WORLD_SIZE"):
        validate_run(output / "run.json", root, parent, clip)


def test_native_optimizer_save_restore_and_resume_guards(run_inputs):
    root, parent, clip, output = run_inputs
    config = validate_run(output / "run.json", root, parent, clip)
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 / (step + 1)
    )
    x = torch.tensor([[0.3, -0.8], [1.2, 0.4]])

    def step(module, opt, sched):
        opt.zero_grad()
        (module(x).square().sum()).backward()
        opt.step()
        sched.step()

    step(model, optimizer, scheduler)
    checkpoint = output / "1.pt"
    save_checkpoint(
        checkpoint,
        module=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        norm_stats=load_norm_stats(root / "norm_stats.json"),
        batch_size=2,
        data_world=1,
        train_config=dataclasses.asdict(config),
    )
    resumed_config = validate_run(output / "run.json", root, parent, clip, checkpoint)
    assert not resumed_config.load_pretrained
    assert resumed_config.resume_from == str(checkpoint)
    saved = load_safe(checkpoint)
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(saved["model"])
    opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1 / (n + 1))
    restore_training_state(saved, optimizer=opt, scheduler=sched, resume_step=1, rank=0)
    step(model, optimizer, scheduler)
    step(restored, opt, sched)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="explicit resume"):
        validate_run(output / "run.json", root, parent, clip)
    identity = saved["train_config"]["dagger_contract_id"]
    saved["train_config"]["dagger_contract_id"] = "other-experiment"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(checkpoint)
    with pytest.raises(ValueError, match="dagger_contract_id"):
        validate_run(output / "run.json", root, parent, clip, checkpoint)
    saved["train_config"]["dagger_contract_id"] = identity
    saved["train_config"]["optim"]["learning_rate"] *= 2
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(checkpoint)
    with pytest.raises(ValueError, match="resume config differs"):
        validate_run(output / "run.json", root, parent, clip, checkpoint)
    del saved["optimizer"]
    torch.save(saved, temporary)
    temporary.replace(checkpoint)
    with pytest.raises(ValueError, match="full native"):
        validate_run(output / "run.json", root, parent, clip, checkpoint)


def test_check_cli_uses_saved_settings_and_rejects_overrides(run_inputs):
    root, parent, clip, output = run_inputs
    command = [
        sys.executable,
        str(Path(__file__).with_name("launch.py")),
        "check",
        "--round",
        str(root),
        "--parent",
        str(parent),
        "--clip-cache",
        str(clip),
        "--run",
        str(output / "run.json"),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    assert json.loads(result.stdout)["batch_size"] == 2
    result = subprocess.run(
        command + ["--batch-size", "4"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "overrides require a new run" in result.stderr
