"""CPU file sealing and real localhost RPC; no weights or devices are loaded."""

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dagger_offline.deployment import (
    INSTRUCTION,
    SealedModel,
    SealedPolicyClient,
    load_deployment,
    main,
    normalization_id,
    seal,
)
from yam_openpi import OPENPI_REVISION


@pytest.fixture
def candidate(tmp_path):
    checkpoint = tmp_path / "checkpoints/dagger_r1/first/499"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/_METADATA").write_text("{}")
    (checkpoint / "params/weights").write_bytes(b"synthetic-not-model-weights")
    (checkpoint / "_CHECKPOINT_METADATA").write_text(
        json.dumps({"commit_timestamp_nsecs": 1})
    )
    stats = checkpoint / "assets/r1_lerobot/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "mean": [0.0] * 14,
                        "std": [1.0] * 14,
                        "q01": [-3.0] * 14,
                        "q99": [2.0] * 14,
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )
    sidecar = Path(str(checkpoint.parent) + ".dagger.json")
    sidecar.write_text(
        json.dumps(
            {
                "inference": {
                    "model": "pi05",
                    "action_dim": 32,
                    "action_horizon": 30,
                    "adapt_to_pi": False,
                    "delta_joint_actions": True,
                    "config_name": "dagger_r1",
                    "exp_name": "first",
                    "asset_id": "r1_lerobot",
                    "norm_id": normalization_id(stats),
                    "openpi_revision": OPENPI_REVISION,
                }
            }
        )
    )
    return checkpoint, sidecar, tmp_path / "deploy/r1.json"


def test_seal_cli_is_offline_portable_and_detects_payload_drift(
    candidate, tmp_path, monkeypatch
):
    import builtins
    import socket

    checkpoint, _, output = candidate
    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name.split(".")[0] in {"openpi", "jax", "torch", "gello", "client_server"}:
            raise AssertionError(f"CPU sealing imported runtime {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    monkeypatch.setattr(
        socket, "socket", lambda *a, **kw: pytest.fail("CPU seal opened a socket")
    )
    main(
        [
            "seal",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--model-round",
            "1",
        ]
    )
    main(["check", "--deployment", str(output)])
    identity = load_deployment(output)["policy_id"]
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "checkpoints", moved / "checkpoints")
    shutil.copytree(output.parent, moved / "deploy")
    assert load_deployment(moved / "deploy/r1.json")["policy_id"] == identity
    (moved / "checkpoints/dagger_r1/first/499/params/weights").write_bytes(
        b"different-model"
    )
    with pytest.raises(ValueError, match="payload changed"):
        load_deployment(moved / "deploy/r1.json")
    with pytest.raises(FileExistsError):
        seal(checkpoint, output, model_round=1)


@pytest.mark.parametrize(
    "damage", ["uncommitted", "stats", "contract", "legacy", "path", "port"]
)
def test_seal_rejects_unsupported_or_mismatched_checkpoint(candidate, damage):
    checkpoint, sidecar, output = candidate
    run = json.loads(sidecar.read_text())
    if damage == "uncommitted":
        (checkpoint / "_CHECKPOINT_METADATA").write_text(
            '{"commit_timestamp_nsecs": null}'
        )
    elif damage == "stats":
        stats = checkpoint / "assets/r1_lerobot/norm_stats.json"
        values = json.loads(stats.read_text())
        values["norm_stats"]["actions"]["q99"][7] = 3.0
        stats.write_text(json.dumps(values))
    elif damage == "contract":
        run["inference"]["delta_joint_actions"] = False
    elif damage == "legacy":
        run.pop("inference")
    elif damage == "path":
        run["inference"]["exp_name"] = "another-run"
    sidecar.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        seal(checkpoint, output, model_round=1, port=8767 if damage == "port" else 8769)
    assert not output.exists()


class SyntheticModel:
    resets = 0

    def reset(self):
        self.resets += 1

    def update_obs(self, observation):
        self.observation = observation

    def get_action(self):
        return (
            np.tile(self.observation["state"], (30, 1)) + np.arange(30)[:, None] * 0.001
        )


OBSERVATION = {"instruction": INSTRUCTION, "state": np.linspace(-0.2, 0.4, 14)}
POLICY_ID = "a" * 64


def test_real_rpc_checks_identity_and_refuses_legacy_paths():
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    async def run():
        model = SyntheticModel()
        server = PolicyServer(
            SealedModel(model, POLICY_ID), PolicyServerConfig(host="127.0.0.1", port=0)
        )
        await server.start()
        try:
            with pytest.raises(ValueError, match="identity"):
                await asyncio.to_thread(SealedPolicyClient, server.url, "b" * 64)
            assert model.resets == 0

            def use():
                client = SealedPolicyClient(server.url, POLICY_ID)
                try:
                    client.reset()
                    result = client.infer_chunk(OBSERVATION)
                    np.testing.assert_allclose(result[0], np.linspace(-0.2, 0.4, 14))
                    np.testing.assert_allclose(
                        result[-1], np.linspace(-0.2, 0.4, 14) + 0.029
                    )
                finally:
                    client.close()
                with (
                    WsModelClient(
                        url=server.url,
                        evaluation_id="legacy",
                        trial_id="legacy",
                        operation_timeout_s=2,
                    ) as legacy,
                    pytest.raises(Exception, match="model must implement"),
                ):
                    legacy.call("get_action", obs=OBSERVATION)

            await asyncio.to_thread(use)
        finally:
            await server.stop()
        legacy_server = PolicyServer(
            SyntheticModel(), PolicyServerConfig(host="127.0.0.1", port=0)
        )
        await legacy_server.start()
        try:
            with pytest.raises(Exception, match="no model method"):
                await asyncio.to_thread(
                    SealedPolicyClient, legacy_server.url, POLICY_ID
                )
        finally:
            await legacy_server.stop()

    asyncio.run(run())


@pytest.mark.parametrize(
    "damage", ["policy", "session", "missing", "shape", "nonfinite", "reset"]
)
def test_reply_identity_or_actions_failure_is_terminal(damage):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    class Corrupt(SealedModel):
        def infer_sealed(self, observation):
            result = super().infer_sealed(observation)
            if damage == "policy":
                result["policy_id"] = "b" * 64
            elif damage == "session":
                result["serving_session_id"] = "different-session"
            elif damage == "missing":
                result.pop("serving_session_id")
            elif damage == "shape":
                result["actions"] = result["actions"][:15]
            elif damage == "nonfinite":
                result["actions"][4, 8] = np.nan
            return result

        def reset(self):
            result = super().reset()
            if damage == "reset":
                result["policy_id"] = "b" * 64
            return result

    async def run():
        server = PolicyServer(
            Corrupt(SyntheticModel(), POLICY_ID),
            PolicyServerConfig(host="127.0.0.1", port=0),
        )
        await server.start()
        try:

            def use():
                client = SealedPolicyClient(server.url, POLICY_ID)
                try:
                    with pytest.raises(ValueError):
                        if damage == "reset":
                            client.reset()
                        else:
                            client.infer_chunk(OBSERVATION)
                    with pytest.raises(RuntimeError, match="closed"):
                        client.infer_chunk(OBSERVATION)
                finally:
                    client.close()

            await asyncio.to_thread(use)
        finally:
            await server.stop()

    asyncio.run(run())


def test_server_restart_with_same_weights_requires_a_new_client():
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from client_server.ws.protocol.exceptions import ServerRestartedError

    async def run():
        server = PolicyServer(
            SealedModel(SyntheticModel(), POLICY_ID),
            PolicyServerConfig(host="127.0.0.1", port=0),
        )
        await server.start()
        url = server.url
        client = await asyncio.to_thread(SealedPolicyClient, url, POLICY_ID)
        await server.stop()
        server = PolicyServer(
            SealedModel(SyntheticModel(), POLICY_ID),
            PolicyServerConfig(host="127.0.0.1", port=int(url.rsplit(":", 1)[1])),
        )
        await server.start()
        try:
            with pytest.raises(ServerRestartedError):
                await asyncio.to_thread(client.infer_chunk, OBSERVATION)
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.to_thread(client.reset)
        finally:
            await asyncio.to_thread(client.close)
            await server.stop()

    asyncio.run(run())


def test_runner_selects_sealed_round_without_changing_baseline_or_hardware(
    candidate, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    import yaml

    from dagger_offline.test_live_runner import runner

    checkpoint, _, output = candidate
    sealed = seal(checkpoint, output, model_round=1)
    root = tmp_path / "rhos"
    config_path = root / "configs/deployments/pi05_yam_heavier.yaml"
    config_path.parent.mkdir(parents=True)
    baseline = {
        "train_config": "pi05_yam_heavier",
        "repo_id": "local/balance_0922_train",
        "port": 8767,
        "host": "127.0.0.1",
        "action_space": "joints",
        "instruction": INSTRUCTION,
        "profile": "profile.yaml",
        "checkpoint": "checkpoints/pi05_yam_heavier/pi05_yam_20k_20260924_023553/39999",
    }
    config_path.write_text(yaml.safe_dump(baseline))
    reference = config_path.with_name("pi05_yam_heavier_start.json")
    reference.write_text(json.dumps({"state": [0.25] * 14}))
    original_reference = reference.read_bytes()
    profile_path = root / "profile.yaml"
    profile_path.write_text("synthetic")
    (root / baseline["checkpoint"] / "params").mkdir(parents=True)
    old_stats = (
        root
        / baseline["checkpoint"]
        / "assets"
        / baseline["repo_id"]
        / "norm_stats.json"
    )
    old_stats.parent.mkdir(parents=True)
    old_stats.write_bytes(
        (checkpoint / "assets/r1_lerobot/norm_stats.json").read_bytes()
    )
    profile = SimpleNamespace(
        path=profile_path,
        network=SimpleNamespace(host="127.0.0.1"),
        cameras=SimpleNamespace(required=True, config_path="unused"),
    )

    def forbidden(*args, transport_timeout_s=None, operation_timeout_s=None, **kwargs):
        pytest.fail("Offline preflight constructed a runtime client")

    modules = {
        "pyrealsense2": {},
        "client_server.ws.model_client": {"WsModelClient": forbidden},
        "gello.cameras.realsense_rig": {
            "load_camera_rig_config": lambda _: SimpleNamespace(
                cameras={k: None for k in ("main", "left", "right")}
            )
        },
        "gello.dynamixel.driver": {
            "DynamixelDriver": SimpleNamespace(get_joint_sample=forbidden)
        },
        "gello.profile": {"load_profile": lambda _: profile},
        "gello.safety.rpc": {
            "SafetyClient": SimpleNamespace(
                acquire_session=forbidden, release_session=forbidden
            )
        },
        "gello.zmq_core.robot_node": {
            "ZMQClientRobot": SimpleNamespace(
                get_joint_sample=forbidden,
                command_joint_state=lambda *, session_id: None,
            )
        },
        "rhospolicy.inference.yam_pi05": {"Pi05PolicyClient": forbidden},
        "run_pi05_heavier": {
            "load_reference": lambda p: json.loads(p.read_text()),
            "require_control_rate": lambda _: None,
        },
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    from dagger_offline import live_robot

    monkeypatch.setattr(live_robot, "relative_controller", lambda _: None)
    monkeypatch.setattr(runner.sys, "path", list(sys.path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kw: runner.REVISION if args[-1] == "HEAD" else "",
    )
    args = SimpleNamespace(
        rhos_root=root, abc=tmp_path, output=tmp_path / "unused.h5", deployment=output
    )
    selected, _, _, start, metadata = runner.preflight(args)
    assert selected["checkpoint"] == str(checkpoint)
    assert selected["repo_id"] == "r1_lerobot" and selected["port"] == 8769
    assert metadata["model_round"] == 1 and metadata["round"] == 2
    assert metadata["sealed_policy_id"] == sealed["policy_id"]
    assert metadata["baseline"] == baseline
    assert selected["profile"] == baseline["profile"]
    expected_zero = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert start["state"] == expected_zero
    assert metadata["start_reference"]["state"] == expected_zero
    assert not args.output.exists()
    args.deployment = None
    selected, _, _, start, metadata = runner.preflight(args)
    assert selected == baseline and metadata["model_round"] == 0
    assert "sealed_policy_id" not in metadata and metadata["round"] == 1
    assert start["state"] == expected_zero
    assert metadata["start_reference"]["state"] == expected_zero
    assert reference.read_bytes() == original_reference


def test_serving_config_rebuilds_the_trained_rl_token_model(tmp_path):
    pytest.importorskip("openpi")
    import dataclasses

    import yam_openpi
    from dagger_offline.deployment import serving_config
    from dagger_offline.rl_token import RLTPi0Config, config_from_contract

    token = {"alpha": 0.0, "width": 512, "depth": 2, "heads": 8, "stride": 1}
    checkpoint = tmp_path / "run/1999"
    base = yam_openpi.make_config(checkpoint.parent).model
    trained = config_from_contract(base, token)
    inference = {
        "model": "pi05",
        "action_dim": 32,
        "action_horizon": 30,
        "adapt_to_pi": False,
        "delta_joint_actions": True,
        "openpi_revision": yam_openpi.OPENPI_REVISION,
        "config_name": "pi05_yam_hgdagger",
        "exp_name": "r1",
        "asset_id": "r1",
        "rl_token": token,
    }
    deployment = {
        "checkpoint": str(checkpoint),
        "run": {"inference": inference, "model": repr(trained)},
    }
    config = serving_config(deployment)
    assert isinstance(config.model, RLTPi0Config) and config.model.rl_alpha == 0.0
    changed = {**token, "width": 256}
    with pytest.raises(ValueError, match="differs"):
        serving_config(
            {
                **deployment,
                "run": {
                    **deployment["run"],
                    "inference": {**inference, "rl_token": changed},
                },
            }
        )
    plain = {**inference}
    del plain["rl_token"]
    assert not isinstance(
        serving_config(
            {
                **deployment,
                "run": {"inference": plain, "model": repr(dataclasses.replace(base))},
            }
        ).model,
        RLTPi0Config,
    )
