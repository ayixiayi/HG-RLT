"""Content-bound Pi0.5 round deployments; seal/check are CPU-only file operations.

Only the explicit `serve` subcommand loads weights/listens on localhost. Nothing
in this module connects to hardware. Identity is a trusted server's claim, not
remote attestation. Keep the selected checkpoint immutable while serving.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

SCHEMA = "hg-pi05-sealed-v1"
INSTRUCTION = "choose_the_heavier_one"


def content_id(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_id(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalization_id(path):
    """Compare numerical stats, not formatting changed by OpenPI asset saving."""
    raw = json.loads(Path(path).read_text())["norm_stats"]
    stats = {}
    for key in ("state", "actions"):
        stats[key] = {}
        for name in ("mean", "std", "q01", "q99"):
            values = raw[key][name]
            if (
                not isinstance(values, list)
                or len(values) != 14
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in values
                )
            ):
                raise ValueError("Expected finite 14-D parent normalization")
            stats[key][name] = [float(v) for v in values]
        if any(v < 0 for v in stats[key]["std"]) or any(
            lo > hi for lo, hi in zip(stats[key]["q01"], stats[key]["q99"], strict=True)
        ):
            raise ValueError("Invalid parent normalization ranges")
    return content_id(stats)


def _inference_contract(run):
    import yam_openpi

    contract = run.get("inference")
    if not isinstance(contract, dict):
        raise ValueError(  # noqa: TRY004 - invalid on-disk contract, not an API argument
            "Training sidecar lacks structured inference identity; do not guess it"
        )
    required = {
        "model": "pi05",
        "action_dim": 32,
        "action_horizon": 30,
        "adapt_to_pi": False,
        "delta_joint_actions": True,
        "openpi_revision": yam_openpi.OPENPI_REVISION,
    }
    if any(
        type(contract.get(k)) is not type(v) or contract[k] != v
        for k, v in required.items()
    ):
        raise ValueError("Unsupported Pi0.5 inference contract")
    for key in ("config_name", "exp_name", "asset_id"):
        value = contract.get(key)
        if (
            not isinstance(value, str)
            or not value
            or Path(value).name != value
            or value in {".", ".."}
        ):
            raise ValueError(f"Invalid inference {key}")
    if contract["config_name"] in {"pi05_yam_heavier", "pi05_yam_heavier_0928"}:
        raise ValueError("Round deployment must not replace a baseline config")
    return contract


def _checkpoint_files(checkpoint, contract):
    marker = checkpoint / "_CHECKPOINT_METADATA"
    params = checkpoint / "params"
    stats = checkpoint / "assets" / contract["asset_id"] / "norm_stats.json"
    paths = [marker, stats, *sorted(params.rglob("*"))]
    if any(p.is_symlink() for p in [params, stats.parent, stats.parent.parent, *paths]):
        raise ValueError("Checkpoint payload must not contain symlinks")
    for path in (marker, stats, params / "_METADATA"):
        if not path.is_file():
            raise ValueError(f"Missing/nonregular checkpoint metadata: {path}")
    committed = json.loads(marker.read_text()).get("commit_timestamp_nsecs")
    if type(committed) is not int or committed <= 0:
        raise ValueError("Checkpoint has not finished committing")
    if normalization_id(stats) != contract.get("norm_id"):
        raise ValueError("Checkpoint normalization differs from training")
    files = {}
    for path in paths:
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Nonregular checkpoint payload: {path}")
        files[str(path.relative_to(checkpoint))] = file_id(path)
    return files


def _policy_id(deployment):
    # Location/port may change during transfer without changing the model.
    return content_id(
        {k: deployment[k] for k in ("schema", "model_round", "run", "files")}
    )


def seal(checkpoint, output, *, model_round, port=8769):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    if type(model_round) is not int or model_round < 1:
        raise ValueError("model_round must be a positive integer")
    if type(port) is not int or not 1024 <= port <= 65535 or port in {8767, 8768}:
        raise ValueError("Use a separate localhost port, not baseline 8767/8768")
    run = json.loads(Path(str(checkpoint.parent) + ".dagger.json").read_text())
    contract = _inference_contract(run)
    if (
        not checkpoint.name.isdecimal()
        or checkpoint.parent.name != contract["exp_name"]
        or checkpoint.parent.parent.name != contract["config_name"]
    ):
        raise ValueError("Checkpoint path does not match its training run")
    deployment = {
        "schema": SCHEMA,
        "checkpoint": os.path.relpath(checkpoint, output.parent),
        "model_round": model_round,
        "port": port,
        "run": run,
        "files": _checkpoint_files(checkpoint, contract),
    }
    deployment["policy_id"] = _policy_id(deployment)
    encoded = json.dumps(deployment, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded + "\n")
    return load_deployment(output)


def load_deployment(path):
    path = Path(path).resolve()
    deployment = json.loads(path.read_text())
    if deployment.get("schema") != SCHEMA:
        raise ValueError("Unknown sealed deployment schema")
    if type(deployment.get("model_round")) is not int or deployment["model_round"] < 1:
        raise ValueError("Invalid model round")
    port = deployment.get("port")
    if type(port) is not int or not 1024 <= port <= 65535 or port in {8767, 8768}:
        raise ValueError("Use a separate localhost port, not baseline 8767/8768")
    contract = _inference_contract(deployment["run"])
    if deployment.get("policy_id") != _policy_id(deployment):
        raise ValueError("Sealed policy identity mismatch")
    checkpoint = (path.parent / deployment["checkpoint"]).resolve()
    if _checkpoint_files(checkpoint, contract) != deployment["files"]:
        raise ValueError("Checkpoint payload changed after sealing")
    return {**deployment, "checkpoint": str(checkpoint)}


def serving_config(deployment):
    """Use the same YAM transforms as training, with checkpoint-owned assets."""
    from openpi.training.config import AssetsConfig

    import yam_openpi

    inference = _inference_contract(deployment["run"])
    checkpoint = Path(deployment["checkpoint"])
    base = yam_openpi.make_config(checkpoint.parent)
    model = base.model
    if inference.get("rl_token") is not None:
        from dagger_offline.rl_token import config_from_contract

        model = config_from_contract(model, inference["rl_token"])
    if repr(model) != deployment["run"].get("model"):
        raise ValueError("Serving model configuration differs from the training run")
    return dataclasses.replace(
        base,
        model=model,
        name=inference["config_name"],
        exp_name=inference["exp_name"],
        data=dataclasses.replace(
            base.data,
            repo_id="local/dagger",
            assets=AssetsConfig(
                assets_dir=str(checkpoint / "assets"), asset_id=inference["asset_id"]
            ),
        ),
    )


class SealedModel:
    """Only identity-bound calls are exposed; no legacy INFER fallback.

    PolicyServer serializes each entire infer_sealed call under its model lock.
    The wrapped Pi_05 adapter owns image layout and absolute-action conversion.
    """

    def __init__(self, model, policy_id):
        self._model = model
        self._tap = None
        policy = getattr(model, "policy", None)
        if hasattr(getattr(policy, "_model", None), "compute_rl_token"):
            from dagger_offline.rl_token import TokenTap

            # RLT round: each reply also carries the RL token of its inputs.
            self._tap = model.policy = model.model = TokenTap(model.policy)
        self._identity = {
            "policy_id": policy_id,
            "serving_session_id": uuid.uuid4().hex,
        }

    def get_deployment_identity(self):
        return dict(self._identity)

    def reset(self):
        self._model.reset()
        return self.get_deployment_identity()

    def infer_sealed(self, observation):
        import numpy as np

        if (
            not isinstance(observation, dict)
            or observation.get("instruction") != INSTRUCTION
        ):
            raise ValueError(
                "Sealed heavier deployment requires the trained instruction"
            )
        self._model.update_obs(observation)
        actions = np.asarray(self._model.get_action(), dtype=np.float64)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError("Expected finite (30,14) absolute joint actions")
        reply = {**self.get_deployment_identity(), "actions": actions.copy()}
        if self._tap is not None:
            if self._tap.last is None:
                raise ValueError("RL token missing for this inference")
            reply["rl_token"] = self._tap.last.copy()
        return reply


class SealedPolicyClient:
    """Verify before reset and on every reply, using the existing bounded RPC."""

    def __init__(self, url, policy_id, *, timeout_s=2.0):
        from client_server.ws.model_client import WsModelClient

        if (
            not isinstance(policy_id, str)
            or len(policy_id) != 64
            or any(c not in "0123456789abcdef" for c in policy_id)
        ):
            raise ValueError("Expected a SHA256 sealed policy identity")
        self._policy_id = policy_id
        self._session = None
        self.last_rl_token = None
        self._closed = False
        self._client = WsModelClient(
            url=url,
            evaluation_id="hg-pi05-sealed",
            trial_id=uuid.uuid4().hex,
            operation_timeout_s=timeout_s,
        )
        self._request("get_deployment_identity")

    def _request(self, method, observation=None):
        import numpy as np

        if self._closed:
            raise RuntimeError("Sealed policy client is closed")
        try:
            reply = self._client.call(method, obs=observation)
            if not isinstance(reply, dict) or reply.get("policy_id") != self._policy_id:
                raise ValueError("Policy identity mismatch or missing identity")
            session = reply.get("serving_session_id")
            if (
                not isinstance(session, str)
                or not session
                or self._session not in {None, session}
            ):
                raise ValueError("Serving session identity changed or missing")
            self._session = session
            if method == "infer_sealed":
                actions = np.asarray(reply["actions"], dtype=np.float64)
                if actions.shape != (30, 14) or not np.isfinite(actions).all():
                    raise ValueError("Expected finite (30,14) absolute joint actions")
                self.last_rl_token = None
                if "rl_token" in reply:
                    token = np.asarray(reply["rl_token"], dtype=np.float32)
                    if token.ndim != 1 or not np.isfinite(token).all():
                        raise ValueError("Invalid RL token in reply")
                    self.last_rl_token = token.copy()
                return actions.copy()
            return reply
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:  # noqa: BLE001 - preserve the primary failure
                exc.add_note(f"Policy close also failed: {cleanup}")
            raise

    def reset(self):
        self._request("reset")

    def infer_chunk(self, observation):
        return self._request("infer_sealed", observation)

    def close(self):
        self._closed = True
        self._client.close()


def serve(path, rhos_root, openpi):
    """Explicit GPU-capable model startup; never called by seal/check/preflight."""
    import yam_openpi

    deployment = load_deployment(path)
    revision = subprocess.check_output(
        ["git", "-C", str(openpi), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != yam_openpi.OPENPI_REVISION:
        raise ValueError("OpenPI revision mismatch")
    sys.path[:0] = [
        str(p)
        for p in (
            rhos_root,
            rhos_root / "XPolicyLab",
            openpi / "src",
            openpi / "packages/openpi-client/src",
        )
    ]
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from openpi.training import config as registry
    from XPolicyLab.policy.Pi_05.model import Model

    config = serving_config(deployment)
    if config.name in registry._CONFIGS_DICT:
        raise ValueError("Refusing to replace an existing registered model config")
    registry._CONFIGS_DICT[config.name] = config
    try:
        model = Model(
            {
                "task_name": INSTRUCTION,
                "action_type": "joint",
                "env_cfg_type": None,
                "model_path": deployment["checkpoint"],
                "train_config_name": config.name,
                "repo_id": deployment["run"]["inference"]["asset_id"],
            }
        )
    finally:
        del registry._CONFIGS_DICT[config.name]
    if load_deployment(path) != deployment:
        raise ValueError("Deployment changed while loading; refusing to serve")
    server = PolicyServer(
        SealedModel(model, deployment["policy_id"]),
        PolicyServerConfig(host="127.0.0.1", port=deployment["port"]),
    )
    asyncio.run(server.serve_forever())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("seal", help="CPU-only checkpoint/sidecar validation")
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--model-round", type=int, required=True)
    create.add_argument("--port", type=int, default=8769)
    for name in ("check", "serve"):
        sub = commands.add_parser(name)
        sub.add_argument("--deployment", type=Path, required=True)
        if name == "serve":
            sub.add_argument("--rhos-root", type=Path, required=True)
            sub.add_argument("--openpi", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seal":
        deployment = seal(
            args.checkpoint, args.output, model_round=args.model_round, port=args.port
        )
    else:
        deployment = load_deployment(args.deployment)
    print(
        json.dumps(
            {
                k: deployment[k]
                for k in ("policy_id", "checkpoint", "model_round", "port")
            },
            indent=2,
        )
    )
    if args.command == "serve":
        serve(args.deployment, args.rhos_root.resolve(), args.openpi.resolve())


if __name__ == "__main__":
    main()
