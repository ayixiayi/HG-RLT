"""Fail-closed ABC recording of acknowledged absolute targets.

The caller owns synchronization and hardware acknowledgement. This writer does
not infer actions from policy proposals or invent measured velocities. Instances
are single-writer; successful explicit finish is the sole publication point.
"""

import copy
import functools
import json
import queue
import threading
import time
from collections.abc import Mapping

import cv2
import h5py
import numpy as np

# Operator-selected reason for a failed episode (HG analysis, RLT phase choice).
FAILURE_TYPES = {
    "wrong_choice": "选错",
    "grasp": "抓取失败",
    "drop": "掉落",
    "stuck": "卡住",
    "timeout": "超时",
    "other": "其他",
}


def _guard(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        self.raise_if_failed()
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._failed = True
            self._file.attrs["usable"] = False
            self._file.attrs["unusable"] = True
            self._file.flush()
            raise

    return guarded


def _timestamp(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("timestamp_ns must be an integer")
    if not 0 <= value <= np.iinfo(np.uint64).max:
        raise ValueError("timestamp_ns outside uint64 range")
    return int(value)


def _vector(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError("Expected finite 14-D vector")
    return array


class EpisodeRecorder:
    def __init__(self, path, *, task_name, metadata):
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("Nonempty task_name required")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON mapping")
        encoded = json.dumps(dict(metadata), allow_nan=False)
        self._closed = self._failed = self._discarded = False
        self._sample_time = self._event_time = None
        self._follower_time = None
        self._paired = None
        self._checkpoints = []
        self._file = h5py.File(path, "x")
        try:
            self._file.attrs.update(
                usable=False,
                unusable=True,
                complete=False,
                outcome="in_progress",
                recording_type="dagger",
                dagger=True,
                task_name=task_name,
                metadata_json=encoded,
                action_semantics="acknowledged_absolute_targets",
            )
            self._file.create_group("data")
            self._file.create_group("timestamps")
            events = self._file.create_group("dagger_events")
            for name, dtype in (
                ("timestamp_ns", "uint64"),
                ("controller_state", h5py.string_dtype()),
                ("reason", h5py.string_dtype()),
            ):
                events.create_dataset(name, (0,), maxshape=(None,), dtype=dtype)
            self._file.flush()
        except Exception:
            self._file.close()
            self._closed = True
            raise

    def _append(self, group, name, value, dtype):
        shape = (
            np.shape(value) if h5py.check_vlen_dtype(np.dtype(dtype)) is None else ()
        )
        if name not in group:
            group.create_dataset(
                name, (0, *shape), maxshape=(None, *shape), dtype=dtype
            )
        dataset = group[name]
        row = len(dataset)
        dataset.resize(row + 1, axis=0)
        dataset[row] = value

    def raise_if_failed(self):
        if self._failed:
            raise RuntimeError("Recorder failed; close and start a new episode")

    @_guard
    def event(self, timestamp_ns, state, reason=""):
        timestamp_ns = _timestamp(timestamp_ns)
        if state not in {"POLICY_RUNNING", "INTERVENTION", "HOLD", "FAULT"}:
            raise ValueError("Invalid controller state")
        if not isinstance(reason, str) or "\x00" in reason:
            raise ValueError("reason must be a NUL-free string")
        if self._event_time is not None and timestamp_ns < self._event_time:
            raise ValueError("Event times must be nondecreasing")
        group = self._file["dagger_events"]
        for name, value, dtype in (
            ("timestamp_ns", timestamp_ns, "uint64"),
            ("controller_state", state, h5py.string_dtype()),
            ("reason", reason, h5py.string_dtype()),
        ):
            self._append(group, name, value, dtype)
        self._event_time = timestamp_ns
        self._file.flush()

    @_guard
    def sample(
        self,
        timestamp_ns,
        qpos,
        applied_action,
        images,
        *,
        raw_action=None,
        timing=None,
    ):
        """Store an observation paired with its subsequent acknowledged command.

        In live recordings timestamps/* is the pairing index (oldest camera
        host timestamp), NOT a claim of simultaneous sensor acquisition. Raw
        camera host stamps, state RPC interval, per-joint CAN receive times and
        command/ACK times are retained in pairing/*. CAN receive is host decode
        time, not encoder sample time or simultaneous acquisition. HOLD rows
        retain observation/last target but action_valid=False excludes training.
        """
        timestamp_ns = _timestamp(timestamp_ns)
        if self._sample_time is not None and timestamp_ns <= self._sample_time:
            raise ValueError("Sample times must be strictly increasing")
        qpos, applied_action = _vector(qpos), _vector(applied_action)
        raw_action = None if raw_action is None else _vector(raw_action)
        paired = timing is not None
        if self._paired is not None and paired != self._paired:
            raise ValueError("Cannot mix paired and legacy samples")
        if paired:
            required = {
                "camera_ns",
                "state_requested_ns",
                "state_received_ns",
                "follower_received_ns",
                "command_sent_ns",
                "command_ack_ns",
                "action_valid",
            }
            if set(timing) != required:
                raise ValueError("Missing paired sample timing")
            camera = [_timestamp(t) for t in timing["camera_ns"]]
            requested, received, sent, ack = (
                _timestamp(timing[k])
                for k in (
                    "state_requested_ns",
                    "state_received_ns",
                    "command_sent_ns",
                    "command_ack_ns",
                )
            )
            if (
                len(camera) != 3
                or min(camera) != timestamp_ns
                or not max(camera) <= requested <= received
                or not isinstance(timing["action_valid"], bool)
            ):
                raise ValueError("Invalid paired observation timing")
            if timing["action_valid"]:
                if not received <= sent <= ack or ack - timestamp_ns > 100_000_000:
                    raise ValueError("Invalid or late command acknowledgement timing")
            elif sent or ack:
                raise ValueError("HOLD row cannot claim an acknowledged command")
            follower = np.array(
                [_timestamp(t) for t in timing["follower_received_ns"]], dtype=np.uint64
            )
            if (
                follower.shape != (14,)
                or np.any(follower == 0)
                or np.any(follower > received)
                or np.any(
                    (ack if timing["action_valid"] else received) - follower
                    > 100_000_000
                )
                or self._follower_time is not None
                and np.any(follower < self._follower_time)
            ):
                raise ValueError(
                    "Invalid/stale/regressed follower CAN receive timestamps"
                )
        if not isinstance(images, Mapping) or set(images) != {"top", "left", "right"}:
            raise ValueError("Exactly top/left/right RGB images required")
        buffers = {}
        for name, image in images.items():
            image = np.asarray(image)
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError("Images must be nonempty HWC RGB uint8")
            ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("JPEG encoding failed")
            buffers[name] = buffer.reshape(-1)
        values = {
            "q_left": qpos[:6],
            "q_gripper_left": qpos[6:7],
            "q_right": qpos[7:13],
            "q_gripper_right": qpos[13:14],
            "q_des_left": applied_action[:7],
            "q_des_right": applied_action[7:],
        }
        for name, value in {**values, **buffers}.items():
            dtype = h5py.vlen_dtype(np.dtype("uint8")) if name in buffers else "float64"
            self._append(self._file["data"], name, value, dtype)
            self._append(self._file["timestamps"], name, timestamp_ns, "uint64")
        if paired:
            self._file.attrs["sample_alignment"] = "observation_command_pair_v2"
            self._file.attrs["state_timestamp_semantics"] = (
                "per-joint host-monotonic CAN decode/receive time; not encoder sample time"
            )
            group = self._file.require_group("pairing")
            for name, value in timing.items():
                self._append(
                    group, name, value, "bool" if name == "action_valid" else "uint64"
                )
            self._follower_time = follower.copy()
        if raw_action is not None:
            diagnostics = self._file.require_group("diagnostics")
            self._append(diagnostics, "raw_action", raw_action, "float64")
            self._append(diagnostics, "timestamp_ns", timestamp_ns, "uint64")
        self._sample_time = timestamp_ns
        self._paired = paired
        self._file.flush()

    def abort(self, timestamp_ns, reason):
        """Best-effort FAULT journal, even if an earlier append poisoned the file."""
        if self._closed:
            return
        self._file.attrs.update(
            usable=False, unusable=True, complete=False, outcome="aborted"
        )
        self._failed = False
        try:
            self.event(timestamp_ns, "FAULT", reason)
        finally:
            self._failed = True
            self._file.flush()

    @_guard
    def checkpoint(self, timestamp_ns):
        timestamp_ns = _timestamp(timestamp_ns)
        if self._checkpoints and timestamp_ns < self._checkpoints[-1]:
            raise ValueError("Checkpoint times must be nondecreasing")
        group = self._file.require_group("dagger_checkpoints")
        self._append(group, "timestamp_ns", timestamp_ns, "uint64")
        self._append(group, "index", len(self._checkpoints), "int64")
        self._checkpoints.append(timestamp_ns)
        self._file.flush()

    @_guard
    def discard(self, timestamp_ns, after_checkpoint=False):
        timestamp_ns = _timestamp(timestamp_ns)
        eligible = [
            i for i, time in enumerate(self._checkpoints) if time <= timestamp_ns
        ]
        if after_checkpoint and eligible:
            index = eligible[-1]
            group = self._file.require_group("discard_segments")
            for name, value, dtype in (
                ("start_ns", self._checkpoints[index], "uint64"),
                ("end_ns", timestamp_ns, "uint64"),
                ("checkpoint_index", index, "int64"),
            ):
                self._append(group, name, value, dtype)
            self._file.attrs["partial_discarded"] = True
        else:
            self._discarded = True
            self._file.attrs["discarded"] = True
        self._file.flush()

    @_guard
    def policy_chunk(self, observed_ns, accepted_ns, actions, *, continuous, skip):
        """Every accepted policy prediction, before filtering (RLT reference ã).

        actions: the complete (T, 14) absolute-joint chunk the policy returned;
        skip: leading actions dropped because they ran after the observation.
        """
        observed_ns, accepted_ns = _timestamp(observed_ns), _timestamp(accepted_ns)
        actions = np.asarray(actions, dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not np.isfinite(actions).all()
        ):
            raise ValueError("policy chunk must be a finite (T, 14) array")
        if accepted_ns < observed_ns or type(skip) is not int or skip < 0:
            raise ValueError("Invalid policy chunk timing/skip")
        group = self._file.require_group("policy_chunks")
        if "actions" in group and group["actions"].shape[1:] != actions.shape:
            raise ValueError("policy chunk length changed within an episode")
        for name, value, dtype in (
            ("observed_ns", observed_ns, "uint64"),
            ("accepted_ns", accepted_ns, "uint64"),
            ("continuous", bool(continuous), "bool"),
            ("skip", skip, "int32"),
            ("actions", actions, "float64"),
        ):
            self._append(group, name, value, dtype)

    @_guard
    def rl_decision(
        self,
        observed_ns,
        accepted_ns,
        *,
        token,
        proprio,
        reference,
        actor,
        actor_version,
        committed,
        skip,
    ):
        """One RLT actor decision: its state (token, proprio, committed steps
        fixed at its observation), the VLA reference ã and its chunk, both
        aligned to the observation, and how many steps ran before it took over."""
        observed_ns, accepted_ns = _timestamp(observed_ns), _timestamp(accepted_ns)
        token = np.asarray(token, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float64)
        actor = np.asarray(actor, dtype=np.float64)
        committed = np.asarray(committed, dtype=np.float64)
        if (
            committed.ndim != 2
            or committed.shape[1] != 14
            or not np.isfinite(committed).all()
            or type(skip) is not int
            or skip < 0
        ):
            raise ValueError("Invalid RL decision committed steps")
        if (
            token.ndim != 1
            or proprio.shape != (28,)
            or reference.ndim != 2
            or reference.shape != actor.shape
            or reference.shape[1] != 14
            or not all(np.isfinite(x).all() for x in (token, proprio, reference, actor))
            or accepted_ns < observed_ns
            or type(actor_version) is not int
        ):
            raise ValueError("Invalid RL decision")
        group = self._file.require_group("rl_decisions")
        for name, value, dtype in (
            ("observed_ns", observed_ns, "uint64"),
            ("accepted_ns", accepted_ns, "uint64"),
            ("actor_version", actor_version, "int64"),
            ("token", token, "float32"),
            ("proprio", proprio, "float32"),
            ("reference", reference, "float64"),
            ("actor", actor, "float64"),
            ("committed", committed, "float64"),
            ("skip", skip, "int64"),
        ):
            self._append(group, name, value, dtype)

    @_guard
    def finish(self, *, keep, outcome=None, failure_type=None):
        if not isinstance(keep, bool):
            raise TypeError("keep must be bool")
        if outcome not in (None, "success", "fail"):
            raise ValueError("outcome must be success, fail, or None")
        if failure_type is not None and (
            outcome != "fail" or failure_type not in FAILURE_TYPES
        ):
            raise ValueError("failure_type only labels a failed episode")
        usable = keep and not self._discarded
        self._file.attrs.update(
            complete=True,
            usable=usable,
            unusable=not usable,
            outcome=(outcome or "unlabelled") if keep else "discard",
        )
        if failure_type is not None:
            self._file.attrs["failure_type"] = failure_type
        self._file.flush()
        self.close()

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None and not self._closed and not self._failed:
                self.abort(time.monotonic_ns(), f"{exc_type.__name__}: {exc}")
        finally:
            self.close()


class AsyncEpisodeRecorder:
    """Single producer, bounded FIFO, one thread owning all JPEG/HDF5 work.

    Enqueue copies inputs and never waits for disk or queue space. Any failed
    write or overflow poisons the episode; it cannot be published afterwards.
    finish/abort/close can wait for storage. Stop commands and attempt HOLD and
    release before cleanup; publishing additionally requires both confirmations.
    No background heartbeats; a failed stop cannot publish a usable recording.
    Abort retains already queued diagnostic rows when storage is still healthy.
    """

    def __init__(
        self,
        path,
        *,
        task_name,
        metadata,
        queue_size=30,
        writer_factory=EpisodeRecorder,
    ):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        self._commands = queue.Queue(maxsize=queue_size)
        self._error_lock = threading.Lock()
        self._error = None
        self._closed = self._aborted = False
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        self._close_error = None
        ready = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run,
            args=(path, task_name, copy.deepcopy(metadata), writer_factory, ready),
            name="hg-episode-writer",
            daemon=True,
        )
        try:
            self._thread.start()
            error = ready.get()  # File creation must succeed before hardware opens.
            if error is not None:
                raise error
        except BaseException as exc:
            self._closed = True
            self._stopping.set()
            try:
                # start() itself may fail before a worker exists. If interrupted
                # before its ident is visible, the eventual worker still stops.
                if self._thread.ident is not None:
                    self.close()
            except BaseException as cleanup:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "Recorder initialization/cleanup failures", [exc, cleanup]
                ) from None
            raise

    def _set_error(self, error):
        with self._error_lock:
            if self._error is None:
                self._error = error

    def raise_if_failed(self):
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Asynchronous recorder failed: {error}") from error

    def _run(self, path, task_name, metadata, factory, ready):
        writer = None
        try:
            try:
                writer = factory(path, task_name=task_name, metadata=metadata)
            except BaseException as exc:  # noqa: BLE001 - notify the waiting constructor
                self._set_error(exc)
                ready.put(exc)
                return
            ready.put(None)
            while not self._stopping.is_set() or not self._commands.empty():
                try:
                    operation, args, kwargs, response = self._commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                error = None
                try:
                    # Cancel publication, not healthy rows preceding a fault.
                    # The synchronous writer latches its own storage failures.
                    if operation == "finish":
                        self.raise_if_failed()
                    getattr(writer, operation)(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - relay worker exits to the caller
                    error = exc
                    self._set_error(exc)
                finally:
                    if response is not None:
                        response.put(error)
                    self._commands.task_done()
        finally:
            try:
                if writer is not None:
                    writer.close()
            except BaseException as exc:  # noqa: BLE001 - close failure belongs to the caller
                self._close_error = exc
                self._set_error(exc)
            finally:
                self._stopped.set()

    def _enqueue(self, operation, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        self.raise_if_failed()
        try:
            command = (operation, copy.deepcopy(args), copy.deepcopy(kwargs), None)
            self._commands.put_nowait(command)
        except queue.Full as exc:
            error = RuntimeError("Recorder queue is full; recording cannot keep up")
            self._set_error(error)
            raise error from exc
        except Exception as exc:
            self._set_error(exc)
            raise

    def _barrier(self, operation, *args, **kwargs):
        # Only used after motion stops; disk waits must never renew a lease.
        response = queue.Queue(maxsize=1)
        self._commands.put((operation, args, kwargs, response))
        error = response.get()
        if error is not None:
            raise error

    def event(self, timestamp_ns, state, reason=""):
        self._enqueue("event", timestamp_ns, state, reason)

    def sample(
        self,
        timestamp_ns,
        qpos,
        applied_action,
        images,
        *,
        raw_action=None,
        timing=None,
    ):
        self._enqueue(
            "sample",
            timestamp_ns,
            qpos,
            applied_action,
            images,
            raw_action=raw_action,
            timing=timing,
        )

    def policy_chunk(self, observed_ns, accepted_ns, actions, *, continuous, skip):
        self._enqueue(
            "policy_chunk",
            observed_ns,
            accepted_ns,
            np.array(actions, dtype=np.float64, copy=True),
            continuous=continuous,
            skip=skip,
        )

    def rl_decision(self, observed_ns, accepted_ns, **fields):
        self._enqueue(
            "rl_decision",
            observed_ns,
            accepted_ns,
            **{
                k: np.array(v, copy=True) if k != "actor_version" else v
                for k, v in fields.items()
            },
        )

    def checkpoint(self, timestamp_ns):
        self._enqueue("checkpoint", timestamp_ns)

    def discard(self, timestamp_ns, after_checkpoint=False):
        self._enqueue("discard", timestamp_ns, after_checkpoint=after_checkpoint)

    def finish(self, *, keep, outcome=None, failure_type=None):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        try:
            kwargs = {"keep": keep, "outcome": outcome}
            if failure_type is not None:
                kwargs["failure_type"] = failure_type
            self._barrier("finish", **kwargs)
        except BaseException as exc:
            # Cancel a still-queued publication if the caller is interrupted
            # while the writer drains. An already committed file is final.
            self._set_error(exc)
            raise
        else:
            # On failure leave the writer open for the caller's FAULT journal.
            self.close()

    def abort(self, timestamp_ns, reason):
        if not self._closed and not self._aborted:
            # One attempt, even if journal I/O fails: __exit__ must not replace
            # the caller's original exception by repeating the same failed I/O.
            self._aborted = True
            self._set_error(RuntimeError(reason))
            self._barrier("abort", timestamp_ns, reason)

    def close(self):
        self._closed = True
        # Never enqueue shutdown: a full queue plus interrupted put can strand
        # the writer. It drains and closes independently even if this wait ends.
        self._stopping.set()
        self._stopped.wait()
        self._thread.join()
        error, self._close_error = self._close_error, None
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        errors = []
        try:
            if exc is not None and not self._closed and not self._aborted:
                self.abort(time.monotonic_ns(), f"{exc_type.__name__}: {exc}")
        except BaseException as journal:  # noqa: BLE001 - preserve original and cleanup failures
            errors.append(journal)
        try:
            self.close()
        except BaseException as cleanup:  # noqa: BLE001 - preserve original and cleanup failures
            errors.append(cleanup)
        if errors:
            if exc is not None:
                errors.insert(0, exc)
            raise BaseExceptionGroup("Recorder cleanup failures", errors) from None
