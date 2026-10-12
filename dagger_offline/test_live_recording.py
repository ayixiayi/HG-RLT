import json
import signal
import subprocess
import sys
import threading
import time

import h5py
import numpy as np
import pytest
from deploy.recording import io as rio

from dagger_offline.live_recording import AsyncEpisodeRecorder, EpisodeRecorder
from dagger_offline.round_data import read_source

TIMES = np.arange(8, dtype=np.uint64) * 33_333_333 + 1_000_000_000
Q = np.arange(14, dtype=float)
A = Q + 0.5


def images():
    image = np.empty((24, 32, 3), dtype=np.uint8)
    image[:, :16] = [240, 20, 10]
    image[:, 16:] = [10, 30, 220]
    return {name: image.copy() for name in rio.CAMERA_KEYS}


def recorder(path, writer_class=EpisodeRecorder):
    return writer_class(
        path, task_name="stack", metadata={"policy_id": "p1", "round": 2}
    )


def populate(r):
    r.event(TIMES[0], "POLICY_RUNNING")
    for t in TIMES:
        r.sample(t, Q, A, images(), raw_action=Q + 100)


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
def test_official_reader_labels_and_rgb(tmp_path, writer_class):
    path = tmp_path / "episode.h5"
    with recorder(path, writer_class) as r:
        populate(r)
        r.event(TIMES[2], "INTERVENTION", "takeover")
        r.event(TIMES[4], "HOLD")
        r.event(TIMES[5], "POLICY_RUNNING", "handback")
        r.event(TIMES[6], "FAULT")
        r.finish(keep=True)
    with h5py.File(path) as f:
        synced, states, actions, labels, valid, task = read_source(f, "current", 100)
        np.testing.assert_array_equal(states, np.tile(Q, (8, 1)))
        np.testing.assert_array_equal(actions, np.tile(A, (8, 1)))
        assert labels.tolist() == ["POLICY_RUNNING"] * 2 + ["INTERVENTION"] * 2 + [
            "HOLD",
            "POLICY_RUNNING",
            "FAULT",
            "FAULT",
        ]
        assert valid.tolist() == [True] * 4 + [False, True, False, False]
        assert rio.true_ranges(valid & (labels == "INTERVENTION")) == [(2, 4)]
        assert task == "stack"
        assert json.loads(f.attrs["metadata_json"])["policy_id"] == "p1"
        for key in f["data"]:
            np.testing.assert_array_equal(f["timestamps"][key], TIMES)
        assert not any("vel" in key for key in f["data"])
        np.testing.assert_array_equal(f["diagnostics/raw_action"][0], Q + 100)
        decoded = rio.decode_frame(f, "top", 0)
        np.testing.assert_allclose(decoded[10, 5], [10, 20, 240], atol=4)
        np.testing.assert_allclose(decoded[10, 25], [220, 30, 10], atol=4)
        np.testing.assert_array_equal(synced.base_ts_ns, TIMES)


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
@pytest.mark.parametrize("mode", ["close", "exception", "reject"])
def test_incomplete_exclusion(tmp_path, mode, writer_class):
    path = tmp_path / "episode.h5"
    if mode == "exception":
        with pytest.raises(RuntimeError), recorder(path, writer_class) as r:
            populate(r)
            raise RuntimeError("simulated process failure")
    else:
        r = recorder(path, writer_class)
        populate(r)
        if mode == "reject":
            r.finish(keep=False)
        r.close()
    with h5py.File(path) as f:
        assert rio.is_unusable(f)
        with pytest.raises(ValueError, match="unusable"):
            read_source(f, "current", 100)
    with pytest.raises(FileExistsError):
        recorder(path, writer_class)


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
@pytest.mark.parametrize("outcome", ["success", "fail"])
@pytest.mark.parametrize("discarded", [False, True])
def test_task_outcome_is_independent_of_data_usability(
    tmp_path, writer_class, outcome, discarded
):
    path = tmp_path / "labelled.h5"
    with recorder(path, writer_class) as r:
        populate(r)
        if discarded:
            r.discard(TIMES[-1])
        r.finish(keep=True, outcome=outcome)
    with h5py.File(path) as recording:
        assert recording.attrs["outcome"] == outcome
        assert recording.attrs["complete"]
        assert bool(recording.attrs["usable"]) == (not discarded)
        if not discarded:
            assert read_source(recording, "current", 100)[4].all()


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
@pytest.mark.parametrize(
    "partial,checkpoint", [(True, True), (True, False), (False, True)]
)
def test_discard(tmp_path, partial, checkpoint, writer_class):
    path = tmp_path / "episode.h5"
    with recorder(path, writer_class) as r:
        populate(r)
        if checkpoint:
            r.checkpoint(TIMES[2])
        r.discard(TIMES[4], after_checkpoint=partial)
        r.finish(keep=True)
    with h5py.File(path) as f:
        if partial and checkpoint:
            assert rio.discard_segments(f) == [(int(TIMES[2]), int(TIMES[4]))]
            assert rio.valid_mask(f, TIMES).tolist() == [
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
            ]
            assert read_source(f, "current", 100)[4].tolist() == [
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
            ]
        else:
            assert rio.is_unusable(f)


@pytest.mark.parametrize(
    "bad",
    ["duplicate", "qpos", "action", "raw", "image", "timestamp", "event", "event_time"],
)
def test_validation_before_append_and_poison(tmp_path, bad):
    path = tmp_path / "episode.h5"
    r = recorder(path)
    r.event(TIMES[0], "INTERVENTION")
    r.sample(TIMES[0], Q, A, images())
    kwargs = {
        "timestamp_ns": TIMES[1],
        "qpos": Q,
        "applied_action": A,
        "images": images(),
    }
    if bad == "duplicate":
        kwargs["timestamp_ns"] = TIMES[0]
    elif bad == "qpos":
        kwargs["qpos"] = np.zeros(13)
    elif bad == "action":
        kwargs["applied_action"] = np.full(14, np.nan)
    elif bad == "raw":
        kwargs["raw_action"] = np.full(14, np.inf)
    elif bad == "image":
        kwargs["images"]["right"] = np.zeros((2, 2, 3), dtype=float)
    elif bad == "timestamp":
        kwargs["timestamp_ns"] = -1
    with pytest.raises(ValueError):
        if bad == "event":
            r.event(TIMES[1], "UNKNOWN")
        elif bad == "event_time":
            r.event(TIMES[0] - np.uint64(1), "HOLD")
        else:
            r.sample(**kwargs)
    with pytest.raises(RuntimeError, match="failed"):
        r.finish(keep=True)
    r.close()
    with h5py.File(path) as f:
        assert rio.is_unusable(f)
        assert all(len(dataset) == 1 for dataset in f["data"].values())
        assert len(f["dagger_events/timestamp_ns"]) == 1


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
def test_pause_between_ticks_and_equal_event_times(tmp_path, writer_class):
    path = tmp_path / "episode.h5"
    with recorder(path, writer_class) as r:
        populate(r)
        r.event(TIMES[2] + np.uint64(1), "HOLD")
        r.event(TIMES[2] + np.uint64(2), "INTERVENTION")
        r.event(TIMES[2] + np.uint64(2), "POLICY_RUNNING")
        r.finish(keep=True)
    with h5py.File(path) as f:
        assert read_source(f, "current", 100)[4].tolist() == [
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ]


def test_async_sample_returns_before_storage_and_freezes_all_inputs(tmp_path):
    entered, resume = threading.Event(), threading.Event()

    class BlockedWriter(EpisodeRecorder):
        def sample(self, *args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test storage gate was not released")
            super().sample(*args, **kwargs)

    path = tmp_path / "snapshot.h5"
    t = int(TIMES[0])
    timing = {
        "camera_ns": [t, t + 1, t + 2],
        "state_requested_ns": t + 3,
        "state_received_ns": t + 4,
        "follower_received_ns": [t - 20 + i for i in range(14)],
        "command_sent_ns": t + 5,
        "command_ack_ns": t + 6,
        "action_valid": True,
    }
    qpos, action, raw, frames = Q.copy(), A.copy(), Q + 100, images()
    with AsyncEpisodeRecorder(
        path, task_name="stack", metadata={}, writer_factory=BlockedWriter
    ) as r:
        try:
            r.event(t, "INTERVENTION")
            r.sample(t, qpos, action, frames, raw_action=raw, timing=timing)
            assert entered.wait(2) and not resume.is_set()
            qpos += 2
            action += 3
            raw += 4
            delta = int(TIMES[1]) - t
            for key, value in timing.items():
                if isinstance(value, list):
                    value[:] = [stamp + delta for stamp in value]
                elif not isinstance(value, bool):
                    timing[key] = value + delta
            r.sample(TIMES[1], qpos, action, frames, raw_action=raw, timing=timing)
            qpos[:] = -1
            action[:] = -2
            raw[:] = -3
            for image in frames.values():
                image[:] = 0
            timing["camera_ns"][1] = 0
            timing["follower_received_ns"][:] = [0] * 14
            timing["action_valid"] = False
        finally:
            resume.set()
        r.finish(keep=True)
    with h5py.File(path) as f:
        _, states, actions, labels, valid, _ = read_source(f, "current", 100)
        np.testing.assert_array_equal(states, [Q, Q + 2])
        np.testing.assert_array_equal(actions, [A, A + 3])
        assert labels.tolist() == ["INTERVENTION"] * 2 and valid.tolist() == [True] * 2
        np.testing.assert_array_equal(
            f["diagnostics/raw_action"][:], [Q + 100, Q + 104]
        )
        np.testing.assert_array_equal(f["pairing/camera_ns"][0], [t, t + 1, t + 2])
        np.testing.assert_array_equal(
            f["pairing/follower_received_ns"][0], np.arange(14) + t - 20
        )
        for camera in rio.CAMERA_KEYS:
            np.testing.assert_allclose(
                rio.decode_frame(f, camera, 0)[10, 5], [10, 20, 240], atol=4
            )


@pytest.mark.parametrize("failure", ["disk", "invalid", "overflow"])
def test_async_failure_is_sticky_and_cannot_publish(tmp_path, failure):
    entered, resume = threading.Event(), threading.Event()

    class BlockedWriter(EpisodeRecorder):
        def sample(self, *args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test storage gate was not released")
            if failure == "disk":
                raise OSError("disk full")
            super().sample(*args, **kwargs)

    path = tmp_path / "failure.h5"
    r = AsyncEpisodeRecorder(
        path, task_name="stack", metadata={}, queue_size=1, writer_factory=BlockedWriter
    )
    try:
        qpos = np.full(14, np.nan) if failure == "invalid" else Q
        r.sample(TIMES[0], qpos, A, images())
        assert entered.wait(2)
        if failure == "overflow":
            r.sample(TIMES[1], Q, A, images())
            with pytest.raises(RuntimeError, match="queue is full"):
                r.sample(TIMES[2], Q, A, images())
            assert not resume.is_set()  # Overflow never waits for the blocked disk.
        resume.set()
        deadline = time.monotonic() + 2
        while True:
            try:
                r.raise_if_failed()
            except RuntimeError:
                break
            assert time.monotonic() < deadline, "writer failure was not propagated"
            time.sleep(0.001)
        with pytest.raises(RuntimeError, match="recorder failed"):
            r.event(TIMES[3], "HOLD")
        with pytest.raises(RuntimeError, match="recorder failed"):
            r.finish(keep=True)
    finally:
        resume.set()
        r.close()
    with h5py.File(path) as f:
        assert not f.attrs["complete"] and rio.is_unusable(f)
        with pytest.raises(ValueError, match="unusable"):
            read_source(f, "current", 100)


@pytest.mark.parametrize("phase", ["startup", "close"])
def test_async_interruption_releases_file(tmp_path, monkeypatch, phase):
    entered, resume = threading.Event(), threading.Event()

    class BlockedWriter(EpisodeRecorder):
        def sample(self, *args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test storage gate was not released")
            super().sample(*args, **kwargs)

    path = tmp_path / "interrupted.h5"
    kwargs = {
        "task_name": "stack",
        "metadata": {},
        "queue_size": 1,
        "writer_factory": BlockedWriter,
    }
    r = None
    previous = signal.getsignal(signal.SIGALRM)

    def interrupt(signum, frame):
        resume.set()
        raise KeyboardInterrupt("cancel the close wait")

    try:
        if phase == "close":
            r = AsyncEpisodeRecorder(path, **kwargs)
            r.sample(TIMES[0], Q, A, images())
            assert entered.wait(2)
            r.event(TIMES[1], "HOLD")  # Fill the queue while storage is blocked.
            signal.signal(signal.SIGALRM, interrupt)
            # Repeat so a regressed close cannot hang pytest in its finally/join.
            signal.setitimer(signal.ITIMER_REAL, 0.05, 0.2)
            with pytest.raises(KeyboardInterrupt):
                r.close()
        else:
            start = threading.Thread.start

            def interrupted_start(thread):
                start(thread)
                raise KeyboardInterrupt("interrupted at thread startup")

            with monkeypatch.context() as context, pytest.raises(KeyboardInterrupt):
                context.setattr(threading.Thread, "start", interrupted_start)
                r = AsyncEpisodeRecorder(path, **kwargs)
    finally:
        if phase == "close":
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        resume.set()
        if r is not None:
            r.close()
    # Another process must be able to lock/read the file: same-process h5py
    # handles can share an open writer and would not detect the leaked worker.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import h5py, sys\n"
                "with h5py.File(sys.argv[1]) as f:\n"
                "    assert not f.attrs['usable'] and not f.attrs['complete']\n"
                "    if sys.argv[2] == 'close':\n"
                "        assert len(f['data/q_left']) == 1\n"
                "        assert f['dagger_events/controller_state'].asstr()[:].tolist() == ['HOLD']\n"
                "    else:\n"
                "        assert not list(f['data'])\n"
            ),
            str(path),
            phase,
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_async_cleanup_preserves_original_journal_and_close_errors(tmp_path):
    class FailedCleanup(EpisodeRecorder):
        def abort(self, timestamp_ns, reason):
            raise OSError("journal unavailable")

        def close(self):
            super().close()
            raise OSError("close failed")

    path = tmp_path / "cleanup.h5"
    with (
        pytest.raises(BaseExceptionGroup) as error,
        AsyncEpisodeRecorder(
            path, task_name="stack", metadata={}, writer_factory=FailedCleanup
        ),
    ):
        raise ValueError("caller failed")
    assert [str(exc) for exc in error.value.exceptions] == [
        "caller failed",
        "journal unavailable",
        "close failed",
    ]
    with h5py.File(path) as f:
        assert not f.attrs["complete"] and rio.is_unusable(f)
