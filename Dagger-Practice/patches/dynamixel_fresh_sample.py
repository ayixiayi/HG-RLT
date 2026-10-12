"""Hardware-free coverage of leader sample freshness and atomic publication."""

from threading import Event, Lock, Thread
from unittest.mock import Mock

import numpy as np
import pytest
from gello.dynamixel import driver as module


@pytest.fixture
def driver():
    instance = module.DynamixelDriver.__new__(module.DynamixelDriver)
    instance._ids = [1, 2]
    instance._is_fake = False
    instance._lock = Lock()
    instance._sample_lock = Lock()
    instance._joint_angles = np.array([2048, -1024])
    instance._velocities = np.array([1, 2])
    instance._joint_sample_time = 123.5
    instance._stop_thread = Event()
    return instance


def test_sample_is_radians_with_original_timestamp_and_independent_copy(driver):
    angles, stamp = driver.get_joint_sample()
    np.testing.assert_allclose(angles, [np.pi, -np.pi / 2])
    assert stamp == 123.5
    angles[:] = 0
    np.testing.assert_array_equal(driver._joint_angles, [2048, -1024])
    later, later_stamp = driver.get_joint_sample()
    driver._joint_angles[:] = 0
    np.testing.assert_allclose(later, [np.pi, -np.pi / 2])
    assert later_stamp == stamp


@pytest.mark.parametrize(
    "angles, stamp",
    [
        (None, None),
        (np.array([1, 2]), None),
        (None, 1.0),
        (np.array([1]), 1.0),
        (np.array([1, np.nan]), 1.0),
        (np.array([1, 2]), np.nan),
        (np.array([1, 2]), -1.0),
    ],
)
def test_missing_or_invalid_sample_fails_without_startup_wait(
    driver, monkeypatch, angles, stamp
):
    driver._joint_angles = angles
    driver._joint_sample_time = stamp
    monkeypatch.setattr(
        module.time, "sleep", Mock(side_effect=AssertionError("startup wait"))
    )
    with pytest.raises(RuntimeError, match="complete valid"):
        driver.get_joint_sample()
    assert driver._sample_lock.acquire(blocking=False)
    driver._sample_lock.release()


def test_fake_fallback_fails_before_lock(driver):
    driver._is_fake = True
    driver._sample_lock = Mock()
    with pytest.raises(RuntimeError, match="Fake"):
        driver.get_joint_sample()
    driver._sample_lock.acquire.assert_not_called()
    fake = module.FakeDynamixelDriver.__new__(module.FakeDynamixelDriver)
    with pytest.raises(RuntimeError, match="Fake"):
        fake.get_joint_sample()


def test_busy_lock_has_bounded_acquisition(driver):
    driver._sample_lock = Mock()
    driver._sample_lock.acquire.return_value = False
    with pytest.raises(RuntimeError, match="busy"):
        driver.get_joint_sample()
    driver._sample_lock.acquire.assert_called_once_with(timeout=0.01)
    driver._sample_lock.release.assert_not_called()


def test_last_complete_sample_remains_readable_during_serial_io(driver):
    reading, finish = Event(), Event()
    driver._groupSyncRead = Mock()

    def transaction():
        reading.set()
        assert finish.wait(2)
        driver._stop_thread.set()
        return module.COMM_SUCCESS

    driver._groupSyncRead.txRxPacket.side_effect = transaction
    driver._groupSyncRead.isAvailable.return_value = True
    driver._groupSyncRead.getData.side_effect = [10, 4096, 20, 1024]
    reader = Thread(target=driver._read_joint_states, daemon=True)
    reader.start()
    try:
        assert reading.wait(1)
        angles, stamp = driver.get_joint_sample()
        np.testing.assert_allclose(angles, [np.pi, -np.pi / 2])
        assert stamp == 123.5  # An in-flight read must not refresh old feedback.
    finally:
        finish.set()
        reader.join(timeout=2)
    assert not reader.is_alive()
    angles, stamp = driver.get_joint_sample()
    np.testing.assert_allclose(angles, [2 * np.pi, np.pi / 2])
    assert stamp > 123.5


@pytest.mark.parametrize("outcome", ["success", "comm_failure", "incomplete"])
def test_reader_only_publishes_complete_success(driver, monkeypatch, outcome):
    old_angles = driver._joint_angles.copy()
    old_velocities = driver._velocities.copy()
    driver._groupSyncRead = Mock()

    def transaction():
        driver._stop_thread.set()  # Exactly one simulated read iteration.
        return module.COMM_SUCCESS if outcome != "comm_failure" else -1

    driver._groupSyncRead.txRxPacket.side_effect = transaction
    driver._groupSyncRead.isAvailable.side_effect = (
        [True, True, True, False] if outcome == "incomplete" else lambda *args: True
    )
    driver._groupSyncRead.getData.side_effect = [10, 4096, 20, 1024]
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def acquisition_time():
        assert driver._lock.locked()
        return 456.0

    clock = Mock(side_effect=acquisition_time)
    monkeypatch.setattr(module.time, "monotonic", clock)
    if outcome == "incomplete":
        with pytest.raises(RuntimeError, match="joint angles"):
            driver._read_joint_states()
    else:
        driver._read_joint_states()
    if outcome == "success":
        angles, stamp = driver.get_joint_sample()
        np.testing.assert_allclose(angles, [2 * np.pi, np.pi / 2])
        np.testing.assert_array_equal(driver._velocities, [10, 20])
        assert stamp == 456.0
        clock.assert_called_once()
    else:
        np.testing.assert_array_equal(driver._joint_angles, old_angles)
        np.testing.assert_array_equal(driver._velocities, old_velocities)
        assert driver._joint_sample_time == 123.5
        clock.assert_not_called()
