import pytest


def test_report_separates_conditions_and_keeps_failure_denominator():
    from dagger_offline.evaluation import summarize

    attempts = [
        {
            "id": "a",
            "condition": "left_heavier",
            "success": True,
            "duration_s": 10,
            "first_right_action_s": 2,
            "interventions": [[2, 4]],
            "protection_triggers": 0,
        },
        {
            "id": "b",
            "condition": "left_heavier",
            "success": False,
            "duration_s": 30,
            "first_right_action_s": None,
            "interventions": [[1, 4], [8, 13]],
            "protection_triggers": 2,
        },
        {
            "id": "c",
            "condition": "right_heavier",
            "success": False,
            "duration_s": 5,
            "first_right_action_s": 4,
            "interventions": [],
            "protection_triggers": 1,
        },
    ]
    report = summarize(attempts)
    left = report["left_heavier"]
    assert left["success_rate"] == 0.5
    assert left["mean_duration_s"] == 20
    assert left["intervention_fraction"] == 0.25  # 10s / 40s, not mean of ratios
    assert left["intervention_count"] == 3
    assert left["right_action_observed"] == 1
    assert left["mean_first_right_action_s_when_observed"] == 2
    assert report["right_heavier"]["success_rate"] == 0
    assert report["overall"]["success_rate"] == 1 / 3


@pytest.mark.parametrize(
    "change",
    [
        {"success": "false"},
        {"duration_s": float("nan")},
        {"interventions": [[1, 4], [3, 5]]},
        {"first_right_action_s": 11},
        {"protection_triggers": -1},
    ],
)
def test_invalid_annotations_not_silently_counted(change):
    from dagger_offline.evaluation import summarize

    row = {
        "id": "a",
        "condition": "left_heavier",
        "success": False,
        "duration_s": 10,
        "first_right_action_s": None,
        "interventions": [],
        "protection_triggers": 0,
    }
    with pytest.raises(ValueError):
        summarize([{**row, **change}])
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, row])
