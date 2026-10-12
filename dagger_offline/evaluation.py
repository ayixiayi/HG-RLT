"""Aggregate explicitly annotated attempts; never infer success from usable data.

Input is a JSON list; see test_evaluation.py for the small annotation schema.
Times are seconds from rollout start. Include failures and safety-aborted attempts.
This reports observed outcomes, not model inference or autonomous evaluation.
"""

import argparse
import json
import math
from pathlib import Path


def summarize(attempts):
    if not attempts:
        raise ValueError("No attempts")
    seen = set()
    for row in attempts:
        if not row["id"] or row["id"] in seen:
            raise ValueError("Missing or duplicate attempt ID")
        seen.add(row["id"])
        duration = row["duration_s"]
        if (
            row["condition"] not in ("left_heavier", "right_heavier")
            or type(row["success"]) is not bool
            or not math.isfinite(duration)
            or duration <= 0
            or type(row["protection_triggers"]) is not int
            or row["protection_triggers"] < 0
        ):
            raise ValueError("Invalid attempt outcome/condition/duration")
        first = row["first_right_action_s"]
        if first is not None and (
            not math.isfinite(first) or not 0 <= first <= duration
        ):
            raise ValueError("Invalid first right-arm action time")
        previous_end = 0
        for start, end in row["interventions"]:
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and previous_end <= start < end <= duration
            ):
                raise ValueError(
                    "Interventions must be ordered nonoverlapping intervals"
                )
            previous_end = end
    result = {}
    for condition in ("overall", "left_heavier", "right_heavier"):
        rows = [
            r for r in attempts if condition == "overall" or r["condition"] == condition
        ]
        if not rows:
            continue
        total = sum(r["duration_s"] for r in rows)
        human = sum(end - start for r in rows for start, end in r["interventions"])
        firsts = [
            r["first_right_action_s"]
            for r in rows
            if r["first_right_action_s"] is not None
        ]
        result[condition] = {
            "attempts": len(rows),
            "successes": sum(r["success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "mean_duration_s": total / len(rows),
            "intervention_count": sum(len(r["interventions"]) for r in rows),
            "intervention_duration_s": human,
            "intervention_fraction": human / total,
            "protection_triggers": sum(r["protection_triggers"] for r in rows),
            "right_action_observed": len(firsts),
            "mean_first_right_action_s_when_observed": sum(firsts) / len(firsts)
            if firsts
            else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(json.loads(args.attempts.read_text())), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
