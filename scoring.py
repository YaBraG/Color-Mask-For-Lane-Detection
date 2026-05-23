"""Scoring helpers for offline OpenCV/NumPy detector auto-tuning."""

from __future__ import annotations

import math
from statistics import pstdev


def average(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def percent(count, total):
    return 100.0 * count / max(1, total)


def compute_detector_metrics(frame_rows, duration_sec):
    """Convert per-frame detector rows into stable, easy-to-score metrics.

    The optimizer should not reward a giant white mask by accident, so these
    metrics balance detection rate, mask size, ego-component quality, scanline
    quality, and centerline smoothness.
    """
    frame_count = len(frame_rows)
    if frame_count == 0:
        return empty_metrics()

    detected = [row for row in frame_rows if row["road_detected"]]
    qualities = [row["detection_quality"] for row in frame_rows]
    valid_scanlines = [row["valid_scanline_count"] for row in frame_rows]
    rejected_scanlines = [row["rejected_scanlines"] for row in frame_rows]
    mask_areas = [row["mask_area_percent"] for row in frame_rows]
    center_errors = [row["road_center_error_px"] for row in frame_rows if row["road_center_error_px"] is not None]
    curve_errors = [row["curve_error_px"] for row in frame_rows if row["curve_error_px"] is not None]

    center_deltas = absolute_deltas(center_errors)
    curve_deltas = absolute_deltas(curve_errors)
    center_jump_count = sum(1 for delta in center_deltas if delta > 95.0)
    curve_jump_count = sum(1 for delta in curve_deltas if delta > 140.0)
    turn_hint_change_count = count_changes([row["turn_hint"] for row in frame_rows])
    minutes = max(duration_sec / 60.0, 1.0 / 60.0)

    metrics = {
        "frame_count": frame_count,
        "road_detected_percent": percent(len(detected), frame_count),
        "average_detection_quality": average(qualities),
        "average_valid_scanline_count": average(valid_scanlines),
        "average_rejected_scanlines": average(rejected_scanlines),
        "rejected_scanline_problem_percent": percent(sum(1 for value in rejected_scanlines if value >= 5), frame_count),
        "tracked_center_invalid_percent": percent(sum(1 for row in frame_rows if not row["tracked_center_valid"]), frame_count),
        "ego_component_found_percent": percent(sum(1 for row in frame_rows if row["ego_component_found"]), frame_count),
        "average_mask_area_percent": average(mask_areas),
        "mask_area_std": pstdev(mask_areas) if len(mask_areas) >= 2 else 0.0,
        "center_jump_count": center_jump_count,
        "curve_jump_count": curve_jump_count,
        "turn_hint_change_count": turn_hint_change_count,
        "center_jump_count_per_minute": center_jump_count / minutes,
        "curve_jump_count_per_minute": curve_jump_count / minutes,
        "turn_hint_change_count_per_minute": turn_hint_change_count / minutes,
        "average_abs_center_delta": average(center_deltas),
        "average_abs_curve_delta": average(curve_deltas),
    }
    metrics["bad_mask_area_penalty"] = bad_mask_area_penalty(
        metrics["average_mask_area_percent"],
        metrics["mask_area_std"],
    )
    metrics["score"] = score_metrics(metrics)
    return metrics


def empty_metrics():
    return {
        "frame_count": 0,
        "road_detected_percent": 0.0,
        "average_detection_quality": 0.0,
        "average_valid_scanline_count": 0.0,
        "average_rejected_scanlines": 0.0,
        "rejected_scanline_problem_percent": 100.0,
        "tracked_center_invalid_percent": 100.0,
        "ego_component_found_percent": 0.0,
        "average_mask_area_percent": 0.0,
        "mask_area_std": 0.0,
        "center_jump_count": 0,
        "curve_jump_count": 0,
        "turn_hint_change_count": 0,
        "center_jump_count_per_minute": 0.0,
        "curve_jump_count_per_minute": 0.0,
        "turn_hint_change_count_per_minute": 0.0,
        "average_abs_center_delta": 0.0,
        "average_abs_curve_delta": 0.0,
        "bad_mask_area_penalty": 1000.0,
        "score": -1000.0,
    }


def score_metrics(metrics):
    """Single score used for ranking candidate configs.

    Rewards are positive detector behaviors. Penalties represent failure modes:
    too much rejected tracking, flickering turn hints, sudden centerline jumps,
    and masks that are implausibly tiny/huge/unstable.
    """
    score = 0.0
    score += 2.0 * metrics["road_detected_percent"]
    score += 120.0 * metrics["average_detection_quality"]
    score += 8.0 * metrics["average_valid_scanline_count"]
    score += 50.0 * (metrics["ego_component_found_percent"] / 100.0)
    score -= 2.0 * metrics["rejected_scanline_problem_percent"]
    score -= 3.0 * metrics["tracked_center_invalid_percent"]
    score -= 4.0 * metrics["center_jump_count_per_minute"]
    score -= 3.0 * metrics["curve_jump_count_per_minute"]
    score -= 2.0 * metrics["turn_hint_change_count_per_minute"]
    score -= 1.5 * metrics["average_abs_center_delta"]
    score -= 1.0 * metrics["average_abs_curve_delta"]
    score -= metrics["bad_mask_area_penalty"]
    return score


def bad_mask_area_penalty(average_mask_area_percent, mask_area_std):
    penalty = 0.0
    if average_mask_area_percent < 5.0:
        penalty += (5.0 - average_mask_area_percent) * 45.0
    if average_mask_area_percent > 65.0:
        penalty += (average_mask_area_percent - 65.0) * 30.0
    # Very high variation usually means the mask is reacting to lighting or
    # background blobs instead of consistently finding the road.
    if mask_area_std > 18.0:
        penalty += (mask_area_std - 18.0) * 10.0
    return penalty


def absolute_deltas(values):
    return [abs(values[index] - values[index - 1]) for index in range(1, len(values))]


def count_changes(values):
    return sum(1 for index in range(1, len(values)) if values[index] != values[index - 1])
