"""Offline OpenCV/NumPy hyperparameter search for the road detector.

This is not machine learning training. It is a reproducible search over detector
configuration values, centered on a human/manual seed config when available.
"""

from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import main as app
from config import (
    CENTER_DEADBAND_PX,
    CENTERLINE_ALPHA,
    CENTERLINE_SMOOTHING_ALPHA,
    CENTER_STRONG_PX,
    CURVE_DEADBAND_PX,
    CURVE_STRONG_PX,
    DEFAULT_SETTINGS,
    EGO_BOTTOM_BAND_PERCENT,
    EGO_MIN_COMPONENT_AREA_PERCENT,
    EGO_SEED_SEARCH_RADIUS_PX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LANE_WIDTH_MM,
    CAR_WIDTH_MM,
    SIDEWALK_MARGIN_MM,
    LINE_MARGIN_MM,
    SAFE_HALLWAY_WIDTH_MM,
    MIN_VALID_LANE_WIDTH_MM,
    MAX_VALID_LANE_WIDTH_MM,
    SAFE_STEERING_GAIN,
    USE_YELLOW_BOUNDARY_LOCK,
    YELLOW_H_MIN,
    YELLOW_H_MAX,
    YELLOW_S_MIN,
    YELLOW_S_MAX,
    YELLOW_V_MIN,
    YELLOW_V_MAX,
    YELLOW_BOUNDARY_DILATE_PX,
    YELLOW_MAX_CROSSING_PIXELS,
    LANE_SIDE_HOLD_FRAMES,
    USE_RIGHT_LANE_YELLOW_LOCK,
    RIGHT_LANE_FROM_YELLOW,
    YELLOW_LANE_SIDE,
    YELLOW_LANE_SEARCH_MARGIN_PX,
    YELLOW_MIN_PIXELS_PER_SCANLINE,
    YELLOW_RIGHT_LANE_HOLD_FRAMES,
    USE_NO_YELLOW_WIDE_BLOB_GATE,
    NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO,
    NO_YELLOW_MAX_MEASURED_WIDTH_MM,
    ALLOW_NO_YELLOW_BLOB_SPLIT,
    LAST_CENTER_HOLD_FRAMES,
    MAX_CENTER_JUMP_PX,
    MIN_SEGMENT_WIDTH_PX,
)
from scoring import compute_detector_metrics

OPTIMIZER_VERSION = "opencv_numpy_random_grid_v1"


def run_auto_tune(args):
    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Could not find video: {video_path}")

    quick = bool(args.quick)
    max_configs = min(args.max_configs, 12) if quick else max(1, args.max_configs)
    sample_stride = max(args.sample_stride, 90) if quick else max(1, args.sample_stride)
    top_k = min(args.top_k, 5) if quick else max(1, args.top_k)
    full_eval_top_k = min(args.full_eval_top_k, 1) if quick else max(1, args.full_eval_top_k)

    seed_config, seed_source = load_seed_config(args.seed_config)
    run_folders = create_auto_tune_folders(args.output_dir, video_path)
    save_json(run_folders["ai"] / "seed_config_used.json", {"source": seed_source, "config": seed_config})
    search_space = build_search_space(seed_config)
    save_json(run_folders["ai"] / "search_space.json", search_space)

    frames, source_fps, total_frames = load_sampled_frames(video_path, sample_stride)
    print(f"Auto-tune seed config: {seed_source}")
    print(f"Sampled frames loaded: {len(frames)} of {total_frames} (stride {sample_stride})")

    candidates = generate_candidates(seed_config, search_space, max_configs, args.random_seed)
    sampled_results = score_candidates(candidates, frames, source_fps, args.auto_tune_time_budget_hours)
    sampled_results.sort(key=lambda item: item["score"], reverse=True)
    for rank, result in enumerate(sampled_results, start=1):
        result["rank"] = rank

    write_scores_csv(run_folders["ai"] / "auto_tune_scores.csv", sampled_results)
    top_results = sampled_results[:top_k]
    save_json(run_folders["ai"] / "top_configs.json", top_results)

    full_results = []
    if quick:
        # Quick mode is a smoke test. It skips the extra full-video scoring
        # pass and spends its time rendering the best sampled config once.
        best = sampled_results[0]
    else:
        for result in sampled_results[:full_eval_top_k]:
            metrics = evaluate_config_on_video(video_path, result["config"])
            full_results.append(
                {
                    "candidate_id": result["candidate_id"],
                    "sampled_score": result["score"],
                    "score": metrics["score"],
                    "metrics": metrics,
                    "config": result["config"],
                }
            )
        full_results.sort(key=lambda item: item["score"], reverse=True)
        best = full_results[0] if full_results else sampled_results[0]

    best_config_payload = {
        "source_video": str(video_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed_config_path": seed_source,
        "optimizer_version": OPTIMIZER_VERSION,
        "score": best["score"],
        "config": best["config"],
        "notes": "Best OpenCV/NumPy detector config found by offline hyperparameter search.",
    }
    best_config_path = run_folders["ai"] / "best_config.json"
    save_json(best_config_path, best_config_payload)
    save_json(run_folders["ai"] / "best_config_metrics.json", best["metrics"])

    render_best_config_video(video_path, best["config"], run_folders, source_fps)
    write_auto_tune_summaries(
        run_folders,
        args,
        video_path,
        seed_config,
        sampled_results,
        top_results,
        best,
        best_config_path,
        search_space,
        len(frames),
        full_eval_top_k,
    )

    print()
    print("Auto-tune complete.")
    print(f"Auto-tune output folder: {run_folders['root']}")
    print(f"Best score: {best['score']:.3f}")
    print(f"Best config: {best_config_path}")
    return {
        "output_folder": str(run_folders["root"]),
        "best_score": best["score"],
        "best_config_path": str(best_config_path),
    }


def load_seed_config(seed_config_path):
    path = Path(seed_config_path) if seed_config_path else Path("configs/manual_tuned_config.json")
    if path.exists():
        return normalize_config(read_config_file(path)), str(path)
    default_manual = Path("configs/manual_tuned_config.json")
    if default_manual.exists():
        return normalize_config(read_config_file(default_manual)), str(default_manual)
    return normalize_config(DEFAULT_SETTINGS.copy()), "DEFAULT_SETTINGS"


def read_config_file(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "config" in payload and isinstance(payload["config"], dict):
        return payload["config"]
    return payload


def normalize_config(config):
    merged = {
        **DEFAULT_SETTINGS,
        "MIN_SEGMENT_WIDTH_PX": MIN_SEGMENT_WIDTH_PX,
        "MAX_CENTER_JUMP_PX": MAX_CENTER_JUMP_PX,
        "CENTERLINE_SMOOTHING_ALPHA": CENTERLINE_SMOOTHING_ALPHA,
        "CENTERLINE_ALPHA": CENTERLINE_ALPHA,
        "CURVE_DEADBAND_PX": CURVE_DEADBAND_PX,
        "CURVE_STRONG_PX": CURVE_STRONG_PX,
        "CENTER_DEADBAND_PX": CENTER_DEADBAND_PX,
        "CENTER_STRONG_PX": CENTER_STRONG_PX,
        "EGO_SEED_SEARCH_RADIUS_PX": EGO_SEED_SEARCH_RADIUS_PX,
        "EGO_BOTTOM_BAND_PERCENT": EGO_BOTTOM_BAND_PERCENT,
        "EGO_MIN_COMPONENT_AREA_PERCENT": EGO_MIN_COMPONENT_AREA_PERCENT,
        "LANE_WIDTH_MM": LANE_WIDTH_MM,
        "CAR_WIDTH_MM": CAR_WIDTH_MM,
        "SIDEWALK_MARGIN_MM": SIDEWALK_MARGIN_MM,
        "LINE_MARGIN_MM": LINE_MARGIN_MM,
        "SAFE_HALLWAY_WIDTH_MM": SAFE_HALLWAY_WIDTH_MM,
        "MIN_VALID_LANE_WIDTH_MM": MIN_VALID_LANE_WIDTH_MM,
        "MAX_VALID_LANE_WIDTH_MM": MAX_VALID_LANE_WIDTH_MM,
        "SAFE_STEERING_GAIN": SAFE_STEERING_GAIN,
        "USE_YELLOW_BOUNDARY_LOCK": USE_YELLOW_BOUNDARY_LOCK,
        "YELLOW_H_MIN": YELLOW_H_MIN,
        "YELLOW_H_MAX": YELLOW_H_MAX,
        "YELLOW_S_MIN": YELLOW_S_MIN,
        "YELLOW_S_MAX": YELLOW_S_MAX,
        "YELLOW_V_MIN": YELLOW_V_MIN,
        "YELLOW_V_MAX": YELLOW_V_MAX,
        "YELLOW_BOUNDARY_DILATE_PX": YELLOW_BOUNDARY_DILATE_PX,
        "YELLOW_MAX_CROSSING_PIXELS": YELLOW_MAX_CROSSING_PIXELS,
        "LANE_SIDE_HOLD_FRAMES": LANE_SIDE_HOLD_FRAMES,
        "USE_RIGHT_LANE_YELLOW_LOCK": USE_RIGHT_LANE_YELLOW_LOCK,
        "RIGHT_LANE_FROM_YELLOW": RIGHT_LANE_FROM_YELLOW,
        "YELLOW_LANE_SIDE": YELLOW_LANE_SIDE,
        "YELLOW_LANE_SEARCH_MARGIN_PX": YELLOW_LANE_SEARCH_MARGIN_PX,
        "YELLOW_MIN_PIXELS_PER_SCANLINE": YELLOW_MIN_PIXELS_PER_SCANLINE,
        "YELLOW_RIGHT_LANE_HOLD_FRAMES": YELLOW_RIGHT_LANE_HOLD_FRAMES,
        "USE_NO_YELLOW_WIDE_BLOB_GATE": USE_NO_YELLOW_WIDE_BLOB_GATE,
        "NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO": NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO,
        "NO_YELLOW_MAX_MEASURED_WIDTH_MM": NO_YELLOW_MAX_MEASURED_WIDTH_MM,
        "ALLOW_NO_YELLOW_BLOB_SPLIT": ALLOW_NO_YELLOW_BLOB_SPLIT,
    }
    for key in merged:
        if key in config:
            merged[key] = config[key]
    return sanitize_config(merged)


def sanitize_config(config):
    clean = dict(config)
    for low_key, high_key, limit in (("H_min", "H_max", 179), ("S_min", "S_max", 255), ("V_min", "V_max", 255)):
        low = int(clip(clean[low_key], 0, limit))
        high = int(clip(clean[high_key], 0, limit))
        if low > high:
            low, high = high, low
        clean[low_key] = low
        clean[high_key] = high
    clean["ROI_top_percent"] = int(clip(clean["ROI_top_percent"], 35, 75))
    clean["Morph_kernel"] = odd_kernel(clean["Morph_kernel"], allow_zero=False)
    clean["Close_kernel"] = odd_kernel(clean["Close_kernel"], allow_zero=True)
    clean["Min_area_percent"] = int(clip(clean["Min_area_percent"], 0, 6))
    clean["MIN_SEGMENT_WIDTH_PX"] = int(clip(clean["MIN_SEGMENT_WIDTH_PX"], 10, 60))
    clean["MAX_CENTER_JUMP_PX"] = int(clip(clean["MAX_CENTER_JUMP_PX"], 50, 160))
    clean["CENTERLINE_SMOOTHING_ALPHA"] = round(float(clip(clean["CENTERLINE_SMOOTHING_ALPHA"], 0.20, 0.70)), 3)
    clean["CENTERLINE_ALPHA"] = round(float(clip(clean["CENTERLINE_ALPHA"], 0.05, 0.60)), 3)
    clean["CURVE_DEADBAND_PX"] = int(clip(clean["CURVE_DEADBAND_PX"], 20, 100))
    clean["CURVE_STRONG_PX"] = int(clip(clean["CURVE_STRONG_PX"], 80, 220))
    clean["CENTER_DEADBAND_PX"] = int(clip(clean["CENTER_DEADBAND_PX"], 15, 80))
    clean["CENTER_STRONG_PX"] = int(clip(clean["CENTER_STRONG_PX"], 70, 180))
    clean["EGO_SEED_SEARCH_RADIUS_PX"] = int(clip(clean["EGO_SEED_SEARCH_RADIUS_PX"], 80, 180))
    clean["EGO_BOTTOM_BAND_PERCENT"] = int(clip(clean["EGO_BOTTOM_BAND_PERCENT"], 8, 30))
    clean["EGO_MIN_COMPONENT_AREA_PERCENT"] = round(float(clip(clean["EGO_MIN_COMPONENT_AREA_PERCENT"], 0.5, 3.0)), 2)
    clean["MIN_VALID_LANE_WIDTH_MM"] = round(float(clip(clean["MIN_VALID_LANE_WIDTH_MM"], 180, 260)), 1)
    clean["MAX_VALID_LANE_WIDTH_MM"] = round(float(clip(clean["MAX_VALID_LANE_WIDTH_MM"], 260, 360)), 1)
    clean["SAFE_STEERING_GAIN"] = round(float(clip(clean["SAFE_STEERING_GAIN"], 0.001, 0.05)), 4)
    clean["YELLOW_H_MIN"] = int(clip(clean["YELLOW_H_MIN"], 0, 179))
    clean["YELLOW_H_MAX"] = int(clip(clean["YELLOW_H_MAX"], 0, 179))
    clean["YELLOW_S_MIN"] = int(clip(clean["YELLOW_S_MIN"], 0, 255))
    clean["YELLOW_S_MAX"] = int(clip(clean["YELLOW_S_MAX"], 0, 255))
    clean["YELLOW_V_MIN"] = int(clip(clean["YELLOW_V_MIN"], 0, 255))
    clean["YELLOW_V_MAX"] = int(clip(clean["YELLOW_V_MAX"], 0, 255))
    clean["YELLOW_BOUNDARY_DILATE_PX"] = int(clip(clean["YELLOW_BOUNDARY_DILATE_PX"], 1, 21))
    clean["YELLOW_MAX_CROSSING_PIXELS"] = int(clip(clean["YELLOW_MAX_CROSSING_PIXELS"], 0, 200))
    clean["LANE_SIDE_HOLD_FRAMES"] = int(clip(clean["LANE_SIDE_HOLD_FRAMES"], 1, 60))
    return clean


def odd_kernel(value, allow_zero):
    value = int(round(float(value)))
    if allow_zero and value <= 0:
        return 0
    value = max(1, value)
    if value % 2 == 0:
        value += 1
    return int(clip(value, 1, 31))


def clip(value, minimum, maximum):
    return min(max(float(value), minimum), maximum)


def build_search_space(seed):
    return {
        "H_min": [int(clip(seed["H_min"] + delta, 0, 179)) for delta in (-5, 0, 5)],
        "H_max": [int(clip(seed["H_max"] + delta, 0, 179)) for delta in (-5, 0, 5)],
        "S_min": [int(clip(seed["S_min"] + delta, 0, 255)) for delta in (-20, 0, 20)],
        "S_max": [int(clip(seed["S_max"] + delta, 0, 255)) for delta in (-30, 0, 30)],
        "V_min": [int(clip(seed["V_min"] + delta, 0, 255)) for delta in (-25, 0, 25)],
        "V_max": [int(clip(seed["V_max"] + delta, 0, 255)) for delta in (-35, 0, 35)],
        "ROI_top_percent": [int(clip(seed["ROI_top_percent"] + delta, 35, 75)) for delta in (-10, 0, 10)],
        "Morph_kernel": [1, 3, 5, 7, 9],
        "Close_kernel": [0, 1, 3, 5, 7, 9, 11],
        "Min_area_percent": [0, 1, 2, 3, 4, 5, 6],
        "MIN_SEGMENT_WIDTH_PX": [10, 20, 30, 40, 50, 60],
        "MAX_CENTER_JUMP_PX": [50, 80, 95, 120, 160],
        "CENTERLINE_SMOOTHING_ALPHA": [0.20, 0.35, 0.45, 0.55, 0.70],
        "EGO_SEED_SEARCH_RADIUS_PX": [80, 120, 150, 180],
        "EGO_MIN_COMPONENT_AREA_PERCENT": [0.5, 1.0, 1.5, 2.0, 3.0],
    }


def generate_candidates(seed, search_space, max_configs, random_seed):
    rng = random.Random(random_seed)
    candidates = [sanitize_config(seed)]
    important_grid = [
        ("V_min", search_space["V_min"]),
        ("V_max", search_space["V_max"]),
        ("ROI_top_percent", search_space["ROI_top_percent"]),
        ("Morph_kernel", search_space["Morph_kernel"]),
        ("Close_kernel", search_space["Close_kernel"]),
    ]
    for key, values in important_grid:
        for value in values:
            candidate = dict(seed)
            candidate[key] = value
            candidates.append(sanitize_config(candidate))

    while len(candidates) < max_configs:
        candidate = dict(seed)
        for key, values in search_space.items():
            if rng.random() < 0.75:
                candidate[key] = rng.choice(values)
        candidates.append(sanitize_config(candidate))

    unique = []
    seen = set()
    for candidate in candidates:
        signature = json.dumps(candidate, sort_keys=True)
        if signature not in seen:
            unique.append(candidate)
            seen.add(signature)
        if len(unique) >= max_configs:
            break
    return unique


def load_sampled_frames(video_path, stride):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % stride == 0:
                frames.append((frame_index, frame_index / fps, app.resize_frame(frame)))
            frame_index += 1
    finally:
        cap.release()
    return frames, fps, total_frames


def score_candidates(candidates, frames, source_fps, time_budget_hours):
    results = []
    started = time.perf_counter()
    deadline = started + time_budget_hours * 3600.0 if time_budget_hours > 0 else None
    best_score = None
    for index, config in enumerate(candidates, start=1):
        metrics = evaluate_config_on_frames(frames, source_fps, config)
        score = metrics["score"]
        best_score = score if best_score is None else max(best_score, score)
        results.append({"candidate_id": index, "score": score, "metrics": metrics, "config": config})
        if index == 1 or index % 10 == 0:
            elapsed = max(0.001, time.perf_counter() - started)
            rate = index / elapsed
            remaining = (len(candidates) - index) / max(rate, 0.001)
            print(f"Auto-tune candidate {index}/{len(candidates)} best={best_score:.2f} rate={rate:.2f}/sec eta={remaining:.1f}s")
        if deadline is not None and time.perf_counter() >= deadline:
            break
    return results


def evaluate_config_on_frames(frames, source_fps, config):
    rows = []
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    previous_turn_hint = "unknown"
    for frame_index, time_sec, frame in frames:
        result = app.detect_road(
            frame,
            config,
            last_center_x,
            frames_since_valid,
            detector_config=config,
            lane_side_memory=lane_side_memory,
        )
        if result.road_detected and result.road_center_x is not None:
            last_center_x = result.road_center_x
            frames_since_valid = 0
        else:
            frames_since_valid += 1
        turn_hint = turn_hint_from_curve(result.curve_error_px, config, previous_turn_hint)
        previous_turn_hint = turn_hint
        rows.append(metrics_row(result, turn_hint))
    duration_sec = (frames[-1][1] - frames[0][1] + (1.0 / max(1.0, source_fps))) if frames else 0.0
    return compute_detector_metrics(rows, duration_sec)


def evaluate_config_on_video(video_path, config):
    frames, fps, _total = load_sampled_frames(video_path, 1)
    return evaluate_config_on_frames(frames, fps, config)


def metrics_row(result, turn_hint):
    height, width = result.mask.shape[:2]
    mask_area_pixels = int(cv2.countNonZero(result.mask))
    return {
        "road_detected": bool(result.road_detected),
        "detection_quality": float(result.detection_quality),
        "valid_scanline_count": int(result.valid_scanline_count),
        "rejected_scanlines": int(result.rejected_scanlines),
        "tracked_center_valid": bool(result.tracked_center_valid),
        "ego_component_found": bool(result.ego_component_found),
        "mask_area_percent": mask_area_pixels / max(1, width * height) * 100.0,
        "road_center_error_px": result.road_center_error_px,
        "curve_error_px": result.curve_error_px,
        "turn_hint": turn_hint,
    }


def turn_hint_from_curve(curve_error_px, config, previous):
    if curve_error_px is None:
        return previous
    deadband = float(config.get("CURVE_DEADBAND_PX", CURVE_DEADBAND_PX))
    if curve_error_px < -deadband:
        return "left"
    if curve_error_px > deadband:
        return "right"
    return "straight"


def create_auto_tune_folders(output_dir, video_path):
    output_root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_root / f"auto_tune_{video_path.stem}_{timestamp}"
    folders = {
        "root": root,
        "human": root / "human_output",
        "ai": root / "output_for_AI",
        "comparison_frames": root / "human_output" / "comparison_frames",
        "frame_samples": root / "output_for_AI" / "frame_samples",
        "failure_frames": root / "output_for_AI" / "failure_frames",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_auto_tune_run.txt").write_text(str(root.resolve()) + "\n", encoding="utf-8")
    return folders


def render_best_config_video(video_path, config, folders, source_fps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    writer = cv2.VideoWriter(
        str(folders["human"] / "best_config_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create best_config_annotated.mp4")

    telemetry_fields = [
        "frame_index", "time_sec", "road_detected", "road_confidence", "road_center_error_px",
        "curve_error_px", "turn_hint", "near_center_x", "far_center_x", "tracked_center_valid",
        "rejected_scanlines", "valid_scanline_count", "detection_quality", "seed_center_x",
        "first_anchor_x", "first_anchor_distance_px", "ego_component_found", "ego_seed_x",
        "ego_seed_y", "ego_anchor_x", "ego_anchor_y", "ego_component_area_pixels",
        "ego_component_area_percent", "ego_component_fallback_used", "selected_path",
        "safe_corridor_valid", "visual_helper_active", "safe_corridor_width_mm",
        "safe_corridor_width_px", "measured_lane_width_mm", "measured_lane_width_px",
        "lane_width_valid", "left_clearance_mm", "right_clearance_mm",
        "corridor_center_error_mm", "corridor_center_error_px", "visual_steering_correction",
        "safe_scanline_count_valid", "safe_corridor_reason", "yellow_boundary_detected",
        "yellow_boundary_pixel_count", "yellow_boundary_enforced", "selected_lane_side",
        "yellow_crossing_pixels", "yellow_right_edge_x", "right_lane_segment_found",
        "right_lane_segment_left_x", "right_lane_segment_right_x", "right_lane_segment_width_px",
        "right_lane_lock_active", "right_lane_lock_reason", "straight_confidence", "left_confidence", "right_confidence", "mask_area_pixels",
        "mask_area_percent", "centerline_point_count", "processing_fps_estimate",
    ]
    event_fields = ["frame_index", "time_sec", "event_type", "old_value", "new_value", "notes"]

    tracker = app.PathConfidenceTracker()
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    previous_state = None
    next_sample_time = 0.0
    failure_count = 0
    frame_index = 0
    started = time.perf_counter()

    with open(folders["ai"] / "best_run_telemetry.csv", "w", newline="", encoding="utf-8") as telemetry_file, open(
        folders["ai"] / "best_run_events.csv", "w", newline="", encoding="utf-8"
    ) as events_file:
        telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry_fields)
        event_writer = csv.DictWriter(events_file, fieldnames=event_fields)
        telemetry_writer.writeheader()
        event_writer.writeheader()
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = app.resize_frame(frame)
                time_sec = frame_index / source_fps
                result = app.detect_road(
                    frame,
                    config,
                    last_center_x,
                    frames_since_valid,
                    detector_config=config,
                    lane_side_memory=lane_side_memory,
                )
                if result.road_detected and result.road_center_x is not None:
                    last_center_x = result.road_center_x
                    frames_since_valid = 0
                else:
                    frames_since_valid += 1
                confidences, selected_path, smoothed_curve_error_px, turn_hint = tracker.update(
                    result.road_center_error_px,
                    result.curve_error_px,
                    result.road_detected,
                )
                debug_frame = app.draw_visualization(frame, result, confidences, selected_path, smoothed_curve_error_px, turn_hint, False)
                overlay = app.build_road_overlay(frame, result)
                writer.write(debug_frame)
                processing_fps = frame_index / max(0.001, time.perf_counter() - started)
                telemetry_writer.writerow(app.build_telemetry_row(frame_index, time_sec, result, confidences, selected_path, turn_hint, processing_fps))
                events, previous_state = app.detect_frame_events(frame_index, time_sec, result, selected_path, turn_hint, previous_state)
                for event_type, old_value, new_value, notes in events:
                    app.write_event_row(event_writer, frame_index, time_sec, event_type, old_value, new_value, notes)
                reason = app.failure_reason_from_events(result, events)
                failure_count, _saved = app.save_failure_frame(debug_frame, folders["failure_frames"], frame_index, reason, failure_count, 100, True)
                next_sample_time = app.save_periodic_ai_samples(frame, result, overlay, debug_frame, folders["frame_samples"], frame_index, time_sec, next_sample_time, 2.0)
                if frame_index in (0, 300, 900):
                    cv2.imwrite(str(folders["comparison_frames"] / f"frame_{frame_index:06d}_debug.jpg"), debug_frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()


def write_scores_csv(path, results):
    fields = [
        "rank", "candidate_id", "score", "road_detected_percent", "average_detection_quality",
        "average_valid_scanline_count", "average_rejected_scanlines", "rejected_scanline_problem_percent",
        "tracked_center_invalid_percent", "ego_component_found_percent", "average_mask_area_percent",
        "mask_area_std", "center_jump_count", "curve_jump_count", "turn_hint_change_count",
        "average_abs_center_delta", "average_abs_curve_delta", "config_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            row = {field: metrics.get(field, "") for field in fields}
            row["rank"] = result.get("rank", "")
            row["candidate_id"] = result["candidate_id"]
            row["score"] = f"{result['score']:.6f}"
            row["config_json"] = json.dumps(result["config"], sort_keys=True)
            writer.writerow(row)


def write_auto_tune_summaries(run_folders, args, video_path, seed_config, sampled_results, top_results, best, best_config_path, search_space, sampled_frames_used, full_eval_top_k):
    warnings = []
    if best["metrics"]["road_detected_percent"] < 80.0:
        warnings.append("Best config still loses road detection often.")
    if best["metrics"]["average_mask_area_percent"] < 5.0:
        warnings.append("Best mask may be too small.")
    if best["metrics"]["average_mask_area_percent"] > 65.0:
        warnings.append("Best mask may be too large.")

    ai_summary = {
        "input_video": str(video_path),
        "seed_config": seed_config,
        "total_candidates_tested": len(sampled_results),
        "sampled_frames_used": sampled_frames_used,
        "full_eval_top_k": full_eval_top_k,
        "best_score": best["score"],
        "best_config_path": str(best_config_path),
        "best_metrics": best["metrics"],
        "top_config_paths": [str(best_config_path)],
        "search_space": search_space,
        "warnings": warnings,
        "recommendations": ["Use best_config.json for the final analysis run."],
    }
    save_json(run_folders["ai"] / "auto_tune_summary_ai.json", ai_summary)

    command = (
        f"py -3.13 main.py --source video --video {video_path} --config {best_config_path} "
        "--no-display --output-dir outputs --save-failure-frames"
    )
    lines = [
        "QCar2 RGB road detector auto-tune summary",
        "",
        "This was OpenCV/NumPy hyperparameter optimization, not ML/RL training.",
        f"Candidates tested: {len(sampled_results)}",
        f"Best score: {best['score']:.3f}",
        f"Best config: {best_config_path}",
        f"Road detected: {best['metrics']['road_detected_percent']:.1f}%",
        f"Average detection quality: {best['metrics']['average_detection_quality']:.3f}",
        f"Average ego mask area: {best['metrics']['average_mask_area_percent']:.2f}%",
        "",
        "Next command:",
        command,
    ]
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend([f"- {warning}" for warning in warnings])
    (run_folders["human"] / "auto_tune_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_folders["human"] / "best_config_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
