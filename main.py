import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    ALLOW_FIRST_ANCHOR_JUMP,
    CENTERLINE_ALPHA,
    CENTERLINE_SMOOTHING_ALPHA,
    CENTER_DEADBAND_PX,
    CENTER_STRONG_PX,
    CONFIDENCE_ALPHA,
    CONFIG_FILE,
    CURVE_DEADBAND_PX,
    CURVE_STRONG_PX,
    DEFAULT_SETTINGS,
    EGO_BOTTOM_BAND_PERCENT,
    EGO_MIN_COMPONENT_AREA_PERCENT,
    EGO_SEED_SEARCH_RADIUS_PX,
    EGO_SEED_X_RATIO,
    EGO_SEED_Y_RATIO,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LAST_CENTER_HOLD_FRAMES,
    MAX_CENTER_JUMP_PX,
    MIN_SEGMENT_WIDTH_PX,
    REALSENSE_FPS,
    REALSENSE_HEIGHT,
    REALSENSE_WIDTH,
    SCANLINE_COUNT,
    SELECT_CONFIDENCE,
    SELECT_MARGIN,
    TRACKBAR_RANGES,
    USE_EGO_CONNECTED_MASK,
    WINDOW_MAIN,
    WINDOW_MASK,
    WINDOW_TUNING,
)

WINDOW_VIDEO_CONTROL = "Video Control"
MANUAL_TUNING_NOTE = "Manual video tuning baseline for future auto-tuning"
AUTO_TUNING_NOTE = (
    "This config should be used as the center/seed for the future auto-tuning search. "
    "Auto-tuning should explore small ranges around these values first."
)
OPTIONAL_TRACKBAR_RANGES = {
    # Planning-mask controls are not used yet. If they are added to
    # DEFAULT_SETTINGS later, these ranges let manual tuning expose them
    # without breaking older configs that do not contain the keys.
    "Planning_close_w": 101,
    "Planning_close_h": 101,
    "Planning_dilate_iterations": 10,
}


@dataclass
class RoadResult:
    raw_mask: np.ndarray
    mask: np.ndarray
    road_detected: bool
    road_confidence: float
    road_center_x: float | None
    road_center_error_px: float | None
    curve_error_px: float | None
    near_center_x: int | None
    far_center_x: int | None
    tracked_center_valid: bool
    rejected_scanlines: int
    valid_scanline_count: int
    detection_quality: float
    seed_center_x: float
    first_anchor_x: float | None
    first_anchor_distance_px: float | None
    ego_component_found: bool
    ego_seed_x: int
    ego_seed_y: int
    ego_anchor_x: int | None
    ego_anchor_y: int | None
    ego_component_area_pixels: int
    ego_component_area_percent: float
    ego_component_fallback_used: bool
    scan_points: list[tuple[int, int, int, int]]
    boundary_points: list[tuple[int, int]]


class ImageSource:
    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class StaticImageSource(ImageSource):
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.frame = cv2.imread(image_path)
        if self.frame is None:
            raise RuntimeError(f"Could not load image: {image_path}")

    def read(self):
        return True, self.frame.copy()


class WebcamSource(ImageSource):
    def __init__(self, camera_index: int):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RealSenseSource(ImageSource):
    def __init__(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install it with "
                "'py -3.12 -m pip install pyrealsense2', or use "
                "'--source webcam' / '--source image'."
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color,
            REALSENSE_WIDTH,
            REALSENSE_HEIGHT,
            rs.format.rgb8,
            REALSENSE_FPS,
        )
        try:
            self.pipeline.start(self.config)
        except Exception as exc:
            raise RuntimeError(f"Could not start RealSense color stream: {exc}") from exc

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None

        rgb = np.asanyarray(color_frame.get_data())
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self):
        self.pipeline.stop()


class PathConfidenceTracker:
    def __init__(self):
        self.confidences = {"left": 0.33, "straight": 0.34, "right": 0.33}
        self.smoothed_curve_error_px = None

    def update(self, center_error_px: float | None, curve_error_px: float | None, road_detected: bool):
        target = {"left": 0.33, "straight": 0.34, "right": 0.33}

        if road_detected and curve_error_px is not None:
            curve = float(curve_error_px)
            if self.smoothed_curve_error_px is None:
                self.smoothed_curve_error_px = curve
            else:
                self.smoothed_curve_error_px = (
                    (1.0 - CENTERLINE_ALPHA) * self.smoothed_curve_error_px
                    + CENTERLINE_ALPHA * curve
                )
        elif not road_detected:
            self.smoothed_curve_error_px = None

        if road_detected and self.smoothed_curve_error_px is not None:
            curve = self.smoothed_curve_error_px
            abs_curve = abs(curve)

            # curve_error_px is the far road center minus the near road center.
            # Negative means the visible road bends left; positive means it bends right.
            if abs_curve <= CURVE_DEADBAND_PX:
                target = {"left": 0.07, "straight": 0.86, "right": 0.07}
            elif curve < -CURVE_STRONG_PX:
                target = {"left": 0.86, "straight": 0.08, "right": 0.06}
            elif curve > CURVE_STRONG_PX:
                target = {"left": 0.06, "straight": 0.08, "right": 0.86}
            elif curve < 0:
                amount = min(1.0, (abs_curve - CURVE_DEADBAND_PX) / max(1, CURVE_STRONG_PX - CURVE_DEADBAND_PX))
                target = {
                    "left": 0.38 + 0.36 * amount,
                    "straight": 0.48 - 0.26 * amount,
                    "right": 0.14 - 0.10 * amount,
                }
            else:
                amount = min(1.0, (abs_curve - CURVE_DEADBAND_PX) / max(1, CURVE_STRONG_PX - CURVE_DEADBAND_PX))
                target = {
                    "left": 0.14 - 0.10 * amount,
                    "straight": 0.48 - 0.26 * amount,
                    "right": 0.38 + 0.36 * amount,
                }

            if center_error_px is not None:
                target = self.apply_lateral_correction(target, center_error_px)

        for name in self.confidences:
            old_value = self.confidences[name]
            self.confidences[name] = (1.0 - CONFIDENCE_ALPHA) * old_value + CONFIDENCE_ALPHA * target[name]

        total = sum(self.confidences.values())
        if total > 0:
            for name in self.confidences:
                self.confidences[name] /= total

        return self.confidences.copy(), self.selected_path(), self.smoothed_curve_error_px, self.turn_hint()

    def apply_lateral_correction(self, target, center_error_px):
        # road_center_error_px is the detected road center minus the image center.
        # Negative means the drivable corridor is left of the camera; positive means right.
        correction = min(0.12, abs(center_error_px) / max(1, CENTER_STRONG_PX) * 0.12)
        if abs(center_error_px) <= CENTER_DEADBAND_PX:
            return target
        if center_error_px < 0:
            target["left"] += correction
            target["right"] = max(0.02, target["right"] - correction)
        else:
            target["right"] += correction
            target["left"] = max(0.02, target["left"] - correction)

        total = sum(target.values())
        return {name: value / total for name, value in target.items()}

    def turn_hint(self):
        if self.smoothed_curve_error_px is None:
            return "unknown"
        if self.smoothed_curve_error_px < -CURVE_DEADBAND_PX:
            return "left"
        if self.smoothed_curve_error_px > CURVE_DEADBAND_PX:
            return "right"
        return "straight"

    def selected_path(self):
        ordered = sorted(self.confidences.items(), key=lambda item: item[1], reverse=True)
        best_name, best_value = ordered[0]
        second_value = ordered[1][1]
        if best_value > SELECT_CONFIDENCE and best_value - second_value >= SELECT_MARGIN:
            return best_name
        return "none"


def parse_args():
    parser = argparse.ArgumentParser(description="RGB-only QCar2 road/drivable-area detection prototype.")
    parser.add_argument("--source", choices=["image", "webcam", "realsense", "video"], required=True)
    parser.add_argument("--image", help="Path to a static image for --source image.")
    parser.add_argument("--video", help="Path to a video file for --source video.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for --source webcam.")
    parser.add_argument("--output-dir", default="outputs", help="Folder where video-mode outputs are written.")
    parser.add_argument("--no-display", action="store_true", help="Process video without opening OpenCV windows.")
    parser.add_argument(
        "--tune-video",
        action="store_true",
        help="Open an interactive manual tuning mode for --source video instead of full analysis.",
    )
    parser.add_argument("--auto-tune", action="store_true", help="Run offline OpenCV/NumPy hyperparameter search.")
    parser.add_argument("--seed-config", default="configs/manual_tuned_config.json", help="Config JSON used as the auto-tune search center.")
    parser.add_argument("--quick", action="store_true", help="Use fewer candidates and wider frame stride for a fast auto-tune smoke test.")
    parser.add_argument("--max-configs", type=int, default=500, help="Maximum candidate configs for sampled-frame auto-tuning.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of sampled-frame top configs to save.")
    parser.add_argument("--sample-stride", type=int, default=10, help="Process every Nth frame during sampled auto-tune search.")
    parser.add_argument("--full-eval-top-k", type=int, default=5, help="Number of top configs to evaluate on the full video.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible candidate generation.")
    parser.add_argument("--auto-tune-time-budget-hours", type=float, default=0.0, help="Optional wall-clock budget. 0 means use --max-configs.")
    parser.add_argument(
        "--use-default-config",
        action="store_true",
        help="Use DEFAULT_SETTINGS from config.py for video mode and ignore road_config.json.",
    )
    parser.add_argument(
        "--config",
        help="Config JSON to load. Video analysis defaults to road_config.json when it exists; tuning mode loads only explicit paths.",
    )
    parser.add_argument("--config-output", default="configs/manual_tuned_config.json", help="Where manual video tuning saves the baseline config.")
    parser.add_argument("--session-output", default="configs/manual_tuning_session.json", help="Where manual video tuning saves session notes and sample frame lists.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index where manual video tuning starts.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Manual tuning playback speed multiplier.")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete only the new timestamped run folder before writing if it already exists.",
    )
    parser.add_argument(
        "--save-failure-frames",
        action="store_true",
        help="Save debug images for suspicious video frames, up to --max-failure-frames.",
    )
    parser.add_argument(
        "--max-failure-frames",
        type=int,
        default=100,
        help="Maximum suspicious-frame debug images to save when --save-failure-frames is used.",
    )
    parser.add_argument(
        "--ai-sample-interval-sec",
        type=float,
        default=2.0,
        help="Seconds between periodic AI frame samples in video mode.",
    )
    return parser.parse_args()


def print_startup(args):
    print("QCar2 RGB Road Detector")
    print("-----------------------")
    print(f"Source: {args.source}")
    print("Keys: q/ESC quit | s save HSV | l load HSV | r reset HSV | m mask | p pause | c candidates")
    print("Tune HSV until road is white in the mask and non-road is black.")
    print()


def make_source(args):
    if args.source == "image":
        if not args.image:
            raise RuntimeError("--image is required when using --source image")
        return StaticImageSource(args.image)
    if args.source == "webcam":
        return WebcamSource(args.camera_index)
    if args.source == "video":
        if not args.video:
            raise RuntimeError("--video is required when using --source video")
        return None
    return RealSenseSource()


def resize_frame(frame):
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def create_trackbars(settings):
    cv2.namedWindow(WINDOW_TUNING, cv2.WINDOW_NORMAL)
    for name, default_value in settings.items():
        max_value = TRACKBAR_RANGES.get(name, OPTIONAL_TRACKBAR_RANGES.get(name))
        if max_value is None:
            continue
        cv2.createTrackbar(name, WINDOW_TUNING, int(default_value), max_value, nothing)


def nothing(_value):
    pass


def get_trackbar_settings():
    settings = {}
    for name in DEFAULT_SETTINGS:
        try:
            settings[name] = cv2.getTrackbarPos(name, WINDOW_TUNING)
        except cv2.error:
            settings[name] = DEFAULT_SETTINGS[name]

    if settings["H_min"] > settings["H_max"]:
        settings["H_min"], settings["H_max"] = settings["H_max"], settings["H_min"]
    if settings["S_min"] > settings["S_max"]:
        settings["S_min"], settings["S_max"] = settings["S_max"], settings["S_min"]
    if settings["V_min"] > settings["V_max"]:
        settings["V_min"], settings["V_max"] = settings["V_max"], settings["V_min"]

    # OpenCV morphology kernels must be odd. Morph_kernel is always active,
    # while Close_kernel may be 0 to skip closing completely.
    morph_kernel = max(1, settings["Morph_kernel"])
    if morph_kernel % 2 == 0:
        morph_kernel += 1
    settings["Morph_kernel"] = morph_kernel

    close_kernel = max(0, settings["Close_kernel"])
    if close_kernel > 0 and close_kernel % 2 == 0:
        close_kernel += 1
    settings["Close_kernel"] = close_kernel
    settings["ROI_top_percent"] = min(max(settings["ROI_top_percent"], 0), 80)
    settings["Min_area_percent"] = min(max(settings["Min_area_percent"], 0), 30)
    return settings


def set_trackbars(settings):
    for name, value in settings.items():
        try:
            cv2.setTrackbarPos(name, WINDOW_TUNING, int(value))
        except cv2.error:
            pass


def save_settings(settings):
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
    print(f"Saved HSV settings to {CONFIG_FILE}")


def load_settings():
    if not os.path.exists(CONFIG_FILE):
        print(f"No {CONFIG_FILE} found yet.")
        return None

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    settings = DEFAULT_SETTINGS.copy()
    for name in settings:
        if name in loaded:
            settings[name] = int(loaded[name])
    print(f"Loaded HSV settings from {CONFIG_FILE}")
    return settings


def clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def cfg_value(detector_config, name, default):
    # Auto-tuning passes candidate values in a runtime dict. Normal live/video
    # modes leave detector_config as None and use config.py defaults.
    if detector_config is not None and name in detector_config:
        return detector_config[name]
    return default


def cfg_int(detector_config, name, default):
    return int(round(float(cfg_value(detector_config, name, default))))


def cfg_float(detector_config, name, default):
    return float(cfg_value(detector_config, name, default))


def detect_road(frame, settings, last_center_x, frames_since_valid, use_ego_connected_mask=None, detector_config=None):
    height, width = frame.shape[:2]
    roi_top = int(height * settings["ROI_top_percent"] / 100.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([settings["H_min"], settings["S_min"], settings["V_min"]], dtype=np.uint8)
    upper = np.array([settings["H_max"], settings["S_max"], settings["V_max"]], dtype=np.uint8)
    full_mask = cv2.inRange(hsv, lower, upper)

    # Ignore the upper image area; it usually contains walls, signs, and other distractions.
    full_mask[:roi_top, :] = 0

    kernel_size = settings["Morph_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    # Closing fills small black holes inside the road mask. Set Close_kernel to
    # 0 to skip this step when closing over-fills the tuned live road shape.
    close_kernel_size = settings["Close_kernel"]
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, close_kernel)

    raw_mask = full_mask.copy()
    mask, ego_debug = keep_relevant_component(full_mask, settings, use_ego_connected_mask, detector_config)
    area = int(cv2.countNonZero(mask))
    roi_area = max(1, width * (height - roi_top))
    min_area = roi_area * settings["Min_area_percent"] / 100.0
    scan_points, boundary_points, rejected_scanlines, seed_center_x, first_anchor_x, first_anchor_distance_px = (
        estimate_scanline_centers(mask, roi_top, last_center_x, frames_since_valid, detector_config)
    )
    tracked_center_valid = len(scan_points) >= 2

    road_detected = area > 0 and area >= min_area and tracked_center_valid
    mask_area_percent = area / roi_area * 100.0
    # mask_area_score compares the white road mask to a practical 20% ROI
    # reference area. This keeps confidence useful even when Min_area_percent
    # is very small or zero in an experimental config.
    mask_area_score = clamp(mask_area_percent / 20.0, 0.0, 1.0)
    valid_scanline_count = len(scan_points)
    # valid_scanline_score measures how many of the planned horizontal scan
    # rows found a usable road segment.
    valid_scanline_score = clamp(valid_scanline_count / SCANLINE_COUNT, 0.0, 1.0)
    # rejected_scanline_score rewards masks that do not force the tracker to
    # reject many scan rows because of missing road or impossible center jumps.
    rejected_scanline_score = clamp(1.0 - (rejected_scanlines / SCANLINE_COUNT), 0.0, 1.0)
    detection_quality = (
        0.35 * mask_area_score
        + 0.45 * valid_scanline_score
        + 0.20 * rejected_scanline_score
    )
    road_confidence = detection_quality

    road_center_x = None
    road_center_error_px = None
    curve_error_px = None
    near_center_x = None
    far_center_x = None
    if road_detected and scan_points:
        # Weight lower scanlines more because they are closer to the vehicle and more reliable.
        weights = np.linspace(2.0, 1.0, num=len(scan_points))
        centers = np.array([point[0] for point in scan_points], dtype=np.float32)
        road_center_x = float(np.average(centers, weights=weights))

        # road_center_error_px is lateral error: road center minus camera/image center.
        # Negative means the road is left of the camera; positive means it is right.
        road_center_error_px = road_center_x - (width / 2.0)

        near_center_x = scan_points[0][0]
        far_center_x = scan_points[-1][0]
        if len(scan_points) >= 2:
            # curve_error_px compares far road center to near road center.
            # Negative means the road center moves left as it goes forward.
            # Positive means it moves right; near zero means mostly straight.
            curve_error_px = float(far_center_x - near_center_x)
    elif last_center_x is not None and frames_since_valid <= LAST_CENTER_HOLD_FRAMES:
        road_center_x = last_center_x
        road_center_error_px = road_center_x - (width / 2.0)

    return RoadResult(
        raw_mask=raw_mask,
        mask=mask,
        road_detected=road_detected,
        road_confidence=road_confidence,
        road_center_x=road_center_x,
        road_center_error_px=road_center_error_px,
        curve_error_px=curve_error_px,
        near_center_x=near_center_x,
        far_center_x=far_center_x,
        tracked_center_valid=tracked_center_valid,
        rejected_scanlines=rejected_scanlines,
        valid_scanline_count=valid_scanline_count,
        detection_quality=detection_quality,
        seed_center_x=seed_center_x,
        first_anchor_x=first_anchor_x,
        first_anchor_distance_px=first_anchor_distance_px,
        ego_component_found=ego_debug["ego_component_found"],
        ego_seed_x=ego_debug["ego_seed_x"],
        ego_seed_y=ego_debug["ego_seed_y"],
        ego_anchor_x=ego_debug["ego_anchor_x"],
        ego_anchor_y=ego_debug["ego_anchor_y"],
        ego_component_area_pixels=ego_debug["ego_component_area_pixels"],
        ego_component_area_percent=ego_debug["ego_component_area_percent"],
        ego_component_fallback_used=ego_debug["ego_component_fallback_used"],
        scan_points=scan_points,
        boundary_points=boundary_points,
    )


def make_ego_debug(mask, found=False, anchor_x=None, anchor_y=None, area_pixels=0, fallback_used=False):
    height, width = mask.shape[:2]
    seed_x = int(round(width * EGO_SEED_X_RATIO))
    seed_y = int(round(height * EGO_SEED_Y_RATIO))
    seed_x = min(max(0, seed_x), width - 1)
    seed_y = min(max(0, seed_y), height - 1)
    return {
        "ego_component_found": bool(found),
        "ego_seed_x": seed_x,
        "ego_seed_y": seed_y,
        "ego_anchor_x": anchor_x,
        "ego_anchor_y": anchor_y,
        "ego_component_area_pixels": int(area_pixels),
        "ego_component_area_percent": area_pixels / max(1, width * height) * 100.0,
        "ego_component_fallback_used": bool(fallback_used),
    }


def select_ego_connected_component(mask, settings, detector_config=None):
    # HSV sees every road-colored pixel. This filter keeps only the white
    # component connected to the ego/start area near the bottom-center, where
    # the car's immediate drivable road should appear in the camera image.
    height, width = mask.shape[:2]
    seed_x = int(round(width * cfg_float(detector_config, "EGO_SEED_X_RATIO", EGO_SEED_X_RATIO)))
    seed_y = int(round(height * cfg_float(detector_config, "EGO_SEED_Y_RATIO", EGO_SEED_Y_RATIO)))
    seed_x = min(max(0, seed_x), width - 1)
    seed_y = min(max(0, seed_y), height - 1)

    anchor = None
    if mask[seed_y, seed_x] > 0:
        anchor = (seed_x, seed_y)
    else:
        anchor = find_nearest_ego_anchor(mask, seed_x, seed_y, detector_config)

    if anchor is None:
        return None, make_ego_debug(mask)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    anchor_x, anchor_y = anchor
    label = labels[anchor_y, anchor_x]
    if label <= 0 or label >= num_labels:
        return None, make_ego_debug(mask)

    area_pixels = int(stats[label, cv2.CC_STAT_AREA])
    min_area = width * height * cfg_float(detector_config, "EGO_MIN_COMPONENT_AREA_PERCENT", EGO_MIN_COMPONENT_AREA_PERCENT) / 100.0
    if area_pixels < min_area:
        return None, make_ego_debug(mask)

    selected = np.where(labels == label, 255, 0).astype(np.uint8)
    debug = make_ego_debug(mask, True, int(anchor_x), int(anchor_y), area_pixels, False)
    return selected, debug


def find_nearest_ego_anchor(mask, seed_x, seed_y, detector_config=None):
    height, width = mask.shape[:2]
    radius = max(1, cfg_int(detector_config, "EGO_SEED_SEARCH_RADIUS_PX", EGO_SEED_SEARCH_RADIUS_PX))
    x1 = max(0, seed_x - radius)
    x2 = min(width - 1, seed_x + radius)
    y1 = max(0, seed_y - radius)
    y2 = min(height - 1, seed_y + radius)
    search = mask[y1 : y2 + 1, x1 : x2 + 1]
    points = cv2.findNonZero(search)
    if points is None:
        return None

    bottom_band_percent = cfg_float(detector_config, "EGO_BOTTOM_BAND_PERCENT", EGO_BOTTOM_BAND_PERCENT)
    bottom_band_y = int(height * (1.0 - bottom_band_percent / 100.0))
    best_score = None
    best_anchor = None
    for point in points.reshape(-1, 2):
        x = int(point[0]) + x1
        y = int(point[1]) + y1
        dx = x - seed_x
        dy = y - seed_y
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > radius:
            continue
        # Lower pixels are closer to the vehicle. Give them a small bonus so
        # a nearby side blob higher in the image is less likely to become the
        # ego anchor.
        bottom_bonus = 35.0 if y >= bottom_band_y else 0.0
        score = distance - bottom_bonus
        if best_score is None or score < best_score:
            best_score = score
            best_anchor = (x, y)
    return best_anchor


def keep_relevant_component(mask, settings, use_ego_connected_mask=None, detector_config=None):
    if use_ego_connected_mask is None:
        use_ego_connected_mask = USE_EGO_CONNECTED_MASK

    if use_ego_connected_mask:
        ego_mask, ego_debug = select_ego_connected_component(mask, settings, detector_config)
        if ego_mask is not None:
            return ego_mask, ego_debug
    else:
        ego_debug = make_ego_debug(mask)

    fallback_mask = keep_relevant_component_by_score(mask, settings)
    ego_debug["ego_component_fallback_used"] = True
    return fallback_mask, ego_debug


def keep_relevant_component_by_score(mask, settings):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    height, width = mask.shape[:2]
    image_center_x = width / 2.0
    bottom_y = height - 1
    min_area = width * height * settings["Min_area_percent"] / 100.0

    best_label = None
    best_score = -1.0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]

        touches_near_bottom = y + h > height * 0.82
        center_distance = abs(cx - image_center_x) / max(1.0, width / 2.0)
        bottom_bonus = 1.5 if touches_near_bottom else 0.0

        # Prefer a large connected road area that reaches near the bottom-center of the image.
        score = area / 1000.0 + bottom_bonus - center_distance
        if y + h >= bottom_y - 3 and x <= image_center_x <= x + w:
            score += 1.0

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask)

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def estimate_scanline_centers(mask, roi_top, last_center_x=None, frames_since_valid=LAST_CENTER_HOLD_FRAMES + 1, detector_config=None):
    height, width = mask.shape[:2]
    y_values = np.linspace(height - 35, max(roi_top + 10, int(height * 0.52)), SCANLINE_COUNT).astype(int)
    min_segment_width_px = cfg_int(detector_config, "MIN_SEGMENT_WIDTH_PX", MIN_SEGMENT_WIDTH_PX)
    max_center_jump_px = cfg_int(detector_config, "MAX_CENTER_JUMP_PX", MAX_CENTER_JUMP_PX)
    allow_first_anchor_jump = bool(cfg_value(detector_config, "ALLOW_FIRST_ANCHOR_JUMP", ALLOW_FIRST_ANCHOR_JUMP))

    raw_points = []
    boundary_points = []
    rejected_scanlines = 0
    if last_center_x is not None and frames_since_valid <= LAST_CENTER_HOLD_FRAMES:
        seed_center_x = float(last_center_x)
    else:
        seed_center_x = width / 2.0
    previous_center_x = seed_center_x
    first_anchor_x = None
    first_anchor_distance_px = None

    for y in y_values:
        segments = find_road_segments(mask[y, :], min_segment_width_px)
        if not segments:
            rejected_scanlines += 1
            continue

        # Using the full row's leftmost and rightmost white pixel fails at
        # curves/intersections because it averages across every visible road
        # branch. Instead, each white run is a candidate corridor, and we keep
        # the run whose center is closest to the previous tracked center.
        best_segment = min(segments, key=lambda segment: abs(segment[2] - previous_center_x))
        left_x, right_x, center_x = best_segment
        center_jump = center_x - previous_center_x

        if first_anchor_x is None:
            first_anchor_x = float(center_x)
            first_anchor_distance_px = float(abs(center_x - seed_center_x))
            if not allow_first_anchor_jump and abs(center_jump) > max_center_jump_px:
                rejected_scanlines += 1
                first_anchor_x = None
                first_anchor_distance_px = None
                continue
            raw_points.append((float(center_x), int(left_x), int(right_x), int(y)))
            boundary_points.append((left_x, int(y)))
            boundary_points.append((right_x, int(y)))
            previous_center_x = center_x
            continue

        # Reject large jumps so the path does not snap across a wide
        # intersection to a different road branch in one scanline.
        if abs(center_jump) > max_center_jump_px:
            rejected_scanlines += 1
            continue

        raw_points.append((float(center_x), int(left_x), int(right_x), int(y)))
        boundary_points.append((left_x, int(y)))
        boundary_points.append((right_x, int(y)))
        previous_center_x = center_x

    scan_points = smooth_centerline_points(raw_points, detector_config)

    # Preserve bottom-to-top order for drawing a path from the vehicle forward.
    return scan_points, boundary_points, rejected_scanlines, seed_center_x, first_anchor_x, first_anchor_distance_px


def find_road_segments(row_mask, min_width_px):
    segments = []
    in_segment = False
    start_x = 0

    for x, value in enumerate(row_mask):
        if value > 0 and not in_segment:
            start_x = x
            in_segment = True
        elif value == 0 and in_segment:
            add_segment_if_wide_enough(segments, start_x, x - 1, min_width_px)
            in_segment = False

    if in_segment:
        add_segment_if_wide_enough(segments, start_x, len(row_mask) - 1, min_width_px)

    return segments


def add_segment_if_wide_enough(segments, left_x, right_x, min_width_px):
    width = right_x - left_x + 1
    if width < min_width_px:
        return
    center_x = (left_x + right_x) / 2.0
    segments.append((int(left_x), int(right_x), float(center_x)))


def smooth_centerline_points(raw_points, detector_config=None):
    if not raw_points:
        return []

    smoothed_points = []
    smoothed_center_x = raw_points[0][0]

    for center_x, left_x, right_x, y in raw_points:
        # Exponential smoothing keeps the centerline from twitching when mask
        # edges are noisy, while still allowing gradual curves to show up.
        alpha = cfg_float(detector_config, "CENTERLINE_SMOOTHING_ALPHA", CENTERLINE_SMOOTHING_ALPHA)
        smoothed_center_x = ((1.0 - alpha) * smoothed_center_x + alpha * center_x)
        smoothed_points.append((int(round(smoothed_center_x)), left_x, right_x, y))

    return smoothed_points


def make_candidates(width, height):
    start = np.array([width // 2, height - 20], dtype=np.int32)
    return {
        "left": np.array(
            [
                start,
                [int(width * 0.46), int(height * 0.78)],
                [int(width * 0.35), int(height * 0.62)],
                [int(width * 0.24), int(height * 0.48)],
            ],
            dtype=np.int32,
        ),
        "straight": np.array(
            [
                start,
                [int(width * 0.50), int(height * 0.78)],
                [int(width * 0.50), int(height * 0.62)],
                [int(width * 0.50), int(height * 0.46)],
            ],
            dtype=np.int32,
        ),
        "right": np.array(
            [
                start,
                [int(width * 0.54), int(height * 0.78)],
                [int(width * 0.65), int(height * 0.62)],
                [int(width * 0.76), int(height * 0.48)],
            ],
            dtype=np.int32,
        ),
    }


def draw_visualization(frame, result, confidences, selected_path, smoothed_curve_error_px, turn_hint, show_candidates):
    output = frame.copy()
    height, width = output.shape[:2]

    road_color = np.zeros_like(output)
    road_color[:, :] = (180, 120, 0)
    output = np.where(result.mask[:, :, None] > 0, cv2.addWeighted(output, 0.55, road_color, 0.45, 0), output)

    for left_x, y in result.boundary_points:
        cv2.circle(output, (left_x, y), 4, (0, 255, 255), -1)

    if show_candidates:
        draw_candidate_paths(output, confidences, selected_path)

    draw_detected_centerline(output, result)
    draw_ego_anchor_debug(output, result)
    cv2.line(output, (width // 2, height), (width // 2, int(height * 0.45)), (255, 255, 255), 1)

    draw_debug_text(output, result, confidences, selected_path, smoothed_curve_error_px, turn_hint)
    return output


def draw_ego_anchor_debug(output, result):
    # White marks the expected ego seed near the front/bottom of the image.
    # Cyan/magenta marks the actual white road pixel used to choose the
    # connected component, which makes side-blob mistakes easy to inspect.
    cv2.circle(output, (int(result.ego_seed_x), int(result.ego_seed_y)), 6, (255, 255, 255), 2)
    if result.ego_anchor_x is not None and result.ego_anchor_y is not None:
        color = (255, 0, 255) if result.ego_component_fallback_used else (255, 255, 0)
        cv2.circle(output, (int(result.ego_anchor_x), int(result.ego_anchor_y)), 6, color, -1)
        cv2.line(
            output,
            (int(result.ego_seed_x), int(result.ego_seed_y)),
            (int(result.ego_anchor_x), int(result.ego_anchor_y)),
            color,
            1,
            cv2.LINE_AA,
        )


def draw_candidate_paths(output, confidences, selected_path):
    height, width = output.shape[:2]
    candidates = make_candidates(width, height)
    for name, points in candidates.items():
        confidence = confidences[name]
        if name == selected_path:
            color = (80, 220, 255)
            thickness = 3
        else:
            brightness = int(45 + 80 * confidence)
            color = (brightness, brightness, 150)
            thickness = 2
        cv2.polylines(output, [points], False, color, thickness, cv2.LINE_AA)
        cv2.circle(output, tuple(points[-1]), 4, color, -1)


def draw_detected_centerline(output, result):
    if not result.scan_points:
        return

    height, width = output.shape[:2]
    centerline_points = [(int(center_x), int(y)) for center_x, _left_x, _right_x, y in result.scan_points]
    path_points = [(width // 2, height - 8)] + centerline_points

    if len(path_points) >= 2:
        path_array = np.array(path_points, dtype=np.int32)
        # Draw a dark outline first so the real detected path stays readable
        # over both the camera image and the semi-transparent road overlay.
        cv2.polylines(output, [path_array], False, (0, 70, 70), 9, cv2.LINE_AA)
        cv2.polylines(output, [path_array], False, (0, 255, 220), 5, cv2.LINE_AA)

        # Put the arrow head on the final segment so it follows the scanline
        # center path instead of pointing to a single averaged center.
        cv2.arrowedLine(
            output,
            tuple(path_array[-2]),
            tuple(path_array[-1]),
            (0, 255, 220),
            5,
            cv2.LINE_AA,
            tipLength=0.35,
        )

    for x, y in centerline_points:
        cv2.circle(output, (x, y), 5, (0, 255, 0), -1)


def draw_debug_text(output, result, confidences, selected_path, smoothed_curve_error_px, turn_hint):
    error = result.road_center_error_px
    error_text = "None" if error is None else f"{error:.1f}"
    curve_text = "None" if smoothed_curve_error_px is None else f"{smoothed_curve_error_px:.1f}"
    near_text = "None" if result.near_center_x is None else str(result.near_center_x)
    far_text = "None" if result.far_center_x is None else str(result.far_center_x)
    lines = [
        f"road_detected: {result.road_detected}",
        f"road_confidence: {result.road_confidence:.2f}",
        f"road_center_error_px: {error_text}",
        f"curve_error_px: {curve_text}",
        f"turn_hint: {turn_hint}",
        f"near_center_x: {near_text}",
        f"far_center_x: {far_text}",
        f"tracked_center_valid: {result.tracked_center_valid}",
        f"rejected_scanlines: {result.rejected_scanlines}",
        f"ego_found: {result.ego_component_found}",
        f"ego_area_pct: {result.ego_component_area_percent:.2f}",
        f"ego_fallback: {result.ego_component_fallback_used}",
        f"selected_path: {selected_path}",
        f"straight_confidence: {confidences['straight']:.2f}",
        f"left_confidence: {confidences['left']:.2f}",
        f"right_confidence: {confidences['right']:.2f}",
    ]

    x = 12
    y = 28
    line_height = 26
    panel_width = 360
    panel_height = line_height * len(lines) + 12
    overlay = output.copy()
    cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    for line in lines:
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_height


def build_display_grid(original, result, visualization):
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    road_overlay = original.copy()
    mask_pixels = result.mask > 0
    if np.any(mask_pixels):
        road_overlay[mask_pixels] = cv2.addWeighted(
            road_overlay[mask_pixels],
            0.45,
            np.full_like(road_overlay[mask_pixels], (180, 120, 0)),
            0.55,
            0,
        )

    top = np.hstack([label_image(original, "Original RGB"), label_image(mask_bgr, "Road Mask")])
    bottom = np.hstack([label_image(road_overlay, "Road Overlay"), label_image(visualization, "Detected Center Path")])
    grid = np.vstack([top, bottom])
    return cv2.resize(grid, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def label_image(image, label):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (250, 34), (0, 0, 0), -1)
    cv2.putText(labeled, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def handle_key(key, settings):
    if key in (ord("q"), 27):
        return "quit"
    if key == ord("s"):
        save_settings(settings)
    elif key == ord("l"):
        loaded = load_settings()
        if loaded:
            set_trackbars(loaded)
    elif key == ord("r"):
        set_trackbars(DEFAULT_SETTINGS)
        print("Reset HSV settings to defaults.")
    elif key == ord("m"):
        return "toggle_mask"
    elif key == ord("p"):
        return "toggle_pause"
    elif key == ord("c"):
        return "toggle_candidates"
    return None


def get_output_base_name(video_path):
    return Path(video_path).stem


def create_output_folders(output_dir, base_name, clean_output=False):
    # Video output is split in two on purpose:
    # human_output is for a person to watch, while output_for_AI is organized
    # as structured CSV/JSON/text/images for later ChatGPT analysis.
    output_root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_root / f"{base_name}_{timestamp}"
    if root.exists() and clean_output:
        shutil.rmtree(root)
    elif root.exists():
        suffix = 1
        while (output_root / f"{base_name}_{timestamp}_{suffix:02d}").exists():
            suffix += 1
        root = output_root / f"{base_name}_{timestamp}_{suffix:02d}"
    human_dir = root / "human_output"
    ai_dir = root / "output_for_AI"
    key_frames_dir = human_dir / "key_frames"
    frame_samples_dir = ai_dir / "frame_samples"
    failure_frames_dir = ai_dir / "failure_frames"

    for folder in (human_dir, ai_dir, key_frames_dir, frame_samples_dir, failure_frames_dir):
        folder.mkdir(parents=True, exist_ok=True)

    output_root.mkdir(parents=True, exist_ok=True)
    latest_run_path = output_root / "latest_run.txt"
    latest_run_path.write_text(str(root.resolve()) + "\n", encoding="utf-8")

    return {
        "root": root,
        "latest_run": latest_run_path,
        "human": human_dir,
        "ai": ai_dir,
        "key_frames": key_frames_dir,
        "frame_samples": frame_samples_dir,
        "failure_frames": failure_frames_dir,
    }


def load_settings_file(path):
    with open(path, "r", encoding="utf-8") as file:
        loaded = json.load(file)
    if "config" in loaded and isinstance(loaded["config"], dict):
        loaded = loaded["config"]

    settings = DEFAULT_SETTINGS.copy()
    for name, value in loaded.items():
        if isinstance(value, (int, float)):
            settings[name] = value
    for name in DEFAULT_SETTINGS:
        settings[name] = int(settings[name])
    return settings


def load_video_settings(args):
    if args.use_default_config:
        return DEFAULT_SETTINGS.copy(), "DEFAULT_SETTINGS"

    config_path = Path(args.config or CONFIG_FILE)
    if config_path.exists():
        source = CONFIG_FILE if str(config_path) == CONFIG_FILE else str(config_path)
        print(f"Loaded HSV settings from {source}")
        return load_settings_file(config_path), source

    if args.config is not None:
        raise RuntimeError(f"Could not find config file: {config_path}")

    return DEFAULT_SETTINGS.copy(), "DEFAULT_SETTINGS"


def load_manual_tuning_settings(args):
    # Manual tuning is meant to find a fresh baseline. It starts from the code
    # defaults unless the user deliberately points at a previous config file.
    if args.config is None:
        return DEFAULT_SETTINGS.copy(), "DEFAULT_SETTINGS"

    config_path = Path(args.config)
    if not config_path.exists():
        raise RuntimeError(f"Could not find config file: {config_path}")

    return load_settings_file(config_path), str(config_path)


def create_manual_tuning_folders():
    # Good and difficult samples become visual evidence for the future
    # auto-tuner, so they live in stable folders instead of per-analysis runs.
    root = Path("outputs") / "manual_tuning"
    folders = {
        "root": root,
        "good_samples": root / "good_samples",
        "difficult_samples": root / "difficult_samples",
        "debug_snapshots": root / "debug_snapshots",
    }
    Path("configs").mkdir(parents=True, exist_ok=True)
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def save_manual_tuning_config(path, settings, source_video, frame_index):
    # HSV controls select which colors count as road. ROI/morphology controls
    # clean that mask. This saved file is the human-picked seed for later
    # self-tuning, so the optimizer can search near a reasonable baseline.
    payload = {name: int(settings[name]) for name in DEFAULT_SETTINGS}
    payload.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": str(source_video),
            "source_frame_index": int(frame_index),
            "note": MANUAL_TUNING_NOTE,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved manual tuning config to {path}")


def save_manual_tuning_session(path, source_video, last_saved_frame_index, config_output_path, session, settings):
    # The session file records which frames were useful during manual tuning.
    # Future auto-tuning can replay those frames before trying larger searches.
    payload = {
        "source_video": str(source_video),
        "last_saved_frame_index": last_saved_frame_index,
        "config_output_path": str(config_output_path),
        "good_sample_frames": session["good_sample_frames"],
        "difficult_sample_frames": session["difficult_sample_frames"],
        "debug_snapshot_frames": session["debug_snapshot_frames"],
        "final_config": {name: int(settings[name]) for name in DEFAULT_SETTINGS},
        "notes_for_auto_tuning": AUTO_TUNING_NOTE,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_tuning_sample(kind, folders, frame_index, frame, result, overlay, debug_frame):
    # Good samples show frames where the current mask works well. Difficult
    # samples show where tuning or the future auto-tuner still needs attention.
    if kind == "good":
        folder = folders["good_samples"]
        frame_list_name = "good_sample_frames"
        save_overlay = False
    elif kind == "difficult":
        folder = folders["difficult_samples"]
        frame_list_name = "difficult_sample_frames"
        save_overlay = False
    else:
        folder = folders["debug_snapshots"]
        frame_list_name = "debug_snapshot_frames"
        save_overlay = True

    prefix = folder / f"frame_{frame_index:06d}"
    cv2.imwrite(str(prefix) + "_original.jpg", frame)
    cv2.imwrite(str(prefix) + "_mask.jpg", result.mask)
    if save_overlay:
        cv2.imwrite(str(prefix) + "_overlay.jpg", overlay)
    cv2.imwrite(str(prefix) + "_debug.jpg", debug_frame)
    print(f"Saved {kind} tuning sample for frame {frame_index}")
    return frame_list_name


def seek_video_frame(cap, frame_index, total_frames):
    # OpenCV can jump directly by frame index for normal MP4 files. We clamp
    # the requested frame so stepping cannot seek before the start or past end.
    if total_frames > 0:
        frame_index = min(max(0, int(frame_index)), total_frames - 1)
    else:
        frame_index = max(0, int(frame_index))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    return frame_index


def update_video_control_trackbar(frame_index, total_frames):
    if total_frames <= 0:
        return
    cv2.setTrackbarPos("Frame", WINDOW_VIDEO_CONTROL, int(frame_index))


def get_config_used(settings, config_source=None):
    config = {
        "config_source": config_source,
        "H_min": settings["H_min"],
        "H_max": settings["H_max"],
        "S_min": settings["S_min"],
        "S_max": settings["S_max"],
        "V_min": settings["V_min"],
        "V_max": settings["V_max"],
        "ROI_top_percent": settings["ROI_top_percent"],
        "Morph_kernel": settings["Morph_kernel"],
        "Close_kernel": settings["Close_kernel"],
        "Min_area_percent": settings["Min_area_percent"],
        "MIN_SEGMENT_WIDTH_PX": MIN_SEGMENT_WIDTH_PX,
        "MAX_CENTER_JUMP_PX": MAX_CENTER_JUMP_PX,
        "ALLOW_FIRST_ANCHOR_JUMP": ALLOW_FIRST_ANCHOR_JUMP,
        "USE_EGO_CONNECTED_MASK": USE_EGO_CONNECTED_MASK,
        "EGO_SEED_X_RATIO": EGO_SEED_X_RATIO,
        "EGO_SEED_Y_RATIO": EGO_SEED_Y_RATIO,
        "EGO_SEED_SEARCH_RADIUS_PX": EGO_SEED_SEARCH_RADIUS_PX,
        "EGO_BOTTOM_BAND_PERCENT": EGO_BOTTOM_BAND_PERCENT,
        "EGO_MIN_COMPONENT_AREA_PERCENT": EGO_MIN_COMPONENT_AREA_PERCENT,
        "CENTERLINE_SMOOTHING_ALPHA": CENTERLINE_SMOOTHING_ALPHA,
        "CURVE_DEADBAND_PX": CURVE_DEADBAND_PX,
        "CURVE_STRONG_PX": CURVE_STRONG_PX,
        "CENTER_DEADBAND_PX": CENTER_DEADBAND_PX,
        "CENTER_STRONG_PX": CENTER_STRONG_PX,
        "SCANLINE_COUNT": SCANLINE_COUNT,
    }
    return config


def save_config_used(path, settings, config_source):
    # This JSON makes a run reproducible because it records the exact HSV,
    # ROI, scanline, smoothing, and deadband values used for the video.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(get_config_used(settings, config_source), file, indent=2)


def build_road_overlay(frame, mask):
    overlay = frame.copy()
    mask_pixels = mask > 0
    if np.any(mask_pixels):
        overlay[mask_pixels] = cv2.addWeighted(
            overlay[mask_pixels],
            0.45,
            np.full_like(overlay[mask_pixels], (180, 120, 0)),
            0.55,
            0,
        )
    return overlay


def safe_number(value, default=""):
    if value is None:
        return default
    return value


def build_telemetry_row(frame_index, time_sec, result, confidences, selected_path, turn_hint, processing_fps):
    height, width = result.mask.shape[:2]
    mask_area_pixels = int(cv2.countNonZero(result.mask))
    mask_area_percent = mask_area_pixels / max(1, width * height) * 100.0

    # road_center_error_px means road center minus camera/image center.
    # Negative values mean the road is left of the camera; positive values mean it is right.
    # curve_error_px means far road center minus near road center.
    # Negative values hint left curvature; positive values hint right curvature.
    # rejected_scanlines counts scan rows skipped because no usable road segment was found
    # or because the center jumped too far to trust.
    return {
        "frame_index": frame_index,
        "time_sec": f"{time_sec:.3f}",
        "road_detected": result.road_detected,
        "road_confidence": f"{result.road_confidence:.4f}",
        "road_center_error_px": safe_number(result.road_center_error_px),
        "curve_error_px": safe_number(result.curve_error_px),
        "turn_hint": turn_hint,
        "near_center_x": safe_number(result.near_center_x),
        "far_center_x": safe_number(result.far_center_x),
        "tracked_center_valid": result.tracked_center_valid,
        "rejected_scanlines": result.rejected_scanlines,
        "valid_scanline_count": result.valid_scanline_count,
        "detection_quality": f"{result.detection_quality:.4f}",
        "seed_center_x": f"{result.seed_center_x:.2f}",
        "first_anchor_x": safe_number(result.first_anchor_x),
        "first_anchor_distance_px": safe_number(result.first_anchor_distance_px),
        "ego_component_found": result.ego_component_found,
        "ego_seed_x": result.ego_seed_x,
        "ego_seed_y": result.ego_seed_y,
        "ego_anchor_x": safe_number(result.ego_anchor_x),
        "ego_anchor_y": safe_number(result.ego_anchor_y),
        "ego_component_area_pixels": result.ego_component_area_pixels,
        "ego_component_area_percent": f"{result.ego_component_area_percent:.4f}",
        "ego_component_fallback_used": result.ego_component_fallback_used,
        "selected_path": selected_path,
        "straight_confidence": f"{confidences['straight']:.4f}",
        "left_confidence": f"{confidences['left']:.4f}",
        "right_confidence": f"{confidences['right']:.4f}",
        "mask_area_pixels": mask_area_pixels,
        "mask_area_percent": f"{mask_area_percent:.4f}",
        "centerline_point_count": len(result.scan_points),
        "processing_fps_estimate": f"{processing_fps:.2f}",
    }


def write_telemetry_row(writer, row):
    # The telemetry CSV has one row per processed frame so later analysis can
    # find trends, spikes, and frame ranges without watching the whole video.
    writer.writerow(row)


def write_event_row(writer, frame_index, time_sec, event_type, old_value, new_value, notes):
    # The events CSV only records important changes so it is quick to skim.
    writer.writerow(
        {
            "frame_index": frame_index,
            "time_sec": f"{time_sec:.3f}",
            "event_type": event_type,
            "old_value": old_value,
            "new_value": new_value,
            "notes": notes,
        }
    )


def detect_frame_events(frame_index, time_sec, result, selected_path, turn_hint, previous_state):
    events = []
    current_state = {
        "road_detected": result.road_detected,
        "road_center_error_px": result.road_center_error_px,
        "curve_error_px": result.curve_error_px,
        "turn_hint": turn_hint,
        "selected_path": selected_path,
        "tracked_center_valid": result.tracked_center_valid,
        "low_confidence": result.road_confidence < 0.60,
        "rejected_scanlines_high": result.rejected_scanlines >= 5,
    }

    if previous_state is None:
        if current_state["low_confidence"]:
            events.append(("low_confidence_started", "", f"{result.road_confidence:.3f}", "Road confidence dropped below 0.60."))
        if current_state["rejected_scanlines_high"]:
            events.append(("rejected_scanlines_high_started", "", result.rejected_scanlines, "Five or more scanlines were rejected."))
        return events, current_state

    if not previous_state["low_confidence"] and current_state["low_confidence"]:
        events.append(("low_confidence_started", "", f"{result.road_confidence:.3f}", "Road confidence dropped below 0.60."))
    if previous_state["low_confidence"] and not current_state["low_confidence"]:
        events.append(("low_confidence_ended", "", f"{result.road_confidence:.3f}", "Road confidence recovered to 0.60 or higher."))
    if not previous_state["rejected_scanlines_high"] and current_state["rejected_scanlines_high"]:
        events.append(("rejected_scanlines_high_started", "", result.rejected_scanlines, "Five or more scanlines were rejected."))
    if previous_state["rejected_scanlines_high"] and not current_state["rejected_scanlines_high"]:
        events.append(("rejected_scanlines_high_ended", "", result.rejected_scanlines, "Rejected scanlines dropped below five."))

    if previous_state["road_detected"] and not result.road_detected:
        events.append(("road_lost", True, False, "Road detection changed from valid to lost."))
    if not previous_state["road_detected"] and result.road_detected:
        events.append(("road_recovered", False, True, "Road detection recovered after being lost."))

    previous_center = previous_state["road_center_error_px"]
    if previous_center is not None and result.road_center_error_px is not None:
        center_jump = abs(result.road_center_error_px - previous_center)
        # MAX_CENTER_JUMP_PX is already the live detector's guardrail for an unsafe center jump.
        if center_jump > MAX_CENTER_JUMP_PX:
            events.append(("center_jump", f"{previous_center:.2f}", f"{result.road_center_error_px:.2f}", f"Center error jumped by {center_jump:.1f}px."))

    previous_curve = previous_state["curve_error_px"]
    if previous_curve is not None and result.curve_error_px is not None:
        curve_jump = abs(result.curve_error_px - previous_curve)
        # CURVE_STRONG_PX is the threshold where a curve is considered strongly left/right.
        if curve_jump > CURVE_STRONG_PX:
            events.append(("curve_jump", f"{previous_curve:.2f}", f"{result.curve_error_px:.2f}", f"Curve error jumped by {curve_jump:.1f}px."))

    if previous_state["turn_hint"] != turn_hint:
        events.append(("turn_hint_changed", previous_state["turn_hint"], turn_hint, "Curve direction hint changed."))
    if previous_state["selected_path"] != selected_path:
        events.append(("selected_path_changed", previous_state["selected_path"], selected_path, "Selected candidate path changed."))
    return events, current_state


def failure_reason_from_events(result, events):
    event_types = [event[0] for event in events]
    if not result.road_detected:
        return "road_lost"
    if not result.tracked_center_valid:
        return "tracked_center_invalid"
    if "low_confidence_started" in event_types:
        return "low_confidence"
    if "center_jump" in event_types:
        return "center_jump"
    if "curve_jump" in event_types:
        return "curve_jump"
    if "rejected_scanlines_high_started" in event_types:
        return "rejected_scanlines"
    return None


def save_failure_frame(debug_frame, folder, frame_index, reason, saved_count, max_count, enabled):
    # Failure frames are saved so ChatGPT or a human can inspect suspicious
    # moments directly. They are capped because video can create many images.
    if not enabled or reason is None or saved_count >= max_count:
        return saved_count, False
    path = folder / f"frame_{frame_index:05d}_{reason}_debug.jpg"
    cv2.imwrite(str(path), debug_frame)
    return saved_count + 1, True


def save_periodic_ai_samples(frame, result, overlay, debug_frame, folder, frame_index, time_sec, next_sample_time, interval_sec):
    # Periodic samples give AI analysis visual anchors without requiring every
    # frame. The mask, overlay, and debug view explain different parts of the detector.
    if interval_sec <= 0 or time_sec + 1e-9 < next_sample_time:
        return next_sample_time

    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_original.jpg"), frame)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_raw_mask.jpg"), result.raw_mask)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_ego_mask.jpg"), result.mask)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_overlay.jpg"), overlay)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_debug.jpg"), debug_frame)
    return next_sample_time + interval_sec


def save_key_frame(debug_frame, folder, frame_index, label, saved_keys):
    key = (frame_index, label)
    if key in saved_keys:
        return
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_{label}.jpg"), debug_frame)
    saved_keys.add(key)


def build_problem_intervals(problem_frames):
    if not problem_frames:
        return []

    intervals = []
    sorted_items = sorted(problem_frames, key=lambda item: (item["reason"], item["frame_index"]))
    current = None

    for item in sorted_items:
        frame_index = item["frame_index"]
        reason = item["reason"]
        if current is None or current["reason"] != reason or frame_index - current["end_frame"] > 10:
            if current is not None:
                intervals.append(current)
            current = {
                "start_frame": frame_index,
                "end_frame": frame_index,
                "reason": reason,
                "notes": item["notes"],
            }
        else:
            current["end_frame"] = frame_index

    if current is not None:
        intervals.append(current)
    return intervals


def average(values):
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return 0.0
    return sum(clean_values) / len(clean_values)


def most_common_text(values, default="unknown"):
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def build_recommendations(stats):
    recommendations = []
    if stats["road_detected_percent"] < 80.0:
        recommendations.append("Road detection drops often")
    if stats["max_abs_road_center_error_px"] > CENTER_STRONG_PX:
        recommendations.append("Centerline jumps too much")
    if stats["average_abs_curve_error_px"] > CURVE_DEADBAND_PX:
        recommendations.append("Curve detection may need tuning")
    if not recommendations and stats["average_road_confidence"] >= 0.60:
        recommendations.append("Mask looks stable")
    if not recommendations:
        recommendations.append("This video is good enough for next-stage testing")
    return recommendations


def write_human_summary(path, args, processed_frames, duration_sec, stats, recommendations):
    # Human summary is short prose for deciding whether the annotated video
    # looks good enough before digging into CSV/JSON details.
    lines = [
        "QCar2 RGB road detector video summary",
        "",
        f"Input video path: {args.video}",
        f"Total frames processed: {processed_frames}",
        f"Duration: {duration_sec:.2f} sec",
        f"Average road confidence: {stats['average_road_confidence']:.3f}",
        f"Road detected percent: {stats['road_detected_percent']:.1f}%",
        f"Average absolute road_center_error_px: {stats['average_abs_road_center_error_px']:.2f}",
        f"Max absolute road_center_error_px: {stats['max_abs_road_center_error_px']:.2f}",
        f"Average absolute curve_error_px: {stats['average_abs_curve_error_px']:.2f}",
        f"Most common turn_hint: {stats['most_common_turn_hint']}",
        "",
        "Major problems found:",
    ]
    if stats["major_problems"]:
        lines.extend([f"- {problem}" for problem in stats["major_problems"]])
    else:
        lines.append("- None detected from telemetry thresholds.")
    lines.extend(["", "Simple recommendation:"])
    lines.extend([f"- {recommendation}" for recommendation in recommendations])

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def write_ai_summary_json(path, args, processed_frames, total_frames, duration_sec, completed, stopped_early, stats, settings, config_source, problem_intervals):
    # AI summary is structured so ChatGPT can reason over the run without
    # guessing values from the annotated video.
    payload = {
        "input_video": args.video,
        "output_created_at": datetime.now().isoformat(timespec="seconds"),
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_sec": duration_sec,
        "completed_full_video": completed,
        "stopped_early": stopped_early,
        "average_road_confidence": stats["average_road_confidence"],
        "min_road_confidence": stats["min_road_confidence"],
        "max_road_confidence": stats["max_road_confidence"],
        "road_detected_percent": stats["road_detected_percent"],
        "average_abs_road_center_error_px": stats["average_abs_road_center_error_px"],
        "max_abs_road_center_error_px": stats["max_abs_road_center_error_px"],
        "average_abs_curve_error_px": stats["average_abs_curve_error_px"],
        "max_abs_curve_error_px": stats["max_abs_curve_error_px"],
        "low_confidence_frame_count": stats["low_confidence_frame_count"],
        "rejected_scanline_problem_count": stats["rejected_scanline_problem_count"],
        "center_jump_count": stats["center_jump_count"],
        "curve_jump_count": stats["curve_jump_count"],
        "tracked_center_invalid_count": stats["tracked_center_invalid_count"],
        "turn_hint_counts": stats["turn_hint_counts"],
        "selected_path_counts": stats["selected_path_counts"],
        "config_used": get_config_used(settings, config_source),
        "known_problem_intervals": problem_intervals,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_run_notes(path, args, folders):
    # These notes tell a future ChatGPT session what each output means and
    # which files are worth uploading first.
    command = " ".join(sys.argv)
    notes = [
        "QCar2 RGB road detector offline video run notes",
        "",
        f"Command used: {command}",
        f"Requested video path: {args.video}",
        f"Outputs root: {folders['root']}",
        f"Human output folder: {folders['human']}",
        f"AI output folder: {folders['ai']}",
        "",
        "What the detector is supposed to do:",
        "It uses RGB/OpenCV/NumPy HSV masking to detect the dark road surface, tracks scanline centers, estimates lateral center error, estimates curve direction, and writes an annotated result.",
        "",
        "Main output files:",
        "- human_output annotated MP4: watch this for visual inspection.",
        "- telemetry CSV: one row per processed frame with numeric detector state.",
        "- events CSV: only important changes such as road loss, low confidence, jumps, and path changes.",
        "- summary AI JSON: structured run summary and grouped problem intervals.",
        "- config used JSON: exact HSV/ROI/math thresholds used for the run.",
        "- frame_samples: periodic original/raw_mask/ego_mask/overlay/debug images.",
        "- failure_frames: suspicious debug frames, capped by command-line settings.",
        "",
        "Known limitations:",
        "- No machine learning, no training, no YOLO, no PyTorch, no TensorFlow.",
        "- RGB only; no ROS2 and no RealSense depth processing yet.",
        "- Lighting changes and road color changes may require HSV retuning.",
        "- Candidate paths are simple visual hints; the detected centerline is the main output.",
        "",
        "Files to upload to ChatGPT for analysis:",
        "- test_video_telemetry.csv",
        "- test_video_events.csv",
        "- test_video_summary_ai.json",
        "- test_video_config_used.json",
        "- selected frame_samples",
        "- selected failure_frames",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(notes) + "\n")


def collect_stats(rows, event_counts, processed_frames):
    confidences = [float(row["road_confidence"]) for row in rows]
    center_errors = [row["road_center_error_px"] for row in rows if row["road_center_error_px"] != ""]
    curve_errors = [row["curve_error_px"] for row in rows if row["curve_error_px"] != ""]
    road_detected_count = sum(1 for row in rows if row["road_detected"] is True)
    low_confidence_frame_count = sum(1 for row in rows if float(row["road_confidence"]) < 0.60)
    rejected_scanline_problem_count = sum(1 for row in rows if int(row["rejected_scanlines"]) >= 5)
    turn_hint_counts = Counter(row["turn_hint"] for row in rows)
    selected_path_counts = Counter(row["selected_path"] for row in rows)

    abs_center_errors = [abs(float(value)) for value in center_errors]
    abs_curve_errors = [abs(float(value)) for value in curve_errors]
    road_detected_percent = road_detected_count / max(1, processed_frames) * 100.0

    major_problems = []
    if low_confidence_frame_count > 0:
        major_problems.append(f"{low_confidence_frame_count} low-confidence frames")
    if event_counts["road_lost"] > 0:
        major_problems.append(f"{event_counts['road_lost']} road-lost events")
    if event_counts["center_jump"] > 0:
        major_problems.append(f"{event_counts['center_jump']} center-jump events")
    if event_counts["curve_jump"] > 0:
        major_problems.append(f"{event_counts['curve_jump']} curve-jump events")
    if rejected_scanline_problem_count > 0:
        major_problems.append(f"{rejected_scanline_problem_count} high rejected-scanline frames")

    return {
        "average_road_confidence": average(confidences),
        "min_road_confidence": min(confidences) if confidences else 0.0,
        "max_road_confidence": max(confidences) if confidences else 0.0,
        "road_detected_percent": road_detected_percent,
        "average_abs_road_center_error_px": average(abs_center_errors),
        "max_abs_road_center_error_px": max(abs_center_errors) if abs_center_errors else 0.0,
        "average_abs_curve_error_px": average(abs_curve_errors),
        "max_abs_curve_error_px": max(abs_curve_errors) if abs_curve_errors else 0.0,
        "low_confidence_frame_count": low_confidence_frame_count,
        "rejected_scanline_problem_count": rejected_scanline_problem_count,
        "center_jump_count": event_counts["center_jump"],
        "curve_jump_count": event_counts["curve_jump"],
        "tracked_center_invalid_count": sum(1 for row in rows if row["tracked_center_valid"] is False),
        "turn_hint_counts": dict(turn_hint_counts),
        "selected_path_counts": dict(selected_path_counts),
        "most_common_turn_hint": most_common_text([row["turn_hint"] for row in rows]),
        "major_problems": major_problems,
    }


def process_video_tuning_mode(args):
    # Manual video tuning is a human-in-the-loop step before auto-tuning. The
    # user pauses on hard frames, adjusts simple RGB/OpenCV/NumPy mask values,
    # and saves a good baseline instead of letting an optimizer start randomly.
    if not args.video:
        raise RuntimeError("--video is required when using --tune-video")
    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Could not find video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    settings, config_source = load_manual_tuning_settings(args)
    print(f"Manual tuning config source: {config_source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    folders = create_manual_tuning_folders()
    config_output_path = Path(args.config_output)
    session_output_path = Path(args.session_output)
    session = {
        "good_sample_frames": [],
        "difficult_sample_frames": [],
        "debug_snapshot_frames": [],
    }
    last_saved_frame_index = None

    # Trackbars expose the same beginner-friendly parameters as live tuning:
    # HSV chooses road-colored pixels, ROI_top_percent ignores the upper scene,
    # Morph_kernel removes small white noise, and Close_kernel fills small gaps.
    create_trackbars(settings)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_VIDEO_CONTROL, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", WINDOW_VIDEO_CONTROL, 0, max(1, total_frames - 1), nothing)

    current_frame_index = seek_video_frame(cap, args.start_frame, total_frames)
    update_video_control_trackbar(current_frame_index, total_frames)
    playback_speed = max(0.1, float(args.playback_speed))
    paused = False
    show_mask_window = False
    show_candidates = True
    use_ego_connected_mask = USE_EGO_CONNECTED_MASK
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    tracker = PathConfidenceTracker()

    print("Manual video tuning controls:")
    print("q/ESC quit | p/SPACE pause | s save config | l load config-output | r reset")
    print("e toggle ego-connected mask | m mask window | c candidate paths")
    print("g good sample | f difficult sample | d debug snapshot | n/right next | b/left back")

    try:
        while True:
            control_frame = cv2.getTrackbarPos("Frame", WINDOW_VIDEO_CONTROL)
            if total_frames > 0 and control_frame != current_frame_index:
                current_frame_index = seek_video_frame(cap, control_frame, total_frames)
                last_center_x = None
                frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = resize_frame(frame)
            settings = get_trackbar_settings()
            result = detect_road(frame, settings, last_center_x, frames_since_valid, use_ego_connected_mask, settings)
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
            debug_frame = draw_visualization(
                frame,
                result,
                confidences,
                selected_path,
                smoothed_curve_error_px,
                turn_hint,
                show_candidates,
            )
            overlay = build_road_overlay(frame, result.mask)
            display = build_display_grid(frame, result, debug_frame)
            cv2.putText(
                display,
                f"Frame {current_frame_index}/{max(0, total_frames - 1)}  speed {playback_speed:.1f}x  ego_filter {use_ego_connected_mask}",
                (12, FRAME_HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_MAIN, display)
            if show_mask_window:
                cv2.imshow(WINDOW_MASK, result.mask)
            else:
                try:
                    cv2.destroyWindow(WINDOW_MASK)
                except cv2.error:
                    pass

            update_video_control_trackbar(current_frame_index, total_frames)
            delay_ms = 40 if paused else max(1, int(1000.0 / (source_fps * playback_speed)))
            key = cv2.waitKeyEx(delay_ms)

            if key in (ord("q"), 27):
                break
            if key in (ord("p"), 32):
                paused = not paused
                print("Paused." if paused else "Playing.")
            elif key == ord("s"):
                save_manual_tuning_config(config_output_path, settings, video_path, current_frame_index)
                last_saved_frame_index = current_frame_index
                save_manual_tuning_session(
                    session_output_path,
                    video_path,
                    last_saved_frame_index,
                    config_output_path,
                    session,
                    settings,
                )
            elif key == ord("l"):
                if config_output_path.exists():
                    settings = load_settings_file(config_output_path)
                    set_trackbars(settings)
                    print(f"Loaded manual tuning config from {config_output_path}")
                else:
                    print(f"No manual tuning config found at {config_output_path}")
            elif key == ord("r"):
                settings = DEFAULT_SETTINGS.copy()
                set_trackbars(settings)
                last_center_x = None
                frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
                print("Reset tuning settings to DEFAULT_SETTINGS.")
            elif key == ord("m"):
                show_mask_window = not show_mask_window
            elif key == ord("e"):
                use_ego_connected_mask = not use_ego_connected_mask
                last_center_x = None
                frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
                print(f"Ego-connected mask {'on' if use_ego_connected_mask else 'off'}.")
            elif key == ord("c"):
                show_candidates = not show_candidates
            elif key in (ord("n"), 83, 2555904):
                paused = True
                current_frame_index = seek_video_frame(cap, current_frame_index + 1, total_frames)
            elif key in (ord("b"), 81, 2424832):
                paused = True
                current_frame_index = seek_video_frame(cap, current_frame_index - 1, total_frames)
            elif key == ord("g"):
                list_name = save_tuning_sample("good", folders, current_frame_index, frame, result, overlay, debug_frame)
                if current_frame_index not in session[list_name]:
                    session[list_name].append(current_frame_index)
                save_manual_tuning_session(session_output_path, video_path, last_saved_frame_index, config_output_path, session, settings)
            elif key == ord("f"):
                list_name = save_tuning_sample("difficult", folders, current_frame_index, frame, result, overlay, debug_frame)
                if current_frame_index not in session[list_name]:
                    session[list_name].append(current_frame_index)
                save_manual_tuning_session(session_output_path, video_path, last_saved_frame_index, config_output_path, session, settings)
            elif key == ord("d"):
                list_name = save_tuning_sample("debug", folders, current_frame_index, frame, result, overlay, debug_frame)
                if current_frame_index not in session[list_name]:
                    session[list_name].append(current_frame_index)
                save_manual_tuning_session(session_output_path, video_path, last_saved_frame_index, config_output_path, session, settings)
            elif key == ord("["):
                playback_speed = max(0.1, playback_speed / 1.25)
                print(f"Playback speed: {playback_speed:.2f}x")
            elif key == ord("]"):
                playback_speed = min(8.0, playback_speed * 1.25)
                print(f"Playback speed: {playback_speed:.2f}x")

            if not paused and key not in (ord("n"), ord("b"), 83, 81, 2555904, 2424832):
                current_frame_index += 1
                if total_frames > 0 and current_frame_index >= total_frames:
                    current_frame_index = total_frames - 1
                    paused = True

    finally:
        save_manual_tuning_session(
            session_output_path,
            video_path,
            last_saved_frame_index,
            config_output_path,
            session,
            settings,
        )
        cap.release()
        cv2.destroyAllWindows()

    print("Manual video tuning complete.")
    print(f"Manual config output: {config_output_path}")
    print(f"Session output: {session_output_path}")
    print(f"Sample folders: {folders['root']}")
    return 0


def process_video_source(args):
    # Offline video mode is useful because it lets the exact same RGB detector
    # be replayed, measured, and inspected without needing the QCar2 or camera live.
    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Could not find video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    base_name = get_output_base_name(video_path)
    folders = create_output_folders(args.output_dir, base_name, args.clean_output)
    settings, config_source = load_video_settings(args)
    print(f"Video config source: {config_source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    annotated_path = folders["human"] / f"{base_name}_annotated.mp4"
    telemetry_path = folders["ai"] / f"{base_name}_telemetry.csv"
    events_path = folders["ai"] / f"{base_name}_events.csv"
    summary_path = folders["human"] / f"{base_name}_summary.txt"
    ai_summary_path = folders["ai"] / f"{base_name}_summary_ai.json"
    config_path = folders["ai"] / f"{base_name}_config_used.json"
    run_notes_path = folders["ai"] / f"{base_name}_run_notes.txt"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, source_fps, (FRAME_WIDTH, FRAME_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {annotated_path}")

    telemetry_fields = [
        "frame_index",
        "time_sec",
        "road_detected",
        "road_confidence",
        "road_center_error_px",
        "curve_error_px",
        "turn_hint",
        "near_center_x",
        "far_center_x",
        "tracked_center_valid",
        "rejected_scanlines",
        "valid_scanline_count",
        "detection_quality",
        "seed_center_x",
        "first_anchor_x",
        "first_anchor_distance_px",
        "ego_component_found",
        "ego_seed_x",
        "ego_seed_y",
        "ego_anchor_x",
        "ego_anchor_y",
        "ego_component_area_pixels",
        "ego_component_area_percent",
        "ego_component_fallback_used",
        "selected_path",
        "straight_confidence",
        "left_confidence",
        "right_confidence",
        "mask_area_pixels",
        "mask_area_percent",
        "centerline_point_count",
        "processing_fps_estimate",
    ]
    event_fields = ["frame_index", "time_sec", "event_type", "old_value", "new_value", "notes"]

    if not args.no_display:
        create_trackbars(settings)
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    tracker = PathConfidenceTracker()
    show_mask_window = False
    show_candidates = True
    paused = False
    stopped_early = False
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    frame_index = 0
    processed_frames = 0
    failure_saved_count = 0
    last_progress_time = time.perf_counter()
    run_start_time = time.perf_counter()
    next_sample_time = 0.0
    previous_state = None
    rows = []
    event_counts = Counter()
    problem_frames = []
    saved_keys = set()
    middle_frame_index = max(0, total_frames // 2) if total_frames > 0 else None
    last_debug_frame = None

    save_config_used(config_path, settings, config_source)
    write_run_notes(run_notes_path, args, folders)

    with open(telemetry_path, "w", newline="", encoding="utf-8") as telemetry_file, open(
        events_path, "w", newline="", encoding="utf-8"
    ) as events_file:
        telemetry_writer = csv.DictWriter(telemetry_file, fieldnames=telemetry_fields)
        events_writer = csv.DictWriter(events_file, fieldnames=event_fields)
        telemetry_writer.writeheader()
        events_writer.writeheader()

        try:
            while True:
                if paused:
                    key = cv2.waitKey(40) & 0xFF
                    action = handle_key(key, settings)
                    if action == "quit":
                        stopped_early = True
                        break
                    if action == "toggle_pause":
                        paused = False
                        print("Unpaused.")
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame = resize_frame(frame)
                if not args.no_display:
                    settings = get_trackbar_settings()

                time_sec = frame_index / source_fps
                elapsed = max(0.001, time.perf_counter() - run_start_time)
                processing_fps = processed_frames / elapsed

                result = detect_road(frame, settings, last_center_x, frames_since_valid, detector_config=settings)

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
                debug_frame = draw_visualization(
                    frame,
                    result,
                    confidences,
                    selected_path,
                    smoothed_curve_error_px,
                    turn_hint,
                    show_candidates,
                )
                overlay = build_road_overlay(frame, result.mask)
                writer.write(debug_frame)
                last_debug_frame = debug_frame.copy()

                telemetry_row = build_telemetry_row(
                    frame_index,
                    time_sec,
                    result,
                    confidences,
                    selected_path,
                    turn_hint,
                    processing_fps,
                )
                write_telemetry_row(telemetry_writer, telemetry_row)
                rows.append(telemetry_row)

                events, previous_state = detect_frame_events(
                    frame_index,
                    time_sec,
                    result,
                    selected_path,
                    turn_hint,
                    previous_state,
                )
                for event_type, old_value, new_value, notes in events:
                    write_event_row(events_writer, frame_index, time_sec, event_type, old_value, new_value, notes)
                    event_counts[event_type] += 1
                    if event_type in ("turn_hint_changed", "road_lost", "road_recovered"):
                        save_key_frame(debug_frame, folders["key_frames"], frame_index, event_type, saved_keys)

                failure_reason = failure_reason_from_events(result, events)
                if failure_reason is not None:
                    problem_frames.append(
                        {
                            "frame_index": frame_index,
                            "reason": failure_reason,
                            "notes": f"Suspicious frame detected because of {failure_reason}.",
                        }
                    )
                failure_saved_count, _saved = save_failure_frame(
                    debug_frame,
                    folders["failure_frames"],
                    frame_index,
                    failure_reason,
                    failure_saved_count,
                    max(0, args.max_failure_frames),
                    args.save_failure_frames,
                )

                next_sample_time = save_periodic_ai_samples(
                    frame,
                    result,
                    overlay,
                    debug_frame,
                    folders["frame_samples"],
                    frame_index,
                    time_sec,
                    next_sample_time,
                    args.ai_sample_interval_sec,
                )

                if frame_index == 0:
                    save_key_frame(debug_frame, folders["key_frames"], frame_index, "start", saved_keys)
                if middle_frame_index is not None and frame_index == middle_frame_index:
                    save_key_frame(debug_frame, folders["key_frames"], frame_index, "middle", saved_keys)

                processed_frames += 1
                now = time.perf_counter()
                if now - last_progress_time >= 3.0:
                    percent = frame_index / max(1, total_frames) * 100.0 if total_frames > 0 else 0.0
                    print(
                        f"Video progress: frame {frame_index}/{total_frames} "
                        f"({percent:.1f}%), processing {processing_fps:.1f} FPS"
                    )
                    last_progress_time = now

                if not args.no_display:
                    cv2.imshow(WINDOW_MAIN, build_display_grid(frame, result, debug_frame))
                    if show_mask_window:
                        cv2.imshow(WINDOW_MASK, result.mask)
                    else:
                        try:
                            cv2.destroyWindow(WINDOW_MASK)
                        except cv2.error:
                            pass

                    key = cv2.waitKey(1) & 0xFF
                    action = handle_key(key, settings)
                    if action == "quit":
                        stopped_early = True
                        break
                    if action == "toggle_mask":
                        show_mask_window = not show_mask_window
                    elif action == "toggle_pause":
                        paused = True
                        print("Paused.")
                    elif action == "toggle_candidates":
                        show_candidates = not show_candidates
                        print("Candidate paths on." if show_candidates else "Candidate paths off.")

                frame_index += 1
        finally:
            cap.release()
            writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()

    if last_debug_frame is not None:
        save_key_frame(last_debug_frame, folders["key_frames"], max(0, frame_index - 1), "end", saved_keys)

    duration_sec = processed_frames / source_fps if source_fps > 0 else 0.0
    completed = not stopped_early and (total_frames <= 0 or processed_frames >= total_frames)
    stats = collect_stats(rows, event_counts, processed_frames)
    recommendations = build_recommendations(stats)
    problem_intervals = build_problem_intervals(problem_frames)

    write_human_summary(summary_path, args, processed_frames, duration_sec, stats, recommendations)
    write_ai_summary_json(
        ai_summary_path,
        args,
        processed_frames,
        total_frames,
        duration_sec,
        completed,
        stopped_early,
        stats,
        settings,
        config_source,
        problem_intervals,
    )

    print()
    print("Video processing complete.")
    print(f"Output folder: {folders['root']}")
    print(f"Human output: {folders['human']}")
    print(f"AI output: {folders['ai']}")
    print(f"Frames processed: {processed_frames}")
    print(f"Failure frames saved: {failure_saved_count}")
    return 0


def main():
    args = parse_args()
    print_startup(args)

    if args.source == "video":
        try:
            if args.auto_tune:
                from auto_tuner import run_auto_tune

                run_auto_tune(args)
                return 0
            if args.tune_video:
                return process_video_tuning_mode(args)
            return process_video_source(args)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

    try:
        source = make_source(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    create_trackbars(DEFAULT_SETTINGS)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    tracker = PathConfidenceTracker()
    show_mask_window = False
    show_candidates = True
    paused = False
    last_frame = None
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1

    try:
        while True:
            if not paused or last_frame is None:
                ok, frame = source.read()
                if not ok or frame is None:
                    print("No frame received from source.")
                    break
                last_frame = resize_frame(frame)

            frame = last_frame.copy()
            settings = get_trackbar_settings()
            result = detect_road(frame, settings, last_center_x, frames_since_valid)

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
            visualization = draw_visualization(
                frame,
                result,
                confidences,
                selected_path,
                smoothed_curve_error_px,
                turn_hint,
                show_candidates,
            )
            display = build_display_grid(frame, result, visualization)
            cv2.imshow(WINDOW_MAIN, display)

            if show_mask_window:
                cv2.imshow(WINDOW_MASK, result.mask)
            else:
                try:
                    cv2.destroyWindow(WINDOW_MASK)
                except cv2.error:
                    pass

            key = cv2.waitKey(20) & 0xFF
            action = handle_key(key, settings)
            if action == "quit":
                break
            if action == "toggle_mask":
                show_mask_window = not show_mask_window
            elif action == "toggle_pause":
                paused = not paused
                print("Paused." if paused else "Unpaused.")
            elif action == "toggle_candidates":
                show_candidates = not show_candidates
                print("Candidate paths on." if show_candidates else "Candidate paths off.")
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
