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

# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------
# The ROS2-ready detector is intentionally self-contained in this file. Camera
# JSON files override these defaults, but keeping safe fallback values here
# means image/video/webcam modes do not depend on a separate Python config
# module.
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
REALSENSE_WIDTH = 848
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

WINDOW_MAIN = "QCar2 RGB Drift Helper"
WINDOW_TUNING = "Tuning"
WINDOW_MASK = "Road Mask Debug"
CONFIG_FILE = "configs/csi_front_config.json"

LANE_WIDTH_MM = 254.0
CAR_WIDTH_MM = 203.2
SIDEWALK_MARGIN_MM = 12.0
LINE_MARGIN_MM = 12.0
SAFE_HALLWAY_WIDTH_MM = CAR_WIDTH_MM + SIDEWALK_MARGIN_MM + LINE_MARGIN_MM

DEFAULT_SETTINGS = {
    "H_min": 0,
    "H_max": 174,
    "S_min": 0,
    "S_max": 202,
    "V_min": 0,
    "V_max": 101,
    "ROI_top_percent": 67,
    "Morph_kernel": 1,
    "Close_kernel": 1,
    "Min_area_percent": 1,
    "SCANLINE_COUNT": 12,
    "MIN_SEGMENT_WIDTH_PX": 20,
    "MAX_CENTER_JUMP_PX": 95,
    "ALLOW_FIRST_ANCHOR_JUMP": True,
    "USE_EGO_CONNECTED_MASK": True,
    "EGO_SEED_X_RATIO": 0.50,
    "EGO_SEED_Y_RATIO": 0.95,
    "EGO_SEED_SEARCH_RADIUS_PX": 120,
    "EGO_BOTTOM_BAND_PERCENT": 18,
    "EGO_MIN_COMPONENT_AREA_PERCENT": 1.0,
    "CENTERLINE_SMOOTHING_ALPHA": 0.45,
    "LAST_CENTER_HOLD_FRAMES": 12,
    "CENTER_DEADBAND_PX": 35,
    "CENTER_STRONG_PX": 110,
    "CURVE_DEADBAND_PX": 45,
    "CURVE_STRONG_PX": 140,
    "LANE_WIDTH_MM": LANE_WIDTH_MM,
    "CAR_WIDTH_MM": CAR_WIDTH_MM,
    "SIDEWALK_MARGIN_MM": SIDEWALK_MARGIN_MM,
    "LINE_MARGIN_MM": LINE_MARGIN_MM,
    "SAFE_HALLWAY_WIDTH_MM": SAFE_HALLWAY_WIDTH_MM,
    "MIN_VALID_LANE_WIDTH_MM": 220.0,
    "MAX_VALID_LANE_WIDTH_MM": 290.0,
    "SAFE_SCANLINE_COUNT": 6,
    "SAFE_SCANLINE_START_RATIO": 0.62,
    "SAFE_SCANLINE_END_RATIO": 0.95,
    "SAFE_SCANLINE_NEAR_WEIGHT_BIAS": 2.0,
    "SAFE_MIN_VALID_SCANLINES": 4,
    "SAFE_MIN_ROAD_CONFIDENCE": 0.55,
    "SAFE_ERROR_DEADBAND_MM": 5.0,
    "SAFE_STEERING_GAIN": 0.01,
    "SAFE_MAX_STEERING_CORRECTION": 0.20,
    "USE_YELLOW_BOUNDARY_LOCK": True,
    "YELLOW_H_MIN": 18,
    "YELLOW_H_MAX": 45,
    "YELLOW_S_MIN": 80,
    "YELLOW_S_MAX": 255,
    "YELLOW_V_MIN": 80,
    "YELLOW_V_MAX": 255,
    "YELLOW_BOUNDARY_DILATE_PX": 7,
    "YELLOW_MAX_CROSSING_PIXELS": 20,
    "LANE_SIDE_HOLD_FRAMES": 15,
    "USE_RIGHT_LANE_YELLOW_LOCK": True,
    "RIGHT_LANE_FROM_YELLOW": True,
    "YELLOW_LANE_SIDE": "right",
    "YELLOW_LANE_SEARCH_MARGIN_PX": 8,
    "YELLOW_MIN_PIXELS_PER_SCANLINE": 3,
    "YELLOW_RIGHT_LANE_HOLD_FRAMES": 20,
    "USE_NO_YELLOW_WIDE_BLOB_GATE": True,
    "NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO": 0.55,
    "NO_YELLOW_MAX_MEASURED_WIDTH_MM": 310.0,
    "ALLOW_NO_YELLOW_BLOB_SPLIT": False,
    "CAMERA_CENTER_X_RATIO": 0.50,
    "CAMERA_CENTER_OFFSET_PX": 0,
    "CAMERA_CENTER_OFFSET_MM": 0.0,
    "MIN_CLEARANCE_MM": 0.0,
    "MAX_REASONABLE_CORRIDOR_ERROR_MM": 75.0,
    "MAX_REASONABLE_CLEARANCE_MM": 80.0,
    "MAX_STEERING_SATURATION_FRAMES": 15,
    "DRIFT_HELPER_ONLY_ON_STRAIGHT": True,
    "DRIFT_MAX_ABS_CURVE_ERROR_PX": 35,
    "DRIFT_MAX_ABS_CORRIDOR_ERROR_MM": 40,
    "DRIFT_MIN_SAFE_CORRIDOR_VALID_FRAMES": 3,
    "DRIFT_DISABLE_ON_TURN_HINT": True,
    "DRIFT_HELPER_GAIN": 0.01,
    "DRIFT_HELPER_MAX_OUTPUT": 0.20,
    "ROAD_OVERLAY_COLOR_BGR": [120, 255, 120],
    "ROAD_OVERLAY_ALPHA": 0.35,
    "ACTIVE_SAFE_CORRIDOR_COLOR_BGR": [255, 130, 20],
    "ACTIVE_SAFE_CORRIDOR_ALPHA": 0.42,
    "INACTIVE_SAFE_CORRIDOR_COLOR_BGR": [140, 120, 100],
    "INACTIVE_SAFE_CORRIDOR_ALPHA": 0.18,
    "SHOW_INACTIVE_HELPER_DEFAULT": False,
}

TRACKBAR_RANGES = {
    "H_min": 179,
    "H_max": 179,
    "S_min": 255,
    "S_max": 255,
    "V_min": 255,
    "V_max": 255,
    "ROI_top_percent": 80,
    "Morph_kernel": 31,
    "Close_kernel": 51,
    "Min_area_percent": 30,
}

SCANLINE_COUNT = DEFAULT_SETTINGS["SCANLINE_COUNT"]
MIN_SEGMENT_WIDTH_PX = DEFAULT_SETTINGS["MIN_SEGMENT_WIDTH_PX"]
MAX_CENTER_JUMP_PX = DEFAULT_SETTINGS["MAX_CENTER_JUMP_PX"]
ALLOW_FIRST_ANCHOR_JUMP = DEFAULT_SETTINGS["ALLOW_FIRST_ANCHOR_JUMP"]
USE_EGO_CONNECTED_MASK = DEFAULT_SETTINGS["USE_EGO_CONNECTED_MASK"]
EGO_SEED_X_RATIO = DEFAULT_SETTINGS["EGO_SEED_X_RATIO"]
EGO_SEED_Y_RATIO = DEFAULT_SETTINGS["EGO_SEED_Y_RATIO"]
EGO_SEED_SEARCH_RADIUS_PX = DEFAULT_SETTINGS["EGO_SEED_SEARCH_RADIUS_PX"]
EGO_BOTTOM_BAND_PERCENT = DEFAULT_SETTINGS["EGO_BOTTOM_BAND_PERCENT"]
EGO_MIN_COMPONENT_AREA_PERCENT = DEFAULT_SETTINGS["EGO_MIN_COMPONENT_AREA_PERCENT"]
CENTERLINE_SMOOTHING_ALPHA = DEFAULT_SETTINGS["CENTERLINE_SMOOTHING_ALPHA"]
LAST_CENTER_HOLD_FRAMES = DEFAULT_SETTINGS["LAST_CENTER_HOLD_FRAMES"]
CENTER_DEADBAND_PX = DEFAULT_SETTINGS["CENTER_DEADBAND_PX"]
CENTER_STRONG_PX = DEFAULT_SETTINGS["CENTER_STRONG_PX"]
CURVE_DEADBAND_PX = DEFAULT_SETTINGS["CURVE_DEADBAND_PX"]
CURVE_STRONG_PX = DEFAULT_SETTINGS["CURVE_STRONG_PX"]
MIN_VALID_LANE_WIDTH_MM = DEFAULT_SETTINGS["MIN_VALID_LANE_WIDTH_MM"]
MAX_VALID_LANE_WIDTH_MM = DEFAULT_SETTINGS["MAX_VALID_LANE_WIDTH_MM"]
SAFE_SCANLINE_COUNT = DEFAULT_SETTINGS["SAFE_SCANLINE_COUNT"]
SAFE_SCANLINE_START_RATIO = DEFAULT_SETTINGS["SAFE_SCANLINE_START_RATIO"]
SAFE_SCANLINE_END_RATIO = DEFAULT_SETTINGS["SAFE_SCANLINE_END_RATIO"]
SAFE_SCANLINE_NEAR_WEIGHT_BIAS = DEFAULT_SETTINGS["SAFE_SCANLINE_NEAR_WEIGHT_BIAS"]
SAFE_MIN_VALID_SCANLINES = DEFAULT_SETTINGS["SAFE_MIN_VALID_SCANLINES"]
SAFE_MIN_ROAD_CONFIDENCE = DEFAULT_SETTINGS["SAFE_MIN_ROAD_CONFIDENCE"]
SAFE_ERROR_DEADBAND_MM = DEFAULT_SETTINGS["SAFE_ERROR_DEADBAND_MM"]
SAFE_STEERING_GAIN = DEFAULT_SETTINGS["SAFE_STEERING_GAIN"]
SAFE_MAX_STEERING_CORRECTION = DEFAULT_SETTINGS["SAFE_MAX_STEERING_CORRECTION"]
USE_YELLOW_BOUNDARY_LOCK = DEFAULT_SETTINGS["USE_YELLOW_BOUNDARY_LOCK"]
YELLOW_H_MIN = DEFAULT_SETTINGS["YELLOW_H_MIN"]
YELLOW_H_MAX = DEFAULT_SETTINGS["YELLOW_H_MAX"]
YELLOW_S_MIN = DEFAULT_SETTINGS["YELLOW_S_MIN"]
YELLOW_S_MAX = DEFAULT_SETTINGS["YELLOW_S_MAX"]
YELLOW_V_MIN = DEFAULT_SETTINGS["YELLOW_V_MIN"]
YELLOW_V_MAX = DEFAULT_SETTINGS["YELLOW_V_MAX"]
YELLOW_BOUNDARY_DILATE_PX = DEFAULT_SETTINGS["YELLOW_BOUNDARY_DILATE_PX"]
YELLOW_MAX_CROSSING_PIXELS = DEFAULT_SETTINGS["YELLOW_MAX_CROSSING_PIXELS"]
LANE_SIDE_HOLD_FRAMES = DEFAULT_SETTINGS["LANE_SIDE_HOLD_FRAMES"]
USE_RIGHT_LANE_YELLOW_LOCK = DEFAULT_SETTINGS["USE_RIGHT_LANE_YELLOW_LOCK"]
RIGHT_LANE_FROM_YELLOW = DEFAULT_SETTINGS["RIGHT_LANE_FROM_YELLOW"]
YELLOW_LANE_SIDE = DEFAULT_SETTINGS["YELLOW_LANE_SIDE"]
YELLOW_LANE_SEARCH_MARGIN_PX = DEFAULT_SETTINGS["YELLOW_LANE_SEARCH_MARGIN_PX"]
YELLOW_MIN_PIXELS_PER_SCANLINE = DEFAULT_SETTINGS["YELLOW_MIN_PIXELS_PER_SCANLINE"]
YELLOW_RIGHT_LANE_HOLD_FRAMES = DEFAULT_SETTINGS["YELLOW_RIGHT_LANE_HOLD_FRAMES"]
USE_NO_YELLOW_WIDE_BLOB_GATE = DEFAULT_SETTINGS["USE_NO_YELLOW_WIDE_BLOB_GATE"]
NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO = DEFAULT_SETTINGS["NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO"]
NO_YELLOW_MAX_MEASURED_WIDTH_MM = DEFAULT_SETTINGS["NO_YELLOW_MAX_MEASURED_WIDTH_MM"]
ALLOW_NO_YELLOW_BLOB_SPLIT = DEFAULT_SETTINGS["ALLOW_NO_YELLOW_BLOB_SPLIT"]
CAMERA_CENTER_X_RATIO = DEFAULT_SETTINGS["CAMERA_CENTER_X_RATIO"]
CAMERA_CENTER_OFFSET_PX = DEFAULT_SETTINGS["CAMERA_CENTER_OFFSET_PX"]
CAMERA_CENTER_OFFSET_MM = DEFAULT_SETTINGS["CAMERA_CENTER_OFFSET_MM"]
MIN_CLEARANCE_MM = DEFAULT_SETTINGS["MIN_CLEARANCE_MM"]
MAX_REASONABLE_CORRIDOR_ERROR_MM = DEFAULT_SETTINGS["MAX_REASONABLE_CORRIDOR_ERROR_MM"]
MAX_REASONABLE_CLEARANCE_MM = DEFAULT_SETTINGS["MAX_REASONABLE_CLEARANCE_MM"]
MAX_STEERING_SATURATION_FRAMES = DEFAULT_SETTINGS["MAX_STEERING_SATURATION_FRAMES"]

WINDOW_VIDEO_CONTROL = "Video Control"
MANUAL_TUNING_NOTE = "Manual video tuning baseline for the RGB drift helper"
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
    yellow_boundary_mask: np.ndarray
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
    safe_corridor_valid: bool
    visual_helper_active: bool
    safe_corridor_width_mm: float
    safe_corridor_width_px: float | None
    measured_lane_width_mm: float | None
    measured_lane_width_px: float | None
    lane_width_valid: bool
    left_clearance_mm: float | None
    right_clearance_mm: float | None
    corridor_center_error_mm: float | None
    corridor_center_error_px: float | None
    visual_steering_correction: float
    safe_scanline_count_valid: int
    safe_scanline_rows: list[dict]
    safe_corridor_reason: str
    yellow_boundary_detected: bool
    yellow_boundary_pixel_count: int
    yellow_boundary_enforced: bool
    selected_lane_side: str
    yellow_crossing_pixels: int
    yellow_right_edge_x: int | None
    right_lane_segment_found: bool
    right_lane_segment_left_x: int | None
    right_lane_segment_right_x: int | None
    right_lane_segment_width_px: float | None
    right_lane_lock_active: bool
    right_lane_lock_reason: str
    ego_reference_x: float
    camera_center_offset_px: float
    lane_center_x: float | None
    left_space_mm: float | None
    right_space_mm: float | None
    unphysical_corridor_geometry: bool
    steering_saturation_count: int
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
                "'py -3.13 -m pip install pyrealsense2', or use "
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


def compute_turn_hint(curve_error_px, detector_config=None):
    """Return a simple curve hint for safety gating, not path planning.

    The visual helper is drift-only. A left/right turn hint disables the helper
    so the normal QCar2 controller can handle turns, intersections, and route
    choices without camera-helper guesses.
    """
    if curve_error_px is None:
        return "unknown"
    deadband = cfg_float(detector_config, "CURVE_DEADBAND_PX", CURVE_DEADBAND_PX)
    if curve_error_px < -deadband:
        return "left"
    if curve_error_px > deadband:
        return "right"
    return "straight"


def parse_args():
    parser = argparse.ArgumentParser(description="RGB-only QCar2 drift-helper detector.")
    parser.add_argument("--source", choices=["image", "webcam", "realsense", "video"], required=True)
    parser.add_argument("--image", help="Path to a static image for --source image.")
    parser.add_argument("--video", help="Path to a video file for --source video.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for --source webcam.")
    parser.add_argument("--output-dir", default="outputs", help="Folder where video-mode outputs are written.")
    parser.add_argument("--no-display", action="store_true", help="Process video without opening OpenCV windows.")
    parser.add_argument(
        "--show-inactive-helper",
        action="store_true",
        help="Draw faint inactive helper geometry for debugging. Hidden by default.",
    )
    parser.add_argument(
        "--use-default-config",
        action="store_true",
        help="Use built-in defaults instead of loading the camera JSON.",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_FILE,
        help="Camera config JSON to load. Defaults to configs/csi_front_config.json.",
    )
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
    print("Keys: q/ESC quit | s save HSV | l load HSV | r reset HSV | m mask | p pause")
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


def get_trackbar_settings(base_settings=None):
    settings = (base_settings or DEFAULT_SETTINGS).copy()
    for name in TRACKBAR_RANGES:
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


def config_values_from_payload(payload):
    """Extract runtime detector values from either flat or metadata JSON."""
    values = payload.get("settings", payload.get("config", payload))
    settings = DEFAULT_SETTINGS.copy()
    if "camera_type" in payload:
        settings["camera_type"] = payload["camera_type"]
    if "name" in payload:
        settings["config_name"] = payload["name"]
    for name, default in DEFAULT_SETTINGS.items():
        if name in values:
            if isinstance(default, list):
                settings[name] = list(values[name])
            elif isinstance(default, str):
                settings[name] = values[name]
            else:
                settings[name] = type(default)(values[name])
    for name in TRACKBAR_RANGES:
        if name in values:
            settings[name] = int(values[name])
    return settings


def load_settings(config_path=CONFIG_FILE):
    if not config_path or not os.path.exists(config_path):
        print(f"No config found at {config_path}. Using built-in defaults.")
        return None

    with open(config_path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    settings = config_values_from_payload(loaded)
    print(f"Loaded detector settings from {config_path}")
    return settings


def clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def cfg_value(detector_config, name, default):
    # Camera JSON files pass runtime values in a dictionary. Missing values use
    # the conservative defaults above so older tuning files still run.
    if detector_config is not None and name in detector_config:
        return detector_config[name]
    return default


def cfg_int(detector_config, name, default):
    return int(round(float(cfg_value(detector_config, name, default))))


def cfg_float(detector_config, name, default):
    return float(cfg_value(detector_config, name, default))


def cfg_bool(detector_config, name, default=False):
    return bool(cfg_value(detector_config, name, default))


def cfg_color(detector_config, name, default):
    value = cfg_value(detector_config, name, default)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(clamp(int(component), 0, 255)) for component in value)
    return tuple(default)


def build_yellow_boundary_mask(hsv, detector_config=None):
    # Yellow lane paint is detected separately from road. The dilated mask is
    # used as a hard no-cross divider for the safe corridor.
    lower = np.array(
        [
            cfg_int(detector_config, "YELLOW_H_MIN", YELLOW_H_MIN),
            cfg_int(detector_config, "YELLOW_S_MIN", YELLOW_S_MIN),
            cfg_int(detector_config, "YELLOW_V_MIN", YELLOW_V_MIN),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            cfg_int(detector_config, "YELLOW_H_MAX", YELLOW_H_MAX),
            cfg_int(detector_config, "YELLOW_S_MAX", YELLOW_S_MAX),
            cfg_int(detector_config, "YELLOW_V_MAX", YELLOW_V_MAX),
        ],
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, lower, upper)
    dilate_px = max(0, cfg_int(detector_config, "YELLOW_BOUNDARY_DILATE_PX", YELLOW_BOUNDARY_DILATE_PX))
    if dilate_px > 0:
        kernel_size = dilate_px if dilate_px % 2 == 1 else dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def estimate_lane_side(ego_debug, yellow_boundary_mask):
    if ego_debug["ego_anchor_x"] is None or cv2.countNonZero(yellow_boundary_mask) == 0:
        return "unknown"
    anchor_x = int(ego_debug["ego_anchor_x"])
    anchor_y = int(ego_debug["ego_anchor_y"])
    y1 = max(0, anchor_y - 80)
    y2 = min(yellow_boundary_mask.shape[0] - 1, anchor_y + 10)
    points = cv2.findNonZero(yellow_boundary_mask[y1 : y2 + 1, :])
    if points is None:
        return "unknown"
    xs = points.reshape(-1, 2)[:, 0]
    yellow_x = float(np.median(xs))
    if anchor_x < yellow_x:
        return "left"
    if anchor_x > yellow_x:
        return "right"
    return "unknown"


def update_lane_side_memory(candidate_side, lane_side_memory, detector_config=None):
    if lane_side_memory is None:
        return candidate_side
    hold_frames = cfg_int(detector_config, "LANE_SIDE_HOLD_FRAMES", LANE_SIDE_HOLD_FRAMES)
    current_side = lane_side_memory.get("side", "unknown")
    if candidate_side == "unknown":
        lane_side_memory["hold"] = max(0, lane_side_memory.get("hold", 0) - 1)
        return current_side if lane_side_memory.get("hold", 0) > 0 else "unknown"
    if current_side in ("unknown", candidate_side):
        lane_side_memory["side"] = candidate_side
        lane_side_memory["pending"] = None
        lane_side_memory["pending_count"] = 0
        lane_side_memory["hold"] = hold_frames
        return candidate_side
    if lane_side_memory.get("pending") != candidate_side:
        lane_side_memory["pending"] = candidate_side
        lane_side_memory["pending_count"] = 1
    else:
        lane_side_memory["pending_count"] = lane_side_memory.get("pending_count", 0) + 1
    if lane_side_memory["pending_count"] >= hold_frames:
        lane_side_memory["side"] = candidate_side
        lane_side_memory["pending"] = None
        lane_side_memory["pending_count"] = 0
        lane_side_memory["hold"] = hold_frames
        return candidate_side
    return current_side


def update_right_lane_memory(yellow_visible, right_segment_found, lane_side_memory, detector_config=None):
    # For the CSI front view, the desired lane is the right side of the yellow
    # divider. Hold that side briefly if the yellow line disappears so the
    # helper does not flicker between lanes frame-to-frame.
    if lane_side_memory is None:
        return "right" if yellow_visible and right_segment_found else "unknown"
    hold_frames = cfg_int(detector_config, "YELLOW_RIGHT_LANE_HOLD_FRAMES", YELLOW_RIGHT_LANE_HOLD_FRAMES)
    if yellow_visible and right_segment_found:
        lane_side_memory["right_lane_hold"] = hold_frames
        lane_side_memory["side"] = "right"
        return "right"
    remaining = max(0, lane_side_memory.get("right_lane_hold", 0) - 1)
    lane_side_memory["right_lane_hold"] = remaining
    if remaining > 0:
        lane_side_memory["side"] = "right"
        return "right"
    return "unknown"


def find_yellow_right_edge_near_row(yellow_boundary_mask, y, detector_config=None):
    if yellow_boundary_mask is None:
        return None
    height = yellow_boundary_mask.shape[0]
    y1 = max(0, int(y) - 3)
    y2 = min(height - 1, int(y) + 3)
    points = cv2.findNonZero(yellow_boundary_mask[y1 : y2 + 1, :])
    min_pixels = cfg_int(detector_config, "YELLOW_MIN_PIXELS_PER_SCANLINE", YELLOW_MIN_PIXELS_PER_SCANLINE)
    if points is None or len(points) < min_pixels:
        return None
    xs = points.reshape(-1, 2)[:, 0]
    return int(np.max(xs))


def select_right_lane_segment_from_yellow(segments, yellow_right_edge_x, detector_config=None):
    if yellow_right_edge_x is None:
        return None
    allowed_start_x = yellow_right_edge_x + cfg_int(detector_config, "YELLOW_LANE_SEARCH_MARGIN_PX", YELLOW_LANE_SEARCH_MARGIN_PX)
    candidates = [segment for segment in segments if segment[1] >= allowed_start_x]
    if not candidates:
        return None
    # Choose the road segment immediately to the right of the yellow boundary,
    # not the biggest segment or the one nearest image center.
    return min(candidates, key=lambda segment: max(0, segment[0] - allowed_start_x))


def apply_yellow_lane_side_clip(mask, yellow_boundary_mask, selected_lane_side):
    # If the yellow line is visible, remove road pixels on the opposite side
    # row-by-row. This keeps the safe hallway on the ego lane side even when
    # the road mask sees both lanes.
    if cv2.countNonZero(yellow_boundary_mask) == 0:
        return mask
    clipped = mask.copy()
    height, width = mask.shape[:2]
    for y in range(height):
        xs = np.where(yellow_boundary_mask[y, :] > 0)[0]
        if xs.size == 0:
            continue
        divider_x = int(np.median(xs))
        if selected_lane_side == "left":
            clipped[y, min(width - 1, divider_x + 1) :] = 0
        elif selected_lane_side == "right":
            clipped[y, : max(0, divider_x)] = 0
    return clipped


def detect_road(
    frame,
    settings,
    last_center_x,
    frames_since_valid,
    use_ego_connected_mask=None,
    detector_config=None,
    use_yellow_boundary_lock=None,
    lane_side_memory=None,
    safe_corridor_state=None,
):
    height, width = frame.shape[:2]
    roi_top = int(height * settings["ROI_top_percent"] / 100.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_boundary_mask = build_yellow_boundary_mask(hsv, detector_config)
    lower = np.array([settings["H_min"], settings["S_min"], settings["V_min"]], dtype=np.uint8)
    upper = np.array([settings["H_max"], settings["S_max"], settings["V_max"]], dtype=np.uint8)
    full_mask = cv2.inRange(hsv, lower, upper)

    # Ignore the upper image area; it usually contains walls, signs, and other distractions.
    full_mask[:roi_top, :] = 0
    yellow_boundary_mask[:roi_top, :] = 0
    if use_yellow_boundary_lock is None:
        use_yellow_boundary_lock = bool(cfg_value(detector_config, "USE_YELLOW_BOUNDARY_LOCK", USE_YELLOW_BOUNDARY_LOCK))
    # Yellow lane paint is a divider, not drivable road, so yellow pixels are
    # always removed from the road mask. The toggle only controls whether the
    # divider is also enforced as a no-cross boundary for the safe corridor.
    full_mask[yellow_boundary_mask > 0] = 0

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
    yellow_visible = cv2.countNonZero(yellow_boundary_mask) > 0
    right_lane_lock_enabled = bool(cfg_value(detector_config, "USE_RIGHT_LANE_YELLOW_LOCK", USE_RIGHT_LANE_YELLOW_LOCK))
    if right_lane_lock_enabled and yellow_visible:
        selected_lane_side = update_right_lane_memory(True, True, lane_side_memory, detector_config)
    else:
        selected_lane_side = update_lane_side_memory(
            estimate_lane_side(ego_debug, yellow_boundary_mask),
            lane_side_memory,
            detector_config,
        )
    if use_yellow_boundary_lock and selected_lane_side in ("left", "right"):
        mask = apply_yellow_lane_side_clip(mask, yellow_boundary_mask, selected_lane_side)
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
    safe_corridor = estimate_safe_corridor(
        mask,
        road_confidence,
        ego_debug,
        detector_config,
        yellow_boundary_mask,
        use_yellow_boundary_lock,
        selected_lane_side,
    )
    apply_steering_saturation_gate(safe_corridor, safe_corridor_state, detector_config)
    if right_lane_lock_enabled:
        selected_lane_side = update_right_lane_memory(
            yellow_visible,
            safe_corridor["right_lane_segment_found"],
            lane_side_memory,
            detector_config,
        )

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
        yellow_boundary_mask=yellow_boundary_mask,
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
        safe_corridor_valid=safe_corridor["safe_corridor_valid"],
        visual_helper_active=safe_corridor["visual_helper_active"],
        safe_corridor_width_mm=safe_corridor["safe_corridor_width_mm"],
        safe_corridor_width_px=safe_corridor["safe_corridor_width_px"],
        measured_lane_width_mm=safe_corridor["measured_lane_width_mm"],
        measured_lane_width_px=safe_corridor["measured_lane_width_px"],
        lane_width_valid=safe_corridor["lane_width_valid"],
        left_clearance_mm=safe_corridor["left_clearance_mm"],
        right_clearance_mm=safe_corridor["right_clearance_mm"],
        corridor_center_error_mm=safe_corridor["corridor_center_error_mm"],
        corridor_center_error_px=safe_corridor["corridor_center_error_px"],
        visual_steering_correction=safe_corridor["visual_steering_correction"],
        safe_scanline_count_valid=safe_corridor["safe_scanline_count_valid"],
        safe_scanline_rows=safe_corridor["safe_scanline_rows"],
        safe_corridor_reason=safe_corridor["safe_corridor_reason"],
        yellow_boundary_detected=int(cv2.countNonZero(yellow_boundary_mask)) > 0,
        yellow_boundary_pixel_count=int(cv2.countNonZero(yellow_boundary_mask)),
        yellow_boundary_enforced=bool(use_yellow_boundary_lock),
        selected_lane_side=selected_lane_side,
        yellow_crossing_pixels=safe_corridor["yellow_crossing_pixels"],
        yellow_right_edge_x=safe_corridor["yellow_right_edge_x"],
        right_lane_segment_found=safe_corridor["right_lane_segment_found"],
        right_lane_segment_left_x=safe_corridor["right_lane_segment_left_x"],
        right_lane_segment_right_x=safe_corridor["right_lane_segment_right_x"],
        right_lane_segment_width_px=safe_corridor["right_lane_segment_width_px"],
        right_lane_lock_active=safe_corridor["right_lane_lock_active"],
        right_lane_lock_reason=safe_corridor["right_lane_lock_reason"],
        ego_reference_x=safe_corridor["ego_reference_x"],
        camera_center_offset_px=safe_corridor["camera_center_offset_px"],
        lane_center_x=safe_corridor["lane_center_x"],
        left_space_mm=safe_corridor["left_space_mm"],
        right_space_mm=safe_corridor["right_space_mm"],
        unphysical_corridor_geometry=safe_corridor["unphysical_corridor_geometry"],
        steering_saturation_count=safe_corridor["steering_saturation_count"],
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


def estimate_safe_corridor(
    mask,
    road_confidence,
    ego_debug,
    detector_config=None,
    yellow_boundary_mask=None,
    yellow_boundary_enforced=False,
    selected_lane_side="unknown",
):
    # The safe corridor is a physical helper, not the whole road mask. It uses
    # nearby/lower road edges to estimate a lane-like hallway that the QCar2
    # can fit through without touching sidewalk or line margins.
    height, width = mask.shape[:2]
    camera_center_offset_px = cfg_float(detector_config, "CAMERA_CENTER_OFFSET_PX", CAMERA_CENTER_OFFSET_PX)
    ego_reference_x = (
        width * cfg_float(detector_config, "CAMERA_CENTER_X_RATIO", CAMERA_CENTER_X_RATIO)
        + camera_center_offset_px
    )
    rows = np.linspace(
        int(height * cfg_float(detector_config, "SAFE_SCANLINE_END_RATIO", SAFE_SCANLINE_END_RATIO)),
        int(height * cfg_float(detector_config, "SAFE_SCANLINE_START_RATIO", SAFE_SCANLINE_START_RATIO)),
        cfg_int(detector_config, "SAFE_SCANLINE_COUNT", SAFE_SCANLINE_COUNT),
    ).astype(int)

    safe_rows = []
    right_lane_rows = []
    yellow_right_edges = []
    yellow_visible_on_rows = False
    right_lane_lock_active = False
    right_lane_lock_reason = "disabled"
    use_right_lane_lock = (
        yellow_boundary_enforced
        and bool(cfg_value(detector_config, "USE_RIGHT_LANE_YELLOW_LOCK", USE_RIGHT_LANE_YELLOW_LOCK))
        and bool(cfg_value(detector_config, "RIGHT_LANE_FROM_YELLOW", RIGHT_LANE_FROM_YELLOW))
        and str(cfg_value(detector_config, "YELLOW_LANE_SIDE", YELLOW_LANE_SIDE)).lower() == "right"
    )
    if use_right_lane_lock:
        right_lane_lock_reason = "yellow_not_visible"
    for index, y in enumerate(rows):
        y = min(max(0, int(y)), height - 1)
        segments = find_road_segments(mask[y, :], cfg_int(detector_config, "MIN_SEGMENT_WIDTH_PX", MIN_SEGMENT_WIDTH_PX))
        if not segments:
            continue
        yellow_right_edge_x = find_yellow_right_edge_near_row(yellow_boundary_mask, y, detector_config)
        selected_segment = None
        if use_right_lane_lock and yellow_right_edge_x is not None:
            yellow_visible_on_rows = True
            yellow_right_edges.append(yellow_right_edge_x)
            selected_segment = select_right_lane_segment_from_yellow(segments, yellow_right_edge_x, detector_config)
            if selected_segment is not None:
                right_lane_rows.append(selected_segment)
                right_lane_lock_active = True
                right_lane_lock_reason = "yellow_right_lane_selected"
        if selected_segment is None:
            if use_right_lane_lock and yellow_right_edge_x is not None:
                right_lane_lock_reason = "yellow_visible_no_right_segment"
                continue
        selected_segment = min(segments, key=lambda segment: abs(segment[2] - ego_reference_x))
        left_x, right_x, segment_center_x = selected_segment
        lane_width_px = max(1.0, float(right_x - left_x))
        safe_rows.append(
            {
                "y": y,
                "left_x": int(left_x),
                "right_x": int(right_x),
                "center_x": float(segment_center_x),
                "lane_width_px": lane_width_px,
                "yellow_right_edge_x": yellow_right_edge_x,
                # Lower rows are closer to the car, so they should dominate
                # the helper correction more than farther scanlines.
                "weight": cfg_float(detector_config, "SAFE_SCANLINE_NEAR_WEIGHT_BIAS", SAFE_SCANLINE_NEAR_WEIGHT_BIAS) - (index / max(1, len(rows) - 1)),
            }
        )

    if not safe_rows:
        reason = "yellow_visible_no_right_segment" if yellow_visible_on_rows else "edge_detection_failed"
        return make_safe_corridor_result(
            reason,
            [],
            detector_config,
            yellow_right_edge_x=max(yellow_right_edges) if yellow_right_edges else None,
            right_lane_lock_active=False,
            right_lane_lock_reason=right_lane_lock_reason if yellow_visible_on_rows else "yellow_not_visible",
        )

    weights = np.array([row["weight"] for row in safe_rows], dtype=np.float32)
    widths_px = np.array([row["lane_width_px"] for row in safe_rows], dtype=np.float32)
    reference_width_px = float(np.median(widths_px))
    if reference_width_px <= 0:
        return make_safe_corridor_result("edge_detection_failed", safe_rows, detector_config)

    lane_width_mm_values = []
    left_clearances = []
    right_clearances = []
    left_spaces = []
    right_spaces = []
    center_errors_mm = []
    center_errors_px = []
    lane_centers_x = []
    safe_width_px_values = []
    safe_width_mm = cfg_float(detector_config, "SAFE_HALLWAY_WIDTH_MM", SAFE_HALLWAY_WIDTH_MM)
    lane_width_mm = cfg_float(detector_config, "LANE_WIDTH_MM", LANE_WIDTH_MM)
    car_half_width_mm = cfg_float(detector_config, "CAR_WIDTH_MM", CAR_WIDTH_MM) / 2.0

    for row in safe_rows:
        # Row-local scale is intentionally simple: the detected lane width at
        # this row maps to the known physical lane width. This avoids full
        # camera calibration/homography while still producing useful mm values.
        local_mm_per_px = lane_width_mm / max(1.0, row["lane_width_px"])
        measured_lane_width_mm = row["lane_width_px"] * local_mm_per_px
        lane_center_x = (row["left_x"] + row["right_x"]) / 2.0
        corridor_error_px = lane_center_x - ego_reference_x
        corridor_error_mm = corridor_error_px * local_mm_per_px + cfg_float(detector_config, "CAMERA_CENTER_OFFSET_MM", CAMERA_CENTER_OFFSET_MM)
        left_space_mm = (ego_reference_x - row["left_x"]) * local_mm_per_px
        right_space_mm = (row["right_x"] - ego_reference_x) * local_mm_per_px
        left_clearance_mm = left_space_mm - car_half_width_mm
        right_clearance_mm = right_space_mm - car_half_width_mm
        row["mm_per_px"] = local_mm_per_px
        row["lane_center_x"] = lane_center_x
        row["safe_left_x"] = lane_center_x - (safe_width_mm / local_mm_per_px / 2.0)
        row["safe_right_x"] = lane_center_x + (safe_width_mm / local_mm_per_px / 2.0)
        row["measured_lane_width_mm"] = measured_lane_width_mm
        row["left_space_mm"] = left_space_mm
        row["right_space_mm"] = right_space_mm
        row["left_clearance_mm"] = left_clearance_mm
        row["right_clearance_mm"] = right_clearance_mm
        row["corridor_error_mm"] = corridor_error_mm
        lane_width_mm_values.append(measured_lane_width_mm)
        left_spaces.append(left_space_mm)
        right_spaces.append(right_space_mm)
        left_clearances.append(left_clearance_mm)
        right_clearances.append(right_clearance_mm)
        center_errors_mm.append(corridor_error_mm)
        center_errors_px.append(corridor_error_px)
        lane_centers_x.append(lane_center_x)
        safe_width_px_values.append(safe_width_mm / local_mm_per_px)

    measured_lane_width_mm = float(np.average(lane_width_mm_values, weights=weights))
    measured_lane_width_px = float(np.average(widths_px, weights=weights))
    left_clearance_mm = float(np.average(left_clearances, weights=weights))
    right_clearance_mm = float(np.average(right_clearances, weights=weights))
    left_space_mm = float(np.average(left_spaces, weights=weights))
    right_space_mm = float(np.average(right_spaces, weights=weights))
    corridor_center_error_mm = float(np.average(center_errors_mm, weights=weights))
    corridor_center_error_px = float(np.average(center_errors_px, weights=weights))
    lane_center_x = float(np.average(lane_centers_x, weights=weights))
    safe_width_px = float(np.average(safe_width_px_values, weights=weights))

    min_width = cfg_float(detector_config, "MIN_VALID_LANE_WIDTH_MM", MIN_VALID_LANE_WIDTH_MM)
    max_width = cfg_float(detector_config, "MAX_VALID_LANE_WIDTH_MM", MAX_VALID_LANE_WIDTH_MM)
    lane_width_valid = min_width <= measured_lane_width_mm <= max_width
    yellow_crossing_pixels = count_safe_corridor_yellow_crossing(
        safe_rows,
        yellow_boundary_mask,
        height,
        width,
    )
    yellow_visible = yellow_boundary_mask is not None and cv2.countNonZero(yellow_boundary_mask) > 0
    wide_blob_no_yellow = (
        bool(cfg_value(detector_config, "USE_NO_YELLOW_WIDE_BLOB_GATE", USE_NO_YELLOW_WIDE_BLOB_GATE))
        and not yellow_visible
        and (
            measured_lane_width_px > width * cfg_float(detector_config, "NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO", NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO)
            or measured_lane_width_mm > cfg_float(detector_config, "NO_YELLOW_MAX_MEASURED_WIDTH_MM", NO_YELLOW_MAX_MEASURED_WIDTH_MM)
        )
    )
    unphysical_corridor_geometry = (
        left_clearance_mm < cfg_float(detector_config, "MIN_CLEARANCE_MM", MIN_CLEARANCE_MM)
        or right_clearance_mm < cfg_float(detector_config, "MIN_CLEARANCE_MM", MIN_CLEARANCE_MM)
        or abs(corridor_center_error_mm) > cfg_float(detector_config, "MAX_REASONABLE_CORRIDOR_ERROR_MM", MAX_REASONABLE_CORRIDOR_ERROR_MM)
        or abs(left_clearance_mm) > cfg_float(detector_config, "MAX_REASONABLE_CLEARANCE_MM", MAX_REASONABLE_CLEARANCE_MM)
        or abs(right_clearance_mm) > cfg_float(detector_config, "MAX_REASONABLE_CLEARANCE_MM", MAX_REASONABLE_CLEARANCE_MM)
    )
    reason = "valid"
    if not ego_debug["ego_component_found"]:
        reason = "ego_component_missing"
    elif road_confidence < cfg_float(detector_config, "SAFE_MIN_ROAD_CONFIDENCE", SAFE_MIN_ROAD_CONFIDENCE):
        reason = "low_confidence"
    elif use_right_lane_lock and yellow_visible_on_rows and not right_lane_rows:
        reason = "yellow_visible_no_right_segment"
    elif len(safe_rows) < cfg_int(detector_config, "SAFE_MIN_VALID_SCANLINES", SAFE_MIN_VALID_SCANLINES):
        reason = "insufficient_scanlines"
    elif (
        yellow_boundary_enforced
        and yellow_crossing_pixels > cfg_int(detector_config, "YELLOW_MAX_CROSSING_PIXELS", YELLOW_MAX_CROSSING_PIXELS)
    ):
        reason = "crosses_yellow_boundary"
    elif wide_blob_no_yellow:
        reason = "wide_blob_no_yellow"
    elif unphysical_corridor_geometry:
        reason = "unphysical_corridor_geometry"
    elif measured_lane_width_mm < min_width:
        reason = "too_narrow"
    elif measured_lane_width_mm > max_width:
        reason = "too_wide"

    valid = reason == "valid" and lane_width_valid
    steering = 0.0
    if valid:
        deadband = cfg_float(detector_config, "SAFE_ERROR_DEADBAND_MM", SAFE_ERROR_DEADBAND_MM)
        if abs(corridor_center_error_mm) > deadband:
            steering = cfg_float(detector_config, "SAFE_STEERING_GAIN", SAFE_STEERING_GAIN) * corridor_center_error_mm
            steering = clamp(steering, -cfg_float(detector_config, "SAFE_MAX_STEERING_CORRECTION", SAFE_MAX_STEERING_CORRECTION), cfg_float(detector_config, "SAFE_MAX_STEERING_CORRECTION", SAFE_MAX_STEERING_CORRECTION))

    return {
        "safe_corridor_valid": valid,
        "visual_helper_active": valid,
        "safe_corridor_width_mm": safe_width_mm,
        "safe_corridor_width_px": safe_width_px,
        "measured_lane_width_mm": measured_lane_width_mm,
        "measured_lane_width_px": measured_lane_width_px,
        "lane_width_valid": lane_width_valid,
        "left_clearance_mm": left_clearance_mm,
        "right_clearance_mm": right_clearance_mm,
        "left_space_mm": left_space_mm,
        "right_space_mm": right_space_mm,
        "corridor_center_error_mm": corridor_center_error_mm,
        "corridor_center_error_px": corridor_center_error_px,
        "visual_steering_correction": steering if valid else 0.0,
        "safe_scanline_count_valid": len(safe_rows),
        "safe_scanline_rows": safe_rows,
        "safe_corridor_reason": reason,
        "yellow_crossing_pixels": yellow_crossing_pixels,
        "yellow_right_edge_x": max(yellow_right_edges) if yellow_right_edges else None,
        "right_lane_segment_found": bool(right_lane_rows),
        "right_lane_segment_left_x": int(right_lane_rows[0][0]) if right_lane_rows else None,
        "right_lane_segment_right_x": int(right_lane_rows[0][1]) if right_lane_rows else None,
        "right_lane_segment_width_px": float(right_lane_rows[0][1] - right_lane_rows[0][0]) if right_lane_rows else None,
        "right_lane_lock_active": bool(right_lane_lock_active),
        "right_lane_lock_reason": right_lane_lock_reason if use_right_lane_lock else "disabled",
        "ego_reference_x": ego_reference_x,
        "camera_center_offset_px": camera_center_offset_px,
        "lane_center_x": lane_center_x,
        "unphysical_corridor_geometry": bool(unphysical_corridor_geometry),
        "steering_saturation_count": 0,
    }


def make_safe_corridor_result(
    reason,
    rows,
    detector_config=None,
    yellow_right_edge_x=None,
    right_lane_lock_active=False,
    right_lane_lock_reason="disabled",
):
    return {
        "safe_corridor_valid": False,
        "visual_helper_active": False,
        "safe_corridor_width_mm": cfg_float(detector_config, "SAFE_HALLWAY_WIDTH_MM", SAFE_HALLWAY_WIDTH_MM),
        "safe_corridor_width_px": None,
        "measured_lane_width_mm": None,
        "measured_lane_width_px": None,
        "lane_width_valid": False,
        "left_clearance_mm": None,
        "right_clearance_mm": None,
        "left_space_mm": None,
        "right_space_mm": None,
        "corridor_center_error_mm": None,
        "corridor_center_error_px": None,
        "visual_steering_correction": 0.0,
        "safe_scanline_count_valid": len(rows),
        "safe_scanline_rows": rows,
        "safe_corridor_reason": reason,
        "yellow_crossing_pixels": 0,
        "yellow_right_edge_x": yellow_right_edge_x,
        "right_lane_segment_found": False,
        "right_lane_segment_left_x": None,
        "right_lane_segment_right_x": None,
        "right_lane_segment_width_px": None,
        "right_lane_lock_active": right_lane_lock_active,
        "right_lane_lock_reason": right_lane_lock_reason,
        "ego_reference_x": 0.0,
        "camera_center_offset_px": cfg_float(detector_config, "CAMERA_CENTER_OFFSET_PX", CAMERA_CENTER_OFFSET_PX),
        "lane_center_x": None,
        "unphysical_corridor_geometry": False,
        "steering_saturation_count": 0,
    }


def apply_steering_saturation_gate(safe_corridor, safe_corridor_state, detector_config=None):
    # A helper that sits at the steering clamp for many frames is not giving
    # useful local guidance. Turn it off until geometry becomes believable.
    max_correction = cfg_float(detector_config, "SAFE_MAX_STEERING_CORRECTION", SAFE_MAX_STEERING_CORRECTION)
    max_frames = cfg_int(detector_config, "MAX_STEERING_SATURATION_FRAMES", MAX_STEERING_SATURATION_FRAMES)
    saturated = (
        safe_corridor["safe_corridor_valid"]
        and abs(safe_corridor["visual_steering_correction"]) >= max_correction - 1e-9
    )
    if safe_corridor_state is None:
        count = 1 if saturated else 0
    else:
        count = safe_corridor_state.get("steering_saturation_count", 0) + 1 if saturated else 0
        safe_corridor_state["steering_saturation_count"] = count
    safe_corridor["steering_saturation_count"] = count
    if count > max_frames:
        safe_corridor["safe_corridor_valid"] = False
        safe_corridor["visual_helper_active"] = False
        safe_corridor["visual_steering_correction"] = 0.0
        safe_corridor["safe_corridor_reason"] = "steering_saturated_too_long"


def apply_drift_only_gate(result, turn_hint, drift_state, detector_config=None):
    """Clamp the camera helper to drift correction in straight-ish corridors.

    The normal QCar2 controller remains in charge. This helper only nudges
    lateral drift when the blue corridor is physically valid and the road is
    not turning. Any uncertain condition disables the helper and forces the
    correction to exactly zero.
    """
    if drift_state is None:
        drift_state = {}

    if not result.safe_corridor_valid:
        drift_state["valid_frames"] = 0
        result.visual_helper_active = False
        result.visual_steering_correction = 0.0
        if result.safe_corridor_reason == "valid":
            result.safe_corridor_reason = "safe_corridor_invalid"
        return

    drift_state["valid_frames"] = drift_state.get("valid_frames", 0) + 1
    min_valid_frames = cfg_int(detector_config, "DRIFT_MIN_SAFE_CORRIDOR_VALID_FRAMES", 3)
    if drift_state["valid_frames"] < min_valid_frames:
        result.visual_helper_active = False
        result.visual_steering_correction = 0.0
        result.safe_corridor_reason = "waiting_for_stable_corridor"
        return

    if bool(cfg_value(detector_config, "DRIFT_DISABLE_ON_TURN_HINT", True)) and turn_hint in ("left", "right"):
        result.visual_helper_active = False
        result.visual_steering_correction = 0.0
        result.safe_corridor_reason = "turning_helper_disabled"
        return

    max_curve = cfg_float(detector_config, "DRIFT_MAX_ABS_CURVE_ERROR_PX", 35.0)
    if bool(cfg_value(detector_config, "DRIFT_HELPER_ONLY_ON_STRAIGHT", True)):
        if result.curve_error_px is not None and abs(result.curve_error_px) > max_curve:
            result.visual_helper_active = False
            result.visual_steering_correction = 0.0
            result.safe_corridor_reason = "curve_error_too_high"
            return

    max_error_mm = cfg_float(detector_config, "DRIFT_MAX_ABS_CORRIDOR_ERROR_MM", 40.0)
    if result.corridor_center_error_mm is None or abs(result.corridor_center_error_mm) > max_error_mm:
        result.visual_helper_active = False
        result.visual_steering_correction = 0.0
        result.safe_corridor_reason = "corridor_error_too_high"
        return

    gain = cfg_float(detector_config, "DRIFT_HELPER_GAIN", cfg_float(detector_config, "SAFE_STEERING_GAIN", SAFE_STEERING_GAIN))
    max_output = cfg_float(detector_config, "DRIFT_HELPER_MAX_OUTPUT", cfg_float(detector_config, "SAFE_MAX_STEERING_CORRECTION", SAFE_MAX_STEERING_CORRECTION))
    result.visual_helper_active = True
    result.visual_steering_correction = clamp(gain * result.corridor_center_error_mm, -max_output, max_output)


def count_safe_corridor_yellow_crossing(safe_rows, yellow_boundary_mask, height, width):
    if yellow_boundary_mask is None or not safe_rows:
        return 0
    corridor_mask = np.zeros((height, width), dtype=np.uint8)
    left_points = []
    right_points = []
    for row in safe_rows:
        if "safe_left_x" not in row or "safe_right_x" not in row:
            continue
        y = int(row["y"])
        left_points.append((int(round(row["safe_left_x"])), y))
        right_points.append((int(round(row["safe_right_x"])), y))
    if len(left_points) < 2 or len(right_points) < 2:
        return 0
    polygon = np.array(left_points + list(reversed(right_points)), dtype=np.int32)
    cv2.fillPoly(corridor_mask, [polygon], 255)
    return int(cv2.countNonZero(cv2.bitwise_and(corridor_mask, yellow_boundary_mask)))


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


def draw_visualization(frame, result, turn_hint, settings=None, show_inactive_helper=False):
    output = frame.copy()
    height, width = output.shape[:2]

    road_color = np.zeros_like(output)
    road_color[:, :] = cfg_color(settings, "ROAD_OVERLAY_COLOR_BGR", DEFAULT_SETTINGS["ROAD_OVERLAY_COLOR_BGR"])
    road_alpha = cfg_float(settings, "ROAD_OVERLAY_ALPHA", DEFAULT_SETTINGS["ROAD_OVERLAY_ALPHA"])
    output = np.where(result.mask[:, :, None] > 0, cv2.addWeighted(output, 1.0 - road_alpha, road_color, road_alpha, 0), output)

    for left_x, y in result.boundary_points:
        cv2.circle(output, (left_x, y), 4, (0, 255, 255), -1)

    draw_detected_centerline(output, result, show_inactive_helper)
    draw_safe_corridor(output, result, settings, show_inactive_helper)
    draw_yellow_boundary(output, result)
    draw_ego_anchor_debug(output, result)
    cv2.line(output, (width // 2, height), (width // 2, int(height * 0.45)), (255, 255, 255), 1)

    draw_debug_text(output, result, turn_hint)
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


def draw_safe_corridor(output, result, settings=None, show_inactive_helper=False):
    if not result.visual_helper_active and not show_inactive_helper:
        return
    if not result.safe_scanline_rows:
        return

    left_points = []
    right_points = []
    for row in result.safe_scanline_rows:
        if "safe_left_x" not in row or "safe_right_x" not in row:
            continue
        y = int(row["y"])
        left_points.append((int(round(row["safe_left_x"])), y))
        right_points.append((int(round(row["safe_right_x"])), y))

    if len(left_points) < 2 or len(right_points) < 2:
        return

    polygon = np.array(left_points + list(reversed(right_points)), dtype=np.int32)
    overlay = output.copy()
    active_color = cfg_color(settings, "ACTIVE_SAFE_CORRIDOR_COLOR_BGR", DEFAULT_SETTINGS["ACTIVE_SAFE_CORRIDOR_COLOR_BGR"])
    inactive_color = cfg_color(settings, "INACTIVE_SAFE_CORRIDOR_COLOR_BGR", DEFAULT_SETTINGS["INACTIVE_SAFE_CORRIDOR_COLOR_BGR"])
    fill_color = active_color if result.visual_helper_active else inactive_color
    edge_color = (255, 220, 80) if result.visual_helper_active else inactive_color
    cv2.fillPoly(overlay, [polygon], fill_color)
    alpha = (
        cfg_float(settings, "ACTIVE_SAFE_CORRIDOR_ALPHA", DEFAULT_SETTINGS["ACTIVE_SAFE_CORRIDOR_ALPHA"])
        if result.visual_helper_active
        else cfg_float(settings, "INACTIVE_SAFE_CORRIDOR_ALPHA", DEFAULT_SETTINGS["INACTIVE_SAFE_CORRIDOR_ALPHA"])
    )
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)
    thickness = 3 if result.visual_helper_active else 1
    cv2.polylines(output, [np.array(left_points, dtype=np.int32)], False, edge_color, thickness, cv2.LINE_AA)
    cv2.polylines(output, [np.array(right_points, dtype=np.int32)], False, edge_color, thickness, cv2.LINE_AA)

    center_points = []
    for left, right in zip(left_points, right_points):
        center_points.append(((left[0] + right[0]) // 2, left[1]))
    cv2.polylines(output, [np.array(center_points, dtype=np.int32)], False, (255, 255, 255), 2 if result.visual_helper_active else 1, cv2.LINE_AA)
    if not result.visual_helper_active:
        cv2.putText(output, "INACTIVE DEBUG ONLY", center_points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2, cv2.LINE_AA)


def draw_yellow_boundary(output, result):
    if result.yellow_boundary_pixel_count <= 0:
        return
    yellow_pixels = result.yellow_boundary_mask > 0
    output[yellow_pixels] = cv2.addWeighted(
        output[yellow_pixels],
        0.35,
        np.full_like(output[yellow_pixels], (0, 255, 255)),
        0.65,
        0,
    )
    if result.yellow_right_edge_x is not None:
        x = int(result.yellow_right_edge_x)
        cv2.line(output, (x, 0), (x, output.shape[0] - 1), (0, 255, 255), 2, cv2.LINE_AA)
        start_x = x + YELLOW_LANE_SEARCH_MARGIN_PX
        cv2.line(output, (start_x, 0), (start_x, output.shape[0] - 1), (255, 255, 0), 1, cv2.LINE_AA)
    if result.right_lane_lock_active:
        cv2.putText(output, "RIGHT LANE LOCK", (output.shape[1] - 280, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    elif result.safe_corridor_reason == "wide_blob_no_yellow":
        cv2.putText(output, "WIDE BLOB NO YELLOW - HELPER OFF", (output.shape[1] - 520, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2, cv2.LINE_AA)


def draw_detected_centerline(output, result, show_inactive_helper=False):
    if not result.scan_points:
        return

    height, width = output.shape[:2]
    centerline_points = [(int(center_x), int(y)) for center_x, _left_x, _right_x, y in result.scan_points]
    path_points = [(width // 2, height - 8)] + centerline_points

    if result.visual_helper_active and len(path_points) >= 2:
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

    elif show_inactive_helper and len(path_points) >= 2:
        path_array = np.array(path_points, dtype=np.int32)
        cv2.polylines(output, [path_array], False, (120, 120, 120), 1, cv2.LINE_AA)

    for x, y in centerline_points:
        cv2.circle(output, (x, y), 5, (0, 255, 0), -1)


def draw_debug_text(output, result, turn_hint):
    error = result.road_center_error_px
    error_text = "None" if error is None else f"{error:.1f}"
    curve_text = "None" if result.curve_error_px is None else f"{result.curve_error_px:.1f}"
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
        f"safe_corridor_valid: {result.safe_corridor_valid}",
        f"helper_active: {result.visual_helper_active}",
        f"ego_ref_x: {result.ego_reference_x:.1f}",
        f"cam_offset_px: {result.camera_center_offset_px:.1f}",
        f"lane_center_x: {safe_fmt(result.lane_center_x)}",
        f"lane_width_mm: {safe_fmt(result.measured_lane_width_mm)}",
        f"L/R clear_mm: {safe_fmt(result.left_clearance_mm)}/{safe_fmt(result.right_clearance_mm)}",
        f"corridor_err_mm: {safe_fmt(result.corridor_center_error_mm)}",
        f"steer_helper: {result.visual_steering_correction:.3f}",
        f"safe_reason: {result.safe_corridor_reason}",
        f"yellow_lock: {result.yellow_boundary_enforced}",
        f"lane_side: {result.selected_lane_side}",
        f"yellow_cross_px: {result.yellow_crossing_pixels}",
        f"right_lane_lock: {result.right_lane_lock_active}",
        f"right_lane_reason: {result.right_lane_lock_reason}",
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


def safe_fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.1f}"


def build_display_grid(original, result, visualization, settings=None, show_inactive_helper=False):
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    road_overlay = build_road_overlay(original, result, settings, show_inactive_helper)

    top = np.hstack([label_image(original, "Original RGB"), label_image(mask_bgr, "Road Mask")])
    bottom = np.hstack([label_image(road_overlay, "Road Overlay"), label_image(visualization, "Safe Corridor Debug")])
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
    return config_values_from_payload(loaded)


def load_video_settings(args):
    if args.use_default_config:
        return DEFAULT_SETTINGS.copy(), "DEFAULT_SETTINGS"

    config_path = Path(args.config or CONFIG_FILE)
    if config_path.exists():
        source = CONFIG_FILE if str(config_path) == CONFIG_FILE else str(config_path)
        print(f"Loaded HSV settings from {source}")
        return load_settings_file(config_path), source

    raise RuntimeError(f"Could not find config file: {config_path}")


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
    # Good and difficult samples become visual evidence for future camera
    # retuning, so they live in stable folders instead of per-analysis runs.
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
    # clean that mask. The saved JSON also carries physical helper settings so
    # the same file can be used later by the ROS2 node.
    path = Path(path)
    payload = {
        "name": path.stem,
        "camera_type": "csi_front" if "csi" in str(path).lower() else "rgb",
        "created_from": "manual_tuning",
        "notes": "Manual tuning baseline for the RGB drift helper.",
        "units": "mm",
        "settings": settings.copy(),
    }
    payload["settings"].update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": str(source_video),
            "source_frame_index": int(frame_index),
            "note": MANUAL_TUNING_NOTE,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved manual tuning config to {path}")


def save_manual_tuning_session(path, source_video, last_saved_frame_index, config_output_path, session, settings):
    # The session file records which frames were useful during manual tuning.
    # That makes later CSI/RealSense retuning easier to repeat.
    payload = {
        "source_video": str(source_video),
        "last_saved_frame_index": last_saved_frame_index,
        "config_output_path": str(config_output_path),
        "good_sample_frames": session["good_sample_frames"],
        "difficult_sample_frames": session["difficult_sample_frames"],
        "debug_snapshot_frames": session["debug_snapshot_frames"],
        "final_config": {name: value for name, value in settings.items() if is_json_scalar(value)},
        "notes_for_retuning": "Use these saved frames when checking future CSI or RealSense camera adjustments.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_tuning_sample(kind, folders, frame_index, frame, result, overlay, debug_frame):
    # Good samples show frames where the current mask works well. Difficult
    # samples show where manual tuning still needs attention.
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
    config = {"config_source": config_source}
    config.update({key: value for key, value in settings.items() if is_json_scalar(value)})
    return config


def save_config_used(path, settings, config_source):
    # This JSON makes a run reproducible because it records the exact HSV,
    # ROI, scanline, smoothing, and deadband values used for the video.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(get_config_used(settings, config_source), file, indent=2)


def build_road_overlay(frame, mask_or_result, settings=None, show_inactive_helper=False):
    overlay = frame.copy()
    result = mask_or_result if hasattr(mask_or_result, "mask") else None
    mask = result.mask if result is not None else mask_or_result
    mask_pixels = mask > 0
    if np.any(mask_pixels):
        road_color = cfg_color(settings, "ROAD_OVERLAY_COLOR_BGR", DEFAULT_SETTINGS["ROAD_OVERLAY_COLOR_BGR"])
        road_alpha = cfg_float(settings, "ROAD_OVERLAY_ALPHA", DEFAULT_SETTINGS["ROAD_OVERLAY_ALPHA"])
        overlay[mask_pixels] = cv2.addWeighted(
            overlay[mask_pixels],
            1.0 - road_alpha,
            np.full_like(overlay[mask_pixels], road_color),
            road_alpha,
            0,
        )
    if result is not None:
        draw_safe_corridor(overlay, result, settings, show_inactive_helper)
    return overlay


def safe_number(value, default=""):
    if value is None:
        return default
    return value


def is_json_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def build_helper_output(result, config=None, timestamp=None, turn_hint=None):
    """Return the future ROS2 payload as a plain JSON-serializable dict."""
    config = config or {}
    if turn_hint is None:
        turn_hint = compute_turn_hint(result.curve_error_px, config)
    camera_type = str(config.get("camera_type", "unknown"))
    return {
        "timestamp": timestamp,
        "road_detected": bool(result.road_detected),
        "road_confidence": float(result.road_confidence),
        "safe_corridor_valid": bool(result.safe_corridor_valid),
        "visual_helper_active": bool(result.visual_helper_active),
        "visual_steering_correction": float(result.visual_steering_correction),
        "corridor_center_error_mm": result.corridor_center_error_mm,
        "left_clearance_mm": result.left_clearance_mm,
        "right_clearance_mm": result.right_clearance_mm,
        "turn_hint": turn_hint,
        "safe_corridor_reason": result.safe_corridor_reason,
        "yellow_boundary_detected": bool(result.yellow_boundary_detected),
        "right_lane_lock_active": bool(result.right_lane_lock_active),
        "camera_type": camera_type,
    }


def build_telemetry_row(frame_index, time_sec, result, turn_hint, processing_fps, camera_type="unknown", settings=None):
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
        "safe_corridor_valid": result.safe_corridor_valid,
        "visual_helper_active": result.visual_helper_active,
        "safe_corridor_width_mm": f"{result.safe_corridor_width_mm:.3f}",
        "safe_corridor_width_px": safe_number(result.safe_corridor_width_px),
        "measured_lane_width_mm": safe_number(result.measured_lane_width_mm),
        "measured_lane_width_px": safe_number(result.measured_lane_width_px),
        "lane_width_valid": result.lane_width_valid,
        "left_clearance_mm": safe_number(result.left_clearance_mm),
        "right_clearance_mm": safe_number(result.right_clearance_mm),
        "corridor_center_error_mm": safe_number(result.corridor_center_error_mm),
        "corridor_center_error_px": safe_number(result.corridor_center_error_px),
        "visual_steering_correction": f"{result.visual_steering_correction:.5f}",
        "safe_scanline_count_valid": result.safe_scanline_count_valid,
        "safe_corridor_reason": result.safe_corridor_reason,
        "yellow_boundary_detected": result.yellow_boundary_detected,
        "yellow_boundary_pixel_count": result.yellow_boundary_pixel_count,
        "yellow_boundary_enforced": result.yellow_boundary_enforced,
        "selected_lane_side": result.selected_lane_side,
        "yellow_crossing_pixels": result.yellow_crossing_pixels,
        "yellow_right_edge_x": safe_number(result.yellow_right_edge_x),
        "right_lane_segment_found": result.right_lane_segment_found,
        "right_lane_segment_left_x": safe_number(result.right_lane_segment_left_x),
        "right_lane_segment_right_x": safe_number(result.right_lane_segment_right_x),
        "right_lane_segment_width_px": safe_number(result.right_lane_segment_width_px),
        "right_lane_lock_active": result.right_lane_lock_active,
        "right_lane_lock_reason": result.right_lane_lock_reason,
        "ego_reference_x": f"{result.ego_reference_x:.2f}",
        "camera_center_offset_px": f"{result.camera_center_offset_px:.2f}",
        "lane_center_x": safe_number(result.lane_center_x),
        "left_space_mm": safe_number(result.left_space_mm),
        "right_space_mm": safe_number(result.right_space_mm),
        "unphysical_corridor_geometry": result.unphysical_corridor_geometry,
        "steering_saturation_count": result.steering_saturation_count,
        "helper_output_json": json.dumps(
            build_helper_output(result, settings or {"camera_type": camera_type}, timestamp=time_sec, turn_hint=turn_hint),
            sort_keys=True,
        ),
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


def detect_frame_events(frame_index, time_sec, result, turn_hint, previous_state):
    events = []
    current_state = {
        "road_detected": result.road_detected,
        "road_center_error_px": result.road_center_error_px,
        "curve_error_px": result.curve_error_px,
        "turn_hint": turn_hint,
        "tracked_center_valid": result.tracked_center_valid,
        "low_confidence": result.road_confidence < 0.60,
        "rejected_scanlines_high": result.rejected_scanlines >= 5,
        "safe_corridor_valid": result.safe_corridor_valid,
        "safe_corridor_reason": result.safe_corridor_reason,
    }

    if previous_state is None:
        if current_state["low_confidence"]:
            events.append(("low_confidence_started", "", f"{result.road_confidence:.3f}", "Road confidence dropped below 0.60."))
        if current_state["rejected_scanlines_high"]:
            events.append(("rejected_scanlines_high_started", "", result.rejected_scanlines, "Five or more scanlines were rejected."))
        if not current_state["safe_corridor_valid"]:
            events.append(("safe_corridor_invalid_started", "", result.safe_corridor_reason, "Safe corridor helper is inactive."))
        return events, current_state

    if not previous_state["low_confidence"] and current_state["low_confidence"]:
        events.append(("low_confidence_started", "", f"{result.road_confidence:.3f}", "Road confidence dropped below 0.60."))
    if previous_state["low_confidence"] and not current_state["low_confidence"]:
        events.append(("low_confidence_ended", "", f"{result.road_confidence:.3f}", "Road confidence recovered to 0.60 or higher."))
    if not previous_state["rejected_scanlines_high"] and current_state["rejected_scanlines_high"]:
        events.append(("rejected_scanlines_high_started", "", result.rejected_scanlines, "Five or more scanlines were rejected."))
    if previous_state["rejected_scanlines_high"] and not current_state["rejected_scanlines_high"]:
        events.append(("rejected_scanlines_high_ended", "", result.rejected_scanlines, "Rejected scanlines dropped below five."))
    if previous_state["safe_corridor_valid"] and not current_state["safe_corridor_valid"]:
        events.append(("safe_corridor_invalid_started", "valid", result.safe_corridor_reason, "Safe corridor changed from valid to inactive."))
    if not previous_state["safe_corridor_valid"] and current_state["safe_corridor_valid"]:
        events.append(("safe_corridor_invalid_ended", previous_state["safe_corridor_reason"], "valid", "Safe corridor helper became active."))
    if (
        not current_state["safe_corridor_valid"]
        and previous_state["safe_corridor_reason"] != current_state["safe_corridor_reason"]
    ):
        events.append(("safe_corridor_reason_changed", previous_state["safe_corridor_reason"], result.safe_corridor_reason, "Safe corridor inactive reason changed."))

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
    if "safe_corridor_invalid_started" in event_types:
        return f"safe_corridor_{result.safe_corridor_reason}"
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
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_yellow_boundary_mask.jpg"), result.yellow_boundary_mask)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_overlay.jpg"), overlay)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_debug.jpg"), debug_frame)
    cv2.imwrite(str(folder / f"frame_{frame_index:06d}_safe_corridor_debug.jpg"), debug_frame)
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


def stddev(values):
    clean_values = [float(value) for value in values if value not in (None, "")]
    if len(clean_values) < 2:
        return 0.0
    mean = sum(clean_values) / len(clean_values)
    return (sum((value - mean) ** 2 for value in clean_values) / len(clean_values)) ** 0.5


def numeric_values(rows, key):
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


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
        f"Safe corridor valid percent: {stats['safe_corridor_valid_percent']:.1f}%",
        f"Visual helper active percent: {stats['visual_helper_active_percent']:.1f}%",
        f"Average measured lane width: {stats['average_measured_lane_width_mm']:.2f} mm",
        f"Average absolute corridor center error: {stats['average_abs_corridor_center_error_mm']:.2f} mm",
        f"Average valid left clearance: {stats['average_valid_left_clearance_mm']:.2f} mm",
        f"Average valid right clearance: {stats['average_valid_right_clearance_mm']:.2f} mm",
        f"Average valid absolute corridor error: {stats['average_valid_abs_corridor_center_error_mm']:.2f} mm",
        f"Right-lane lock active percent: {stats['right_lane_lock_active_percent']:.1f}%",
        f"Right-lane segment found percent: {stats['right_lane_segment_found_percent']:.1f}%",
        "",
        "Major problems found:",
    ]
    if stats["major_problems"]:
        lines.extend([f"- {problem}" for problem in stats["major_problems"]])
    else:
        lines.append("- None detected from telemetry thresholds.")
    lines.extend(["", "Simple recommendation:"])
    lines.extend([f"- {recommendation}" for recommendation in recommendations])
    if stats["inactive_reason_counts"]:
        lines.extend(["", "Safe corridor inactive reasons:"])
        lines.extend([f"- {reason}: {count}" for reason, count in stats["inactive_reason_counts"].items()])

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
        "safe_corridor_valid_percent": stats["safe_corridor_valid_percent"],
        "visual_helper_active_percent": stats["visual_helper_active_percent"],
        "average_measured_lane_width_mm": stats["average_measured_lane_width_mm"],
        "lane_width_std_mm": stats["lane_width_std_mm"],
        "average_left_clearance_mm": stats["average_left_clearance_mm"],
        "average_right_clearance_mm": stats["average_right_clearance_mm"],
        "average_abs_corridor_center_error_mm": stats["average_abs_corridor_center_error_mm"],
        "max_abs_corridor_center_error_mm": stats["max_abs_corridor_center_error_mm"],
        "inactive_reason_counts": stats["inactive_reason_counts"],
        "unphysical_corridor_geometry_count": stats["unphysical_corridor_geometry_count"],
        "steering_saturated_too_long_count": stats["steering_saturated_too_long_count"],
        "average_valid_left_clearance_mm": stats["average_valid_left_clearance_mm"],
        "average_valid_right_clearance_mm": stats["average_valid_right_clearance_mm"],
        "average_valid_abs_corridor_center_error_mm": stats["average_valid_abs_corridor_center_error_mm"],
        "visual_steering_saturation_percent": stats["visual_steering_saturation_percent"],
        "right_lane_lock_active_percent": stats["right_lane_lock_active_percent"],
        "right_lane_segment_found_percent": stats["right_lane_segment_found_percent"],
        "wide_blob_no_yellow_count": stats["wide_blob_no_yellow_count"],
        "yellow_visible_no_right_segment_count": stats["yellow_visible_no_right_segment_count"],
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
        "- events CSV: only important changes such as road loss, low confidence, jumps, and helper state changes.",
        "- summary AI JSON: structured run summary and grouped problem intervals.",
        "- config used JSON: exact HSV/ROI/math thresholds used for the run.",
        "- frame_samples: periodic original/raw_mask/ego_mask/overlay/debug/safe_corridor_debug images.",
        "- failure_frames: suspicious debug frames, capped by command-line settings.",
        "",
        "Known limitations:",
        "- No machine learning, no training, no YOLO, no PyTorch, no TensorFlow.",
        "- RGB only; no ROS2 and no RealSense depth processing yet.",
        "- Lighting changes and road color changes may require HSV retuning.",
        "- The blue safe corridor is a drift helper only; it is not a path planner.",
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
    safe_valid_count = sum(1 for row in rows if row["safe_corridor_valid"] is True)
    helper_active_count = sum(1 for row in rows if row["visual_helper_active"] is True)
    lane_widths_mm = numeric_values(rows, "measured_lane_width_mm")
    left_clearances_mm = numeric_values(rows, "left_clearance_mm")
    right_clearances_mm = numeric_values(rows, "right_clearance_mm")
    corridor_errors_mm = numeric_values(rows, "corridor_center_error_mm")
    steering_corrections = numeric_values(rows, "visual_steering_correction")
    valid_safe_rows = [row for row in rows if row["safe_corridor_valid"] is True]
    valid_left_clearances = numeric_values(valid_safe_rows, "left_clearance_mm")
    valid_right_clearances = numeric_values(valid_safe_rows, "right_clearance_mm")
    valid_corridor_errors = numeric_values(valid_safe_rows, "corridor_center_error_mm")
    inactive_reason_counts = Counter(
        row["safe_corridor_reason"] for row in rows if row["safe_corridor_reason"] != "valid"
    )
    right_lane_lock_active_count = sum(1 for row in rows if row.get("right_lane_lock_active") is True)
    right_lane_segment_found_count = sum(1 for row in rows if row.get("right_lane_segment_found") is True)

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
    if inactive_reason_counts:
        reason, count = inactive_reason_counts.most_common(1)[0]
        major_problems.append(f"{count} safe-corridor inactive frames, most often {reason}")

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
        "most_common_turn_hint": most_common_text([row["turn_hint"] for row in rows]),
        "safe_corridor_valid_percent": safe_valid_count / max(1, processed_frames) * 100.0,
        "visual_helper_active_percent": helper_active_count / max(1, processed_frames) * 100.0,
        "average_measured_lane_width_mm": average(lane_widths_mm),
        "lane_width_std_mm": stddev(lane_widths_mm),
        "average_left_clearance_mm": average(left_clearances_mm),
        "average_right_clearance_mm": average(right_clearances_mm),
        "average_abs_corridor_center_error_mm": average([abs(value) for value in corridor_errors_mm]),
        "max_abs_corridor_center_error_mm": max([abs(value) for value in corridor_errors_mm]) if corridor_errors_mm else 0.0,
        "average_abs_visual_steering_correction": average([abs(value) for value in steering_corrections]),
        "average_valid_left_clearance_mm": average(valid_left_clearances),
        "average_valid_right_clearance_mm": average(valid_right_clearances),
        "average_valid_abs_corridor_center_error_mm": average([abs(value) for value in valid_corridor_errors]),
        "unphysical_corridor_geometry_count": inactive_reason_counts["unphysical_corridor_geometry"],
        "steering_saturated_too_long_count": inactive_reason_counts["steering_saturated_too_long"],
        "visual_steering_saturation_percent": (
            sum(
                1
                for row in rows
                if abs(float(row["visual_steering_correction"])) >= SAFE_MAX_STEERING_CORRECTION - 1e-9
            )
            / max(1, processed_frames)
            * 100.0
        ),
        "inactive_reason_counts": dict(inactive_reason_counts),
        "right_lane_lock_active_percent": right_lane_lock_active_count / max(1, processed_frames) * 100.0,
        "right_lane_segment_found_percent": right_lane_segment_found_count / max(1, processed_frames) * 100.0,
        "wide_blob_no_yellow_count": inactive_reason_counts["wide_blob_no_yellow"],
        "yellow_visible_no_right_segment_count": inactive_reason_counts["yellow_visible_no_right_segment"],
        "major_problems": major_problems,
    }


def process_video_tuning_mode(args):
    # Manual video tuning is the human-in-the-loop camera setup step. The
    # user pauses on hard frames, adjusts simple RGB/OpenCV/NumPy mask values,
    # and saves a camera config for later video analysis or ROS2 porting.
    if not args.video:
        raise RuntimeError("--video is required when tuning a video source")
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
    config_output_path = Path(args.config_output or args.config)
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
    use_ego_connected_mask = USE_EGO_CONNECTED_MASK
    use_yellow_boundary_lock = USE_YELLOW_BOUNDARY_LOCK
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    safe_corridor_state = {}
    drift_state = {}

    print("Manual video tuning controls:")
    print("q/ESC quit | p/SPACE pause | s save config | l load config-output | r reset")
    print("e toggle ego-connected mask | y toggle yellow-boundary lock | m mask window")
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
            settings = get_trackbar_settings(settings)
            result = detect_road(
                frame,
                settings,
                last_center_x,
                frames_since_valid,
                use_ego_connected_mask,
                settings,
                use_yellow_boundary_lock,
                lane_side_memory,
                safe_corridor_state,
            )
            if result.road_detected and result.road_center_x is not None:
                last_center_x = result.road_center_x
                frames_since_valid = 0
            else:
                frames_since_valid += 1

            turn_hint = compute_turn_hint(result.curve_error_px, settings)
            apply_drift_only_gate(result, turn_hint, drift_state, settings)
            show_inactive_helper = getattr(args, "show_inactive_helper", False) or cfg_bool(settings, "SHOW_INACTIVE_HELPER_DEFAULT", False)
            debug_frame = draw_visualization(frame, result, turn_hint, settings, show_inactive_helper)
            overlay = build_road_overlay(frame, result, settings, show_inactive_helper)
            display = build_display_grid(frame, result, debug_frame, settings, show_inactive_helper)
            cv2.putText(
                display,
                f"Frame {current_frame_index}/{max(0, total_frames - 1)}  speed {playback_speed:.1f}x  ego {use_ego_connected_mask} yellow {use_yellow_boundary_lock}",
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
            elif key == ord("y"):
                use_yellow_boundary_lock = not use_yellow_boundary_lock
                lane_side_memory.clear()
                last_center_x = None
                frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
                print(f"Yellow-boundary lock {'on' if use_yellow_boundary_lock else 'off'}.")
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
        "safe_corridor_valid",
        "visual_helper_active",
        "safe_corridor_width_mm",
        "safe_corridor_width_px",
        "measured_lane_width_mm",
        "measured_lane_width_px",
        "lane_width_valid",
        "left_clearance_mm",
        "right_clearance_mm",
        "corridor_center_error_mm",
        "corridor_center_error_px",
        "visual_steering_correction",
        "safe_scanline_count_valid",
        "safe_corridor_reason",
        "yellow_boundary_detected",
        "yellow_boundary_pixel_count",
        "yellow_boundary_enforced",
        "selected_lane_side",
        "yellow_crossing_pixels",
        "yellow_right_edge_x",
        "right_lane_segment_found",
        "right_lane_segment_left_x",
        "right_lane_segment_right_x",
        "right_lane_segment_width_px",
        "right_lane_lock_active",
        "right_lane_lock_reason",
        "ego_reference_x",
        "camera_center_offset_px",
        "lane_center_x",
        "left_space_mm",
        "right_space_mm",
        "unphysical_corridor_geometry",
        "steering_saturation_count",
        "helper_output_json",
        "mask_area_pixels",
        "mask_area_percent",
        "centerline_point_count",
        "processing_fps_estimate",
    ]
    event_fields = ["frame_index", "time_sec", "event_type", "old_value", "new_value", "notes"]

    if not args.no_display:
        create_trackbars(settings)
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    show_mask_window = False
    paused = False
    stopped_early = False
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    safe_corridor_state = {}
    drift_state = {}
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
                    settings = get_trackbar_settings(settings)

                time_sec = frame_index / source_fps
                elapsed = max(0.001, time.perf_counter() - run_start_time)
                processing_fps = processed_frames / elapsed

                result = detect_road(
                    frame,
                    settings,
                    last_center_x,
                    frames_since_valid,
                    detector_config=settings,
                    lane_side_memory=lane_side_memory,
                    safe_corridor_state=safe_corridor_state,
                )

                if result.road_detected and result.road_center_x is not None:
                    last_center_x = result.road_center_x
                    frames_since_valid = 0
                else:
                    frames_since_valid += 1

                turn_hint = compute_turn_hint(result.curve_error_px, settings)
                apply_drift_only_gate(result, turn_hint, drift_state, settings)
                show_inactive_helper = args.show_inactive_helper or cfg_bool(settings, "SHOW_INACTIVE_HELPER_DEFAULT", False)
                debug_frame = draw_visualization(frame, result, turn_hint, settings, show_inactive_helper)
                overlay = build_road_overlay(frame, result, settings, show_inactive_helper)
                writer.write(debug_frame)
                last_debug_frame = debug_frame.copy()

                telemetry_row = build_telemetry_row(
                    frame_index,
                    time_sec,
                    result,
                    turn_hint,
                    processing_fps,
                    camera_type=str(settings.get("camera_type", "unknown")),
                    settings=settings,
                )
                write_telemetry_row(telemetry_writer, telemetry_row)
                rows.append(telemetry_row)

                events, previous_state = detect_frame_events(
                    frame_index,
                    time_sec,
                    result,
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
                    cv2.imshow(WINDOW_MAIN, build_display_grid(frame, result, debug_frame, settings, show_inactive_helper))
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
            return process_video_source(args)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

    try:
        source = make_source(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    settings, _config_source = load_video_settings(args)
    create_trackbars(settings)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    show_mask_window = False
    paused = False
    last_frame = None
    last_center_x = None
    frames_since_valid = LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    safe_corridor_state = {}
    drift_state = {}

    try:
        while True:
            if not paused or last_frame is None:
                ok, frame = source.read()
                if not ok or frame is None:
                    print("No frame received from source.")
                    break
                last_frame = resize_frame(frame)

            frame = last_frame.copy()
            settings = get_trackbar_settings(settings)
            result = detect_road(
                frame,
                settings,
                last_center_x,
                frames_since_valid,
                lane_side_memory=lane_side_memory,
                safe_corridor_state=safe_corridor_state,
            )

            if result.road_detected and result.road_center_x is not None:
                last_center_x = result.road_center_x
                frames_since_valid = 0
            else:
                frames_since_valid += 1

            turn_hint = compute_turn_hint(result.curve_error_px, settings)
            apply_drift_only_gate(result, turn_hint, drift_state, settings)
            show_inactive_helper = args.show_inactive_helper or cfg_bool(settings, "SHOW_INACTIVE_HELPER_DEFAULT", False)
            visualization = draw_visualization(frame, result, turn_hint, settings, show_inactive_helper)
            display = build_display_grid(frame, result, visualization, settings, show_inactive_helper)
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
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
