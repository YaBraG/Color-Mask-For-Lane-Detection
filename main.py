import argparse
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    CENTER_DEADBAND_PX,
    CENTER_STRONG_PX,
    CONFIDENCE_ALPHA,
    CONFIG_FILE,
    DEFAULT_SETTINGS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LAST_CENTER_HOLD_FRAMES,
    REALSENSE_FPS,
    REALSENSE_HEIGHT,
    REALSENSE_WIDTH,
    SCANLINE_COUNT,
    SELECT_CONFIDENCE,
    SELECT_MARGIN,
    TRACKBAR_RANGES,
    WINDOW_MAIN,
    WINDOW_MASK,
    WINDOW_TUNING,
)


@dataclass
class RoadResult:
    mask: np.ndarray
    road_detected: bool
    road_confidence: float
    road_center_x: float | None
    road_center_error_px: float | None
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

    def update(self, center_error_px: float | None, road_detected: bool):
        target = {"left": 0.33, "straight": 0.34, "right": 0.33}

        if road_detected and center_error_px is not None:
            error = float(center_error_px)
            abs_error = abs(error)

            if abs_error <= CENTER_DEADBAND_PX:
                target = {"left": 0.12, "straight": 0.76, "right": 0.12}
            elif error < -CENTER_STRONG_PX:
                target = {"left": 0.78, "straight": 0.12, "right": 0.10}
            elif error > CENTER_STRONG_PX:
                target = {"left": 0.10, "straight": 0.12, "right": 0.78}
            elif error < 0:
                amount = min(1.0, (abs_error - CENTER_DEADBAND_PX) / max(1, CENTER_STRONG_PX - CENTER_DEADBAND_PX))
                target = {
                    "left": 0.35 + 0.35 * amount,
                    "straight": 0.50 - 0.25 * amount,
                    "right": 0.15 - 0.10 * amount,
                }
            else:
                amount = min(1.0, (abs_error - CENTER_DEADBAND_PX) / max(1, CENTER_STRONG_PX - CENTER_DEADBAND_PX))
                target = {
                    "left": 0.15 - 0.10 * amount,
                    "straight": 0.50 - 0.25 * amount,
                    "right": 0.35 + 0.35 * amount,
                }

        for name in self.confidences:
            old_value = self.confidences[name]
            self.confidences[name] = (1.0 - CONFIDENCE_ALPHA) * old_value + CONFIDENCE_ALPHA * target[name]

        total = sum(self.confidences.values())
        if total > 0:
            for name in self.confidences:
                self.confidences[name] /= total

        return self.confidences.copy(), self.selected_path()

    def selected_path(self):
        ordered = sorted(self.confidences.items(), key=lambda item: item[1], reverse=True)
        best_name, best_value = ordered[0]
        second_value = ordered[1][1]
        if best_value > SELECT_CONFIDENCE and best_value - second_value >= SELECT_MARGIN:
            return best_name
        return "none"


def parse_args():
    parser = argparse.ArgumentParser(description="RGB-only QCar2 road/drivable-area detection prototype.")
    parser.add_argument("--source", choices=["image", "webcam", "realsense"], required=True)
    parser.add_argument("--image", help="Path to a static image for --source image.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for --source webcam.")
    return parser.parse_args()


def print_startup(args):
    print("QCar2 RGB Road Detector")
    print("-----------------------")
    print(f"Source: {args.source}")
    print("Keys: q/ESC quit | s save HSV | l load HSV | r reset HSV | m toggle mask window | p pause")
    print("Tune HSV until road is white in the mask and non-road is black.")
    print()


def make_source(args):
    if args.source == "image":
        if not args.image:
            raise RuntimeError("--image is required when using --source image")
        return StaticImageSource(args.image)
    if args.source == "webcam":
        return WebcamSource(args.camera_index)
    return RealSenseSource()


def resize_frame(frame):
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def create_trackbars(settings):
    cv2.namedWindow(WINDOW_TUNING, cv2.WINDOW_NORMAL)
    for name, default_value in settings.items():
        cv2.createTrackbar(name, WINDOW_TUNING, int(default_value), TRACKBAR_RANGES[name], nothing)


def nothing(_value):
    pass


def get_trackbar_settings():
    settings = {}
    for name in DEFAULT_SETTINGS:
        settings[name] = cv2.getTrackbarPos(name, WINDOW_TUNING)

    if settings["H_min"] > settings["H_max"]:
        settings["H_min"], settings["H_max"] = settings["H_max"], settings["H_min"]
    if settings["S_min"] > settings["S_max"]:
        settings["S_min"], settings["S_max"] = settings["S_max"], settings["S_min"]
    if settings["V_min"] > settings["V_max"]:
        settings["V_min"], settings["V_max"] = settings["V_max"], settings["V_min"]

    # OpenCV morphology needs a positive odd kernel size.
    kernel = max(1, settings["Morph_kernel"])
    if kernel % 2 == 0:
        kernel += 1
    settings["Morph_kernel"] = kernel
    settings["ROI_top_percent"] = min(max(settings["ROI_top_percent"], 0), 80)
    settings["Min_area_percent"] = min(max(settings["Min_area_percent"], 0), 30)
    return settings


def set_trackbars(settings):
    for name, value in settings.items():
        cv2.setTrackbarPos(name, WINDOW_TUNING, int(value))


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


def detect_road(frame, settings, last_center_x, frames_since_valid):
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
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    mask = keep_relevant_component(full_mask, settings)
    area = int(cv2.countNonZero(mask))
    roi_area = max(1, width * (height - roi_top))
    min_area = roi_area * settings["Min_area_percent"] / 100.0
    scan_points, boundary_points = estimate_scanline_centers(mask, roi_top)

    road_detected = area > 0 and area >= min_area and len(scan_points) > 0
    road_confidence = min(1.0, area / max(1.0, min_area * 3.0)) if road_detected else 0.0


    road_center_x = None
    road_center_error_px = None
    if road_detected and scan_points:
        # Weight lower scanlines more because they are closer to the vehicle and more reliable.
        weights = np.linspace(1.0, 2.0, num=len(scan_points))
        centers = np.array([point[0] for point in scan_points], dtype=np.float32)
        road_center_x = float(np.average(centers, weights=weights))
        road_center_error_px = road_center_x - (width / 2.0)
    elif last_center_x is not None and frames_since_valid <= LAST_CENTER_HOLD_FRAMES:
        road_center_x = last_center_x
        road_center_error_px = road_center_x - (width / 2.0)

    return RoadResult(
        mask=mask,
        road_detected=road_detected,
        road_confidence=road_confidence,
        road_center_x=road_center_x,
        road_center_error_px=road_center_error_px,
        scan_points=scan_points,
        boundary_points=boundary_points,
    )


def keep_relevant_component(mask, settings):
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


def estimate_scanline_centers(mask, roi_top):
    height, _width = mask.shape[:2]
    y_values = np.linspace(height - 35, max(roi_top + 10, int(height * 0.52)), SCANLINE_COUNT).astype(int)

    scan_points = []
    boundary_points = []
    for y in y_values:
        xs = np.where(mask[y, :] > 0)[0]
        if xs.size < 8:
            continue
        left_x = int(xs[0])
        right_x = int(xs[-1])
        center_x = int((left_x + right_x) / 2)
        scan_points.append((center_x, left_x, right_x, int(y)))
        boundary_points.append((left_x, int(y)))
        boundary_points.append((right_x, int(y)))

    # Preserve bottom-to-top order for drawing a path from the vehicle forward.
    return scan_points, boundary_points


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


def draw_visualization(frame, result, confidences, selected_path):
    output = frame.copy()
    height, width = output.shape[:2]

    road_color = np.zeros_like(output)
    road_color[:, :] = (180, 120, 0)
    output = np.where(result.mask[:, :, None] > 0, cv2.addWeighted(output, 0.55, road_color, 0.45, 0), output)

    for left_x, y in result.boundary_points:
        cv2.circle(output, (left_x, y), 4, (0, 255, 255), -1)

    if result.scan_points:
        centerline_points = [(int(center_x), int(y)) for center_x, _left_x, _right_x, y in result.scan_points]
        if len(centerline_points) >= 2:
            cv2.polylines(output, [np.array(centerline_points, dtype=np.int32)], False, (0, 255, 0), 3)

    candidates = make_candidates(width, height)
    for name, points in candidates.items():
        confidence = confidences[name]
        if name == selected_path:
            color = (0, 255, 255)
            thickness = 6
        else:
            brightness = int(80 + 130 * confidence)
            color = (brightness, brightness, 255)
            thickness = 3
        cv2.polylines(output, [points], False, color, thickness, cv2.LINE_AA)
        cv2.circle(output, tuple(points[-1]), 6, color, -1)

    if result.road_center_x is not None:
        start = (width // 2, height - 1)
        end = (int(result.road_center_x), int(height * 0.58))
        cv2.arrowedLine(output, start, end, (255, 255, 0), 4, cv2.LINE_AA, tipLength=0.12)
        cv2.line(output, (width // 2, height), (width // 2, int(height * 0.45)), (255, 255, 255), 1)

    draw_debug_text(output, result, confidences, selected_path)
    return output


def draw_debug_text(output, result, confidences, selected_path):
    error = result.road_center_error_px
    error_text = "None" if error is None else f"{error:.1f}"
    lines = [
        f"road_detected: {result.road_detected}",
        f"road_confidence: {result.road_confidence:.2f}",
        f"road_center_error_px: {error_text}",
        f"selected_path: {selected_path}",
        f"straight_confidence: {confidences['straight']:.2f}",
        f"left_confidence: {confidences['left']:.2f}",
        f"right_confidence: {confidences['right']:.2f}",
    ]

    x = 12
    y = 28
    line_height = 26
    panel_width = 330
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
    bottom = np.hstack([label_image(road_overlay, "Road Overlay"), label_image(visualization, "Path Candidates")])
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


def main():
    args = parse_args()
    print_startup(args)

    try:
        source = make_source(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    create_trackbars(DEFAULT_SETTINGS)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    tracker = PathConfidenceTracker()
    show_mask_window = False
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

            confidences, selected_path = tracker.update(result.road_center_error_px, result.road_detected)
            visualization = draw_visualization(frame, result, confidences, selected_path)
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
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
