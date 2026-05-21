import argparse
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    CENTERLINE_ALPHA,
    CENTER_DEADBAND_PX,
    CENTER_STRONG_PX,
    CONFIDENCE_ALPHA,
    CONFIG_FILE,
    CURVE_DEADBAND_PX,
    CURVE_STRONG_PX,
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
    curve_error_px: float | None
    near_center_x: int | None
    far_center_x: int | None
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
    parser.add_argument("--source", choices=["image", "webcam", "realsense"], required=True)
    parser.add_argument("--image", help="Path to a static image for --source image.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for --source webcam.")
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

    # OpenCV morphology needs positive odd kernel sizes.
    for kernel_name in ("Morph_kernel", "Close_kernel"):
        kernel = max(1, settings[kernel_name])
        if kernel % 2 == 0:
            kernel += 1
        settings[kernel_name] = kernel
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

    # Closing fills small black holes inside the road mask. A separate, larger
    # close kernel is useful when the road threshold is good but has pinholes.
    close_kernel_size = settings["Close_kernel"]
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, close_kernel)

    mask = keep_relevant_component(full_mask, settings)
    area = int(cv2.countNonZero(mask))
    roi_area = max(1, width * (height - roi_top))
    min_area = roi_area * settings["Min_area_percent"] / 100.0
    scan_points, boundary_points = estimate_scanline_centers(mask, roi_top)

    road_detected = area > 0 and area >= min_area and len(scan_points) > 0
    road_confidence = min(1.0, area / max(1.0, min_area * 3.0)) if road_detected else 0.0


    road_center_x = None
    road_center_error_px = None
    curve_error_px = None
    near_center_x = None
    far_center_x = None
    if road_detected and scan_points:
        # Weight lower scanlines more because they are closer to the vehicle and more reliable.
        weights = np.linspace(1.0, 2.0, num=len(scan_points))
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
        mask=mask,
        road_detected=road_detected,
        road_confidence=road_confidence,
        road_center_x=road_center_x,
        road_center_error_px=road_center_error_px,
        curve_error_px=curve_error_px,
        near_center_x=near_center_x,
        far_center_x=far_center_x,
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
    cv2.line(output, (width // 2, height), (width // 2, int(height * 0.45)), (255, 255, 255), 1)

    draw_debug_text(output, result, confidences, selected_path, smoothed_curve_error_px, turn_hint)
    return output


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
