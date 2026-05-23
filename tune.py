import argparse
import time
from pathlib import Path

import cv2

import main as detector


def parse_args():
    parser = argparse.ArgumentParser(description="Manual tuning tool for the QCar2 RGB drift helper.")
    parser.add_argument("--source", choices=["video", "webcam", "realsense"], required=True)
    parser.add_argument("--video", help="Path to a video file when --source video is used.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for --source webcam.")
    parser.add_argument("--config", default=detector.CONFIG_FILE, help="Camera config JSON to load and save.")
    parser.add_argument("--config-output", help="Optional alternate config path to save when pressing s.")
    parser.add_argument("--session-output", default="configs/manual_tuning_session.json")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--show-inactive-helper", action="store_true", help="Draw faint inactive helper geometry for debugging.")
    return parser.parse_args()


def tune_live_source(args):
    settings, config_source = detector.load_video_settings(args)
    config_output_path = Path(args.config_output or args.config)
    source = detector.make_source(args)

    detector.create_trackbars(settings)
    cv2.namedWindow(detector.WINDOW_MAIN, cv2.WINDOW_NORMAL)

    paused = False
    last_frame = None
    last_center_x = None
    frames_since_valid = detector.LAST_CENTER_HOLD_FRAMES + 1
    lane_side_memory = {}
    safe_corridor_state = {}
    drift_state = {}
    use_ego_connected_mask = detector.cfg_bool(settings, "USE_EGO_CONNECTED_MASK", True)
    use_yellow_boundary_lock = detector.cfg_bool(settings, "USE_YELLOW_BOUNDARY_LOCK", True)
    debug_folder = detector.create_manual_tuning_folders()["debug_snapshots"]
    frame_index = 0

    print(f"Tuning config source: {config_source}")
    print("Controls: q/ESC quit | p/SPACE pause | s save | d debug snapshot | y yellow lock | e ego mask")

    try:
        while True:
            if not paused or last_frame is None:
                ok, frame = source.read()
                if not ok or frame is None:
                    print("No frame received from source.")
                    break
                last_frame = detector.resize_frame(frame)

            frame = last_frame.copy()
            settings = detector.get_trackbar_settings(settings)
            result = detector.detect_road(
                frame,
                settings,
                last_center_x,
                frames_since_valid,
                use_ego_connected_mask=use_ego_connected_mask,
                detector_config=settings,
                use_yellow_boundary_lock=use_yellow_boundary_lock,
                lane_side_memory=lane_side_memory,
                safe_corridor_state=safe_corridor_state,
            )

            if result.road_detected and result.road_center_x is not None:
                last_center_x = result.road_center_x
                frames_since_valid = 0
            else:
                frames_since_valid += 1

            turn_hint = detector.compute_turn_hint(result.curve_error_px, settings)
            detector.apply_drift_only_gate(result, turn_hint, drift_state, settings)
            show_inactive = args.show_inactive_helper or detector.cfg_bool(settings, "SHOW_INACTIVE_HELPER_DEFAULT", False)
            debug_frame = detector.draw_visualization(frame, result, turn_hint, settings, show_inactive)
            display = detector.build_display_grid(frame, result, debug_frame, settings, show_inactive)
            cv2.imshow(detector.WINDOW_MAIN, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("p"), 32):
                paused = not paused
                print("Paused." if paused else "Unpaused.")
            elif key == ord("s"):
                detector.save_manual_tuning_config(config_output_path, settings, args.source, frame_index)
            elif key == ord("d"):
                prefix = debug_folder / f"live_{int(time.time())}_{frame_index:06d}"
                cv2.imwrite(str(prefix) + "_original.jpg", frame)
                cv2.imwrite(str(prefix) + "_mask.jpg", result.mask)
                cv2.imwrite(str(prefix) + "_debug.jpg", debug_frame)
                print(f"Saved debug snapshot {prefix}")
            elif key == ord("y"):
                use_yellow_boundary_lock = not use_yellow_boundary_lock
                lane_side_memory.clear()
                print(f"Yellow lock {'on' if use_yellow_boundary_lock else 'off'}.")
            elif key == ord("e"):
                use_ego_connected_mask = not use_ego_connected_mask
                print(f"Ego mask {'on' if use_ego_connected_mask else 'off'}.")
            frame_index += 1
    finally:
        source.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.source == "video":
        if not args.video:
            print("ERROR: --video is required when --source video is used.")
            return 1
        return detector.process_video_tuning_mode(args)
    try:
        tune_live_source(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
