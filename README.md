# QCar2 RGB Drift Helper

This repository contains a final-prep RGB/OpenCV/NumPy detector for QCar2 lane-drift assistance. It is not a full autonomous planner. The normal QCar2 controller/path follower remains in charge.

The camera helper only tries to answer one local question: is the car drifting left or right inside a believable, straight-ish drivable corridor? When the visual geometry is turning, wide, ambiguous, physically impossible, or missing important cues, the helper turns off and outputs `visual_steering_correction = 0`.

There is no machine learning, no reinforcement learning, no PyTorch, no TensorFlow, no YOLO, no ROS2, and no depth processing.

## Install

Use Python 3.13 on Windows:

```powershell
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r requirements.txt
```

`pyrealsense2` is optional at runtime. If it is unavailable, image, video, and webcam modes still work.

## Final Runtime Files

The active detector runtime is intentionally simple:

- `main.py`: normal runtime/analyzer and ROS2-ready helper-output dictionary
- `tune.py`: manual tuning tool for video/webcam/RealSense RGB
- `configs/csi_front_config.json`: CSI/front camera config for `assets/test_video_2.mp4`
- `configs/realsense_config.json`: RealSense D435 RGB config for `assets/test_video.mp4`

Generated `outputs/` and local videos under `assets/*.mp4` are ignored by git.

## Run Commands

CSI/front video analysis:

```powershell
py -3.13 main.py --source video --video assets/test_video_2.mp4 --config configs/csi_front_config.json --no-display --output-dir outputs --save-failure-frames
```

CSI/front manual tuning:

```powershell
py -3.13 tune.py --source video --video assets/test_video_2.mp4 --config configs/csi_front_config.json
```

RealSense RGB video analysis:

```powershell
py -3.13 main.py --source video --video assets/test_video.mp4 --config configs/realsense_config.json --no-display --output-dir outputs --save-failure-frames
```

Live RealSense RGB:

```powershell
py -3.13 main.py --source realsense --config configs/realsense_config.json
```

Webcam/CSI capture testing:

```powershell
py -3.13 main.py --source webcam --camera-index 0 --config configs/csi_front_config.json
```

Live camera tuning:

```powershell
py -3.13 tune.py --source webcam --camera-index 0 --config configs/csi_front_config.json
py -3.13 tune.py --source realsense --config configs/realsense_config.json
```

## Manual Tuning

Manual tuning is kept because lighting and camera mounting can change. It opens the video, shows the 2x2 debug view, and exposes HSV/mask trackbars.

Useful keys:

- `p` or `SPACE`: pause/unpause
- `n`: next frame while paused
- `b`: previous frame
- `s`: save current config
- `d`: save a debug snapshot
- `y`: toggle yellow-boundary/right-lane enforcement
- `e`: toggle ego-connected road mask
- `q` or `ESC`: quit

When `--config-output` is omitted, pressing `s` saves back to the file passed with `--config`.

## Visual Colors

The default view keeps the final helper signal clean:

- light green: detected asphalt/road mask
- blue: active safe hallway/path only
- yellow: yellow lane divider/no-cross boundary

When `visual_helper_active = False`, the blue hallway and helper arrow are hidden by default. Use `--show-inactive-helper` with `main.py` or `tune.py` only when debugging inactive geometry; inactive helper drawings are faint and labeled debug-only.

## Blue Safe Corridor

The road mask can be wider than the usable lane. The blue hallway is the constrained safe corridor, not the whole road blob.

Physical values are in millimeters:

- `LANE_WIDTH_MM = 254.0`
- `CAR_WIDTH_MM = 203.2`
- `SAFE_HALLWAY_WIDTH_MM = 227.2`

The helper estimates nearby left/right road edges from lower scanlines, converts row-local pixels to millimeters using the known lane width, and computes:

- `corridor_center_error_mm`
- `left_clearance_mm`
- `right_clearance_mm`
- `visual_steering_correction`

If clearances are negative, the lane is too wide/narrow, the center error is too large, steering saturates too long, or the corridor is not straight-ish, the helper disables itself.

## Yellow Right-Lane Lock

The yellow line is treated as a hard lane divider. When yellow is visible, the helper builds the blue corridor only from the road segment to the right of the yellow boundary. It does not cross the yellow line unless future route logic explicitly allows that.

If yellow is missing and the ego road blob is too wide, the helper turns off with `safe_corridor_reason = "wide_blob_no_yellow"` instead of guessing.

## Drift-Only Gating

The helper is intentionally conservative. It disables correction when:

- `turn_hint` is `left` or `right`
- `abs(curve_error_px)` is above `DRIFT_MAX_ABS_CURVE_ERROR_PX`
- the safe corridor is invalid
- yellow/right-lane geometry is invalid
- no-yellow wide-blob gate is active
- physical geometry is impossible
- steering correction saturates too long

When disabled, `visual_helper_active = False` and `visual_steering_correction = 0`.

## Outputs

Each video run creates:

```text
outputs/<video_name_YYYYMMDD_HHMMSS>/
  human_output/
  output_for_AI/
```

`human_output` contains the annotated video and readable summary. `output_for_AI` contains telemetry CSV, events CSV, config used JSON, summary AI JSON, frame samples, and failure frames.

Telemetry includes a `helper_output_json` field. That dictionary is the future ROS2 message payload shape:

```json
{
  "road_detected": true,
  "road_confidence": 0.9,
  "safe_corridor_valid": true,
  "visual_helper_active": true,
  "visual_steering_correction": 0.03,
  "corridor_center_error_mm": 3.0,
  "left_clearance_mm": 24.0,
  "right_clearance_mm": 27.0,
  "turn_hint": "straight",
  "safe_corridor_reason": "valid",
  "yellow_boundary_detected": true,
  "right_lane_lock_active": true,
  "camera_type": "csi_front"
}
```

## Future ROS2 Port

The next step is to wrap `build_helper_output(result, config)` in a ROS2 node and publish it as a small helper topic. The ROS2 controller should keep treating this as optional drift assistance, not as route/path planning.
