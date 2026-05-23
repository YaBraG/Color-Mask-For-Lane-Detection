# QCar2 RGB Drift Helper

This repository is a Python 3.13 RGB/OpenCV/NumPy drift-helper module for QCar2. It is not a route planner and it is not a full autonomy stack. The normal QCar2 controller/path follower stays in charge.

The helper detects road/asphalt, detects the yellow divider, builds a blue safe corridor only when the local geometry is valid, and outputs a small drift correction only in a safe straight-ish corridor. During turns, intersections, wide black road blobs, yellow crossings, low confidence, or unphysical geometry, the helper shuts off and returns `visual_steering_correction = 0`.

There is no ML, no RL, no ROS2 yet, and no depth processing.

## Files

Active programs:

- `main.py`: final runtime/local test program
- `tune.py`: manual tuning tool

Active configs:

- `configs/csi_front_config.json` default CSI/front camera config
- `configs/realsense_config.json` RealSense D435 RGB config

Default config:

```text
configs/csi_front_config.json
```

## Install

```powershell
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r requirements.txt
```

`pyrealsense2` is optional unless you use `--source realsense`.

## Runtime

CSI video test:

```powershell
py -3.13 main.py --source video --video assets/test_video_2.mp4
```

CSI video explicit:

```powershell
py -3.13 main.py --source video --video assets/test_video_2.mp4 --config configs/csi_front_config.json
```

RealSense video:

```powershell
py -3.13 main.py --source video --video assets/test_video.mp4 --config configs/realsense_config.json
```

Webcam/CSI:

```powershell
py -3.13 main.py --source webcam --camera-index 0
```

Live RealSense:

```powershell
py -3.13 main.py --source realsense --config configs/realsense_config.json
```

Useful runtime flags:

- `--display` / `--no-display`: show or hide OpenCV windows
- `--show-debug`: draw scanlines, anchor points, and extra geometry
- `--show-inactive-helper`: draw faint inactive helper geometry for debugging

## Tuning

Use `tune.py` to adjust HSV/mask values and save the selected config JSON.

```powershell
py -3.13 tune.py --source video --video assets/test_video_2.mp4
py -3.13 tune.py --source video --video assets/test_video_2.mp4 --config configs/csi_front_config.json
py -3.13 tune.py --source video --video assets/test_video.mp4 --config configs/realsense_config.json
py -3.13 tune.py --source webcam --camera-index 0
py -3.13 tune.py --source realsense --config configs/realsense_config.json
```

Keys:

- `s`: save config
- `d`: save one debug snapshot to `outputs/debug_snapshots/`
- `p` or `SPACE`: pause/unpause
- `n` / `b`: step video forward/back
- `y`: toggle yellow lock for testing
- `e`: toggle ego-connected mask for testing
- `q` or `ESC`: quit

## Visuals

Default display is intentionally clean:

- light green: detected asphalt/road area
- yellow: yellow divider/no-cross boundary
- blue: active safe hallway/path only

If `visual_helper_active` is false, the blue hallway and helper arrow are hidden. Use `--show-inactive-helper` only when debugging; inactive helper drawings are faint and labeled debug-only.

## Helper Output

`build_helper_output(result, config)` returns the future ROS2 payload shape:

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
  "right_lane_lock_active": true
}
```

The next step is a ROS2 node that subscribes to camera images and publishes this helper output. The controller should still treat it as optional drift assistance, not path planning.
