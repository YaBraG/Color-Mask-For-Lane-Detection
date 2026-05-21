# QCar2 RGB Road Detector Prototype

This project is a simple RGB-only road/drivable-area detector for a QCar2-style lane/path helper using an Intel RealSense D435 color stream, a regular webcam, or a static test image.

It uses classical OpenCV and NumPy only. There is no ROS2, no depth processing, no machine learning, no training, and no neural network dependency.

## Why Road Detection Instead Of Lane Lines

The indoor road map has a dark gray/black drivable surface, while sidewalks, buildings, and background areas are light gray/white. Yellow lane markings are useful visual cues, but they stop at intersections and forks.

For that reason this prototype detects the road surface itself, treats non-road as unsafe, estimates the drivable corridor, and draws a suggested center path. At forks or intersections, it shows simple demo path candidates for left, straight, and right, then highlights the candidate that best matches the current road-center trend.

## Install

Use Python 3.12 on Windows. A virtual environment is not required.

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

`pyrealsense2` is listed in `requirements.txt` for RealSense support. If it is not installed, image and webcam modes still work.

## Run

Static image tuning mode:

```powershell
py -3.12 main.py --source image --image test.jpg
```

Webcam mode:

```powershell
py -3.12 main.py --source webcam --camera-index 0
```

RealSense D435 RGB-only mode:

```powershell
py -3.12 main.py --source realsense
```

## Controls

- `q` or `ESC`: quit
- `s`: save current HSV and ROI settings to `road_config.json`
- `l`: load settings from `road_config.json`
- `r`: reset HSV settings to defaults
- `m`: toggle separate mask debug window
- `p`: pause/unpause frame processing
- `c`: toggle demo candidate paths on/off

## Tuning

1. Put the camera in the QCar2 point of view.
2. Run image, webcam, or RealSense mode.
3. Adjust the HSV trackbars until the road is white in the mask and the sidewalk/background is black.
4. Adjust `ROI_top_percent` if the upper image contains distracting walls, signs, windows, or traffic lights.
5. Adjust `Morph_kernel` to remove noise.
6. Adjust `Close_kernel` to fill small black holes inside the road mask.
7. Press `s` to save settings.

Default dark-road settings:

- `H_min = 0`
- `H_max = 179`
- `S_min = 0`
- `S_max = 80`
- `V_min = 20`
- `V_max = 120`
- `ROI_top_percent = 35`
- `Morph_kernel = 5`
- `Close_kernel = 11`
- `Min_area_percent = 3`

## Display

The main window shows:

- Original RGB frame
- Road mask
- Road overlay
- Detected drivable center path from road-mask scanlines
- Optional demo candidate path visualization
- Debug values for road detection, center error, selected path, and path confidences
- Curve debug values: `curve_error_px`, `turn_hint`, `near_center_x`, and `far_center_x`

## Known Limitations

- This is not ML and does not understand objects semantically.
- It depends on color and brightness contrast between road and non-road.
- Lighting changes may require HSV tuning.
- Candidate paths are demo paths only; the bright detected centerline is the main visual output.
- It uses RGB only and ignores RealSense depth.
