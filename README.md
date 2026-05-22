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

## Video Processing Mode

Offline video processing runs a local video through the same RGB road-mask, centerline, curve, confidence, and candidate-path logic used by the live modes. This is useful for repeatable testing because the same frames can be processed again after tuning changes.

Run the local test video:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4
```

Run without opening OpenCV display windows:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --no-display
```

Run with the built-in defaults from `config.py`, ignoring any saved tuning:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --use-default-config --no-display
```

Load a specific saved tuning file:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --config path\to\road_config.json --no-display
```

Choose the output folder:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --output-dir outputs
```

By default, video mode loads `road_config.json` when that file exists. This lets a saved live tuning carry into offline analysis, but it also means `road_config.json` overrides the defaults in `config.py` unless `--use-default-config` is passed. The file `test_video_config_used.json` records the exact values and `config_source` used for each run.

Video mode writes each run into a timestamped folder so old samples, telemetry, and failure frames do not mix with new results:

```text
outputs/
  test_video_YYYYMMDD_HHMMSS/
    human_output/
    output_for_AI/
  latest_run.txt
```

`outputs/latest_run.txt` points to the most recent run folder. `output_for_AI` lives inside the per-run folder so uploaded AI analysis files all come from the same video pass. Previous runs are kept automatically. Use `--clean-output` only to delete the newly selected timestamped run folder before writing if that exact folder already exists.

Each timestamped run creates two output folders:

- `human_output` is for watching and visually checking the result. It contains the annotated MP4, a short human-readable summary, and representative key frames.
- `output_for_AI` is for uploading to ChatGPT for deeper analysis. It contains structured CSV/JSON files, run notes, periodic frame samples, and optional failure frames.

Files to upload to ChatGPT for analysis:

- `test_video_telemetry.csv`
- `test_video_events.csv`
- `test_video_summary_ai.json`
- `test_video_config_used.json`
- selected `failure_frames`
- selected `frame_samples`

The annotated video is best for human visual inspection. The CSV, JSON, frame samples, and failure frames are better for AI debugging because they preserve the detector's numeric state and selected visual evidence. The local file `assets/test_video.mp4` is intentionally not committed and is ignored by git because it is large.

## Ego-Connected Road Mask

HSV thresholding detects every road-colored pixel in the frame. That can include parking lots, side roads, or disconnected dark areas that look like the road but are not the road directly in front of the QCar2.

The ego-connected mask keeps only the white road component connected to the bottom-center/front area of the image. The seed point is near the lower center of the frame, and if that exact pixel is black, the detector searches nearby for the closest white road pixel, preferring pixels in the bottom band because they are closer to the car.

This filtered mask is what centerline tracking uses, so disconnected side-road blobs should not affect scanline centers, `road_center_error_px`, `curve_error_px`, `turn_hint`, or path confidences. Video samples now include both `frame_XXXXXX_raw_mask.jpg` and `frame_XXXXXX_ego_mask.jpg` so the raw HSV result can be compared with the driving/path-extraction mask.

If a side area is physically connected to the main road, ego filtering alone may still include it. The scanline tracker still enforces centerline continuity with `MAX_CENTER_JUMP_PX` after the first anchor, but future map/path guidance may be needed for complex connected branches.

## Manual Video Tuning Mode

Manual video tuning plays a recorded QCar2 ride with the same 2x2 display used by the detector: original RGB, road mask, road overlay, and detected center path/debug view. It is for finding a good RGB/OpenCV/NumPy baseline config before building the future auto-tuning optimizer.

Run manual tuning:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --tune-video
```

Save the current tuning values by pressing `s`. By default, the config is saved to:

```text
configs/manual_tuned_config.json
```

You can choose a different output path:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --tune-video --config-output configs/manual_tuned_config.json
```

Manual tuning starts from `DEFAULT_SETTINGS` unless `--config` is explicitly provided. To start from an existing saved config:

```powershell
py -3.12 main.py --source video --video assets/test_video.mp4 --tune-video --config configs/manual_tuned_config.json
```

Useful keys:

- `p` or `SPACE`: pause/unpause
- `s`: save `configs/manual_tuned_config.json`
- `l`: load from `--config-output`
- `r`: reset to `DEFAULT_SETTINGS`
- `e`: toggle ego-connected mask filtering for comparison
- `n` or right arrow: step forward while paused
- `b` or left arrow: step backward
- `g`: save a good tuning sample
- `f`: save a difficult tuning sample
- `d`: save a full debug snapshot
- `[` and `]`: decrease/increase playback speed
- `q` or `ESC`: quit

Samples are saved here:

```text
outputs/manual_tuning/
  good_samples/
  difficult_samples/
  debug_snapshots/
```

`g` and `f` save original, mask, and debug images. `d` saves original, mask, overlay, and debug images. The session file `configs/manual_tuning_session.json` records saved sample frame numbers and notes for future auto-tuning.

This mode does not train anything, does not use machine learning, and does not control the QCar2. The future optimizer should use `configs/manual_tuned_config.json` as its starting point instead of searching randomly.

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
6. Adjust `Close_kernel` to fill small black holes inside the road mask. `Close_kernel = 0` disables close morphology.
7. Press `s` to save settings.

Default dark-road settings from live QCar2 RealSense camera testing:

- `H_min = 0`
- `H_max = 179`
- `S_min = 0`
- `S_max = 80`
- `V_min = 20`
- `V_max = 120`
- `ROI_top_percent = 58`
- `Morph_kernel = 5`
- `Close_kernel = 0`
- `Min_area_percent = 3`

## Development Log Rule

Every time Codex modifies this repository, it must update `CHANGELOG.md` with:

- Date
- Files changed
- Summary of what changed
- Why the change was made
- Any known issues or follow-up work

## Display

The main window shows:

- Original RGB frame
- Road mask
- Road overlay
- Detected drivable center path from road-mask scanlines
- Optional demo candidate path visualization
- Debug values for road detection, center error, selected path, and path confidences
- Curve debug values: `curve_error_px`, `turn_hint`, `near_center_x`, and `far_center_x`
- Tracking debug values: `tracked_center_valid` and `rejected_scanlines`

The center path is tracked from the bottom of the image upward. Each scanline is split into continuous road segments, and the first usable segment is allowed to anchor away from image center when `ALLOW_FIRST_ANCHOR_JUMP = True`; later scanlines still obey `MAX_CENTER_JUMP_PX`. This helps curves and edge-of-road views lock onto the visible road before enforcing continuity.

## Known Limitations

- This is not ML and does not understand objects semantically.
- It depends on color and brightness contrast between road and non-road.
- Lighting changes may require HSV tuning.
- Candidate paths are demo paths only; the bright detected centerline is the main visual output.
- It uses RGB only and ignores RealSense depth.
