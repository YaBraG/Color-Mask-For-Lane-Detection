# Changelog

## 2026-05-22

Files changed:

- `config.py`
- `main.py`
- `auto_tuner.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added the blue safe drivable corridor overlay as the primary local visual steering helper.
- Added physical lane/car constraints: lane width, car width, sidewalk/line margins, safe hallway width, and valid lane-width thresholds.
- Added corridor-based helper outputs including safe-corridor validity, helper active state, measured lane width, left/right clearance, corridor center error, and visual steering correction.
- Added gating behavior so the helper turns off and outputs zero correction in wide, narrow, low-confidence, ego-missing, or insufficient-scanline areas.
- Added safe-corridor telemetry, AI summary metrics, human summary metrics, frame samples, and failure events.
- Reduced old candidate arrows to secondary debug output; they are no longer the main local steering helper.

Known limitations:

- The mm conversion is a simple row-local calibration from detected lane width, not a full camera calibration or homography.
- The helper is logged and visualized only; it is not connected to QCar control yet.
- Wide physically connected branches may still need map/path guidance.

Follow-up work:

- Later integrate `visual_steering_correction` with the actual QCar controller after validating the blue corridor on front CSI camera video.

## 2026-05-22

Files changed:

- `main.py`
- `auto_tuner.py`
- `scoring.py`
- `README.md`
- `requirements.txt`
- `CHANGELOG.md`

Summary:

- Added offline auto-tuning with `--auto-tune`, `--seed-config`, `--quick`, `--max-configs`, `--top-k`, `--sample-stride`, `--full-eval-top-k`, `--random-seed`, and `--auto-tune-time-budget-hours`.
- Added `auto_tuner.py` for candidate generation, sampled-frame evaluation, full-video evaluation of top configs, timestamped auto-tune output folders, best-config rendering, and human/AI summaries.
- Added `scoring.py` with readable OpenCV/NumPy detector metrics and score math that balances detection rate, detection quality, scanlines, ego-component quality, mask area, and stability penalties.
- Refactored detector calls so runtime candidate configs can override detector parameters such as center jump, smoothing, segment width, and ego-mask settings without editing `config.py`.
- Updated README examples and workflow for Python 3.13.
- Kept RealSense optional and removed `pyrealsense2` from required install dependencies because Python 3.13 wheels may not be available on every laptop.
- Disabled candidate-path drawing in the final auto-tune render so best-config output focuses on the actual detected road centerline instead of demo branch visuals.

What was removed as useless/dead code:

- Removed `pyrealsense2` from `requirements.txt` as a hard dependency; the lazy optional import remains for live RealSense use.
- No videos or generated output folders were added to version control; existing `.gitignore` already excludes local videos and generated outputs.

Why:

- Manual tuning gives a human baseline, but the detector now needs a repeatable optimizer that searches around that seed and scores balanced detector behavior instead of optimizing only road-detected percent.

Known limitations:

- This is still classical OpenCV/NumPy hyperparameter search, not ML training or reinforcement learning.
- The first implementation is single-process for Windows simplicity; long searches can take time.
- Candidate path confidence logic still exists for live/manual visualization, but auto-tuning scores the detected road mask and centerline directly.

Follow-up work:

- Split more of `main.py` into detector/video/output modules once the auto-tuning behavior stabilizes.
- Add a tiny synthetic video fixture for automated tests.

## 2026-05-22

Files changed:

- `config.py`
- `main.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added ego-connected road mask filtering so the detector keeps the white road component connected to the bottom-center/front-of-car area.
- Added ego seed/search config values for seed position, search radius, bottom-band preference, and minimum connected component area.
- Added telemetry and debug visualization for ego seed point, selected anchor point, component area, and fallback use.
- Updated video frame samples to include both raw HSV masks and ego-connected masks.
- Added `e` in manual video tuning mode to toggle ego-connected filtering for comparison.

Why ego-connected road filtering was added:

- HSV can detect parking lots, side roads, and other road-colored blobs. The centerline tracker should follow the component closest to the vehicle instead of being pulled toward disconnected side areas.

Known limitation:

- If the side area is physically connected to the main road, ego filtering alone may not fully reject it; centerline continuity and map/path guidance are still needed.

## 2026-05-22

Files changed:

- `main.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added manual video tuning mode with `--tune-video` for pausing recorded QCar2 video, adjusting HSV/mask trackbars, and saving a baseline config.
- Added `--config-output`, `--session-output`, `--start-frame`, and `--playback-speed` for manual tuning sessions.
- Added tuning keyboard controls for pause/play, frame stepping, config save/load/reset, candidate/mask toggles, playback speed, and good/difficult/debug sample capture.
- Added a video frame control trackbar for jumping through the recorded video.

Why:

- The detector needs a human-tuned baseline on difficult recorded frames before building a full self-tuning optimizer.
- The future optimizer should search around a known useful manual config instead of starting randomly.

Output files/folders added:

- `configs/manual_tuned_config.json`
- `configs/manual_tuning_session.json`
- `outputs/manual_tuning/good_samples/`
- `outputs/manual_tuning/difficult_samples/`
- `outputs/manual_tuning/debug_snapshots/`

Known limitations:

- Manual tuning is interactive and requires OpenCV display windows, so it is not suitable for headless validation.
- It still uses RGB/OpenCV/NumPy only; no machine learning, reinforcement learning, ROS2, or depth processing is involved.

Follow-up work:

- Use `configs/manual_tuned_config.json` as the seed for auto-tuning.

## 2026-05-22

Files changed:

- `config.py`
- `main.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added video config controls with `--use-default-config`, `--config`, terminal config-source output, and `config_source` in `test_video_config_used.json`.
- Fixed video confidence scoring so `road_confidence` and `detection_quality` combine mask area, valid scanlines, and rejected scanlines instead of depending only on `Min_area_percent`.
- Fixed first centerline anchor behavior so the tracker can lock onto the first visible road segment away from image center before applying `MAX_CENTER_JUMP_PX` to later scanlines.
- Added `ALLOW_FIRST_ANCHOR_JUMP = True` and telemetry for `valid_scanline_count`, `seed_center_x`, `first_anchor_x`, and `first_anchor_distance_px`.
- Added timestamped per-run output folders under `outputs/test_video_YYYYMMDD_HHMMSS/`, plus `outputs/latest_run.txt`, reducing stale output and mixed-run risk.
- Improved failure-frame selection to keep one prioritized reason per frame and improved events CSV to record low-confidence and rejected-scanline intervals instead of repeated rows.

Why:

- The first offline AI review showed that saved `road_config.json` values were silently overriding intended defaults, confidence collapsed into unhelpful 0/1 behavior, centerline tracking rejected visible curved-road anchors, and old output files could be mixed into new analysis uploads.

Known issues or follow-up:

- Add automated tests for config-source selection, timestamped output creation, and event interval logging once a small synthetic video fixture is available.

## 2026-05-22

Files changed:

- `main.py`
- `README.md`
- `.gitignore`
- `CHANGELOG.md`

Summary:

- Added offline video-processing mode with `--source video`, `--video`, `--output-dir`, `--no-display`, `--save-failure-frames`, `--max-failure-frames`, and `--ai-sample-interval-sec`.
- Reused the existing RGB/OpenCV/NumPy road-mask, centerline, curve, confidence, and candidate-path logic for each video frame.
- Added annotated video output, human summary text, key frames, telemetry CSV, events CSV, AI summary JSON, config JSON, run notes, periodic AI samples, and optional capped failure frames.
- Updated README documentation for video mode and clarified which files should be uploaded to ChatGPT for analysis.
- Updated `.gitignore` so local videos and generated output folders are not committed.

Why video mode was added:

- Offline video mode makes the detector easier to test repeatedly on the same local footage without needing a live QCar2, webcam, or RealSense session.
- It creates frame-by-frame evidence that can be inspected by a human and later analyzed by ChatGPT.

Human output vs output_for_AI:

- `human_output` is for visual inspection, especially the annotated MP4, summary text, and representative key frames.
- `output_for_AI` is for structured analysis with CSV, JSON, run notes, periodic frame samples, and optional failure frames.

Local video note:

- `assets/test_video.mp4` is local only, intentionally not committed, and ignored by git because it is too large for the repository.

Known limitations:

- The detector is still RGB/OpenCV/NumPy only; it does not use machine learning, training, YOLO, PyTorch, TensorFlow, ROS2, or depth processing.
- HSV tuning may still need adjustment for lighting changes or different road material.
- Failure-frame saving is capped and only saves many images when `--save-failure-frames` is provided.

Follow-up work:

- Add automated tests around video summary/event generation with a tiny synthetic video fixture.
- Consider adding optional frame-range processing for faster tuning passes.

## 2026-05-21

Files changed:

- `config.py`
- `main.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Updated default HSV/ROI/morphology tuning based on real QCar2/RealSense lab testing.
- Changed `ROI_top_percent` from `35` to `58`.
- Changed `Close_kernel` from `11` to `0`.
- Added behavior where `Close_kernel = 0` disables close morphology instead of being converted to `1`.
- Documented the permanent development log rule for future Codex edits.

Why:

- The tuned live camera result works better with a higher ROI cutoff and close morphology disabled.
- Skipping close morphology keeps the road mask closer to the tuned live result and avoids over-filling or distorting the road shape.

Known issues or follow-up:

- Lighting or camera height changes may still require retuning HSV and ROI values.
