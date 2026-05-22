# Changelog

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
