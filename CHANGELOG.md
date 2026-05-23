# Changelog

## 2026-05-23

Files changed:

- `main.py`
- `tune.py`
- `configs/csi_front_config.json`
- `configs/realsense_config.json`
- `README.md`
- `CHANGELOG.md`

Summary:

- Split the final workflow into two active programs: `main.py` for runtime/video analysis and `tune.py` for manual tuning.
- Removed manual tuning from the `main.py` CLI so normal runtime stays focused on detection, visualization, telemetry, and output writing.
- Kept active runtime on JSON config files only; `config.py` remains removed from the active workflow.
- Hid inactive helper hallway/arrow visuals by default. Inactive geometry is only drawn when `--show-inactive-helper` is passed and is labeled debug-only.
- Changed asphalt/road mask overlay to light green so blue is reserved for the active safe hallway/path.
- Added visualization config values for road overlay color/alpha, safe corridor color/alpha, and inactive-helper default behavior.
- Kept `helper_output_json` telemetry and `build_helper_output(...)` as the future ROS2 payload shape.

Removed/de-emphasized prototype features:

- Candidate path/arrow/path-confidence logic remains removed from primary behavior.
- Auto-tuning/trainer code remains removed from the active workflow.

Known limitations:

- Manual tuning still uses OpenCV desktop windows and should be run locally, not headless.
- No ROS2 publisher exists yet; telemetry only writes the future helper payload as JSON.

Follow-up work:

- Convert `build_helper_output(...)` into a ROS2 node/topic while keeping the normal QCar2 controller in charge.

## 2026-05-23

Files changed:

- `main.py`
- `configs/csi_front_config.json`
- `configs/realsense_config.json`
- `README.md`
- `CHANGELOG.md`
- Removed `config.py`
- Removed `auto_tuner.py`
- Removed `scoring.py`

Summary:

- Cleaned the repository for the final pre-ROS2 RGB drift-helper workflow.
- Consolidated the active detector runtime into `main.py` so the runnable detector no longer imports `config.py`, `auto_tuner.py`, or `scoring.py`.
- Added final camera-specific JSON configs: `configs/csi_front_config.json` and `configs/realsense_config.json`.
- Removed the old auto-tune/trainer CLI and modules from the active workflow. Auto-tuning was useful during development; its learned values are preserved in the final config JSON files.
- Removed old candidate-arrow/path-confidence behavior from the primary runtime. The main helper output is now the blue safe corridor, `corridor_center_error_mm`, `visual_steering_correction`, `visual_helper_active`, and `safe_corridor_reason`.
- Added drift-only helper gating so the correction disables during turns, high curve error, unstable corridors, ambiguous/wide blobs, unphysical geometry, and repeated steering saturation.
- Added `build_helper_output(...)`, a ROS2-ready dictionary payload for the future node/topic.
- Rewrote README.md around the final QCar2 RGB drift-helper purpose and Python 3.13 commands.

Removed/de-emphasized prototype features:

- Removed active auto-tuning/trainer code from the final runtime path.
- Removed candidate path selection as a steering concept.
- Removed dependency on `road_config.json` and `config.py` for the final workflow.

Known limitations:

- Still RGB-only and row-local; there is no camera calibration, homography, depth, ROS2 publishing, or route-level permission to cross yellow boundaries yet.
- The helper is intentionally conservative and may disable often when visual geometry is ambiguous.

Follow-up work:

- Port `build_helper_output(...)` into a ROS2 node/topic and keep the normal QCar2 controller in charge.

## 2026-05-23

Files changed:

- `config.py`
- `main.py`
- `auto_tuner.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Fixed safe-corridor steering math to use the selected lane segment center relative to a tunable camera/car reference point.
- Added camera center offset config values for future CSI camera alignment tuning.
- Recomputed left/right clearances from car half-width so centered clearance matches the physical lane/car dimensions.
- Added physical sanity gates for negative clearance, unreasonable corridor error, and unreasonable clearance.
- Added steering saturation protection so the visual helper shuts off instead of commanding hard correction for too many consecutive frames.
- Added telemetry/debug fields for ego reference, lane center, side spaces, unphysical geometry, and saturation count.

Known limitation:

- This still uses row-local lane-width scaling, not full camera calibration or homography.

Follow-up work:

- Tune camera center offset on measured CSI camera mounting and validate before connecting `visual_steering_correction` to the QCar controller.

## 2026-05-23

Files changed:

- `config.py`
- `main.py`
- `auto_tuner.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added forced right-lane yellow lock for safe-corridor scanlines.
- When yellow is visible, the safe corridor now uses the road segment immediately to the right of the yellow boundary and ignores left-lane road segments.
- Added lane-side hold behavior for the right-lane lock so selected side does not flicker when yellow briefly disappears.
- Added a no-yellow wide-blob gate that disables the helper with `safe_corridor_reason = "wide_blob_no_yellow"` instead of guessing in open/ambiguous black road areas.
- Added telemetry/debug fields for yellow right edge, right-lane segment geometry, right-lane lock state, and right-lane lock reason.

Why:

- The blue safe corridor could still focus on the wrong lane when yellow was visible or become confused by very wide road blobs with no visible divider.

Known limitation:

- If the yellow line is not detected correctly, the helper may disable itself instead of guessing.

Follow-up work:

- Add route-level logic that can intentionally allow crossing the yellow boundary when the planned path requires it.

## 2026-05-22

Files changed:

- `config.py`
- `main.py`
- `auto_tuner.py`
- `README.md`
- `CHANGELOG.md`

Summary:

- Added yellow lane-line detection as a separate boundary mask with default yellow-boundary lock enabled.
- Removed yellow pixels from the drivable road mask so lane paint is not treated as road.
- Added lane-side clipping and lane-side memory so the ego lane side does not flicker frame-to-frame.
- Added safe-corridor crossing checks; if the blue hallway overlaps the yellow divider, the helper becomes inactive with `safe_corridor_reason = "crosses_yellow_boundary"` and zero steering correction.
- Added manual tuning debug toggle `y` for yellow-boundary enforcement.
- Added yellow-boundary telemetry and `yellow_boundary_mask` frame samples for AI/debug review.

Why:

- The safe corridor could choose the wrong lane when the road mask detected both sides of a visible yellow line. The yellow line now acts as a hard no-cross divider unless future route logic explicitly permits crossing.

Known limitations:

- Yellow detection is HSV-based and may need tuning if lighting changes or lane paint appears washed out.
- This lock prevents accidental crossing for the visual helper only; future route logic still needs an explicit way to allow intentional lane changes or turns across a yellow boundary.

Follow-up work:

- Add route-level permission for intentional yellow-line crossing when the QCar mission requires it.

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
