# Changelog

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
