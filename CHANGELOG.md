# Changelog

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
