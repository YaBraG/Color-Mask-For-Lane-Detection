"""Default settings for the RGB road detector prototype."""

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# RealSense color stream settings. The D435 supports several color resolutions;
# 848x480 is a useful balance for speed and field of view on small robots.
REALSENSE_WIDTH = 848
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

WINDOW_MAIN = "QCar2 RGB Road Detector"
WINDOW_TUNING = "Tuning"
WINDOW_MASK = "Road Mask Debug"

CONFIG_FILE = "road_config.json"

# HSV defaults for dark gray/black indoor road material:
# broad hue range, low saturation, and relatively dark value.
DEFAULT_SETTINGS = {
    "H_min": 0,
    "H_max": 179,
    "S_min": 0,
    "S_max": 80,
    "V_min": 20,
    "V_max": 120,
    "ROI_top_percent": 35,
    "Morph_kernel": 5,
    "Min_area_percent": 3,
}

TRACKBAR_RANGES = {
    "H_min": 179,
    "H_max": 179,
    "S_min": 255,
    "S_max": 255,
    "V_min": 255,
    "V_max": 255,
    "ROI_top_percent": 80,
    "Morph_kernel": 31,
    "Min_area_percent": 30,
}

SCANLINE_COUNT = 12
LAST_CENTER_HOLD_FRAMES = 12
CONFIDENCE_ALPHA = 0.18
CENTER_DEADBAND_PX = 35
CENTER_STRONG_PX = 110
SELECT_CONFIDENCE = 0.80
SELECT_MARGIN = 0.25
