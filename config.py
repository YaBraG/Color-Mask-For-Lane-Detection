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
    "ROI_top_percent": 58,
    "Morph_kernel": 1,
    "Close_kernel": 0,
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
    "Close_kernel": 51,
    "Min_area_percent": 30,
}

SCANLINE_COUNT = 12
MIN_SEGMENT_WIDTH_PX = 20
MAX_CENTER_JUMP_PX = 95
ALLOW_FIRST_ANCHOR_JUMP = True
USE_EGO_CONNECTED_MASK = True
EGO_SEED_X_RATIO = 0.50
EGO_SEED_Y_RATIO = 0.95
EGO_SEED_SEARCH_RADIUS_PX = 120
EGO_BOTTOM_BAND_PERCENT = 18
EGO_MIN_COMPONENT_AREA_PERCENT = 1.0
CENTERLINE_SMOOTHING_ALPHA = 0.45
LAST_CENTER_HOLD_FRAMES = 12
CONFIDENCE_ALPHA = 0.18
CENTERLINE_ALPHA = 0.25
CENTER_DEADBAND_PX = 35
CENTER_STRONG_PX = 110
CURVE_DEADBAND_PX = 45
CURVE_STRONG_PX = 140
SELECT_CONFIDENCE = 0.80
SELECT_MARGIN = 0.25

# Physical safe-corridor values in millimeters. The blue corridor is narrower
# than the full road: it represents the car body plus practical margins.
LANE_WIDTH_MM = 254.0
CAR_WIDTH_MM = 203.2
SIDEWALK_MARGIN_MM = 12.0
LINE_MARGIN_MM = 12.0
SAFE_HALLWAY_WIDTH_MM = CAR_WIDTH_MM + SIDEWALK_MARGIN_MM + LINE_MARGIN_MM
MIN_VALID_LANE_WIDTH_MM = 220.0
MAX_VALID_LANE_WIDTH_MM = 290.0
SAFE_SCANLINE_COUNT = 6
SAFE_SCANLINE_START_RATIO = 0.62
SAFE_SCANLINE_END_RATIO = 0.95
SAFE_SCANLINE_NEAR_WEIGHT_BIAS = 2.0
SAFE_MIN_VALID_SCANLINES = 4
SAFE_MIN_ROAD_CONFIDENCE = 0.55
SAFE_ERROR_DEADBAND_MM = 5.0
SAFE_STEERING_GAIN = 0.01
SAFE_MAX_STEERING_CORRECTION = 0.20

# Yellow boundary lock keeps the blue safe corridor on the current lane side.
# HSV yellow pixels are treated as a no-cross divider, not as road.
USE_YELLOW_BOUNDARY_LOCK = True
YELLOW_H_MIN = 18
YELLOW_H_MAX = 45
YELLOW_S_MIN = 80
YELLOW_S_MAX = 255
YELLOW_V_MIN = 80
YELLOW_V_MAX = 255
YELLOW_BOUNDARY_DILATE_PX = 7
YELLOW_MAX_CROSSING_PIXELS = 20
LANE_SIDE_HOLD_FRAMES = 15
USE_RIGHT_LANE_YELLOW_LOCK = True
RIGHT_LANE_FROM_YELLOW = True
YELLOW_LANE_SIDE = "right"
YELLOW_LANE_SEARCH_MARGIN_PX = 8
YELLOW_MIN_PIXELS_PER_SCANLINE = 3
YELLOW_RIGHT_LANE_HOLD_FRAMES = 20
USE_NO_YELLOW_WIDE_BLOB_GATE = True
NO_YELLOW_MAX_BLOB_WIDTH_PX_RATIO = 0.55
NO_YELLOW_MAX_MEASURED_WIDTH_MM = 310.0
ALLOW_NO_YELLOW_BLOB_SPLIT = False
