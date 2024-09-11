INPUT_VIDEO_PATH = "input_samples/Roundabout_shortened_1.mp4"
STUB_PATH = 'tracker_stubs/vehicle_detections_1.pkl'
MODEL_PATH = "models/best.pt"

BBOX_ENABLE = True
VELOCITY_ENABLE = True
TRAINS_ENABLE = True
HEATMAP_ENABLE = True

ROUNDABOUT_INNER_LINE = [(830, 560), (1125, 560)]
# PIXELS_PER_METER = (ROUNDABOUT_INNER_LINE[1][0] - ROUNDABOUT_INNER_LINE[0][0]) / 30           # REVESE VARIABLE NAME
METERS_PER_PIXEL = 1 / ((ROUNDABOUT_INNER_LINE[1][0] - ROUNDABOUT_INNER_LINE[0][0]) / 30)           # REVESE VARIABLE NAME
MPF_TO_KMH = 108

TRAIL_COLOR_VALUES = [
    (0, 0, 255),      # Red
    (0, 6, 255),
    (0, 18, 255),
    (0, 24, 255),
    (0, 30, 255),
    (0, 36, 255),
    (0, 42, 255),
    (0, 60, 255),
    (0, 66, 255),
    (0, 72, 255),
    (0, 78, 255),
    (0, 90, 255),
    (0, 96, 255),
    (0, 102, 255),
    (0, 108, 255),
    (0, 114, 255),
    (0, 120, 255),
    (0, 138, 255),
    (0, 144, 255),
    (0, 156, 255),
    (0, 162, 255),
    (0, 168, 255),
    (0, 180, 255),
    (0, 198, 255),
    (0, 204, 255),
    (0, 216, 255),
    (0, 222, 255),
    (0, 228, 255),
    (0, 234, 255),
    (0, 246, 255),
    (0, 252, 255),
    (0, 255, 255),    # Yellow (midway)
    (0, 255, 240),
    (0, 255, 225),
    (0, 255, 210),
    (0, 255, 195),
    (0, 255, 180),
    (0, 255, 150),
    (0, 255, 135),
    (0, 255, 120),
    (0, 255, 90),
    (0, 255, 75),
    (0, 255, 60),
    (0, 255, 45),
    (0, 255, 15),
    (0, 255, 0)       # Green
]

LINE_ALPHA = 0.4
LINE_COLOR = (0, 102, 255, LINE_ALPHA)

LINE_1 = [(730, 630), (750, 430)]
LINE_2 = [LINE_1[1], (875, 320)]
LINE_3 = [LINE_2[1], (1090, 315)]
LINE_4 = [(1210, 445), LINE_3[1]]
LINE_5 = [(1225, 685), LINE_4[0]]
LINE_6 = [(1060, 830), LINE_5[0]]
LINE_7 = [LINE_6[0], (835, 810)]
LINE_8 = [LINE_7[1], LINE_1[0]]

FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920

OUTPUT_VIDEO_PATH = "output_videos/output_video_8.avi"