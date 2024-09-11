import const
import math
import cv2
import numpy as np

# calculating speed of object based on its last two instance positions
def apply_speed(object_data, frame):
    # filter object data by its last two instances
    object_data = object_data.tail(2)
    # further filter by only coordinates
    coords = object_data['coords'].tolist()

    # converting coordinates to centroids
    centroids = [
        [(coords[0][0] + coords[0][2]) / 2,  # Centroid x for first box
         (coords[0][1] + coords[0][3]) / 2], # Centroid y for first box

        [(coords[1][0] + coords[1][2]) / 2,  # Centroid x for second box
         (coords[1][1] + coords[1][3]) / 2]  # Centroid y for second box
    ]

    # calculating velocity of object 
    velocity = (
        math.sqrt(
            ((centroids[1][0] - centroids[0][0]) ** 2) + 
            ((centroids[1][1] - centroids[0][1]) ** 2)
        ) * const.METERS_PER_PIXEL
    ) * const.MPF_TO_KMH

    # coordinates for text positioning
    text_coords = [int(coords[1][0]), int(coords[1][3])]

    # adding object velocity to frame
    frame = cv2.putText(
        frame, f"{round(velocity, 2)} km/h", (text_coords[0] - 10, text_coords[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 
        0.5, (0, 255, 0), 2
    )

    # returning frame
    return frame

# drawing trail of object
def draw_trails(object_data, frame):
    # if object data spans more than 45 instances
    if len(object_data) >= 45:
        # cut down data to its last 45 instances
        object_data = object_data.tail(45)

    # filtering data to coordinates
    coords = object_data['coords'].tolist()

    # for all coordinates of object (this function draws the trails by segments)
    for i in range(0, len(coords) - 2):
        # convering initial segment position to centroids
        start_point = [
            round((coords[i][0] + coords[i][2]) / 2),  # Centroid x for first box
            round((coords[i][1] + coords[i][3]) / 2)] # Centroid y for first box

        # converting previous segment position to centroids
        end_point = [
            round((coords[i + 1][0] + coords[i + 1][2]) / 2),  # Centroid x for first box
            round((coords[i + 1][1] + coords[i + 1][3]) / 2)]

        # color selected based on segment position in trail
        color = const.TRAIL_COLOR_VALUES[i]  # Get the corresponding color from the array

        # adding trail to frame
        frame = cv2.line(frame, start_point, end_point, color, 2)

    # returning frame
    return frame

# detecting if object has entered/exited the roundabout
# this function is based on linear algebra described here: https://www.youtube.com/watch?v=5FkOO1Wwb8w
def has_crossed_line(object_data, line):
    # if less than two instances of object
    if len(object_data) < 2:
        # cannot be determined dut to lack of info and therefore false
        return False
    
    # filtering data by its last two instances of object
    object_data = object_data.tail(2)
    # futher filtering data by only coordinates
    coords = object_data['coords'].tolist()

    # convering coordinates into centroids
    centroids = [
    [(coords[0][0] + coords[0][2]) / 2,
        (coords[0][1] + coords[0][3]) / 2],

    [(coords[1][0] + coords[1][2]) / 2,
        (coords[1][1] + coords[1][3]) / 2
    ]]

    # convering points into necessary vectors
    ac = np.array([line[0][0] - centroids[0][0], line[0][1] - centroids[0][1]])
    ab = np.array([centroids[1][0] - centroids[0][0], centroids[1][1] - centroids[0][1]])
    cd = np.array([line[1][0] - line[0][0], line[1][1] - line[0][1]])

    # calculating the denominator
    temp = np.cross(ab, cd)

    # if denominator is zero
    if temp == 0:
        # cannot divide by zero therefore false
        return False

    # calculating the "t" value as per video
    cross = np.cross(ac, cd) / temp
    
    # if "t" value is between 0 and 1, and object is within the y-bounds of line
    if 0 < cross and cross <= 1 and (line[0][1] >= centroids[1][1] and centroids[1][1] >= line[1][1]):
        # vehicle has cross the line and therefore true
        return True
    # otherwise false
    return False

# determining heatmap of objects
def overlay_heatmap(object_data, frame, heatmap):
    # filtering data by only coordinates
    object_data = object_data['coords']

    # for all coordinates in data 
    for obj_coords in object_data:
        x_min, y_min, x_max, y_max = obj_coords
        # adding depth to heatmap for all pixels object appears in
        heatmap[round(y_min):round(y_max), round(x_min):round(x_max)] += 1  # accumulate heatmap data

    # appling Gaussian blur to smooth the heatmap
    heatmap_smoothed = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # normalizing heatmap for the current frame relative to the minimum value
    heatmap_min = heatmap_smoothed.min()
    heatmap_max = heatmap_smoothed.max()
    
    # avoid division by zero and normalizing relative to min
    if heatmap_max > heatmap_min:
        normalized_heatmap = (heatmap_smoothed - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)
    else:
        normalized_heatmap = heatmap_smoothed  # if all values are the same, skip normalization

    # appling color map
    heatmap_colored = cv2.applyColorMap((normalized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # ensuring heatmap_colored and frame are the same size
    if heatmap_colored.shape != frame.shape:
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # blending heatmap with the current video frame
    frame = cv2.addWeighted(frame, 0.98, heatmap_colored, 0.02, 0)

    # returning frame and heatmap
    return frame, heatmap