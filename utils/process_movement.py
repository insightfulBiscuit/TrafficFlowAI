import const
import math
import cv2
import numpy as np

def apply_speed(object_data, frame):
    object_data = object_data.tail(2)
    coords = object_data['coords'].tolist()

    centroids = [
        [(coords[0][0] + coords[0][2]) / 2,  # Centroid x for first box
         (coords[0][1] + coords[0][3]) / 2], # Centroid y for first box

        [(coords[1][0] + coords[1][2]) / 2,  # Centroid x for second box
         (coords[1][1] + coords[1][3]) / 2]  # Centroid y for second box
    ]

    velocity = (
        math.sqrt(
            ((centroids[1][0] - centroids[0][0]) ** 2) + 
            ((centroids[1][1] - centroids[0][1]) ** 2)
        ) * const.METERS_PER_PIXEL                          # VARIABLE NAME NEEDS TO BE RENAMED
    ) * const.MPF_TO_KMH

    # Return the current bounding box (coords[1]) for accurate text positioning
    text_coords = [int(coords[1][0]), int(coords[1][3])]

    # print(text_coords)

    frame = cv2.putText(
        frame, f"{round(velocity, 2)} km/h", (text_coords[0] - 10, text_coords[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 
        0.5, (0, 255, 0), 2
    )

    return frame


def draw_trails(object_data, frame):
    if len(object_data) >= 45:
        object_data = object_data.tail(45)

    coords = object_data['coords'].tolist()

    for i in range(0, len(coords) - 2):
        # start_point = (trail[i][0], trail[i][1])
        start_point = [
            round((coords[i][0] + coords[i][2]) / 2),  # Centroid x for first box
            round((coords[i][1] + coords[i][3]) / 2)] # Centroid y for first box

        # end_point = (trail[i + 1][0], trail[i + 1][1])
        end_point = [
            round((coords[i + 1][0] + coords[i + 1][2]) / 2),  # Centroid x for first box
            round((coords[i + 1][1] + coords[i + 1][3]) / 2)]

        color = const.TRAIL_COLOR_VALUES[i]  # Get the corresponding color from the array

        frame = cv2.line(frame, start_point, end_point, color, 2)

    return frame

def has_crossed_line(object_data, line):
    # Ensure we have at least two points to check
    if len(object_data) < 2:
        return False
    
    # Get the last two centroids (current and previous positions)
    object_data = object_data.tail(2)
    coords = object_data['coords'].tolist()

    centroids = [
    [(coords[0][0] + coords[0][2]) / 2,  # Centroid x for first box
        (coords[0][1] + coords[0][3]) / 2], # Centroid y for first box

    [(coords[1][0] + coords[1][2]) / 2,  # Centroid x for second box
        (coords[1][1] + coords[1][3]) / 2]  # Centroid y for second box
    ]

    ac = np.array([line[0][0] - centroids[0][0], line[0][1] - centroids[0][1]])
    ab = np.array([centroids[1][0] - centroids[0][0], centroids[1][1] - centroids[0][1]])
    cd = np.array([line[1][0] - line[0][0], line[1][1] - line[0][1]])

    temp = np.cross(ab, cd)

    if temp == 0:
        return False

    cross = np.cross(ac, cd) / temp

    obj_id = object_data['id'].tail(1).tolist()[0]
    
    # # If the sign of the cross product changes and neither is zero, the point has crossed the line
    if 0 < cross and cross <= 1 and (line[0][1] >= centroids[1][1] and centroids[1][1] >= line[1][1]):
        return True
    return False

def overlay_heatmap(object_data, frame, heatmap):
    # Convert the 'coords' column to a list
    object_data = object_data['coords']

    # Loop over the bounding box coordinates
    for obj_coords in object_data:
        x_min, y_min, x_max, y_max = obj_coords
        heatmap[round(y_min):round(y_max), round(x_min):round(x_max)] += 1  # Accumulate heatmap data

       # Apply Gaussian blur to smooth the heatmap
    heatmap_smoothed = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalize heatmap for the current frame relative to the minimum value
    heatmap_min = heatmap_smoothed.min()
    heatmap_max = heatmap_smoothed.max()
    
    # Avoid division by zero and normalize relative to min
    if heatmap_max > heatmap_min:
        normalized_heatmap = (heatmap_smoothed - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)
    else:
        normalized_heatmap = heatmap_smoothed  # If all values are the same, skip normalization

    # Apply color map
    heatmap_colored = cv2.applyColorMap((normalized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Ensure heatmap_colored and frame are the same size
    if heatmap_colored.shape != frame.shape:
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # Blend heatmap with the current video frame
    frame = cv2.addWeighted(frame, 0.98, heatmap_colored, 0.02, 0)

    return frame, heatmap