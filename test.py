from ultralytics import YOLO
from utils import read_video, save_video, apply_speed, draw_trails

import const
import cv2
import numpy as np
import pandas as pd

def main():
    # breaking video down to frames
    video_frames = read_video(const.INPUT_VIDEO_PATH)
    # loading custom trained model
    # model = YOLO('models/best.pt')
    # model.conf = 0.1

    # creating a dataframe of ids and coordinates
    df = pd.DataFrame(columns=['id', 'coords', 'centroid'])

    # results = model.predict(video_frames[0])

    for frame in video_frames:
        frame = cv2.line(frame, (770, 500), (750, 600), (0, 255, 0), 3)

    # # for all frames
    # for i in range (0, len(video_frames)):
    #     frame = video_frames[i]
    #     # track vehicles
    #     results = model.track(frame, persist=True)
    #     obj_in_frame = []

    #     # for all vehicles tracked
    #     for box in results[0].boxes:
    #         # extract coordinates
    #         x1, y1, x2, y2 = box.xyxy[0]

    #         coords = [int(x1), int(y1), int(x2), int(y2)]
    #         centroid = [int(x1 + (abs(x2 - x1) / 2)), int(y1 + (abs(y2 - y1)) / 2)]

    #         # save object id and centroid
    #         df.loc[len(df)] = [int(box.id), coords, centroid]

    #         #draw box surrounding object
    #         frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #         frame = cv2.putText(frame, f"ID: {int(box.id)}", (int(x1)-10, int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    #         obj_in_frame.append(int(box.id))

    #         velocity = 0

    #     for obj in obj_in_frame:
    #         object_data = df[df['id'] == obj]

    #         if len(object_data) >= 2:
    #             velocity, coords = apply_speed(object_data)

    #             frame = cv2.putText(frame, f"{round(velocity, 2)}", (coords[0] - 10, coords[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    #             frame = cv2.putText(frame, "km/hr", (coords[0] + 40, coords[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    #         if const.ENABLE_TRAILS:
    #             # pts = apply_trail(object_data, df, frame)
    #             # frame = cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    #             frame = draw_trails(object_data, frame)

    # convert frames back into a video
    save_video(video_frames[1:], const.OUTPUT_VIDEO_PATH)


if __name__ == '__main__':
    main()