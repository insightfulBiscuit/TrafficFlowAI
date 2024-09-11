from ultralytics import YOLO
from utils import read_video, save_video
from trackers import VehicleTracker

import const
import cv2
import numpy as np
import pandas as pd

def main():
    # breaking video down to frames
    video_frames = read_video(const.INPUT_VIDEO_PATH)

    vehicle_tracker = VehicleTracker(const.MODEL_PATH)
    vehicle_detections = vehicle_tracker.detect_Vehicles(video_frames, read_from_stub=False, stub_path=const.STUB_PATH)
    
    video_frames = vehicle_tracker.outline_objects(video_frames, vehicle_detections,
                                                   bbox_enable=const.BBOX_ENABLE,
                                                   speed_enable=const.VELOCITY_ENABLE,
                                                   trails_enable=const.TRAINS_ENABLE,
                                                   heatmap_enable=const.HEATMAP_ENABLE)
    
    # for frame in video_frames:
    #     frame = cv2.line(frame, (const.LINE_1[0][0], const.LINE_1[0][1]), (const.LINE_1[1][0], const.LINE_1[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_2[0][0], const.LINE_2[0][1]), (const.LINE_2[1][0], const.LINE_2[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_3[0][0], const.LINE_3[0][1]), (const.LINE_3[1][0], const.LINE_3[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_4[0][0], const.LINE_4[0][1]), (const.LINE_4[1][0], const.LINE_4[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_5[0][0], const.LINE_5[0][1]), (const.LINE_5[1][0], const.LINE_5[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_6[0][0], const.LINE_6[0][1]), (const.LINE_6[1][0], const.LINE_6[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_7[0][0], const.LINE_7[0][1]), (const.LINE_7[1][0], const.LINE_7[1][1]), const.LINE_COLOR, 3)
    #     frame = cv2.line(frame, (const.LINE_8[0][0], const.LINE_8[0][1]), (const.LINE_8[1][0], const.LINE_8[1][1]), const.LINE_COLOR, 3)

    # convert frames back into a video
    save_video(video_frames[1:], const.OUTPUT_VIDEO_PATH)

if __name__ == '__main__':
    main()