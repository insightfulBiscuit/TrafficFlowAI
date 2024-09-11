from ultralytics import YOLO
from utils import read_video, save_video, apply_speed, draw_trails, has_crossed_line, overlay_heatmap

import const
import cv2
import numpy as np
import pandas as pd
import pickle

class VehicleTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_Vehicles(self, video_frames, read_from_stub=True, stub_path=None):
        vehicle_detections = pd.DataFrame(columns=['frame', 'id', 'coords', 'obj_id_in_frame'])

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                vehicle_detections = pickle.load(f)
            return vehicle_detections

        for i in range(0, len(video_frames)-1):
            print('Detecting Frame', i, '/', len(video_frames)-1)
            obj_id_in_frame = []
            results = self.model.track(video_frames[i], conf=0.2, persist=True)[0]
            for box in results.boxes:
                obj_id_in_frame.append(int(box.id))
                vehicle_detections.loc[len(vehicle_detections)] = [i, int(box.id), box.xyxy.tolist()[0], obj_id_in_frame]
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(vehicle_detections, f)
        
        return vehicle_detections
    
    def outline_objects(self, video_frames, vehicle_detections, bbox_enable=False, speed_enable=False, trails_enable=False, heatmap_enable=False):
        obj_cross_line = [[], [], [], [], [], [], [], []]
        heatmap = np.zeros((const.FRAME_HEIGHT, const.FRAME_WIDTH), dtype=np.float32)  # Single persistent heatmap
        for i in range(1, len(video_frames) - 1):
            print("Processing Frame", i, "/", len(video_frames)-2)

            obj_id_in_frame = vehicle_detections[vehicle_detections['frame'] == i]['obj_id_in_frame'].tail(1).tolist()[0]

            for obj_id in obj_id_in_frame:
                if heatmap_enable:
                    video_frames[i], heatmap = overlay_heatmap(vehicle_detections[vehicle_detections['frame'] == i], video_frames[i], heatmap)

                object_data = vehicle_detections[vehicle_detections['frame'] <= i][vehicle_detections[vehicle_detections['frame'] <= i]['id'] == obj_id]
                
                if len(object_data) >= 2:
                    if speed_enable:
                        video_frames[i] = apply_speed(object_data, video_frames[i])
                    if trails_enable:
                        video_frames[i] = draw_trails(object_data, video_frames[i])
                    if trails_enable:
                        if (has_crossed_line(object_data, const.LINE_1) and not obj_id in obj_cross_line[0]):
                            obj_cross_line[0].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_2) and not obj_id in obj_cross_line[1]):
                            obj_cross_line[1].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_3) and not obj_id in obj_cross_line[2]):
                            obj_cross_line[2].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_4) and not obj_id in obj_cross_line[3]):
                            obj_cross_line[3].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_5) and not obj_id in obj_cross_line[4]):
                            obj_cross_line[4].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_6) and not obj_id in obj_cross_line[5]):
                            obj_cross_line[5].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_7) and not obj_id in obj_cross_line[6]):
                            obj_cross_line[6].append(obj_id)
                        elif (has_crossed_line(object_data, const.LINE_8) and not obj_id in obj_cross_line[7]):
                            obj_cross_line[7].append(obj_id)

                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[0])}", 
                            (round((const.LINE_1[0][0] + const.LINE_1[1][0]) / 2), round((const.LINE_1[0][1] + const.LINE_1[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[1])}", 
                            (round((const.LINE_2[0][0] + const.LINE_2[1][0]) / 2), round((const.LINE_2[0][1] + const.LINE_2[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[2])}", 
                            (round((const.LINE_3[0][0] + const.LINE_3[1][0]) / 2), round((const.LINE_3[0][1] + const.LINE_3[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[3])}", 
                            (round((const.LINE_4[0][0] + const.LINE_4[1][0]) / 2), round((const.LINE_4[0][1] + const.LINE_4[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[4])}", 
                            (round((const.LINE_5[0][0] + const.LINE_5[1][0]) / 2), round((const.LINE_5[0][1] + const.LINE_5[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[5])}", 
                            (round((const.LINE_6[0][0] + const.LINE_6[1][0]) / 2), round((const.LINE_6[0][1] + const.LINE_6[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[6])}", 
                            (round((const.LINE_7[0][0] + const.LINE_7[1][0]) / 2), round((const.LINE_7[0][1] + const.LINE_7[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"{len(obj_cross_line[7])}", 
                            (round((const.LINE_8[0][0] + const.LINE_8[1][0]) / 2), round((const.LINE_8[0][1] + const.LINE_8[1][1]) / 2)), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2
                        )

                    if bbox_enable:
                        coords = object_data['coords'].tail(1).tolist()[0]
                        video_frames[i] = cv2.putText(
                            video_frames[i], f"ID: {obj_id}", (round(coords[0]) - 10, round(coords[1]) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2
                        )
                        video_frames[i] = cv2.rectangle(
                            video_frames[i], (round(coords[0]), round(coords[1])), (round(coords[2]), round(coords[3])), (0, 180, 255), 2
                        )

        return video_frames