from utils import read_video, save_video
from trackers import VehicleTracker

import const

def main():
    # breaking video down to frames
    video_frames = read_video(const.INPUT_VIDEO_PATH)

    # creating object to utilize the data collected from frames
    vehicle_tracker = VehicleTracker(const.MODEL_PATH)
    # create a large dataframe with the information detections from frames
    vehicle_detections = vehicle_tracker.detect_Vehicles(video_frames, read_from_stub=True, stub_path=const.STUB_PATH)
    
    # proccess video frames based on detections
    video_frames = vehicle_tracker.outline_objects(video_frames, vehicle_detections,
                                                   bbox_enable=const.BBOX_ENABLE,
                                                   speed_enable=const.VELOCITY_ENABLE,
                                                   trails_enable=const.TRAILS_ENABLE,
                                                   of_ramp_enable=const.OF_RAMP_ENABLE,
                                                   heatmap_enable=const.HEATMAP_ENABLE)

    # convert frames back into a video
    save_video(video_frames[1:], const.OUTPUT_VIDEO_PATH)

if __name__ == '__main__':
    main()