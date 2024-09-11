import cv2

# frame-by-frame slicing
def read_video(video_path):
    # to store frames in array
    frames = []

    # opening video connection
    vid = cv2.VideoCapture(video_path)

    # for all frames in the video
    while True:
        # read frame
        continues, frame = vid.read()

        # if last frame
        if not continues:
            # end function
            break
        
        # add frame to the list
        frames.append(frame)

    # release video
    vid.release()

    # return frames
    return frames

# splicing frames back into video
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        vid_out.write(frame)

    vid_out.release