import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import argparse

import os



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
            description='Extrapolate and manipulate images to obtain leaf dimension')

    parser.add_argument("-f", "--folder", 
            default="C:/Users/david/Pictures/realsense/video",
            type=str,
            help="Path to folder that contains video streams")

    parser.add_argument("-s", "--saving", 
            default="C:/Users/david/Pictures/realsense/frames",
            type=str,
            help="Path to folder that contains video streams")



    return parser.parse_args(argv)


def get_frames(video_name, saving_folder, frame, start=0):

    print(f"Start analyzing video n° {frame}")

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    rs.config.enable_device_from_file(config, video_name)


    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    device = pipeline_profile.get_device()

    sensors = device.query_sensors()

    fps_d = sensors[0].get_stream_profiles()[0].fps()
    fps_rgb = sensors[1].get_stream_profiles()[0].fps()

    format_d = sensors[0].get_stream_profiles()[0].format()
    format_rgb = sensors[1].get_stream_profiles()[0].format()

    width_d = sensors[0].get_stream_profiles()[0].as_video_stream_profile().width()
    width_rgb = sensors[1].get_stream_profiles()[0].as_video_stream_profile().width()

    height_d = sensors[0].get_stream_profiles()[0].as_video_stream_profile().height()
    height_rgb = sensors[1].get_stream_profiles()[0].as_video_stream_profile().height()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, width_d, height_d, format_d, fps_d)

    config.enable_stream(rs.stream.color, width_rgb, height_rgb, format_rgb, fps_rgb)



    # Start streaming
    profile = pipeline.start(config)

    profile = pipeline.get_active_profile()
    print(profile)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    print(depth_profile)
    depth_intrinsics = depth_profile.get_intrinsics()
    print(f"intrinsics: {depth_intrinsics}")
    exit()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    # clipping_distance_in_meters = 1 #1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_number = 0



    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # aligned_depth_frame = colorizer.colorize(aligned_depth_frame)
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            if color_frame.frame_number > frame_number:
                frame_number = color_frame.frame_number
                color_frame_to_get = color_image
                depth_frame_to_get = depth_image

            if frame_number > color_frame.frame_number:
                cv2.imwrite(f"{saving_folder}/depth/depth_{start+frame:03d}.png", depth_frame_to_get)
                cv2.imwrite(f"{saving_folder}/color/color_{start+frame:03d}.png", color_frame_to_get)
                break
                
    finally:
        pipeline.stop()




if __name__ == "__main__":
    args = parse_args()
    start = 0

    if not os.path.isdir(args.saving):
        os.mkdir(args.saving)
        os.mkdir(f"{args.saving}/color")
        os.mkdir(f"{args.saving}/depth")
    else:
        if not os.path.isdir(f"{args.saving}/color"):
            os.mkdir(f"{args.saving}/color")
        else:
            last = os.listdir(f"{args.saving}/color")[-1]
            last = last.split("_")[-1].split(".")[0]
            start = int(last) + 1
        if not os.path.isdir(f"{args.saving}/depth"):
            os.mkdir(f"{args.saving}/depth")

    videos = os.listdir(args.folder)

    for idx, video in enumerate(videos):
        if ".bag" in video:
            get_frames(f"{args.folder}/{video}", args.saving, idx, start)




