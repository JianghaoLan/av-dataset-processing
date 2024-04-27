import argparse
import os
import os.path as osp
import cv2
from tqdm import tqdm
from align import NoLandmarksFoundException, align_image

from lib.landmarks_pytorch import LandmarksEstimation

# IMAGE_EXT = ('.jpg', '.jpeg', '.png')
VIDEO_EXT = ('.mp4')

# def align_video(le, video_path, size, conf_threshold, keep_largest):
#     # Open input image
#     # img = read_image_opencv(video_file).copy()
    
#     video_capture = cv2.VideoCapture(video_path)
#     if not video_capture.isOpened():
#         print("Error: Could not open video file.")
#         exit()
#     frame_idx = -1
#     while True:
#         frame_idx += 1
        
#         # 读取一帧
#         ret, frame = video_capture.read()

#         # 检查是否成功读取帧
#         if not ret:
#             # print("Error: Could not read frame.")
#             break
        
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         try:
#             img = align_image(le, img, size, conf_threshold, keep_largest)
#         except NoLandmarksFoundException:
#             print("#. Warning: No landmarks found in {}:{}".format(video_path, frame_idx))
#             with open('issues.txt', 'a' if osp.exists('issues.txt') else 'w') as f:
#                 f.write("{}\n".format(video_path))
#             continue
        
#         # Save output image
#         output_path = osp.join(output_dir, osp.basename(osp.splitext(video_path)[0]), f'{frame_idx}.png')
#         os.makedirs(osp.dirname(output_path), exist_ok=True)
#         cv2.imwrite(output_path, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))

#     # 释放视频捕捉对象和关闭窗口
#     video_capture.release()

def main():
    """TODO: add docstring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='set input image directory')
    parser.add_argument('--output-dir', type=str, help='set output image directory')
    parser.add_argument('--size', type=int, default=512, help='set output size of cropped image')
    parser.add_argument('--conf-threshold', type=float, default=0.99, help='confidence threshold')
    parser.add_argument('--keep-largest', action='store_true', help='Only keep largest face instead of center face')
    args = parser.parse_args()

    # Get input/output directories
    input_dir = osp.abspath(osp.expanduser(args.input_dir))
    if args.output_dir:
        output_dir = osp.abspath(osp.expanduser(args.output_dir))
    else:
        output_dir = osp.join(osp.split(input_dir)[0], "{}_aligned".format(osp.split(input_dir)[1]))
    # Create output directory
    print("#. Create output directory: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get input images paths
    input_videos = [osp.join(input_dir, dI) for dI in os.listdir(input_dir)
                    if osp.isfile(osp.join(input_dir, dI)) and osp.splitext(dI)[-1] in VIDEO_EXT]
    input_videos.sort()

    # Build landmark estimator
    le = LandmarksEstimation(type='2D')

    for video_file in tqdm(input_videos, desc='Preprocess {} images'.format(len(input_videos))):
        # Open input image
        # img = read_image_opencv(video_file).copy()
        
        video_capture = cv2.VideoCapture(video_file)
        if not video_capture.isOpened():
            print("Error: Could not open video file.")
            exit()
        frame_idx = -1
        while True:
            frame_idx += 1
            
            # 读取一帧
            ret, frame = video_capture.read()

            # 检查是否成功读取帧
            if not ret:
                # print("Error: Could not read frame.")
                break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                img = align_image(le, img, args.size, args.conf_threshold, args.keep_largest)
            except NoLandmarksFoundException:
                print("#. Warning: No landmarks found in {}:{}".format(video_file, frame_idx))
                with open('issues.txt', 'a' if osp.exists('issues.txt') else 'w') as f:
                    f.write("{}\n".format(video_file))
                continue
            
            # Save output image
            output_path = osp.join(output_dir, osp.basename(osp.splitext(video_file)[0]), f'{frame_idx}.png')
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))

        # 释放视频捕捉对象和关闭窗口
        video_capture.release()


if __name__ == "__main__":
    main()
