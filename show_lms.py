import os
import argparse
import random
import cv2
import json

import numpy as np

from crop_dataset import gen_src_data
from utils.lms68_utils import get_left_eye, get_mouth, get_right_eye, get_square_roi


def load_json(lms_path):
    with open(lms_path) as f:
        return json.load(f)
    

def draw_rectangle(image, roi, color=(255, 191, 0)):
    roi = roi.astype(int)
    pt1 = roi[:2]
    pt2 = roi[2:]
    cv2.rectangle(image, pt1, pt2, color, 4)


def draw_lm(image, lm, roi):
    assert len(lm) == 68
    assert roi is None or len(roi) == 5
    
    # 给定坐标和半径画一个点
    radius = 5  # 半径
    # radius = 1  # 半径
    thickness = -1  # 填充圆
    font = cv2.FONT_HERSHEY_SIMPLEX
    lm_law = lm[:17]
    lm_eye_brow = lm[17:27]
    lm_nose = lm[27:36]
    lm_eyes = lm[36:48]
    lm_mouth = lm[48:]
    i = 0
    for points, color in zip([lm_law, lm_eye_brow, lm_nose, lm_eyes, lm_mouth], [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 165, 255)]):
        for p in points:
            p = int(p[0]), int(p[1])
            cv2.circle(image, p, radius, color, thickness)
            # cv2.putText(image, str(i), p, font, 0.3,(0, 238, 118), 1, cv2.LINE_AA)
            i += 1
            
    # draw roi
    if roi is not None:
        draw_rectangle(image, roi, (255, 191, 0))
    
    # draw components
    mouth_roi = get_square_roi(get_mouth(lm), enlarge_ratio=1.4)
    l_eye_roi = get_square_roi(get_left_eye(lm))
    r_eye_roi = get_square_roi(get_right_eye(lm))
    draw_rectangle(image, mouth_roi, (0, 0, 0))
    draw_rectangle(image, l_eye_roi, (0, 0, 0))
    draw_rectangle(image, r_eye_roi, (0, 0, 0))
    


def show_video_lms(src_path, src_lms_path, src_rois_path, dst_path):
    # 打开视频文件
    video_capture = cv2.VideoCapture(src_path)

    # 获取视频的帧率和大小
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(dst_path, fourcc, fps, (frame_width, frame_height))
    
    lms = load_json(src_lms_path)
    rois = load_json(src_rois_path) if src_rois_path is not None else None
    lms_it = iter(lms)
    rois_it = iter(rois) if rois is not None else None

    # 读取视频帧并修改每一帧
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break  # 视频结束，退出循环
        
        lm = next(lms_it)
        roi = next(rois_it) if rois_it is not None else None
        draw_lm(frame, np.asarray(lm, dtype=np.float32), roi)

        # 将修改后的帧写入输出视频文件
        output_video.write(frame)

    # 释放 VideoCapture 和 VideoWriter 对象
    video_capture.release()
    output_video.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--dst_root', required=True, type=str)
    args = parser.parse_args()
    
    filelist = args.filelist
    dst_root = args.dst_root
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)
    
    # with open(filelist) as f:
    #     lines = f.readlines()
    # src_list = map(lambda x: x.strip(), lines)
    src_list = [os.path.join(os.path.abspath(r[2]), 'ori_video.mp4') for r in gen_src_data('/data2/CN-CVS/cncvs-stylegan/final/', '/data2/CN-CVS/cncvs-stylegan/result.txt')]
    src_list = random.choices(src_list, k=10)
    
    for src in src_list:
        lms_path = os.path.join(os.path.dirname(src), 'landmarks.json')
        rois_path = os.path.join(os.path.dirname(src), 'rois.json')
        dst = os.path.join(dst_root, os.path.basename(os.path.dirname(src)) + '.mp4')
        show_video_lms(src, lms_path, rois_path, dst)
        print(dst)


if __name__ == '__main__':
    # main()
    # show_video_lms('/data2/CN-CVS/cncvs-stylegan/final/s02148/s02148_001_00020/ori_video.mp4', 
    #                '/data2/CN-CVS/cncvs-stylegan/final/s02148/s02148_001_00020/landmarks.json', 
    #                '/data2/CN-CVS/cncvs-stylegan/final/s02148/s02148_001_00020/rois.json',
    #                './lms_test.mp4')
    show_video_lms('/data2/CN-CVS/synced/s00005/s00005_001_00006/video.mp4', 
                   '/data2/CN-CVS/synced/s00005/s00005_001_00006/aligned_lms.json', 
                   None,
                   './output/lms_test5.mp4')
