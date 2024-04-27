import os
import argparse
import random
import cv2
import json

from crop_dataset import gen_src_data


def load_json(lms_path):
    with open(lms_path) as f:
        return json.load(f)


def draw_lm(image, lm, roi):
    assert len(lm) == 68
    assert len(roi) == 5
    
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
    x1, y1, x2, y2 = map(int, roi[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 191, 0), 4)


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
    rois = load_json(src_rois_path)
    lms_it = iter(lms)
    rois_it = iter(rois)

    # 读取视频帧并修改每一帧
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break  # 视频结束，退出循环
        
        lm = next(lms_it)
        roi = next(rois_it)
        draw_lm(frame, lm, roi)

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
    show_video_lms('/data2/CN-CVS/cncvs-stylegan/final/s01222/s01222_001_00021/ori_video.mp4', 
                   '/data2/CN-CVS/cncvs-stylegan/final/s01222/s01222_001_00021/landmarks.json', 
                   '/data2/CN-CVS/cncvs-stylegan/final/s01222/s01222_001_00021/rois.json',
                   './lms_test2.mp4')
