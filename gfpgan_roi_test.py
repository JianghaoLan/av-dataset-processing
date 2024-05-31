import os
import torch
import cv2
import numpy as np


def get_box(roi):
    center = roi[:2]
    half_len = roi[2]
    return ((center - half_len), (center + half_len))


def draw_rect(img, roi, color, thickness):
    start, end = get_box(roi)
    cv2.rectangle(img, start.astype(int), end.astype(int), color, thickness)


def show_roi(img_path, rois, out_path):
    img = cv2.imread(img_path)
    resize_ratio = img.shape[0] / 512
    roi_le = np.array(rois['left_eye'], dtype=float) * resize_ratio
    roi_re = np.array(rois['right_eye'], dtype=float) * resize_ratio
    roi_m = np.array(rois['mouth'], dtype=float) * resize_ratio
    draw_rect(img, roi_le, (255, 0, 0), 2)
    draw_rect(img, roi_re, (0, 255, 0), 2)
    draw_rect(img, roi_m, (0, 0, 255), 2)
    cv2.imwrite(out_path, img)


def get_rois(roi_dict, img_name):
    idx = int(os.path.splitext(img_name)[0])
    return roi_dict[f'{idx:08d}']


def main():
    roi_path = 'assets\\FFHQ_eye_mouth_landmarks_512.pth'
    img_dir = 'D:\\dataset\\FFHQ\\00000'
    out_dir = 'output\\gfpgan_roi_test'
    
    roi_dict = torch.load(roi_path)
    print(list(roi_dict.keys())[:5])
    
    os.makedirs(out_dir, exist_ok=True)
    for img_name in os.listdir(img_dir)[:3]:
        img_path = os.path.join(img_dir, img_name)
        out_path = os.path.join(out_dir, img_name)
        
        rois = get_rois(roi_dict, img_name)
        show_roi(img_path, rois, out_path)
        # print(rois)


if __name__ == '__main__':
    main()
