import os
import os.path as osp
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import scipy
from tqdm import tqdm


class FaceAlignment:
    def __init__(self, ori_image, lms, quad, to_size, cropped_face):
        self._ori_image = ori_image
        self._lms = lms
        self._quad = np.float32(quad)
        self._to_size = to_size
        self._cropped_face = cropped_face
        
    def _get_inverse_transform(self):
        pts_src = np.float32([[0, 0], [0, self._to_size], [self._to_size, self._to_size], [self._to_size, 0]])
        pts_dst = self._quad + 0.5
        return cv2.getAffineTransform(pts_src[:3], pts_dst[:3])
    
    def _get_transform(self):
        pts_src = self._quad + 0.5
        pts_dst = np.float32([[0, 0], [0, self._to_size], [self._to_size, self._to_size], [self._to_size, 0]])
        return cv2.getAffineTransform(pts_src[:3], pts_dst[:3])
        
    def get_lms(self):
        return self._lms
    
    def get_cropped_lms(self):
        M = self._get_transform()
        p1 = np.hstack([self._lms, np.ones((len(self._lms), 1))])
        return np.dot(M, p1.T).T
    
    def get_cropped_face(self):
        return self._cropped_face

    def get_synthesized_image(self, cropped_face):
        M = self._get_transform()
        transformed_face = cv2.warpAffine(cropped_face, M, (self._ori_image.shape[1], self._ori_image.shape[0]))
        
        # 创建一个掩码
        mask = np.zeros_like(self._ori_image, dtype=np.float32)
        pts_dst = self._quad + 0.5
        cv2.fillConvexPoly(mask, pts_dst.astype(int), (1.0, 1.0, 1.0))
        # 定义结构元素
        kernel = np.ones((3, 3), np.uint8)
        # 掩码稍微收缩，解决黑边问题
        mask = cv2.erode(mask, kernel, iterations=1)
        
        syn_img = mask * transformed_face + (1 - mask) * self._ori_image
        return syn_img
    

def get_face_alignment(image, landmarks, transform_size=256) -> FaceAlignment:
    # Get estimated landmarks
    lm = landmarks
    lm_chin = lm[0: 17]            # left-right
    lm_eyebrow_left = lm[17: 22]   # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]           # top-down
    lm_nostrils = lm[31: 36]       # top-down
    lm_eye_left = lm[36: 42]       # left-clockwise
    lm_eye_right = lm[42: 48]      # left-clockwise
    lm_mouth_outer = lm[48: 60]    # left-clockwise
    lm_mouth_inner = lm[60: 68]    # left-clockwise

    # Calculate auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    ori_quad = quad.copy()
    qsize = np.hypot(*x) * 2

    img = Image.fromarray(image)
    shrink = int(np.floor(qsize / transform_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    enable_padding = True
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        # mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
        #                   1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / (pad[0] + 1e-12), np.float32(w - 1 - x) / (pad[2] + 1e-12)),
                          1.0 - np.minimum(np.float32(y) / (pad[1] + 1e-12), np.float32(h - 1 - y) / (pad[3] + 1e-12)))

        blur = qsize * 0.01
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

        quad += pad[:2]
    
    img = np.array(img)
    pts_src = np.float32(quad + 0.5)
    pts_dst = np.float32([[0, 0], [0, transform_size], [transform_size, transform_size], [transform_size, 0]])
    M = cv2.getAffineTransform(pts_src[:3], pts_dst[:3])
    cropped_face = cv2.warpAffine(img, M, (transform_size, transform_size))

    return FaceAlignment(image, landmarks, ori_quad, transform_size, cropped_face)

    # Transform
    res = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(),
                        Image.Resampling.BILINEAR)
    
    return np.array(res)


VIDEO_EXT = ('.mp4')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def convert_video_to_images(src_video_path, dst_dir, suffix=''):
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(dst_dir, f"{frame_count:3}{suffix}.png")
        cv2.imwrite(output_path, frame)
        frame_count += 1
    # 释放视频对象
    cap.release()


def draw_line(img, p1, p2):
    # 绘制线条
    color = (0, 255, 0)  # 线条颜色为绿色
    thickness = 3  # 线条宽度为3像素
    cv2.line(img, p1, p2, color, thickness)


def main():
    """TODO: add docstring
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input-dir', type=str, required=True, help='set input image directory')
    # parser.add_argument('--output-dir', type=str, help='set output image directory')
    parser.add_argument('--src-video-path', type=str, required=True)
    parser.add_argument('--src-landmarks-path', type=str, required=True)
    parser.add_argument('--dst-dir', type=str, required=True)
    # parser.add_argument('--copy', action='store_true')
    parser.add_argument('--size', type=int, default=512, help='set output size of cropped image')
    # parser.add_argument('--conf-threshold', type=float, default=0.99, help='confidence threshold')
    # parser.add_argument('--keep-largest', action='store_true', help='Only keep largest face instead of center face')
    args = parser.parse_args()
    
    src_video_path = args.src_video_path
    src_lm_path = args.src_landmarks_path
    dst_dir = args.dst_dir

    # # Get input/output directories
    # input_dir = osp.abspath(osp.expanduser(args.input_dir))
    # if args.output_dir:
    #     output_dir = osp.abspath(osp.expanduser(args.output_dir))
    # else:
    #     output_dir = osp.join(osp.split(input_dir)[0], "{}_aligned".format(osp.split(input_dir)[1]))
    # Create output directory
    print("#. Create output directory: {}".format(dst_dir))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # # Get input images paths
    # input_videos = [osp.join(input_dir, dI) for dI in os.listdir(input_dir)
    #                 if osp.isfile(osp.join(input_dir, dI)) and osp.splitext(dI)[-1] in VIDEO_EXT]
    # input_videos.sort()

    lms = np.array(load_json(src_lm_path))

    # Open input image
    # img = read_image_opencv(video_file).copy()
    
    video_capture = cv2.VideoCapture(src_video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    basename = osp.basename(osp.splitext(src_video_path)[0])
    os.makedirs(osp.join(dst_dir, basename), exist_ok=True)
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

        # # img = align_image(le, img, args.size, args.conf_threshold, args.keep_largest)
        # cropped_img, quad = align_crop_image(image=img,
        #         landmarks=lms[frame_idx],
        #         transform_size=args.size,
        #         return_quad=True)

        # # Save output image
        # output_path = osp.join(dst_dir, basename, f'{frame_idx:03}.png')
        # cv2.imwrite(output_path, cv2.cvtColor(cropped_img.copy(), cv2.COLOR_RGB2BGR))
        
        # def save_temp_img_with_quad():
        #     temp_with_quad = img.copy()
        #     for i in range(len(quad)):
        #         draw_line(temp_with_quad, quad[i].astype(int), quad[(i + 1) % len(quad)].astype(int)) 
        #     output_path_ori = osp.join(dst_dir, basename, f'{frame_idx:03}-ori.png')
        #     cv2.imwrite(output_path_ori, cv2.cvtColor(temp_with_quad.copy(), cv2.COLOR_RGB2BGR))
        # save_temp_img_with_quad()
        
        align = get_face_alignment(img.copy(), lms[frame_idx], args.size)

        cropped = align.get_cropped_face()
        output_path = osp.join(dst_dir, basename, f'{frame_idx:03}-affine.png')
        cv2.imwrite(output_path, cv2.cvtColor(cropped.copy(), cv2.COLOR_RGB2BGR))
        
        cropped_lms = align.get_cropped_lms()
        cropped_with_lms = cropped.copy()
        for p in cropped_lms:
            cv2.circle(cropped_with_lms, p.astype(int), radius=2, color=(255, 0, 0), thickness=-1)  # 用绿色点标记，半径为5
        output_path = osp.join(dst_dir, basename, f'{frame_idx:03}-lms.png')
        cv2.imwrite(output_path, cv2.cvtColor(cropped_with_lms.copy(), cv2.COLOR_RGB2BGR))
        
        syn_img = align.get_synthesized_image(cropped)
        output_path = osp.join(dst_dir, basename, f'{frame_idx:03}-syn.png')
        cv2.imwrite(output_path, cv2.cvtColor(syn_img.copy(), cv2.COLOR_RGB2BGR))
        
        output_path = osp.join(dst_dir, basename, f'{frame_idx:03}-ori1.png')
        cv2.imwrite(output_path, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
        
        break

    # 释放视频捕捉对象和关闭窗口
    video_capture.release()


if __name__ == "__main__":
    main()
