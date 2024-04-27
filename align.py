import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2
import PIL.Image
import PIL.ImageFile
from PIL import Image
import scipy.ndimage
from lib.landmarks_pytorch import LandmarksEstimation

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def align_crop_image(image, landmarks, transform_size=256):
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
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

        quad += pad[:2]

    # Transform
    img = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(),
                        Image.Resampling.BILINEAR)

    return np.array(img)


def read_image_opencv(image_path):
    # Read image in BGR order
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')


def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class NoLandmarksFoundException(Exception):
    pass


class MultiFacesDetectedException(Exception):
    pass


def align_image(le, img, size, conf_threshold, keep_largest=False, disable_multi_faces=False, return_roi_and_landmarks=False, 
                landmarks=None, detected_faces=None):
    # Landmark estimation
    img_tensor = torch.tensor(np.transpose(img, (2, 0, 1))).float().to(le.device)
    with torch.no_grad():
        landmarks, detected_faces = le.detect_landmarks(img_tensor.unsqueeze(0), detected_faces=None, conf_threshold=conf_threshold)
        landmarks, detected_faces = landmarks[0], detected_faces[0]
    # Align and crop face
    if len(landmarks) > 0:
        if len(landmarks) > 1 and disable_multi_faces:
            raise MultiFacesDetectedException
        
        if len(landmarks) == 1:
            landmarks = landmarks[0]
            detected_face = detected_faces[0]
        elif keep_largest:
            _, h, w = img_tensor.shape   
            detected_face, face_index = get_largest_face(detected_faces, h, w)
            landmarks = landmarks[face_index]
        else:       # keep center
            _, h, w = img_tensor.shape   
            detected_face, face_index = get_center_face(detected_faces, h, w)
            landmarks = landmarks[face_index]
        landmarks=np.asarray(landmarks[0].detach().cpu().numpy())
                             
        img = align_crop_image(image=img,
                                landmarks=landmarks,
                                transform_size=size)
        if return_roi_and_landmarks:
            return img, detected_face, landmarks
        return img
    else:
        raise NoLandmarksFoundException()


def main():
    """TODO: add docstring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='set input image directory')
    parser.add_argument('--output-dir', type=str, help='set output image directory')
    parser.add_argument('--size', type=int, default=256, help='set output size of cropped image')
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
    input_images = [osp.join(input_dir, dI) for dI in os.listdir(input_dir)
                    if osp.isfile(osp.join(input_dir, dI)) and osp.splitext(dI)[-1] in IMAGE_EXT]
    input_images.sort()

    # Build landmark estimator
    le = LandmarksEstimation(type='2D')

    for img_file in tqdm(input_images, desc='Preprocess {} images'.format(len(input_images))):
        # Open input image
        img = read_image_opencv(img_file).copy()

        # # Landmark estimation
        # img_tensor = torch.tensor(np.transpose(img, (2, 0, 1))).float().cuda()
        # with torch.no_grad():
        #     landmarks, detected_faces = le.detect_landmarks(img_tensor.unsqueeze(0), detected_faces=None, conf_threshold=args.conf_threshold)
        #     landmarks, detected_faces = landmarks[0], detected_faces[0]
        # # Align and crop face
        # if len(landmarks) > 0:
        #     if args.keep_largest:
        #         _, h, w = img_tensor.shape   
        #         _, face_index = get_largest_face(detected_faces, h, w)
        #         landmarks = landmarks[face_index]
        #     else:       # keep center
        #         _, h, w = img_tensor.shape   
        #         _, face_index = get_center_face(detected_faces, h, w)
        #         landmarks = landmarks[face_index]
            
        #     img = align_crop_image(image=img,
        #                            landmarks=np.asarray(landmarks[0].detach().cpu().numpy()),
        #                            transform_size=args.size)
        # else:
        #     print("#. Warning: No landmarks found in {}".format(img_file))
        #     with open('issues.txt', 'a' if osp.exists('issues.txt') else 'w') as f:
        #         f.write("{}\n".format(img_file))
        
        try:
            img = align_image(le, img, args.size, args.conf_threshold, args.keep_largest)
        except NoLandmarksFoundException:
            print("#. Warning: No landmarks found in {}".format(img_file))
            with open('issues.txt', 'a' if osp.exists('issues.txt') else 'w') as f:
                f.write("{}\n".format(img_file))
            continue

        # Save output image
        cv2.imwrite(osp.join(output_dir, osp.split(img_file)[-1]), cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
