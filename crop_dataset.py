"""
Crop processed cncvs dataset by rois.
"""
import json
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import argparse
import shutil
import traceback
import cv2
import numpy as np

from utils.lms68_utils import get_center, get_left_eye, get_right_eye


class FileWritter:
    def __init__(self, filepath, new_file=False):
        self.filepath = filepath
        self.mutex = threading.RLock()
        if new_file and os.path.exists(filepath):
            os.remove(filepath)

    def append(self, msg):
        if self.filepath is None:
            return
        with self.mutex:
            with open(self.filepath, 'a') as f:
                f.write(msg + '\n')


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def gen_src_data(root, data_list_path):
    with open(data_list_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        id, vid = line.split('/')
        yield id, vid, os.path.join(root, line)


def crop_image(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    pad_l = max(-x1, 0)
    pad_r = max(x2 - w, 0)
    pad_t = max(-y1, 0)
    pad_b = max(y2 - h, 0)
    image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant')
    x1 = x1 + pad_l
    x2 = x2 + pad_l
    y1 = y1 + pad_t
    y2 = y2 + pad_t

    return image[y1:y2, x1:x2]


def crop_mouth_no_jitter(image, x1, y1, x2, y2):
    width = x2 - x1
    new_x1 = x1
    new_x2 = x2
    new_y1 = y1 + width // 2
    new_y2 = new_y1 + width
    cropped = crop_image(image, new_x1, new_y1, new_x2, new_y2)
    assert cropped.shape[0] == cropped.shape[1]
    return cropped


def crop_face_no_jitter(image, x1, y1, x2, y2, lm):
    width = x2 - x1
    left_eye = get_left_eye(lm)
    right_eye = get_right_eye(lm)
    eye_center_y = get_center(np.concatenate((left_eye, right_eye), axis=0))[1]
    new_x1 = x1
    new_x2 = x2
    new_y1 = int(eye_center_y)
    new_y2 = new_y1 + width
    cropped = crop_image(image, new_x1, new_y1, new_x2, new_y2)
    assert cropped.shape[0] == cropped.shape[1]
    return cropped


def _crop(crop_type, frame, x1, y1, x2, y2, lm):
    if crop_type == 'face':
        return crop_image(frame, x1, y1, x2, y2)
    elif crop_type == 'mouth_no_jitter':
        return crop_mouth_no_jitter(frame, x1, y1, x2, y2)
    elif crop_type == 'face_no_jitter':
        lm = np.asarray(lm, dtype=np.float32)
        return crop_face_no_jitter(frame, x1, y1, x2, y2, lm)
    else:
        raise NotImplementedError()


def crop_video(src_video_path, rois_path, lms_path, crop_type, resize):
    """
    resize: (w, h)
    """
    try:
        rois = load_json(rois_path)
        lms = load_json(lms_path)
        video = cv2.VideoCapture(src_video_path)
        for roi, lm in zip(rois, lms):
            success, frame = video.read()
            if not success:
                raise Exception(f'Read video frame error: {src_video_path}')

            x1, y1, x2, y2 = map(int, roi[:4])
            cropped = _crop(crop_type, frame, x1, y1, x2, y2, lm)
            if resize:
                cropped = cv2.resize(cropped, resize)
            yield cropped
    except cv2.error as e:
        print('[Error] An error happened when cropping images.')
        print('frame shape:', frame.shape)
        print('cropped frame shape:', cropped.shape)
        print('roi:', x1, y1, x2, y2)
        print('src_video_path:', src_video_path)
        traceback.print_exc()
        raise e
    except Exception as e:
        print('[Error] An error happened when cropping video.')
        traceback.print_exc()
        raise e
    finally:
        video.release()


def crop_video_to_images(src_video_path, rois_path, lms_path, crop_type, resize, dst_dir):
    # cmd = ['ffmpeg', '-i', src_video_path, os.path.join(dst_dir, r'%d.png')]   # 使用png无损图片格式
    for i, cropped in enumerate(crop_video(src_video_path, rois_path, lms_path, crop_type, resize)):
        cv2.imwrite(os.path.join(dst_dir, f'{i}.jpg'), cropped)


def crop_video_to_video(src_video_path, rois_path, lms_path, crop_type, resize, dst_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        dst_path,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(resize[0]), int(resize[1])),
        isColor=True,
    )
    try:
        for frame in crop_video(src_video_path, rois_path, lms_path, crop_type, resize):
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
    finally:
        writer.release()


def convert_one(data_path, to_path, crop_type, resize, output_type, cp_audio=True):
    assert output_type in ['images', 'video']
    if isinstance(resize, int):
        resize = (resize, resize)
    
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    os.makedirs(to_path, exist_ok=True)

    video_path = os.path.join(data_path, 'ori_video.mp4')
    audio_path = os.path.join(data_path, 'audio.wav')
    # landmarks_path = os.path.join(data_path, 'landmarks.json')
    rois_path = os.path.join(data_path, 'rois.json')
    lms_path = os.path.join(data_path, 'landmarks.json')
    
    if output_type == 'images':
        crop_video_to_images(video_path, rois_path, lms_path, crop_type, resize, to_path)
    elif output_type == 'video':
        crop_video_to_video(video_path, rois_path, lms_path, crop_type, resize, os.path.join(to_path, 'video.mp4'))

    if cp_audio:
        shutil.copyfile(audio_path, os.path.join(to_path, 'audio.wav'))


def crop_dataset(dataset_root, data_list_path, crop_type, resize, output_root, output_data_list_path, max_workers=4, output_type='images', cp_audio=True):
    assert output_type in ['images', 'video']
    assert crop_type in ['face', 'mouth_no_jitter', 'face_no_jitter']
    
    src_datas = list(gen_src_data(dataset_root, data_list_path))
    id_num = len(set(map(lambda x: x[0], src_datas)))
    
    print('Total video num:', len(src_datas))
    print('From dir:', dataset_root)
    print('Crop type:', crop_type)
    print('Resize:', resize)
    print('Output type:', output_type)
    print('To dir:', output_root)
    print('Total id num:', id_num)
    print('Copy audio:', cp_audio)
    print('Processing...')
    
    out_filelist = None
    if output_data_list_path is not None:
        out_filelist = FileWritter(output_data_list_path)
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        done_count = 0
        for id, vid, data_path in src_datas:
            dst_dir = os.path.join(output_root, id, vid)

            future = executor.submit(convert_one, data_path, dst_dir, crop_type, resize, output_type, cp_audio)
            futures.append(future)
            
            if out_filelist is not None:
                def done_callback(f, _id=id, _vid=vid):
                    if f.exception is None:
                        out_filelist.append(f'{_id}/{_vid}')
                future.add_done_callback(done_callback)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print('Exception occured in subprocess:', str(e))
                traceback.print_exc()
            done_count += 1
            if done_count % 100 == 0:
                print(f'{done_count} / {len(src_datas)} completed.')

    print('Success.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='root of processed dataset.')
    parser.add_argument('--data_list_path', type=str, required=True, help='path of data list file.')
    parser.add_argument('--crop_type', type=str, default='face', help='face / mouth_no_jitter / face_no_jitter.')
    parser.add_argument('--resize', type=int, default=None, help='resize.')
    parser.add_argument('--output_root', type=str, required=True, help='root of output dataset.')
    parser.add_argument('--output_type', type=str, default='images', help='images / video .')
    parser.add_argument('--output_data_list_path', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers in process pool')
    parser.add_argument('--no_audio', action='store_false', dest='cp_audio', help='Donot copy audio to output dir.')
    args = parser.parse_args()

    dataset_root = args.dataset_root
    data_list_path = args.data_list_path
    crop_type = args.crop_type
    resize = args.resize
    output_root = args.output_root
    output_type = args.output_type
    output_data_list_path = args.output_data_list_path
    cp_audio = args.cp_audio
    max_workers = args.max_workers

    crop_dataset(dataset_root, data_list_path, crop_type, resize, output_root, output_data_list_path, max_workers=max_workers, output_type=output_type, cp_audio=cp_audio)


if __name__ == '__main__':
    main()
