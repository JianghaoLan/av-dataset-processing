"""
convert processed cncvs dataset to wav2lip format.
"""
import json
import os
import argparse
import subprocess
import shutil
import concurrent.futures
from traceback import print_exc
from typing import List
import cv2
import numpy as np

from utils.img_utils import pad_crop
from utils.lms68_utils import get_left_eye, get_mouth, get_right_eye, get_square_roi
from utils.roi_utils import get_offset


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gen_src_data(root, data_list_path):
    with open(data_list_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        data_path = line.split()[0]
        id, vid = data_path.split('/')
        yield id, vid, os.path.join(root, data_path)
        

_get_comp_dict = {
    'mouth': get_mouth,
    'left_eye': get_left_eye,
    'right_eye': get_right_eye
}
def get_component_lms(lms, comp_type: str):
    return _get_comp_dict[comp_type](lms)


def crop_component(src_video_path: str, src_lms_path: str, component: str, dst_dir: str, 
                   lm_img_size: int=None, out_size: int=None, enlarge_ratio: float=1.0, ext: str='.jpg', offset_y: float=0.):
    # # test
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以使用其他编码，如 'MJPG', 'X264' 等
    # out = cv2.VideoWriter('output/crop_component_test.mp4', fourcc, 25.0, (64, 64))
    
    lms = load_json(src_lms_path)
    lms_it = iter(lms)
    
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        assert lm_img_size is None or frame.shape[:2] == (lm_img_size, lm_img_size)
        frame_lms = next(lms_it)
        frame_lms = np.asarray(frame_lms, dtype=np.float32)
        comp_lms = get_component_lms(frame_lms, component)
        comp_roi = get_square_roi(comp_lms, enlarge_ratio=enlarge_ratio)
        comp_roi = get_offset(comp_roi, y_ratio=offset_y)
        
        comp_img = pad_crop(frame, comp_roi)
        if out_size is not None and comp_img.shape[:2] != (out_size, out_size):
            comp_img = cv2.resize(comp_img, (out_size, out_size))

        # out.write(cv2.resize(comp_img, (64, 64)))
        output_path = os.path.join(dst_dir, f"{frame_count}{ext}")
        cv2.imwrite(output_path, comp_img)
        frame_count += 1
    # 释放视频对象
    cap.release()


def crop_component_to_video(src_video_path: str, src_lms_path: str, component: str, dst_dir: str,
                            lm_img_size: int=None, out_size: int=None, enlarge_ratio: float=1.0, ext: str='.mp4', offset_y: float=0.):
    assert out_size is not None
    lms = load_json(src_lms_path)
    lms_it = iter(lms)
    
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    dst_path = os.path.join(dst_dir, f'video{ext}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    out = cv2.VideoWriter(dst_path, fourcc, fps, (out_size, out_size))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        assert lm_img_size is None or frame.shape[:2] == (lm_img_size, lm_img_size)
        frame_lms = next(lms_it)
        frame_lms = np.asarray(frame_lms, dtype=np.float32)
        comp_lms = get_component_lms(frame_lms, component)
        comp_roi = get_square_roi(comp_lms, enlarge_ratio=enlarge_ratio)
        comp_roi = get_offset(comp_roi, y_ratio=offset_y)
        
        comp_img = pad_crop(frame, comp_roi)
        if out_size is not None and comp_img.shape[:2] != (out_size, out_size):
            comp_img = cv2.resize(comp_img, (out_size, out_size))

        out.write(comp_img)
    # 释放视频对象
    cap.release()
    out.release()
    

def get_source_paths(src_type, data_path):
    assert src_type in ['ffhq', 'ori']
    if src_type == 'ffhq':
        video_path = os.path.join(data_path, 'video.mp4')
        audio_path = os.path.join(data_path, 'audio.wav')
        landmarks_path = os.path.join(data_path, 'aligned_lms.json')
    elif src_type == 'ori':
        video_path = os.path.join(data_path, 'ori_video.mp4')
        audio_path = os.path.join(data_path, 'audio.wav')
        landmarks_path = os.path.join(data_path, 'landmarks.json')
    return video_path, audio_path, landmarks_path


def worker(data_path, to_path, component, out_size=None, enlarge_ratio=1.0, out_type='images', src_type='ffhq', cp_audio=True, offset_y=0.0):
    assert out_type in ['images', 'video']
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    os.makedirs(to_path)
    
    # video_path = os.path.join(data_path, 'video.mp4')
    # audio_path = os.path.join(data_path, 'audio.wav')
    # landmarks_path = os.path.join(data_path, 'aligned_lms.json')
    video_path, audio_path, landmarks_path = get_source_paths(src_type, data_path)
    # rois_path = os.path.join(data_path, 'rois.json')
    if out_type == 'images':
        crop_component(video_path, landmarks_path, component, to_path, None, out_size=out_size, enlarge_ratio=enlarge_ratio, offset_y=offset_y)
    elif out_type == 'video':
        crop_component_to_video(video_path, landmarks_path, component, to_path, None, out_size=out_size, enlarge_ratio=enlarge_ratio, offset_y=offset_y)
    if cp_audio:
        shutil.copyfile(audio_path, os.path.join(to_path, 'audio.wav'))


def process_dataset(dataset_root, output_root, data_list_path, src_type, component, max_workers=8, resize=None, 
                    enlarge_ratio=1.0, out_type: str='images', cp_audio: bool=True, offset_y=0.0):
    assert out_type in ['images', 'video']

    src_datas = list(gen_src_data(dataset_root, data_list_path))
    id_num = len(set(map(lambda x: x[0], src_datas)))
    
    print('Total video num:', len(src_datas))
    print('Total id num:', id_num)
    print('From dir:', dataset_root)
    print('Src type:', src_type)
    print('Offset y:', offset_y)
    print('To dir:', output_root)
    print('Out type:', out_type)
    print('Resize to:', resize)
    print('Copy audio:', cp_audio)
    print('Processing...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for id, vid, data_path in src_datas:
            dst_dir = os.path.join(output_root, id, vid)

            f = executor.submit(worker, data_path, dst_dir, component, resize, enlarge_ratio, out_type, src_type, cp_audio, offset_y=offset_y)
            # # convert_one(data_path, dst_dir, cp_rois_and_lms)
            futures.append(f)
        
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                f.result()
            except BaseException:
                print_exc()
                executor.shutdown(wait=False)
                exit(0)
            if i % 100 == 0:
                print(f'{i} / {len(src_datas)} completed.')
    print('Finished.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='root of processed dataset.')
    parser.add_argument('--output_root', type=str, required=True, help='root of output wav2lip format dataset.')
    parser.add_argument('--data_list_path', type=str, required=True, help='root of result filelist generated by download_and_process.py')
    parser.add_argument('--src_type', type=str, default='ffhq', help='ffhq or ori .')
    parser.add_argument('--component', type=str, required=True, help='left_eye, right_eye or mouth')
    parser.add_argument('--max_workers', type=int, help='max num of workers of process pool')
    parser.add_argument('--resize', type=int, default=None, help='resize each image to specific size')
    parser.add_argument('--enlarge_ratio', type=float, default=1.2, help='enlarge ratio.')
    parser.add_argument('--out_type', type=str, default='images', help='Out type: images or video .')
    parser.add_argument('--no_audio', action='store_false', dest='cp_audio', help='Donot copy audio to output dir.')
    parser.add_argument('--offset_y', type=float, default=0.0)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_root = args.output_root
    data_list_path = args.data_list_path
    src_type = args.src_type
    component = args.component
    max_workers = args.max_workers
    resize = args.resize
    enlarge_ratio = args.enlarge_ratio
    out_type = args.out_type
    cp_audio = args.cp_audio
    offset_y = args.offset_y

    process_dataset(dataset_root, output_root, data_list_path, src_type, component, max_workers=max_workers, resize=resize, 
                    enlarge_ratio=enlarge_ratio, out_type=out_type, cp_audio=cp_audio, offset_y=offset_y)
    
    # crop_component('/data2/CN-CVS/synced/s00005/s00005_001_00006/video.mp4', 
    #                '/data2/CN-CVS/synced/s00005/s00005_001_00006/aligned_lms.json', 
    #                'mouth',
    #                None,
    #                512,
    #                None)


if __name__ == '__main__':
    main()
