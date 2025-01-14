"""
convert processed cncvs dataset to wav2lip format.
"""
import os
import argparse
import subprocess
import shutil
import concurrent.futures
from traceback import print_exc
import cv2


def gen_src_data(root, data_list_path):
    with open(data_list_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        data_path = line.split()[0]
        id, vid = data_path.split('/')
        yield id, vid, os.path.join(root, data_path)


def convert_video_to_images(src_video_path, dst_dir, resize=None, format='.jpg'):
    # cmd = ['ffmpeg', '-i', src_video_path, os.path.join(dst_dir, r'%d.png')]   # 使用png无损图片格式
    # try:
    #     subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError as e:
    #     print('[Error] An error happened when convert video. FFmpeg output:')
    #     print(e.stdout)
    #     raise e

    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize and frame.shape[:2] != (resize, resize):
            try:
                frame = cv2.resize(frame, (resize, resize))
            except Exception:
                return None

        output_path = os.path.join(dst_dir, f"{frame_count}{format}")
        cv2.imwrite(output_path, frame)
        frame_count += 1
    # 释放视频对象
    cap.release()


def convert_one(data_path, to_path, cp_rois_and_lms=False, resize=None, format='.jpg'):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    os.makedirs(to_path)
    
    video_path = os.path.join(data_path, 'video.mp4')
    audio_path = os.path.join(data_path, 'audio.wav')
    landmarks_path = os.path.join(data_path, 'landmarks.json')
    rois_path = os.path.join(data_path, 'rois.json')
    
    convert_video_to_images(video_path, to_path, resize, format)
    shutil.copyfile(audio_path, os.path.join(to_path, 'audio.wav'))
    if cp_rois_and_lms:
        shutil.copyfile(landmarks_path, os.path.join(to_path, 'landmarks.json'))
        shutil.copyfile(rois_path, os.path.join(to_path, 'rois.json'))


def convert_dataset(dataset_root, output_root, data_list_path, cp_rois_and_lms=False, max_workers=8, resize=None, format='.jpg'):
    src_datas = list(gen_src_data(dataset_root, data_list_path))
    id_num = len(set(map(lambda x: x[0], src_datas)))
    
    print('Total video num:', len(src_datas))
    print('Total id num:', id_num)
    print('From dir:', dataset_root)
    print('To dir:', output_root)
    print('Resize to:', resize)
    print('Processing...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for id, vid, data_path in src_datas:
            dst_dir = os.path.join(output_root, id, vid)

            f = executor.submit(convert_one, data_path, dst_dir, cp_rois_and_lms, resize, format)
            # convert_one(data_path, dst_dir, cp_rois_and_lms)
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
    parser.add_argument('--copy_rois_and_lms', action='store_true', help='copy rois and landmarks to the target dataset.')
    parser.add_argument('--max_workers', type=int, help='max num of workers of process pool')
    parser.add_argument('--resize', type=int, default=None, help='resize each image to specific size')
    parser.add_argument('--format', type=str, default='.jpg', help='Output image format, like \'.jpg\', \'.png\'')
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    output_root = args.output_root
    data_list_path = args.data_list_path
    cp_rois_and_lms = args.copy_rois_and_lms
    max_workers = args.max_workers
    resize = args.resize
    format = args.format
    
    convert_dataset(dataset_root, output_root, data_list_path, cp_rois_and_lms, max_workers=max_workers, resize=resize, format=format)


if __name__ == '__main__':
    main()
