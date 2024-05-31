import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
from traceback import print_exc
from typing import Counter
import numpy as np
from utils.media_utils import get_media_duration, get_video_fps, get_video_frame_count, trim_audio, trim_video


class FileWritter:
    def __init__(self, filepath, new_file=False):
        self.filepath = filepath
        self.mode = 'w' if new_file else 'a'
        self.file = None
        # self.mutex = threading.RLock()
        # if new_file and os.path.exists(filepath):
        #     os.remove(filepath)

    def append(self, msg):
        if self.file is None:
            self.file = open(self.filepath, self.mode)
        self.file.write(msg + '\n')
        self.file.flush()
    
    def __del__(self):
        if self.file is not None:
            self.file.close()


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def save_json(obj, json_path):
    with open(json_path, 'w') as f:
        json.dump(obj, f)


def get_best_sync(scores, cur_idx, filter_kernel: np.ndarray=None):
    scores = np.array(scores)
    if filter_kernel is not None:
        scores = np.convolve(scores, filter_kernel, mode='valid')
        cur_idx = cur_idx - filter_kernel.shape[0] // 2
    best_sync_idx = int(scores.argmax())
    best_offset = best_sync_idx - cur_idx
    score = float(scores[best_sync_idx])
    # left = avsync_config['window_size'] // 2
    return best_offset, score


def float_equal(value, other, eps=1e-9):
    return abs(value - other) < eps


def correct_data(ori_video_path, video_path, audio_path, rois_path, lms_path, offset_frame, 
                 out_ori_video_path, out_video_path, out_audio_path, out_rois_path, out_lms_path):
    def offset_frame2second(fps, offset_frame):
        return offset_frame / fps
    
    fps = get_video_fps(video_path)
    offset_second = abs(offset_frame2second(fps, offset_frame))
    video_duration = get_media_duration(video_path)
    audio_duration = get_media_duration(audio_path)
    if offset_frame == 0 and video_duration == audio_duration:
        shutil.copyfile(ori_video_path, out_ori_video_path)
        shutil.copyfile(video_path, out_video_path)
        shutil.copyfile(audio_path, out_audio_path)
        shutil.copyfile(rois_path, out_rois_path)
        shutil.copyfile(lms_path, out_lms_path)
        return

    if offset_frame >= 0:    # 音频应该后移
        # |<-  ori video  ->|
        #    |<-  ori audio  ->|
        #    |<- reserved ->|
        video_trim_st = offset_second
        # video_trim_ed = None
        video_duration -= offset_second
        duration = min(video_duration, audio_duration)
        audio_trim_st = None
        # audio_trim_ed = offset_second
    else:                   # 音频应该前移
        #    |<-  ori video  ->|
        # |<-  ori audio  ->|
        #    |<- reserved ->|
        video_trim_st = None
        audio_trim_st = offset_second
        audio_duration -= offset_second
        duration = min(video_duration, audio_duration)
    frame_duration = 1. / fps
    duration = duration // frame_duration * frame_duration

    trim_video(ori_video_path, out_ori_video_path, trim_start=video_trim_st, duration=duration)
    trim_video(video_path, out_video_path, trim_start=video_trim_st, duration=duration)
    trim_audio(audio_path, out_audio_path, trim_start=audio_trim_st, duration=duration)
    
    frame_count = get_video_frame_count(out_ori_video_path)
    assert frame_count == get_video_frame_count(out_video_path)
    assert float_equal(duration, get_media_duration(out_video_path)) and float_equal(duration, get_media_duration(out_audio_path)), \
            f'Expected duration: {duration}, out video: {get_media_duration(out_video_path)}, out audio: {get_media_duration(out_audio_path)}' + \
            f'\nout video path: {out_video_path}, out audio path: {out_audio_path}'
    # sync rois and lms
    start_frame = max(offset_frame, 0)
    rois = load_json(rois_path)
    rois = rois[start_frame:start_frame + frame_count]
    assert len(rois) == frame_count
    lms = load_json(lms_path)
    lms = lms[start_frame:start_frame + frame_count]
    assert len(lms) == frame_count
    save_json(rois, out_rois_path)
    save_json(lms, out_lms_path)


def process_worker(id, vid, data_path, output_dataset_root, score_thres, min_num_sampels, filter_kernel=None):
    ori_video_path = os.path.join(data_path, 'ori_video.mp4')
    video_path = os.path.join(data_path, 'video.mp4')
    audio_path = os.path.join(data_path, 'audio.wav')
    rois_path = os.path.join(data_path, 'rois.json')
    lms_path = os.path.join(data_path, 'landmarks.json')
    avsync_result_path = os.path.join(data_path, 'avsync.json')
    
    output_data_path = os.path.join(output_dataset_root, id, vid)
    output_ori_video_path = os.path.join(output_data_path, 'ori_video.mp4')
    output_video_path = os.path.join(output_data_path, 'video.mp4')
    output_audio_path = os.path.join(output_data_path, 'audio.wav')
    output_rois_path = os.path.join(output_data_path, 'rois.json')
    output_lms_path = os.path.join(output_data_path, 'landmarks.json')
    
    avsync_result = load_json(avsync_result_path)
    if avsync_result['num_samples'] < min_num_sampels:
        return id, vid, 'no_enough_samples', 0, 0., avsync_result['num_samples']
    best_offset, best_score = get_best_sync(avsync_result['scores'], avsync_result['current_idx'], filter_kernel)
    if best_score < score_thres:
        return id, vid, 'low_score', best_offset, best_score, avsync_result['num_samples']

    if os.path.exists(output_data_path):
        shutil.rmtree(output_data_path)
    os.makedirs(output_data_path, exist_ok=True)
    correct_data(ori_video_path, video_path, audio_path, rois_path, lms_path, best_offset, 
                 output_ori_video_path, output_video_path, output_audio_path, output_rois_path, output_lms_path)
    return id, vid, 'success', best_offset, best_score, avsync_result['num_samples']


def gen_data_item(avsync_result_list, dataset_root, resume: str=None):
    if resume:
        with open(resume) as f:
            lines = f.readlines()
        completed_set = set(map(lambda line: line.strip().split()[0], lines))
    else:
        completed_set = set()
    
    with open(avsync_result_list) as f:
        lines = f.readlines()
    # lines = [
    #     's00027/s00027_001_00006 success',
    #     's00515/s00515_001_00006 success',
    #     's00700/s00700_001_00020 success',
    #     's01405/s01405_001_00010 success',
    #     's01516/s01516_001_00004 success',
    #     's02280/s02280_001_00012 success',
    #     's02506/s02506_001_00002 success',
    #     's02528/s02528_001_00001 success'
    # ]
    
    for line in lines:
        line = line.strip()
        data_path, result_desp = line.split()
        assert result_desp in ['success', 'short_failure', 'dirty_failure']
        if result_desp == 'success':
            if data_path in completed_set:
                continue
            id, vid = data_path.split('/')
            yield id, vid, os.path.join(dataset_root, data_path)


def main():
    parser = argparse.ArgumentParser(description='Code to sync audio-video speech')
    parser.add_argument('--avsync_result_list', help='Avsync result list file path', type=str, required=True)
    parser.add_argument('--dataset_root', help='Dataset root', type=str, required=True)
    parser.add_argument('--output_dataset_root', help='Output dataset root', type=str, required=True)
    parser.add_argument('--output_result_list', help='Output data list path', type=str, required=True)
    parser.add_argument('--max_workers', help='max_workers for thread pool executor', type=int, default=4)
    parser.add_argument('--score_thres', help='Score threshold', type=float, default=0)
    parser.add_argument('--min_num_samples', help='Mininum samples num for data to reserve', type=int, default=8)
    parser.add_argument('--filter', help='Filter or not', type=bool, default=True)
    args = parser.parse_args()
    avsync_result_list = args.avsync_result_list
    dataset_root = args.dataset_root
    output_dataset_root = args.output_dataset_root
    output_result_list = args.output_result_list
    max_workers = args.max_workers
    score_thres = args.score_thres
    min_num_sampels = args.min_num_samples
    filter_ = args.filter
    print('Output dataset root:', output_dataset_root)
    print('Output list path:', output_result_list)
    
    filter_kernel = np.array([1, 3, 1], dtype=float) if filter_ else None
    if filter_kernel is not None:
        filter_kernel = filter_kernel / filter_kernel.sum()
    
    data_items = list(gen_data_item(avsync_result_list, dataset_root, resume=output_result_list))
    print(f'Found {len(data_items)} data to process.')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        result_list_writter = FileWritter(output_result_list)
        for id, vid, data_path in data_items:
            future = executor.submit(process_worker, 
                                     id, vid, data_path, output_dataset_root, 
                                     score_thres, min_num_sampels, filter_kernel=filter_kernel)
            futures.append(future)
        result_counter = Counter()
        for i, future in enumerate(as_completed(futures)):
            try:
                id ,vid, result_desp, best_offset, best_score, num_samples = future.result()
            except Exception:
                print_exc()
                executor.shutdown(wait=False)
                exit(1)
            try:
                assert result_desp in ['no_enough_samples', 'low_score', 'success']
                result_counter[result_desp] += 1
                result_list_writter.append(f'{id}/{vid} {result_desp} {best_offset} {best_score} {num_samples}')
                if (i + 1) % 100 == 0:
                    print(f'Process {i + 1} / {len(futures)} completed.' + 
                        f' | {" / ".join([f"{item[0]}: {item[1]}" for item in result_counter.items()])}')
            except Exception:
                print_exc()
                exit(1)
    print('Finished.')
    print(f'{" / ".join([f"{item[0]}: {item[1]}" for item in result_counter.items()])}')


if __name__ == '__main__':
    main()
