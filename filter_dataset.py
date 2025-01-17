import argparse
from collections import Counter
import json
import os

import numpy as np


def load_rois(video_path):
    with open(os.path.join(video_path, 'rois.json')) as f:
        return json.load(f)
    
    
class WidthDataFilter:
    def __init__(self, min_width, num_select=5):
        self.min_width = min_width
        self.num_select = num_select
    
    def filter(self, id, vid, video_path):
        rois = load_rois(video_path)
        
        if self.num_select and self.num_select > 0:
            rois = rois[:self.num_select]
    
        rois = np.array(rois)
        assert len(rois.shape) == 2 and rois.shape[-1] == 5
        x1, x2 = rois[:, 0], rois[:, 2]
        assert x1.shape == (len(rois), ) and x2.shape == (len(rois), )
        width = np.abs(x2 - x1).mean()
        return width >= self.min_width


class BlacklistDataFilter:
    def __init__(self, blacklist_path):
        self.blacklist_path = blacklist_path
        with open(blacklist_path) as f:
            lines = f.readlines()
        self.blacklist = set(map(lambda line: line.strip().split(maxsplit=1)[0].split('/')[1], lines))
        print(f'Found {len(self.blacklist)} records in blacklist.')

    def filter(self, id, vid, video_path):
        return vid not in self.blacklist


_possible_result_desp = {
    'Success', 'Download failed', 'transcode failed', 'Unexcepted EOF', 'Clip too short', 'No landmarks found'
}
def gen_src_data(root, result_list_path):
    result_dict = {}
    with open(result_list_path) as f:
        for line in f.readlines():
            vid, result_desp = line.strip().split(' ', 1)

            assert result_desp in _possible_result_desp
            if len(vid.split('_')) == 3:
                assert vid not in result_dict
            result_dict[vid] = result_desp

    for id in os.listdir(root):
        id_path = os.path.join(root, id)
        if not os.path.isdir(id_path):
            continue

        for vid in os.listdir(id_path):
            assert vid in result_dict
            if result_dict[vid] == 'Success':
                data_path = os.path.join(id_path, vid)
                assert os.path.isdir(data_path)
                assert os.path.isfile(os.path.join(data_path, 'video.mp4'))
                assert os.path.isfile(os.path.join(data_path, 'audio.wav'))
                assert os.path.isfile(os.path.join(data_path, 'landmarks.json'))
                assert os.path.isfile(os.path.join(data_path, 'rois.json'))
                assert os.path.isfile(os.path.join(data_path, 'ori_video.mp4'))
                yield id, vid, data_path


def write_to_file(datas, file_path):
    with open(file_path, 'x') as f:
        f.writelines(map(lambda data: f'{data[0]}/{data[1]}\n', datas))   # data: (id, vid)
    print(f'Save to {file_path} .')
        

def do_filter(data_list, filters):
    def _filter(_data, _filters):
        for filter_ in _filters:
            if not filter_.filter(*_data):
                return False, filter_
        return True, None
    
    filtered = []
    filter_counter = Counter()
    for data in data_list:
        res, filter_ = _filter(data, filters)
        if res:
            filtered.append(data)
        else:
            filter_counter[filter_] += 1
    return filtered, filter_counter


def filter_dataset(root, result_list, filters, out_path, dry_run=False):
    data_list = list(gen_src_data(root, result_list))
    id_list = list(set(map(lambda data: data[0], data_list)))
    print('Dataset root:', root)
    print('Dataset total video num:', len(data_list))
    print('Dataset total id num:', len(id_list))
    
    data_list, counter = do_filter(data_list, filters)
    id_list = list(set(map(lambda data: data[0], data_list)))
    # print('Filtered Dataset root:', root)
    print('Filtered Dataset total video num:', len(data_list))
    print('Filtered Dataset total id num:', len(id_list))
    for f in counter:
        print(f'Video number filtered by [{f.__class__.__name__}]: {counter[f]}')
    data_list.sort()
    if not dry_run:
        write_to_file(data_list, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='root of processed dataset.')
    parser.add_argument('--result_list_path', type=str, required=True, help='root of result filelist generated by download_and_process.py')
    parser.add_argument('--output_path', type=str, required=True, help='root of filtered wrong data.')
    parser.add_argument('--min_width', default=None, type=float, help='Filter data with a min width.')
    parser.add_argument('--blacklist', default=None, type=str, help='Blacklist file path.')
    parser.add_argument('--dry_run', action='store_true', help='Run this script with no output.')
    args = parser.parse_args()

    root = args.dataset_root
    result_list_path = args.result_list_path
    output_path = args.output_path
    min_width: float = args.min_width
    blacklist: str = args.blacklist
    dry_run: bool = args.dry_run

    filters = []
    if min_width is not None:
        filters.append(WidthDataFilter(min_width))
    if blacklist is not None:
        filters.append(BlacklistDataFilter(blacklist))
    filter_dataset(root, result_list_path, filters, output_path, dry_run)


if __name__ == '__main__':
    main()
