

import argparse
import random


def get_data_list(filelist):
    with open(filelist) as f:
        data_list = f.readlines()
    return [line.strip() for line in data_list]


def save_data_list(data_list, save_path):
    with open(save_path, 'x') as f:
        f.writelines(map(lambda data: data + '\n', data_list))
    

def get_data_dict(data_list):
    data_dict = {}
    for data_path in data_list:
        id, vid = data_path.split('/')
        data_dict.setdefault(id, []).append(vid)
    return data_dict


def data_dict2list(data_dict):
    data_list = []
    for id in data_dict:
        for vid in data_dict[id]:
            data_list.append(f'{id}/{vid}')
    data_list.sort()
    return data_list


def subset(filelist, max_videos_per_id=5, out_filelist=None):
    data_list = get_data_list(filelist)
    data_dict = get_data_dict(data_list)
    
    print('File list:', filelist)
    print('Total video num:', len(data_list))
    print('Total id num:', len(data_dict))
    print('Max videos per id:', max_videos_per_id)

    for id in data_dict.keys():
        video_list = data_dict[id]
        data_dict[id] = random.sample(video_list, k=min(max_videos_per_id, len(video_list)))
    
    new_data_list = data_dict2list(data_dict)
    assert len(set(new_data_list)) == len(new_data_list)
    print('Subset total video num:', len(new_data_list))
    print('Save path:', out_filelist)
    if out_filelist is not None:
        save_data_list(new_data_list, out_filelist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--max_videos_per_id', default=5, type=int)
    parser.add_argument('--out_filelist', default=None, type=str, help='Dir of output filelist')
    args = parser.parse_args()
    
    filelist = args.filelist
    max_videos_per_id = args.max_videos_per_id
    out_filelist = args.out_filelist
    subset(filelist, max_videos_per_id, out_filelist)


if __name__ == '__main__':
    main()
