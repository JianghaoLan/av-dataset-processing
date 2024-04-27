import argparse
import json
import os
import random
import numpy as np


def read_data_list(path):
    with open(path) as f:
        lines = f.readlines()
    return list(map(lambda line: line.strip().split(maxsplit=1)[0].split('/'), lines))
    

def write_split_to_file(datas, file_path):
    with open(file_path, 'x') as f:
        f.writelines(map(lambda data: f'{data[0]}/{data[1]}\n', datas))   # data: (id, vid)


def split(data_list_path, name_list, num_list, out_filelist_dir):
    name_num_list = sorted(zip(name_list, num_list), key=lambda pair: pair[1], reverse=True)
    assert not (len(name_num_list) > 1 and name_num_list[-2][1] < 0), 'At most one split number can be negative'
    
    data_list = read_data_list(data_list_path)
    id_list = list(set(map(lambda data: data[0], data_list)))
    print('Data list:', data_list_path)
    print('Dataset total video num:', len(data_list))
    print('Dataset total id num:', len(id_list))
    
    random.shuffle(id_list)

    id_splits = {}
    i = 0
    for name, num in name_num_list:
        end = i + num if num >= 0 else len(id_list)
        id_splits[name] = id_list[i : end]
        i = end
        
    id2datas_dict = {}
    for id, vid in data_list:
        id2datas_dict.setdefault(id, []).append(vid)
    
    data_splits = {}
    for name, ids in id_splits.items():
        datas = []
        for id in sorted(ids):
            datas.extend(map(lambda d: (id, d), sorted(id2datas_dict[id])))
        data_splits[name] = datas

    for name, datas in data_splits.items():
        print(f'Split[{name}]: {len(datas)} videos ({len(id_splits[name])} ids)')
        
    for name, datas in data_splits.items():
        write_split_to_file(datas, os.path.join(out_filelist_dir, name + '.txt'))
        
    return data_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--data_list', required=True, type=str, help='Root path to wav2lip-format dataset root.')
    parser.add_argument('-s', '--split_names', required=True, type=str, help='Names of each split, separated by commas. E.g. train,test')
    parser.add_argument('-n', '--split_nums', required=True, type=str, help='Id nums of each split, separated by commas. E.g. 30,-1')
    parser.add_argument('-o', '--out_filelist_dir', required=True, type=str, help='Dir of output filelist')
    args = parser.parse_args()

    data_list_path = args.data_list
    split_names: str = args.split_names
    split_nums: str = args.split_nums
    out_filelist_dir = args.out_filelist_dir
    
    def parse_split_names(_split_names):
        return list(map(lambda x: x.strip(), _split_names.split(',')))
    
    def parse_split_nums(_split_nums):
        return list(map(lambda x: int(x), _split_nums.split(',')))
    
    name_list = parse_split_names(split_names)
    num_list = parse_split_nums(split_nums)
    assert len(name_list) == len(num_list) and len(name_list) > 0
    split(data_list_path, name_list, num_list, out_filelist_dir)


if __name__ == '__main__':
    main()
