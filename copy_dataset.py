import argparse
import os
import shutil


def get_data_list(filelist):
    with open(filelist) as f:
        data_list = f.readlines()
    return [line.strip() for line in data_list]


def copy_dataset(filelist, dataset_root, out_dataset_root):
    data_list = get_data_list(filelist)
    
    print('File list:', filelist)
    print('Dataset root:', dataset_root)
    print('Total video num:', len(data_list))
    print('Out dataset root:', out_dataset_root)

    if out_dataset_root is None:
        return
    os.makedirs(out_dataset_root, exist_ok=True)
    for i, data in enumerate(data_list, 1):
        data_dir = os.path.join(dataset_root, data)
        dst_data_dir = os.path.join(out_dataset_root, data)
        os.makedirs(os.path.dirname(dst_data_dir), exist_ok=True)
        shutil.copytree(data_dir, dst_data_dir)
        if i % 100 == 0:
            print(f'{i} / {len(data_list)} finished.')
    print('Finished.')
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--out_dataset_root', default=None, type=str)
    args = parser.parse_args()

    filelist = args.filelist
    dataset_root = args.dataset_root
    out_dataset_root = args.out_dataset_root
    # out_dataset_root = None
    copy_dataset(filelist, dataset_root, out_dataset_root)


if __name__ == '__main__':
    main()



