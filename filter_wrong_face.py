import argparse
import json
import math
import os
import threading

import cv2


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

                
def get_box(points):
    xs, ys = list(zip(*points))
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min, y_min, x_max, y_max


def get_center(points):
    x_min, y_min, x_max, y_max = get_box(points)
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calc_adjacent_points_mean_distance(points, end_to_end=False):
    dis = 0
    for i in range(0, len(points) - 1):
        p = points[i]
        next_p = points[i + 1]
        dis += get_distance(p, next_p)
    if end_to_end:
        dis += get_distance(points[0], points[-1])
    count = len(points) - 1
    if end_to_end:
        count += 1
    return dis / count


def is_components_position_correct(lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth):
    lm_jaw_l = get_center(lm_jaw[:7])
    lm_jaw_r = get_center(lm_jaw[-7:])
    p_eye_brow_l = get_center(lm_eye_brow_l)
    p_eye_brow_r = get_center(lm_eye_brow_r)
    p_nose = get_center(lm_nose)
    p_eye_l = get_center(lm_eye_l)
    p_eye_r = get_center(lm_eye_r)
    p_mouth = get_center(lm_mouth)
    
    try:
        assert p_eye_brow_l[0] < p_eye_brow_r[0]     # the left eye brow is to the left of the right eye brow
        assert p_eye_brow_l[1] < p_eye_l[1] and p_eye_brow_r[1] < p_eye_r[1]   # eyebrows are above the eyes
        assert p_eye_l[0] < p_eye_r[0]     # the left eye is to the left of the right eye
        assert p_eye_l[1] < p_nose[1] and p_eye_r[1] < p_nose[1]   # eyes are above the nose
        assert p_nose[1] < p_mouth[1]    # nose is above the mouth
        assert lm_jaw_l[0] < lm_jaw_r[0]   # the left jaw is to the left of the right jaw
    except AssertionError:
        return False
    return True


def is_points_adjacency_correct(roi, lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth):
    lm_nose_bridge = lm_nose[:4]
    lm_nose_bottom = lm_nose[4:]
    
    x1, y1, x2, y2, _ = roi
    refer = (x2 - x1) + (y2 - y1)
    try:
        assert calc_adjacent_points_mean_distance(lm_jaw, end_to_end=False) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_eye_brow_l, end_to_end=False) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_eye_brow_r, end_to_end=False) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_nose_bridge, end_to_end=False) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_nose_bottom, end_to_end=False) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_eye_l, end_to_end=True) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_eye_r, end_to_end=True) < refer * 0.1
        assert calc_adjacent_points_mean_distance(lm_mouth, end_to_end=True) < refer * 0.1
    except AssertionError:
        return False
    return True


def calc_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    v1_norm = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v2_norm = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_theta = dot_product / (v1_norm * v2_norm)
    theta = math.acos(cos_theta)
    return theta


def vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def is_face_direction_correct(lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth):
    p_bottom = lm_jaw[8]
    p_eye_l = get_center(lm_eye_l)
    p_eye_r = get_center(lm_eye_r)
    p_eye_center = ((p_eye_l[0] + p_eye_r[0]) / 2, (p_eye_l[1] + p_eye_r[1]) / 2)
    try:
        assert calc_angle(vector(p_eye_center, p_bottom), (0, 1)) < math.pi / 4
        assert calc_angle(vector(p_eye_l, p_eye_r), (1, 0)) < math.pi / 4
    except AssertionError:
        return False
    return True


def is_wrong_face(roi, lm):
    assert len(lm) == 68
    assert len(roi) == 5
    
    lm_jaw = lm[:17]
    # lm_eye_brow = lm[17:27]
    lm_eye_brow_l = lm[17:22]
    lm_eye_brow_r = lm[22:27]
    lm_nose = lm[27:36]
    # lm_eyes = lm[36:48]
    lm_eye_l = lm[36:42]
    lm_eye_r = lm[42:48]
    lm_mouth = lm[48:]
    
    res = is_points_adjacency_correct(roi, lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth)
    if not res:
        return True
    
    res = is_components_position_correct(lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth)
    if not res:
        return True
    
    res = is_face_direction_correct(lm_jaw, lm_eye_brow_l, lm_eye_brow_r, lm_nose, lm_eye_l, lm_eye_r, lm_mouth)
    if not res:
        return True
    
    return False


def get_video_info(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    # 获取视频的长宽
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 关闭视频文件
    video.release()
    return width, height, total_frames


def is_in_bounds(roi, lm, width, height):
    x_min, y_min, x_max, y_max = get_box(lm)
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        return False
    x1, y1, x2, y2, _ = roi
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    return True
    

def check_video(video_path, rois_path, lms_path):
    """
    return true if video is bad.
    """
    out_bounds_ratio = 0.7
    wrong_face_ratio = 0.1
    
    width, height, total_frames = get_video_info(video_path)
    rois = load_json(rois_path)
    lms = load_json(lms_path)
    assert len(rois) == len(lms)
    
    out_bounds_thres = int(total_frames * out_bounds_ratio)
    wrong_face_thres = int(total_frames * wrong_face_ratio)
    
    count_out_bounds = 0
    count_wrong_face = 0
    for roi, lm in zip(rois, lms):
        if not is_in_bounds(roi, lm, width, height):
            count_out_bounds += 1
            if count_out_bounds >= out_bounds_thres:
                return False, 'out_bounds'
        if is_wrong_face(roi, lm):
            count_wrong_face += 1
            if count_wrong_face >= wrong_face_thres:
                return False, 'wrong_face'
    return True, None


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--filelist', required=True, type=str)
    parser.add_argument('--dataset_root', type=str, required=True, help='root of processed dataset.')
    parser.add_argument('--result_list_path', type=str, required=True, help='root of result filelist generated by download_and_process.py')
    parser.add_argument('--output_path', type=str, required=True, help='root of filtered wrong data.')
    # parser.add_argument('--dst_root', required=True, type=str)
    args = parser.parse_args()
    
    # filelist = args.filelist
    dataset_root = args.dataset_root
    result_list_path = args.result_list_path
    output_path = args.output_path
    
    # with open(filelist) as f:
    #     lines = f.readlines()
    # src_list = map(lambda x: x.strip(), lines)

    src_list = gen_src_data(dataset_root, result_list_path)
    # src_list = random.choices(src_list, k=200)
    # dst_root = args.dst_root
    # if not os.path.exists(dst_root):
    #     os.mkdir(dst_root)
    
    output_writter = FileWritter(output_path)
    print('output:', output_path)
    for src in src_list:
        src_path = src[2]
        video_path = os.path.join(src_path, 'ori_video.mp4')
        lms_path = os.path.join(src_path, 'landmarks.json')
        rois_path = os.path.join(src_path, 'rois.json')
        # dst = os.path.join(dst_root, os.path.basename(os.path.dirname(src)) + '.mp4')
        # show_video_lms(src, lms_path, rois_path, dst)
        res, reason = check_video(video_path, rois_path, lms_path)
        if not res:
            output_writter.append(f'{src_path} {reason}')
            # print(f'{src}: {reason}')
        # if res:
        #     print(src)

    
if __name__ == '__main__':
    main()
