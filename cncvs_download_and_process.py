import copy
import math
import os
import json
import argparse
import logging
import time
from pathlib import Path
import multiprocessing
import subprocess
from align import MultiFacesDetectedException, NoLandmarksFoundException, align_image
import cv2
from lib.landmarks_pytorch import LandmarksEstimation


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', required=True, help='download & transcode files in this path')
# parser.add_argument('-j', '--metadata', required=True, help='given json file')
parser.add_argument('-m', '--metadata_root', required=True, help='dataset metadata root')
# parser.add_argument('-r', '--roi_dir', required=True, help='given roi json file folder')
parser.add_argument('-f', '--frame_width', default=512, type=int, help='size of output video')
parser.add_argument('--download_worker', default=1, type=int, help='num of download process')
parser.add_argument('--transcode_worker', default=2, type=int, help='num of transcode process')
parser.add_argument('--process_worker', default=2, type=int, help='num of process process')
parser.add_argument('-l', '--loglevel', default=0, type=int, choices=[0, 1, 2, 3, 4], help='logger output level')
parser.add_argument('--max_clips_per_speaker', default=-1, type=int, help='max clip num per speaker')
parser.add_argument('--cookies', default=None, type=str, help='cookie file pass to yt-dlp')
parser.add_argument('--gpus', default='0', type=str, help='gpu ids')
# parser.add_argument('--result_filelist', type=str, default='./')
parser.add_argument('--save_origin_video',
                    default=False,
                    action='store_true',
                    help='save or remove origin video file after transcoding')
parser.add_argument('--save_transcoded_video',
                    default=False,
                    action='store_true',
                    help='save or remove transcoded video file after processing')

loglevel_list = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]


start_time = time.time()


class FileWritter:
    def __init__(self, filepath, new_file=True):
        self.filepath = filepath
        self.mutex = multiprocessing.RLock()
        if new_file and os.path.exists(filepath):
            os.remove(filepath)

    def append(self, msg):
        if self.filepath is None:
            return
        with self.mutex:
            with open(self.filepath, 'a') as f:
                f.write(msg + '\n')


def write_video(output_file: str, frames: list, frame_size: tuple) -> None:
    '''
    description: write frames to mp4 file 
    param {str}   output_file target output mp4 file
    param {list}  frames      list of frame images, same size with frame_size 
    param {tuple} frame_size  (weight, height)
    return {*}
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_file,
        fourcc=fourcc,
        fps=25,
        frameSize=(int(frame_size[0]), int(frame_size[1])),
        isColor=True,
    )
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


class ReadVideoException(Exception):
    pass


def calc_area(roi):
    return (roi[2] - roi[0]) * (roi[3] - roi[1])


def calc_center_distance(roi, other):
    c1 = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
    c2 = (other[0] + other[2]) / 2, (other[1] + other[3]) / 2
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def read_video(src: str, frame_boundary: tuple, le: LandmarksEstimation, frame_size: tuple = (224, 224), conf_threshold: float = 0.97, return_roi_and_landmarks=False) -> list:
    ori_imgs = []
    imgs = []
    rois = []
    landmarkss = []
    video = cv2.VideoCapture(src)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_boundary[0])
    for idx in range(frame_boundary[0], frame_boundary[1]):
        # l, r, t, b = rois[idx - frame_boundary[0]]
        success, frame = video.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                face, roi, landmarks = align_image(le, frame, frame_size, conf_threshold, 
                                                   disable_multi_faces=False, 
                                                   keep_largest=True, 
                                                   return_roi_and_landmarks=True)

            # except MultiFacesDetectedException:
            #     raise ReadVideoException('Multi faces detected')
            except NoLandmarksFoundException:
                raise ReadVideoException('No landmarks found')
            
            # head = frame[t:b, l:r, :]
            # head = cv2.resize(head, frame_size)
            # imgs.append(head)
            
            # 判断是否连贯，不连贯的丢弃后半段（视频足够长时）或丢弃整个视频
            if len(rois) > 0:
                prev = rois[-1]
                prev_area = calc_area(prev)
                cur_area = calc_area(roi)
                distance = calc_center_distance(prev, roi)
                distance_thres = ((roi[2] - roi[0]) + (roi[3] - roi[1])) / 4
                if not (cur_area > prev_area * 0.33 and cur_area < prev_area * 3 and distance < distance_thres):
                    break

            rois.append(roi)
            landmarkss.append(landmarks)
            ori_imgs.append(frame)
            imgs.append(face)
        else:
            raise ReadVideoException('Unexcepted EOF')
    if len(imgs) < 25:
        raise ReadVideoException('Clip too short')
    if return_roi_and_landmarks:
        return imgs, ori_imgs, rois, landmarkss
    return imgs, ori_imgs


def trancode_worker(meta, src_dir, dst_dir, save_ori):
    print(f'### transcode {meta["spkid"]}_{meta["videoid"]}')
    src = os.path.join(src_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
    dst = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
    # if os.path.exists(dst):
    #     logger.debug(f'{dst} exists')
    #     return 0

    get_fps = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "
    cmd = f"{get_fps} {src}"
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if str(res.stdout[:-1], 'utf-8').split('/') == ['25', '1']:
        cmd = f"cp {src} {dst}"
        res = subprocess.call(cmd, shell=True)
    else:
        cmd = f'ffmpeg -v error -i {src} -qscale 0 -r 25 -y {dst}'
        logging.debug(cmd)
        res = subprocess.call(cmd, shell=True)
    if res != 0:
        logging.warning(f'error occur when transcoding {src}')
        if os.path.exists(dst):
            os.remove(dst)
    else:
        logging.debug(f'finish transcoding {dst}')
    if not save_ori and os.path.exists(src):
        os.remove(src)
    return res


def process_worker(meta, roi_dir, src_dir, dst_dir, frame_size, save_ori, le, result_writter):
    print(f'### process {meta["spkid"]}_{meta["videoid"]}')
    try:
        src = os.path.join(src_dir, f'{meta["spkid"]}_{meta["videoid"]}.mp4')
        os.makedirs(os.path.join(dst_dir, meta["spkid"]), exist_ok=True)
        for _, utt_info in enumerate(meta['uttlist']):
            uid = utt_info['uttid']
            # af = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.wav')
            # vf = os.path.join(dst_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.mp4')
            video_dir = os.path.join(dst_dir, meta["spkid"], f'{meta["spkid"]}_{meta["videoid"]}_{uid}')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            af = os.path.join(video_dir, 'audio.wav')
            vf = os.path.join(video_dir, 'video.mp4')
            vof = os.path.join(video_dir, 'ori_video.mp4')
            rf = os.path.join(video_dir, 'rois.json')
            lf = os.path.join(video_dir, 'landmarks.json')
            
            # extract silent video
            # rois = json.load(open(os.path.join(roi_dir, f'{meta["spkid"]}_{meta["videoid"]}_{uid}.json')))
            # frames = None
            # rois = None
            # landmarks = None
            try:
                frames, ori_frames, rois, landmarks = read_video(src, (utt_info['video_frame_ss'], utt_info['video_frame_ed']), 
                                                                 le, frame_size=frame_size, return_roi_and_landmarks=True)
            except ReadVideoException as e:
                result_writter.append(f'{meta["spkid"]}_{meta["videoid"]}_{uid} {str(e)}')
                continue
            write_video(vf, frames, frame_size=(frame_size, frame_size))
            write_video(vof, ori_frames, frame_size=(ori_frames[0].shape[1], ori_frames[0].shape[0]))
            logger.debug(f'write_video({vf}, frames, frame_size={frame_size})')
            
            # save rois and landmarks
            with open(rf, mode='w') as f:
                json.dump([each.tolist() for each in rois], f)
            with open(lf, mode='w') as f:
                json.dump([each.tolist() for each in landmarks], f)
            
            # extract audio
            ss = utt_info['ss']
            t = utt_info['t']
            cmd = f'ffmpeg -v error -y -accurate_seek -i {src} -ss {ss} -t {t} -avoid_negative_ts 1 -b:a 256k -ar 16000 -ac 1 -acodec pcm_s16le -strict -2 {af}'
            logger.debug(cmd)
            subprocess.call(cmd, shell=True)
            
            result_writter.append(f'{meta["spkid"]}_{meta["videoid"]}_{uid} Success')
        if not save_ori and os.path.exists(src):
            os.remove(src)
    except Exception as e:
        logger.error(f'Unknown error occured in process worker. Processing {meta["spkid"]}_{meta["videoid"]}_{uid}.')
        logger.exception(e)


def download_worker(meta, dst_dir, cookie_path=None):
    url = meta['videourl']
    save_name = f'{meta["spkid"]}_{meta["videoid"]}'
    print(f'### download {save_name} from {url}')
    try:
        if True or not os.path.exists(os.path.join(dst_dir, f"{save_name}.mp4")):    # always True
            print('downloading ' + os.path.join(dst_dir, f"{save_name}.mp4"))
            # cmd = f'you-get "{url}" -o "{dst_dir}" -O "{save_name}"'
            # cmd = f'youtube-dl "{url}" -o "{os.path.join(dst_dir, f"{save_name}.mp4")}"'
            cookies = ''
            if cookie_path is not None:
                cookies = f'--cookies "{cookie_path}"'
            cmd = f'yt-dlp "{url}" {cookies} -P "{dst_dir}" -o "{save_name}.mp4"'
            res = subprocess.call(cmd, shell=True)
        else:
            print(os.path.join(dst_dir, f"{save_name}.mp4") + 'exist')
        if not os.path.exists(os.path.join(dst_dir, f"{save_name}.mp4")):
            logger.error(f'{save_name} download failed (file not exists)')
            return -2
        return 0
    except Exception as e:
        logger.error(f'{save_name} download failed')
        logger.exception(e)
        return -1


def pipeline(le, result_writter, meta, rpath, dpath, tpath, fpath, frame_size, save_download, save_transcoded, cookie_path=None, download_sem=None):
    if download_sem is not None:
        with download_sem:
            res = download_worker(meta, dpath, cookie_path)
    else:
        res = download_worker(meta, dpath, cookie_path)
    if res != 0:
        result_writter.append(f'{meta["spkid"]}_{meta["videoid"]} Download failed')
        return res
    res = trancode_worker(meta, dpath, tpath, save_download)
    if res != 0:
        result_writter.append(f'{meta["spkid"]}_{meta["videoid"]} transcode failed')
        return res
    process_worker(meta, rpath,  tpath, fpath, frame_size, save_transcoded, le, result_writter)

        
def download_loop(in_queue, out_queue, result_writter):
    while True:
        cur = in_queue.get()
        if cur is None:
            in_queue.put(None)
            print('***Download process finished***')
            return
        meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video, cookie_path = cur
        res = download_worker(meta, dpath, cookie_path)
        if res != 0:
            result_writter.append(f'{meta["spkid"]}_{meta["videoid"]} Download failed')
            continue
        out_queue.put(cur)
        

def transcode_loop(in_queue, out_queue, result_writter):
      while True:
        cur = in_queue.get()
        if cur is None:
            in_queue.put(None)   # notify other transcode process to stop
            print('***Transcode process finished***')
            return
        meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video, cookie_path = cur
        res = trancode_worker(meta, dpath, tpath, save_origin_video)
        if res != 0:
            result_writter.append(f'{meta["spkid"]}_{meta["videoid"]} transcode failed')
            continue
        out_queue.put(cur)
        
def process_loop(in_queue, result_writter, count, total_num, gpu_id: int=0):
    le = LandmarksEstimation(type='2D', device=f'cuda:{gpu_id}')
    while True:
        cur = in_queue.get()
        if cur is None:
            in_queue.put(None)   # notify other process process to stop
            print('***Process process finished***')
            return
        meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video, cookie_path = cur
        process_worker(meta, rpath,  tpath, fpath, frame_size, save_transcoded_video, le, result_writter)
        count.value += 1

        running_time = time.time() - start_time
        minutes, _ = divmod(running_time, 60)
        hours, minutes = divmod(minutes, 60)
        print(f'*** {count.value} / {total_num} finished. Running time: {int(hours)} h {int(minutes)} min ***')


def get_skip_set(filelist_path):
    with open(filelist_path) as f:
        lines = f.readlines()
    return set(map(lambda x: x.split(' ', 1)[0], lines))


def get_filtered_metadata(metadata, skip_set):
    new_metadata = []
    for meta in metadata:
        new_meta = copy.copy(meta)
        
        new_meta['uttlist'] = list(filter(lambda x: f'{meta["spkid"]}_{meta["videoid"]}_{x["uttid"]}' not in skip_set, meta['uttlist']))
        if len(new_meta['uttlist']) == 0:
            continue
        new_metadata.append(new_meta)
    return new_metadata


def main(out_filelist, metadata, rpath, dpath, tpath, fpath, frame_size, 
         num_download_workers, num_transcode_workers, num_process_workers,
         save_origin_video, save_transcoded_video, cookie_path=None, gpu_ids=[0], skip_set=None):
    # le = LandmarksEstimation(type='2D')
    # result filelist writter
    result_writter = FileWritter(out_filelist, new_file=False)
    
    metadata = list(metadata)
    # skip metadata already processed
    skip_set = get_skip_set(out_filelist)
    metadata = get_filtered_metadata(metadata, skip_set)
    print(f'*** {len(skip_set)} videos have been finished by {out_filelist} and will be skipped ***')
    
    sync = False
    if sync:
        le = LandmarksEstimation(type='2D')
        for meta in metadata:
            pipeline(le, result_writter, meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video, cookie_path)
    else:   
        download_queue = multiprocessing.Queue()
        transcode_queue = multiprocessing.Queue(num_transcode_workers)
        process_queue = multiprocessing.Queue(num_process_workers)
        total_num = 0
        for meta in metadata:
            download_queue.put((meta, rpath, dpath, tpath, fpath, frame_size, save_origin_video, save_transcoded_video, cookie_path))
            total_num += 1
        download_queue.put(None)
        print(f'*** Total task num: {total_num} ***')
        
        download_processes = []
        transcode_processes = []
        process_processes = []
        count = multiprocessing.Value('i', 0)
        for i in range(num_download_workers):
            p = multiprocessing.Process(name=f'Download Process {i}', target=download_loop, args=(download_queue, transcode_queue, result_writter))
            p.start()
            download_processes.append(p)
        for i in range(num_transcode_workers):
            p = multiprocessing.Process(name=f'Transcode Process {i}', target=transcode_loop, args=(transcode_queue, process_queue, result_writter))
            p.start()
            transcode_processes.append(p)
        for i in range(num_process_workers):
            p = multiprocessing.Process(name=f'Process Process {i}', target=process_loop, args=(process_queue, result_writter, count, total_num, gpu_ids[i % len(gpu_ids)]))
            p.start()
            process_processes.append(p)
        
        # wait all task done
        for p in download_processes:
            p.join()
        print('*** All download task finished ***')
        transcode_queue.put(None)    # notify transcode processes all task done
        for p in transcode_processes:
            p.join()
        print('*** All transcode task finished ***')
        process_queue.put(None)    # notify process processes all task done
        for p in process_processes:
            p.join()
        print('*** All process task finished ***')


def gen_metadata_cncvs(ori_path, max_clips_per_speaker=-1):
    for spkid in sorted(os.listdir(os.path.join(ori_path, 'data'))):
        with open(os.path.join(ori_path, 'data', spkid, spkid + '.json')) as f:
            spk_clip_count = 0
            for each in json.load(f):
                if max_clips_per_speaker <= 0:
                    yield each
                    continue
                if spk_clip_count < max_clips_per_speaker:
                    each['uttlist'] = each['uttlist'][ : max_clips_per_speaker - spk_clip_count]
                    yield each
                    spk_clip_count += len(each['uttlist'])
                else:
                    break


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    # worker = args.worker
    num_download_workers = args.download_worker
    num_transcode_workers = args.transcode_worker
    num_process_workers = args.process_worker
    # jsonfile = args.metadata
    metadata_root= args.metadata_root
    # roi_dir = args.roi_dir
    roi_dir = None
    frame_width = args.frame_width
    frame_size = frame_width
    gpu_ids = list(map(lambda x: int(x), args.gpus.split(',')))
    try:
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        logging.basicConfig(
            level=loglevel_list[args.loglevel],
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s %(message)s',
            filename=os.path.join(data_dir, f'logs_{cur_time}.log'),
            filemode='a',
        )
        logger = logging.getLogger(__file__)
    except Exception as e:
        print(e)
        print("can't init logger")
    try:
        # # load metadata from json file
        # if not os.path.exists(jsonfile):
        #     logger.critical('metadata file not exist')
        #     exit(-1)
        # metadata = json.load(open(jsonfile, 'r'))
        metadata = gen_metadata_cncvs(metadata_root, max_clips_per_speaker=args.max_clips_per_speaker)
        # build folder
        download_data_dir = os.path.join(data_dir, 'download')
        transcoded_data_dir = os.path.join(data_dir, 'transcoded')
        final_data_dir = os.path.join(data_dir, 'final')
        Path(download_data_dir).mkdir(exist_ok=True, parents=True)
        Path(transcoded_data_dir).mkdir(exist_ok=True, parents=True)
        Path(final_data_dir).mkdir(exist_ok=True, parents=True)   
        # do download and transcode and process
        main(
            os.path.join(data_dir, 'result.txt'),
            metadata,
            roi_dir,
            download_data_dir,
            transcoded_data_dir,
            final_data_dir,
            frame_size,
            num_download_workers,
            num_transcode_workers,
            num_process_workers,
            args.save_origin_video,
            args.save_transcoded_video,
            args.cookies, 
            gpu_ids=gpu_ids,
        )

    except Exception as e:
        logger.exception(e)
