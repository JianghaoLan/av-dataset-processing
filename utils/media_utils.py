import glob
import os
import re
import subprocess
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import cv2


def get_video_fps(video_path):
    try:
        # 构建 ffmpeg 命令以获取视频信息
        ffmpeg_command = [
            'ffmpeg',           # 命令名称
            '-i',               # 输入选项
            video_path,         # 输入视频文件路径
            '-hide_banner',     # 隐藏 ffmpeg 的冗长输出
            '-f',               # 输出格式选项
            'null',             # 使用 null 输出
            '-'
        ]
        # print("###", ' '.join(ffmpeg_command))
        
        # 执行 ffmpeg 命令并捕获输出
        output = subprocess.check_output(ffmpeg_command, stderr=subprocess.STDOUT, universal_newlines=True)

        # 使用正则表达式查找帧率信息
        fps_match = re.search(r"(\d+(\.\d+)?) fps,", output)

        if fps_match:
            fps = float(fps_match.group(1))
            return fps
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    

def get_video_frame_count(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    # 获取视频的总帧数
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 释放视频对象
    video.release()
    return total_frame_count

    
def get_media_duration(input_file_path):
    try:
        # 构建 ffmpeg 命令
        ffmpeg_command = [
            'ffmpeg',              # 命令名称
            '-i',                  # 输入选项
            input_file_path,        # 输入文件路径
            '-f',                  # 输出格式选项
            'null',                # 使用 null 输出
            '-'
        ]
        # print("###", ' '.join(ffmpeg_command))
        
        # 执行 ffmpeg 命令并捕获输出
        output = subprocess.run(ffmpeg_command, stderr=subprocess.PIPE, universal_newlines=True).stderr

        # 使用正则表达式提取时长信息（以秒为单位）
        duration_match = re.search(r"Duration: (\d+:\d+:\d+\.\d+)", output)

        if duration_match:
            duration_str = duration_match.group(1)
            # 将时长字符串转换为秒
            hours, minutes, seconds = map(float, duration_str.split(':'))
            total_duration_seconds = hours * 3600 + minutes * 60 + seconds
            return total_duration_seconds
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    

def get_wav_sample_rate(filepath):
    try:
        # 使用 AudioSegment.from_file() 方法读取 WAV 文件
        audio = AudioSegment.from_file(filepath)
        # 获取采样率
        sample_rate = audio.frame_rate      
        return sample_rate
    except Exception as e:
        print(f"读取采样率时出现错误: {e}")
        return None
    

def get_video_frame_rate(video_path):
    try:
        # 使用 VideoFileClip 函数加载视频
        video_clip = VideoFileClip(video_path)
        # 获取视频的帧率
        frame_rate = video_clip.fps
        return frame_rate
    except Exception as e:
        print(f"获取视频帧率时出现错误: {e}")
        return None


def trim_video(video_path, output_path, trim_start=None, trim_end=None, duration=None, start_eps=0.001):
    if trim_end is not None and duration is not None:
        raise ValueError('`trim_end` and `duration` should not be both set.')
    clip = VideoFileClip(video_path)
    start = trim_start or 0
    end = -trim_end if trim_end is not None else None
    end = start + duration if duration is not None else end
    start = start + start_eps       # 防止有时前一帧被包含在内
    # subclip = clip.set_start(start, change_end=False).set_end(end)
    subclip = clip.subclip(start, end)
    subclip.write_videofile(output_path, logger=None)
    clip.close()


def trim_audio(audio_path, output_path, trim_start=None, trim_end=None, duration=None):
    if trim_end is not None and duration is not None:
        raise ValueError('`trim_end` and `duration` should not be both set.')
    audio = AudioSegment.from_file(audio_path, format="wav")
    # duration = audio.duration_seconds
    start = trim_start * 1000 if trim_start is not None else 0
    end = -trim_end * 1000 if trim_end is not None else None
    end = start + duration * 1000 if duration is not None else end
    audio = audio[start:end]
    audio.export(output_path, format="wav")


if __name__ == '__main__':
    def check_cncvs_data_format():
        from tqdm import tqdm
        original_root = '/hdd1/ljh/CN-CVS/processed'
        all_audios = glob.glob(os.path.join(original_root, '*/*/audio/*.wav'))
        print('Checking audios...')
        pbar = tqdm(all_audios)
        for audio in pbar:
            pbar.set_description(f"Checking {audio}")
            
            ### What to check
            sample_rate = get_wav_sample_rate(audio)
            if sample_rate != 16000:
                print(f"{audio} sample rate: {sample_rate}")

        all_videos = glob.glob(os.path.join(original_root, '*/*/video/*.mp4'))
        print('Checking videos...')
        pbar = tqdm(all_videos)
        for video in pbar:
            pbar.set_description(f"Checking {video}")
            
            frame_rate = get_video_frame_rate(video)
            if frame_rate != 25.0:
                print(f"{video} frame rate: {frame_rate}")

    check_cncvs_data_format()
