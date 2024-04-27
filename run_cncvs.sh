nohup python -u cncvs_download_and_process.py \
-d /data2/CN-CVS/cncvs-stylegan \
-m /app/cncvs/speech/ \
-l 1 \
--download_worker 1 \
--transcode_worker 3 \
--process_worker 3 \
--max_clips_per_speaker 30 \
--cookies bilibili_cookies.txt \
--gpus 0 \
> output.log 2>&1 &
