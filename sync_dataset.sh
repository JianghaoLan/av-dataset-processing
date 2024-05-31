touch sync_dataset_output.log
nohup python -u sync_dataset.py \
--dataset_root /data2/CN-CVS/cncvs-stylegan/final \
--avsync_result_list /data2/CN-CVS/cncvs-stylegan/avsync_result.txt \
--output_dataset_root /data2/CN-CVS/synced \
--output_result_list /data2/CN-CVS/dataset_sync_result.txt \
--max_workers 8 --score_thres 0.33 \
--min_num_samples 12 \
--filter true >> sync_dataset_output.log 2>&1 &
tail -f sync_dataset_output.log
