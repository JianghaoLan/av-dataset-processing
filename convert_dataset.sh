touch convert_dataset_output.log
nohup python -u convert_dataset.py \
--dataset_root /data2/CN-CVS/synced \
--output_root /data2/CN-CVS/wav2lip_format_synced \
--data_list_path /data2/CN-CVS/filtered_synced.txt \
--copy_rois_and_lms \
--max_workers 12 >> convert_dataset_output.log 2>&1 &
tail -f convert_dataset_output.log
