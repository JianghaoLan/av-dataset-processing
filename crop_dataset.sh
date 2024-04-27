nohup python -u crop_dataset.py \
--dataset_root /data2/CN-CVS/cncvs-stylegan/final \
--data_list_path /data2/CN-CVS/filtered.txt \
--output_root /data2/CN-CVS/cropped \
--output_data_list_path /data2/CN-CVS/cropped_data_list.txt \
--max_workers 4 >> crop.log 2>&1 &
