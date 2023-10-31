#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/semi-refuge400.yaml
save_path=exp/semi-refuge400

mkdir -p $save_path
python -W ignore train_semi_refuge400.py \
    --config=$config\
    --save-path $save_path  --port 28990 | tee $save_path/$now.txt

