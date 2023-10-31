#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/semi-refuge400.yaml


save_path=exp/semi-refuge400/version1
mkdir -p $save_path
python -W ignore train_semi_refuge400.py \
    --config=$config\
    --save-path $save_path  --port 28990 --info '伪标签来自teacher，没有对teacher的batchnorm进行有标记维护'


save_path=exp/semi-refuge400/version2
mkdir -p $save_path
python -W ignore train_semi_refuge400.py \
    --config=$config\
    --pseudo_from prototype \
    --save-path $save_path  --port 28990 --info '伪标签来自prototype，没有对teacher的batchnorm进行有标记维护'

save_path=exp/semi-refuge400/version3
mkdir -p $save_path
python -W ignore train_semi_refuge400.py \
    --config=$config\
    --pseudo_from mix \
    --save-path $save_path  --port 28990 --info '伪标签来自teacher和prototype两个部分的相交部分，没有对teacher的batchnorm进行有标记维护'


