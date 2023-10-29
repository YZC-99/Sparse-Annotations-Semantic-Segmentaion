#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/semi-drishti.yaml
save_path=exp/semi-drishti

mkdir -p $save_path
python -W ignore train_semi_drishti.py \
    --config=$config\
    --save-path $save_path  --port 28990 | tee $save_path/$now.txt
# python -m torch.distributed.launch \
#     --nproc_per_node=$1 \
#     --master_addr=localhost \
#     --master_port=$2 \
#     train_drishti.py \
#     --config=$config\
#     --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt

