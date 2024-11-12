#!/bin/bash

set -e
set -x

#    --warmup_model_dir 'dev_outputs/simgcd/log/nwpu_test/checkpoints/model.pt' \

#CUDA_VISIBLE_DEVICES=0 python train.py \
python test.py \
    --dataset_name 'mstar' \
    --batch_size 128 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --transform 'imagenet' \
    --eval_funcs 'v2b' \
    --save_name mstar_test