#!/bin/bash

set -e
set -x

#    --warmup_model_dir 'dev_outputs/simgcd/log/nwpu_test/checkpoints/model.pt' \
# test6 0.08+0.04
#gamma
#test1 0.07+0.05
#test2 0.07+0.06

#CUDA_VISIBLE_DEVICES=0 python train.py \
python train_lora.py \
    --dataset_name 'mstar' \
    --batch_size 128 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2b' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name mstar_nwpu_lora_gamma_ada4