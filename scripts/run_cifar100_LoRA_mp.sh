#!/bin/bash

set -e
set -x

NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12355 train_lora_mp.py \
    --dataset_name 'cifar100' \
    --batch_size 64 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name cifar100_lora_mp \
    --lora_rank 4
