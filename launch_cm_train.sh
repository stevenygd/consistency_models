#!/bin/bash

# AFHQ-64x64
# 100 microbatch size takes about 46G memories
# Originally it's 800_000
OMPI_MCA_opal_cuda_support=true CUDA_VISIBLE_DEVICES="4,5,6,7" GPUS_PER_NODE=4 \
   mpiexec -n 4 python cm_train.py \
    --training_mode consistency_training \
    --target_ema_mode adaptive \
    --start_ema 0.95 \
    --ema_rate 0.999,0.9999,0.9999432189950708 \
    --scale_mode progressive \
    --start_scales 2 --end_scales 200 \
    --total_training_steps 100000 \
    --loss_norm l2 \
    --lr_anneal_steps 0 \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --use_scale_shift_norm True \
    --dropout 0.0 \
    --teacher_dropout 0.1 \
    --microbatch 64 \
    --global_batch_size 256 \
    --image_size 128 \
    --lr 0.0001 \
    --num_channels 128 \
    --num_head_channels 32 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --schedule_sampler uniform \
    --use_fp16 True \
    --weight_decay 0.0 \
    --weight_schedule uniform \
    --log_interval 100 \
    --save_interval 5000 \
    --result_dir results/tri_planes \
    --data_dir /home/rlpo/triplane_autodecoder/data/tri_planes \
    --stats_dir /home/rlpo/triplane_autodecoder/data/planes_triplanes_stats

# # AFHQ-256x256
# CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 mpiexec -n 1 python cm_train.py \
#     --training_mode consistency_training \
#     --target_ema_mode adaptive \
#     --start_ema 0.95 \
#     --scale_mode progressive \
#     --start_scales 2 --end_scales 150 \
#     --total_training_steps 1000000 \
#     --loss_norm lpips \
#     --lr_anneal_steps 0 \
#     --attention_resolutions 32,16,8 \
#     --class_cond False \
#     --use_scale_shift_norm False \
#     --dropout 0.0 \
#     --teacher_dropout 0.1 \
#     --ema_rate 0.9999,0.99994,0.9999432189950708 \
#     --microbatch 4 \
#     --global_batch_size 256 \
#     --image_size 256 \
#     --lr 0.00005 \
#     --num_channels 256 \
#     --num_head_channels 64 \
#     --num_res_blocks 2 \
#     --resblock_updown True \
#     --schedule_sampler uniform \
#     --use_fp16 True \
#     --weight_decay 0.0 \
#     --weight_schedule uniform \
#     --log_interval 100\
#     --data_dir /home/guandao/stargan-v2/afhq-v2-256

# CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 mpiexec -n 1 python cm_train.py \
#     --training_mode consistency_training \
#     --target_ema_mode adaptive \
#     --start_ema 0.95 \
#     --scale_mode progressive \
#     --start_scales 2 --end_scales 150 \
#     --total_training_steps 1000000 \
#     --loss_norm lpips \
#     --lr_anneal_steps 0 \
#     --attention_resolutions 32,16,8 \
#     --class_cond False \
#     --use_scale_shift_norm False \
#     --dropout 0.0 \
#     --teacher_dropout 0.1 \
#     --ema_rate 0.9999,0.99994,0.9999432189950708 \
#     --global_batch_size 256 \
#     --image_size 256 \
#     --lr 0.00005 \
#     --num_channels 256 \
#     --num_head_channels 64 \
#     --num_res_blocks 2 \
#     --resblock_updown True \
#     --schedule_sampler uniform \
#     --use_fp16 True \
#     --weight_decay 0.0 \
#     --weight_schedule uniform \
#     --log_interval 100\
#     --data_dir /data/guandao/lsun/bedroom_train_images/
