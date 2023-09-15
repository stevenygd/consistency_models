# !/bin/bash

# AFHQ-128x128
# Class is WRONG! It should be three classes, but now only two: "flickr, pixabay"
OMPI_MCA_opal_cuda_support=true CUDA_VISIBLE_DEVICES="8" GPUS_PER_NODE=1 \
    mpiexec -n 1 python triplane_sample.py \
    --batch_size 2 \
    --training_mode consistency_training \
    --sampler onestep \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --num_classes 2 \
    --use_scale_shift_norm True \
    --dropout 0.0 \
    --num_channels 128 \
    --num_head_channels 32 \
    --num_res_blocks 2 \
    --image_size 128 \
    --resblock_updown True \
    --use_fp16 True \
    --weight_schedule uniform \
    --stats_dir /home/guandao/consistency_models/data/triplanes/stats \
    --in_channels 96 \
    --out_channels 96 \
    --num_samples ${2} \
    --result_dir `dirname ${1}` \
    --model_path ${1}
# /home/guandao/consistency_models/results/afhq-128/ema_0.999_185000.pt