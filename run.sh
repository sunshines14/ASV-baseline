#!/bin/bash

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License

stage=1

# train
if [ $stage -le 1 ]; then
    CUDA_VISIBLE_DEVICES=1,2,3 python3 main.py \
        --version=vox2 \
        --win_size=0 \
        --feature=fbank \
        --feature_dims=64 \
        --train_batch_size=96 \
        --num_epochs=100 \
        --embedding_size=512 \
        --speakers=1211 \
        --model_comment=baselineSAMAFRN02 \
        --loss=ce \
        --optim=sgd \
        --lr=0.1 \
        --wd=0.0001 \
        --sched_factor=0.1 \
        --sched_patience=0 \
        --sched_min_lr=0.0001
exit 0
fi

# test
if [ $stage -le 2 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --version=vox2 \
        --win_size=0 \
        --feature=fbank \
        --feature_dims=64 \
        --train_batch_size=96 \
        --num_epochs=100 \
        --embedding_size=512 \
        --speakers=1211 \
        --model_comment=baselineSAMAFRN02 \
        --loss=ce \
        --optim=sgd \
        --lr=0.1 \
        --wd=0.0001 \
        --sched_factor=0.1 \
        --sched_patience=0 \
        --sched_min_lr=0.0001 \
        --test_mode \
        --test_num_checkpoint=48 \
        --test_protocol=protocol_vox1.txt
exit 0
fi

# result
if [ $stage -le 3 ]; then
    python3 eval.py \
        --ground_truth=protocols/protocol_vox1.txt \
        --prediction=models/model_vox1_0_fbank_64_96_100_512_baselineSAMAFRN02/48.result \
        --positive=1
exit 0
fi