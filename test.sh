#!/bin/bash

for ((i=1;i<=1;i++))
do
    echo "Running Test "$i
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
        --test_num_checkpoint=$i \
        --test_protocol=protocol_vox1_sub.txt
        
    python3 eval.py \
        --ground_truth=protocols/protocol_vox1_sub.txt \
        --prediction=models/model_vox1_0_fbank_64_96_200_512_baselineSAMAFRN02/$i.result \
        --positive=1
done
