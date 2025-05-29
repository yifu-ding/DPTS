#!/bin/bash

WORKSPACE=../
cd $WORKSPACE
export BENCHMARK_ROOT='./benchmark'

export CUDA_VISIBLE_DEVICES=0
export GPU=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

work_dir=./results   
exp_name=test

mkdir -p ./results/math-qwen-1.5b/${exp_name}

DATASET_NAME=math 
MODEL_NAME=qwen-1.5b
REWARD_MODEL=mistral_prm-7b

python3 main.py \
    --config configs/inference/DPTS.json \
    --work-dir $work_dir \
    --exp-name $exp_name \
    --data $DATASET_NAME \
    --model $MODEL_NAME \
    --reward_model $REWARD_MODEL \
    --dtype bfloat16 \
    --flash-attn \
    --debug 
