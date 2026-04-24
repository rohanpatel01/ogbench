#!/bin/bash

export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench/impls

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3

RESTORE_PATH=/data/rohanp/ogbench/data_gen_scripts/exp/OGBench/ogbench/sd000_20260422_021540

SAVE_DIR=/data/rohanp/ogbench/data_gen_scripts/data/our_ant_policy_clean/
SAVE_FILE_NAME=visual-antmaze-medium-stitch-v0.npz

cd /data/rohanp/ogbench/data_gen_scripts

# visual-antmaze-medium-stitch-v0
python generate_locomaze.py \
    --env_name=visual-antmaze-medium-v0 \
    --save_dir=$SAVE_DIR \
    --save_file_name=$SAVE_FILE_NAME \
    --dataset_type=stitch \
    --num_episodes=5000 \
    --max_episode_steps=201 \
    --restore_path=$RESTORE_PATH \
    --restore_epoch=400000

