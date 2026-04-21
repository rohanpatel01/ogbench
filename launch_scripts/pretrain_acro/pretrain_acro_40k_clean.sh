#!/bin/bash

export WANDB_API_KEY=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

export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench

export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=2

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

# Single distractor bear data
# DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/bear_single_distractor_dataset/new/visual-antmaze-medium-stitch-v0-distracted.npz
# DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/bear_single_distractor_dataset/new/visual-antmaze-medium-stitch-v0-distracted-val.npz

# Clean data
DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/clean/visual-antmaze-medium-stitch-v0.npz
DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/clean/visual-antmaze-medium-stitch-v0-val.npz

DAVIS_DATASET_PATH=/data/rohanp/DAVIS/JPEGImages/480p
STEPS_PRE_TRAIN_ACRO=40000
# 40000
TRAIN_STEPS=1
#210000
EVAL_INTERVAL=1
# 25000
ENV_NAME=visual-antmaze-medium-stitch-v0
SEED=0
cd /data/rohanp/ogbench/impls


python main.py \
    --seed=$SEED \
    --exp_name='pretrain_acro_40k_clean' \
    --using_distractions_dataset=0 \
    --steps_pre_train_acro=$STEPS_PRE_TRAIN_ACRO \
    --train_steps=$TRAIN_STEPS \
    --use_acro_rep=0 \
    --use_acro_for_reward=0 \
    --eval_interval=$EVAL_INTERVAL \
    --eval_episodes=1 \
    --save_interval=25000 \
    --dataset_path_train=$DATASET_PATH_TRAIN \
    --dataset_path_val=$DATASET_PATH_VAL \
    --env_name=$ENV_NAME \
    --agent=agents/hiql.py \
    --agent.high_alpha=3.0 \
    --agent.low_alpha=3.0 \
    --agent.encoder=impala_small \
    --agent.low_actor_rep_grad=True \
    --agent.batch_size=256 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_p_trajgoal=0.5 \
    --log_interval=1 \
    --eval_on_cpu=0 \
    --freeze_acro_rep=0 \
    --davis_dataset_path=$DAVIS_DATASET_PATH \
    --dataset_download_dir=/data/rohanp/ogbench/data_gen_scripts/data
