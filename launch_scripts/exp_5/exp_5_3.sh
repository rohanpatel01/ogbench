#!/bin/bash

export WANDB_API_KEY=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

export PYTHONPATH=$PYTHONPATH:/home/ekuo/ogbench/
export MUJOCO_GL=egl

export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco210/bin
export MUJOCO_PATH=$HOME/.mujoco/mujoco210
export CUDA_VISIBLE_DEVICES=2

source /data/ekuo/miniconda3/etc/profile.d/conda.sh
conda activate /data/ekuo/miniconda3/envs/ogbench
cd /home/ekuo/ogbench/impls

DATASET_PATH_TRAIN=/home/ekuo/ogDatasets/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/home/ekuo/ogDatasets/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

for SEED in 0 1; do
    python main.py \
        --seed=$SEED \
        --exp_name='exp_5_3' \
        --using_distractions_dataset=1 \
        --train_steps=210000 \
    	--steps_pre_train_acro=100000 \
        --use_acro_rep=0 \
        --use_acro_for_reward=1 \
        --eval_interval=25000 \
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
        --freeze_acro_rep=1 \
        --specific_distractor=bear \
        --davis_dataset_path=/home/ekuo/ogDatasets/DAVIS/JPEGImages/480p \
        --dataset_download_dir=/home/ekuo/ogbench/data_gen_scripts/data
done