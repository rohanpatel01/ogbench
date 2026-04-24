#!/bin/bash

export WANDB_API_KEY=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

export PYTHONPATH=$PYTHONPATH:/home/ekuo/ogbench

export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=1

source /data/ekuo/miniconda3/etc/profile.d/conda.sh
conda activate /data/ekuo/miniconda3/envs/ogbench

# distracted data
DATASET_PATH_TRAIN=/home/ekuo/ogDatasets/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/home/ekuo/ogDatasets/visual-antmaze-medium-stitch-v0-distracted-val.npz

DAVIS_DATASET_PATH=/home/ekuo/ogDatasets/DAVIS/JPEGImages/480p

STEPS_PRE_TRAIN_ACRO=40000
TRAIN_STEPS=210000

# STEPS_PRE_TRAIN_ACRO=10
# TRAIN_STEPS=10

ENV_NAME=visual-antmaze-medium-stitch-v0
SEED=0

cd /home/ekuo/ogbench/impls

# For this run we do p_aug = 1.0 in acro.py
for SEED in 0 1 2 3; do
	python main.py \
		--seed=$SEED \
		--exp_name='distracted_hiql_acro' \
		--using_distractions_dataset=1 \
		--steps_pre_train_acro=$STEPS_PRE_TRAIN_ACRO \
		--train_steps=$TRAIN_STEPS \
		--use_acro_rep=1 \
		--use_acro_for_reward=0 \
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
		--freeze_acro_rep=0 \
		--davis_dataset_path=$DAVIS_DATASET_PATH \
		--dataset_download_dir=/home/ekuo/ogbench/data_gen_scripts/data
done