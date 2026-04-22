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
# DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/clean/visual-antmaze-medium-stitch-v0.npz
# DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/clean/visual-antmaze-medium-stitch-v0-val.npz

# ENV_NAME=visual-antmaze-medium-stitch-v0

cd /data/rohanp/ogbench/impls

# visual-antmaze-medium-stitch-v0 (GCIQL)
# I changed eval_episodes from 50 to 5 for sake of compute and seeing results faster
python main.py \
    --exp_name='gciql_clean_baseline' \
    --env_name=visual-antmaze-medium-stitch-v0 \
    --train_steps=500000 \
    --eval_episodes=5 \
    --eval_on_cpu=0 \
    --agent=agents/gciql.py \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.alpha=0.3 \
    --agent.batch_size=256 \
    --agent.encoder=impala_small