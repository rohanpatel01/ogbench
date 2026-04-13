#!/bin/bash

# TODO: If getting "TypeError: ImageDistractionWrapper.__init__() got an unexpected keyword argument 'specific_distractor'"
# Likely means a path in here is wrong

export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export MUJOCO_GL=egl

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/bear_single_distractor_dataset/new/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/bear_single_distractor_dataset/new/visual-antmaze-medium-stitch-v0-distracted-val.npz

# Smaller dataset just to test compilation
# TODO: Note that these were not generated from their downloaded data rather we generated these. So only use to test compilation faster - not actual debugging
# DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/intermediate_save_3/visual-antmaze-medium-stitch-v0-distracted.npz
# DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/intermediate_save_3/visual-antmaze-medium-stitch-v0-distracted-val.npz


DAVIS_DATASET_PATH=/data/rohanp/DAVIS/JPEGImages/480p
SPECIFIC_DISTRACTOR=bear

ENV_NAME=visual-antmaze-medium-stitch-v0

cd /data/rohanp/ogbench/impls

STEPS_PRE_TRAIN_ACRO=20000

python main.py \
    --seed=0 \
    --exp_name='exp_4_2' \
    --davis_dataset_path=$DAVIS_DATASET_PATH \
    --specific_distractor=$SPECIFIC_DISTRACTOR \
    --freeze_acro_rep=0 \
    --using_distractions_dataset=1 \
    --train_steps=2 \
    --steps_pre_train_acro=$STEPS_PRE_TRAIN_ACRO \
    --use_acro_rep=0 \
    --use_acro_for_reward=1 \
    --eval_interval=50000 \
    --save_interval=50000 \
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
    --eval_on_cpu=0
