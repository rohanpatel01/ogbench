#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

export WANDB_API_KEY=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
# /work/10993/rohanpatel01/vista/ogbench/
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export MUJOCO_GL=egl

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

DATASET_PATH_TRAIN=/data/rohanp/ogbench/data_gen_scripts/data/bear_dog_distractor_dataset/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/data/rohanp/ogbench/data_gen_scripts/data/bear_dog_distractor_dataset/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

DAVIS_DATASET_PATH=/data/rohanp/DAVIS/JPEGImages/480p
SPECIFIC_DISTRACTOR=dog,bear

cd /data/rohanp/ogbench/impls

# seed 0 1 2
# train_steps=200000
# steps_pre_train_acro=100000
# change back eval to include i == 1 in main.py
for SEED in {0..2}
do
    python main.py \
        --seed=$SEED \
        --exp_name='exp_4_2_old_dog_bear_distractions_sanity_check_after_fix_eval' \
        --davis_dataset_path=$DAVIS_DATASET_PATH \
        --specific_distractor=$SPECIFIC_DISTRACTOR \
        --freeze_acro_rep=0 \
        --using_distractions_dataset=1 \
        --train_steps=200000 \
        --steps_pre_train_acro=100000 \
        --use_acro_rep=0 \
        --use_acro_for_reward=0 \
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


done