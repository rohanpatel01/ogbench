#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench

DATASET_PATH_TRAIN=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

cd /work/10993/rohanpatel01/vista/ogbench/impls

python main.py \
    --seed=0 \
    --exp_name='exp_4_1' \
    --using_distractions_dataset=1 \
    --train_steps=200000 \
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
