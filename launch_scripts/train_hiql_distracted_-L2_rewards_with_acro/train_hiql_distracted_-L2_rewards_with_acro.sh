#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench

# TODO: Using smaller bad distracted dataset for now, but before submitting job change these two to use the actual data:
# DATASET_PATH_TRAIN=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted.npz
# DATASET_PATH_VAL=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted-val.npz

DATASET_PATH_TRAIN=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/intermediate_save_3/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/intermediate_save_3/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

cd /work/10993/rohanpatel01/vista/ogbench/impls

# TODO: Update train_steps to be what we need to converge (note this will be the same for ACRO and HIQL so see if we need specify for each or keep same)
python main.py \
    --exp_name='exp_4_2' \
    --train_steps=1 \
    --use_acro_rep=1 \
    --use_acro_for_reward=0 \
    --seed=0 \
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
    --using_distractions_dataset=1 \
    --log_interval=1 \
    --eval_interval=100000 \
    --save_interval=100000 \
    --eval_on_cpu=0

