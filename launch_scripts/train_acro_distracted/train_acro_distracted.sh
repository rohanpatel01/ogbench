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
    --dataset_path_train=$DATASET_PATH_TRAIN \
    --dataset_path_val=$DATASET_PATH_VAL \
    --env_name=$ENV_NAME \
    --agent=agents/acro.py \
    --agent.batch_size=256 \
    --using_distractions_dataset=1 \
    --log_interval=1 \
    --eval_interval=10000000 \
    --save_interval=100000 \
    --train_steps=300000

