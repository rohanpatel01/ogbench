#!/bin/bash

export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

ENV_NAME=visual-antmaze-medium-stitch-v0
DISTRACTION_IMAGES_DIR=/data/rohanp/DAVIS/JPEGImages/480p
SPECIFIC_DISTRACTOR=dog
NUM_TRAJ=1

cd /data/rohanp/ogbench/data_gen_scripts

python gen_distracted_from_downloaded.py \
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR \
    --specific_distractor=$SPECIFIC_DISTRACTOR \
    --env_name=$ENV_NAME \
    --num_traj=$NUM_TRAJ