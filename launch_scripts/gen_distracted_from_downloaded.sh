#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench/impls

export MUJOCO_GL=egl

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

ENV_NAME=visual-antmaze-medium-stitch-v0
DISTRACTION_IMAGES_DIR=/data/rohanp/DAVIS/JPEGImages/480p
AGENT=/data/rohanp/ogbench/impls/agents/hiql.py
SPECIFIC_DISTRACTOR=bear

cd /data/rohanp/ogbench/data_gen_scripts

python gen_distracted_from_downloaded.py \
    --agent=$AGENT \
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR \
    --specific_distractor=$SPECIFIC_DISTRACTOR \
    --env_name=$ENV_NAME \