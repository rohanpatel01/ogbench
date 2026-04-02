#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/impls/

export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench

ENV_NAME=visual-antmaze-medium-stitch-v0
DISTRACTION_IMAGES_DIR=/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p
AGENT=/work/10993/rohanpatel01/vista/ogbench/impls/agents/hiql.py

cd /work/10993/rohanpatel01/vista/ogbench/data_gen_scripts

python gen_distracted_from_downloaded.py \
    --agent=$AGENT \
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR \
    --env_name=$ENV_NAME \