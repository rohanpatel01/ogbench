#!bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/impls/
export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export MUJOCO_GL=egl


SAVE_DIR=undistracted_data/
SAVE_FILE_NAME=visual-antmaze-medium-stitch-v0.npz
ENV_NAME=visual-antmaze-medium-v0
NUM_EPISODES=5000
SAVE_PERIOD=500

MAX_EPISODE_STEPS=201
RESTORE_PATH=/work/10993/rohanpatel01/vista/ogbench/impls/data_gen_scripts/exp/OGBench/ogbench/sd000_s_626713.0.20260316_183527
RESTORE_EPOCH=1000000


pwd
cd ../impls/data_gen_scripts

python generate_locomaze.py\
    --env_name=$ENV_NAME\
    --dataset_type=stitch\
    --num_episodes=$NUM_EPISODES\
    --max_episode_steps=$MAX_EPISODE_STEPS\
    --restore_path=$RESTORE_PATH\
    --restore_epoch=$RESTORE_EPOCH\
    --save_dir=$SAVE_DIR\
    --save_period=$SAVE_PERIOD\
    --save_file_name=$SAVE_FILE_NAME
