#!bin/bash


export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3


SAVE_DIR=data/
SAVE_FILE_NAME=visual-antmaze-medium-stitch-v0-single-distractor-dog.npz
ENV_NAME=visual-antmaze-medium-v0
NUM_EPISODES=5000
SAVE_PERIOD=500

MAX_EPISODE_STEPS=201
RESTORE_PATH=/work/10993/rohanpatel01/vista/ogbench/impls/data_gen_scripts/exp/OGBench/ogbench/sd000_s_626713.0.20260316_183527
RESTORE_EPOCH=1000000


DISTRACTION_IMAGES_DIR=/data/rohanp/480p

pwd
cd ../impls/data_gen_scripts

python generate_distracted_locomaze.py\
    --env_name=$ENV_NAME\
    --dataset_type=stitch\
    --num_episodes=$NUM_EPISODES\
    --max_episode_steps=$MAX_EPISODE_STEPS\
    --restore_path=$RESTORE_PATH\
    --restore_epoch=$RESTORE_EPOCH\
    --distraction=image\
    --distraction_difficulty=easy\
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR\
    --save_dir=$SAVE_DIR\
    --save_period=$SAVE_PERIOD\
    --save_file_name=$SAVE_FILE_NAME

