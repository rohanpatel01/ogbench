#!bin/bash



SAVE_PATH=data/visual-antmaze-medium-stitch-v0-distracted.npz
ENV_NAME=visual-antmaze-medium-v0
NUM_EPISODES=10
RESTORE_PATH=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/exp/OGBench/ogbench/sd000_s_626713.0.20260316_183527
RESTORE_EPOCH=1000000

DISTRACTION_IMAGES_DIR=/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p

MUJOCO_GL=egl 

cd ../data_gen_scripts

python generate_distracted_locomaze.py\
    --env_name=$ENV_NAME\
    --save_path=$SAVE_PATH\
    --dataset_type=stitch\
    --num_episodes=$NUM_EPISODES\
    --max_episode_steps=201\
    --restore_path=$RESTORE_PATH\
    --restore_epoch=$RESTORE_EPOCH\
    --distraction=image\
    --distraction_difficulty=easy\
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR

