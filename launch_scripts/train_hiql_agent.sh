#!bin/bash


'''
Note:If you are running on a remote/headless server without a display, 
you can use EGL for rendering by setting the MUJOCO_GL environment variable
'''


MUJOCO_GL=egl 

# ENV_NAME=visual-antmaze-medium-v0
ENV_NAME=visual-antmaze-medium-stitch-v0

# Generate dataset using our expert trained ant policy
# cd data_gen_scripts
# python generate_locomaze.py \
#     --env_name=$ENV_NAME\
#     --save_path=data/visual-antmaze-medium-v0.npz\
#     --num_episodes=100\
#     --restore_path=exp/OGBench/ogbench/sd000_s_626713.0.20260316_183527\
#     --restore_epoch=1000000             # This tells the script how many train steps the expert agent we're using was trained to


# Train a HIQL agent using our dataset - just to see if we can and also we can treat this as our baseline once we get everything working
# TODO: do we need to be worried about these hyperparams and being tailored to the task?

DATASET_PATH_TRAIN=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/visual-antmaze-medium-stitch-v0.npz
DATASET_PATH_VAL=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/visual-antmaze-medium-stitch-v0-val.npz

cd impls
#  -m pdb
python main.py\
    --dataset_path_train=$DATASET_PATH_TRAIN\
    --dataset_path_val=$DATASET_PATH_VAL\
    --env_name=$ENV_NAME\
    --agent=agents/hiql.py\
    --agent.high_alpha=3.0\
    --agent.low_alpha=3.0

