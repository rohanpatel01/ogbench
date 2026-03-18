#!/bin/bash


conda activate ogbench

export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

cd data_gen_scripts
MUJOCO_GL=egl

# Generate expert policy for ant so we can generate data
python main_sac.py --env_name=online-ant-xy-v0

