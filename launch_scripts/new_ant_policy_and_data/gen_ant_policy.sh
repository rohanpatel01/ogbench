#!/bin/bash

export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench
export PYTHONPATH=$PYTHONPATH:/data/rohanp/ogbench/impls

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3

source /data/rohanp/miniconda3/etc/profile.d/conda.sh
conda activate /data/rohanp/miniconda3/envs/ogbench

cd /data/rohanp/ogbench/data_gen_scripts

# ant (online-ant-xy-v0)
python main_sac.py --env_name=online-ant-xy-v0 --train_steps=400000 --eval_interval=100000 --save_interval=400000 --log_interval=5000