#!/bin/bash

export WANDB_API_KEY=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

# export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export PYTHONPATH=$PYTHONPATH:/work/11247/evankuo/vista/ogbench/
export MUJOCO_GL=egl

# source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
# conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench
# cd /work/10993/rohanpatel01/vista/ogbench/impls

source /work/11247/evankuo/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/11247/evankuo/vista/miniconda3/envs/ogbench
cd /work/11247/evankuo/vista/ogbench/impls

DATASET_PATH_TRAIN=/work/11247/evankuo/vista/visual-antmaze-medium-stitch-v0-distracted.npz
DATASET_PATH_VAL=/work/11247/evankuo/vista/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

# python main.py \
#     --seed=0 \
#     --exp_name='exp_4_2_1' \
#     --using_distractions_dataset=1 \
#     --train_steps=200000 \
#     --steps_pre_train_acro=100000 \
#     --use_acro_rep=0 \
#     --use_acro_for_reward=1 \
#     --eval_interval=50000 \
#     --save_interval=50000 \
#     --dataset_path_train=$DATASET_PATH_TRAIN \
#     --dataset_path_val=$DATASET_PATH_VAL \
#     --env_name=$ENV_NAME \
#     --agent=agents/hiql.py \
#     --agent.high_alpha=3.0 \
#     --agent.low_alpha=3.0 \
#     --agent.encoder=impala_small \
#     --agent.low_actor_rep_grad=True \
#     --agent.batch_size=256 \
#     --agent.actor_p_randomgoal=0.5 \
#     --agent.actor_p_trajgoal=0.5 \
#     --log_interval=1 \
#     --eval_on_cpu=0 \
#     --eval_episodes=20

python main.py \
    --seed=0 \
    --exp_name='exp_4_2_1_compilation_test' \
    --using_distractions_dataset=1 \
    --train_steps=2 \
    --steps_pre_train_acro=2 \
    --use_acro_rep=0 \
    --use_acro_for_reward=1 \
    --eval_interval=1 \
    --save_interval=50000 \
    --dataset_path_train=$DATASET_PATH_TRAIN \
    --dataset_path_val=$DATASET_PATH_VAL \
    --env_name=$ENV_NAME \
    --agent=agents/hiql.py \
    --agent.high_alpha=3.0 \
    --agent.low_alpha=3.0 \
    --agent.encoder=impala_small \
    --agent.low_actor_rep_grad=True \
    --agent.batch_size=256 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_p_trajgoal=0.5 \
    --log_interval=1 \
    --eval_on_cpu=0 \
    --eval_episodes=2
