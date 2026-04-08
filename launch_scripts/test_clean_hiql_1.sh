#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench
cd /work/10993/rohanpatel01/vista/ogbench/impls

# source /work/11247/evankuo/vista/miniconda3/etc/profile.d/conda.sh
# conda activate /work/11247/evankuo/vista/miniconda3/envs/ogbench
# cd /work/11247/evankuo/vista/ogbench/impls

ENV_NAME=visual-antmaze-medium-stitch-v0

python main.py \
    --exp_name='clean_hiql_test1' \
    --seed=0 \
    --env_name=$ENV_NAME \
    --agent=agents/hiql.py \
    --agent.high_alpha=3.0 \
    --agent.low_alpha=3.0 \
    --agent.encoder=impala_small \
    --agent.low_actor_rep_grad=True \
    --agent.batch_size=256 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_p_trajgoal=0.5 \
    --using_distractions_dataset=0 \
    --log_interval=1 \
    --eval_interval=100000 \
    --save_interval=100000 \
	--train_steps=120000


