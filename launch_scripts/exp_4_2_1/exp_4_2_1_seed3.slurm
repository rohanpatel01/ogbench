#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
#
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's PyLauncher utility to run multiple serial
#       executables at the same time, execute "module load pylauncher"
#       followed by "module help pylauncher".
#----------------------------------------------------

#SBATCH -J exp_4_2_1           # Job name
#SBATCH -o exp_4_2_1.o%j       # Name of stdout output file
#SBATCH -e exp_4_2_1.e%j       # Name of stderr error file
#SBATCH -p gh           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 4:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A ASC24079       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=evan.kuo@utexas.edu

# Any other commands must follow all #SBATCH directives...
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

python main.py \
    --seed=3 \
    --exp_name='exp_4_2_1' \
    --using_distractions_dataset=1 \
    --train_steps=200000 \
    --steps_pre_train_acro=100000 \
    --use_acro_rep=0 \
    --use_acro_for_reward=1 \
    --eval_interval=50000 \
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
    --eval_on_cpu=0
