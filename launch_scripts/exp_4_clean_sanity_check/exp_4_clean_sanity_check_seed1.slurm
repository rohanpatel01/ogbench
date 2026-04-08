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

#SBATCH -J hiql_train_ACRO_Representation           # Job name
#SBATCH -o hiql_train_ACRO_Representation.o%j       # Name of stdout output file
#SBATCH -e hiql_train_ACRO_Representation.e%j       # Name of stderr error file
#SBATCH -p gh           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 1:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A ASC24079       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=rpp689@eid.utexas.edu

# Any other commands must follow all #SBATCH directives...
export WANDB_API_KEY=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

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench

# DATASET_PATH_TRAIN=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted.npz
# DATASET_PATH_VAL=/work/10993/rohanpatel01/vista/ogbench/data_gen_scripts/data/distracted_from_downloaded/new/visual-antmaze-medium-stitch-v0-distracted-val.npz

ENV_NAME=visual-antmaze-medium-stitch-v0

cd /work/10993/rohanpatel01/vista/ogbench/impls

python main.py \
    --seed=1 \
    --exp_name='exp_4_clean_sanity_check' \
    --using_distractions_dataset=0 \
    --train_steps=100000 \
    --use_acro_rep=0 \
    --use_acro_for_reward=0 \
    --eval_interval=25000 \
    --save_interval=25000 \
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