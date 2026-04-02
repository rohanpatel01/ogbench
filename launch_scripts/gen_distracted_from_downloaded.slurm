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

#SBATCH -J gen_distracted_from_downloaded_new           # Job name
#SBATCH -o gen_distracted_from_downloaded_new.o%j       # Name of stdout output file
#SBATCH -e gen_distracted_from_downloaded_new.e%j       # Name of stderr error file
#SBATCH -p gh           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 16:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A ASC24079       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=rpp689@eid.utexas.edu

#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/
export PYTHONPATH=$PYTHONPATH:/work/10993/rohanpatel01/vista/ogbench/impls/

export MUJOCO_GL=egl

source /work/10993/rohanpatel01/vista/miniconda3/etc/profile.d/conda.sh
conda activate /work/10993/rohanpatel01/vista/miniconda3/envs/ogbench

ENV_NAME=visual-antmaze-medium-stitch-v0
DISTRACTION_IMAGES_DIR=/work/10993/rohanpatel01/vista/DAVIS/JPEGImages/480p
AGENT=/work/10993/rohanpatel01/vista/ogbench/impls/agents/hiql.py

cd /work/10993/rohanpatel01/vista/ogbench/data_gen_scripts

python gen_distracted_from_downloaded.py \
    --agent=$AGENT \
    --distraction_images_dir=$DISTRACTION_IMAGES_DIR \
    --env_name=$ENV_NAME \
