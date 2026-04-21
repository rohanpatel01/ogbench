#!/bin/bash

# Launch pre-training ACRO on clean dataset with 40k timesteps
bash /data/rohanp/ogbench/launch_scripts/pretrain_acro/pretrain_acro_40k_clean.sh

# Launch pre-training ACRO on single bear distracted dataset with 40k timesteps
bash /data/rohanp/ogbench/launch_scripts/pretrain_acro/pretrain_acro_40k_distracted.sh