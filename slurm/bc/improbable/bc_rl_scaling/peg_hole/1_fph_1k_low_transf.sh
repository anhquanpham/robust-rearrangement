#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_fph_1k_low_transf

python -m src.train.bc +experiment=state/scaling/peg_hole/1k \
    actor/diffusion_model=transformer \
    wandb.name=fph-1k-transf-steps-h16-28 \
    wandb.continue_run_id=c62fe1c8 \
    dryrun=false
