#!/usr/bin/env bash
#SBATCH --job-name=KNN-gen
#SBATCH --output=/home/rajarshi/Dropbox/research/cbr-weak-supervision/wandb/wandb-%A_%a.out
#SBATCH --partition=rtx8000-short
#SBATCH --gres=gpu:1
#SBATCH --time=00-04:00:00
#SBATCH --mem=80G
#SBATCH --array=0-3

conda activate pygnn
# Set to scratch/work since server syncing will occur from here
# Ensure sufficient space else runs crash without error message

# Flag --count specifies number of hyperparam settings (runs) tried per job
# If SBATCH --array flag is say 0-7 (8 jobs) then total (8 x count)
# hyperparam settings will be tried
wandb agent rajarshd/cbr-weak-supervision/66r26jn3

