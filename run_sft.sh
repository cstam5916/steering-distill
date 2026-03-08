#!/bin/bash
##  Asking for 1 node, and 6 cores - the next line asks for 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -o KDoutLog
#SBATCH -e KDerrLog
#SBATCH --time=16:00:00

python -m train --loss "token_ce" --output_dir "sft_batched" --batch_size 16

# python -m train --loss "kd" --output_dir "first-kd" 