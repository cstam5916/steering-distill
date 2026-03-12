#!/bin/bash
##  Asking for 1 node, and 6 cores - the next line asks for 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -o DeepSeekKDoutLog
#SBATCH -e DeepSeekKDerrLog
#SBATCH --time=2:00:00

# python -m train --loss "token_ce" --output_dir "no-distil-2" 

# python -m train --loss "kd" --output_dir "deepseek-kd" --batch_size 8

python -m train --loss "steer_kd" --output_dir "steering-save-projector" --batch_size 4

# python -m teacher_steering_evaluation