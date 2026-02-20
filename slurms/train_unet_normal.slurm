#!/bin/bash
#SBATCH --job-name=nowcasting
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00     # HHI allows us to train the model only 3 days at a time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate weather

cd ~/projects/weather-learning
export PYTHONPATH=$PWD/src

python -u train_unet_normal.py