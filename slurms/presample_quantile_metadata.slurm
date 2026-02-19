#!/bin/bash
#SBATCH --job-name=nowcasting
#SBATCH --partition=gpu
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate weather

cd ~/projects/weather-learning
export PYTHONPATH=$PWD/src

python -u src/preprocess/presample_quantile_metadata.py \
  --p99_top_ratio 0.20 \
  --prob 0.985 \
  --train_seed 42 \
  --val_seed 43 \
  --test_seed 44