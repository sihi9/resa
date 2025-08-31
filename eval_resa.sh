#!/bin/bash
#SBATCH --job-name=resa                   # Name of your job
#SBATCH --partition=IMLcuda1             # GPU partition
#SBATCH --nodelist=nodeicuda1            # Specific node
#SBATCH --gres=gpu:RTX4000Ada:1          # Request 1 GPU
#SBATCH --time=06:00:00                  # Max wall time
#SBATCH --output=logs/resa_job_%j.out     # Stdout + stderr log file (%j = job ID)

# Load conda (adjust if needed for your setup)
source ~/.bashrc
conda activate resa

# Optional: print environment info
echo "Running on node: $(hostname)"

# Run your Python script
python ~/resa/main.py configs/carla.py --gpus 0 --load_from work_dirs/CarlaLaneDataset/20250827_152104_lr_2e-02_b_16/ckpt/best.pth