#!/bin/bash

### Job options
#BSUB -q gpuv100                    # GPU queue
#BSUB -J train_job                  # job name
#BSUB -W 04:00                      # max runtime (4 hours)
#BSUB -n 4                          # number of CPU cores
#BSUB -R "span[hosts=1]"            # all cores on same node
#BSUB -R "rusage[mem=8GB]"          # memory per core (32GB total)
#BSUB -M 8GB                        # kill if exceeds this per core
#BSUB -gpu "num=1"                  # request 1 GPU
#BSUB -u your_email@student.dtu.dk  # your DTU email
#BSUB -B                            # notify when job starts
#BSUB -N                            # notify when job ends
#BSUB -o logs/%J.out                # stdout log
#BSUB -e logs/%J.err                # stderr log

# Load CUDA (required for JAX GPU support)
module load cuda/12.6

# Navigate to project folder
cd $HOME/Desktop/ProjectWork/JAX-OPTIMAL-TRANSPORT-FAGPROJECT

# Make sure we're on the right branch
git checkout main  # change this to your branch name

# Install/sync dependencies
uv sync

# Run training
uv run python src/main_project/train.py
