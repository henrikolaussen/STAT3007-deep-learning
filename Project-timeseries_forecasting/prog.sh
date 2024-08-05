#!/bin/bash
#SBATCH --mem=16G # request at least 16G memory
#SBATCH --time=4-00:00 # maximum running time is 4 days
#SBATCH --partition=,gpu # run on GPUs from either the smp or the gpu partition; delete this line and the next if you don't need to use a gpu
#SBATCH --gres=gpu:1 # request one GPU
#SBATCH --output=prog.log # all outputs will be written to prog.log

module load anaconda3
# check PyTorch version and whether GPU is available 
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
# comment out the line below if you just want to check whether GPU is available
python prog.py
