#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=LSTM_4096
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=lars.ostberg.moan@gmail.com
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=p100
#SBATCH --gres=gpu:1

conda activate torch-gpu

python stat3007_timeseries_forecasting/lstm/run.py