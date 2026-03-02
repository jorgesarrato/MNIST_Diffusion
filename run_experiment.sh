#!/bin/bash
#############################
#SBATCH -J FM
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH -t 0-12:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres gpu:2
#SBATCH --mem 30G

#SBATCH -D .
#############################

source mnist_env/bin/activate
python -u src/run_experiment.py
