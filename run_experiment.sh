#!/bin/bash
#############################
#SBATCH -J FM
#SBATCH -n 1
#SBATCH --cpus-per-task=5
#SBATCH -t 0-36:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres gpu:2 
#SBATCH --mem 32G
#SBATCH -D .
#############################

source mnist_env/bin/activate

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_STEP_GPUS" ]; then
    NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
fi

echo "Detected $NUM_GPUS GPUs requested."

# 
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching with torchrun for $NUM_GPUS GPUs (Distributed Mode)..."
    torchrun --nproc_per_node=$NUM_GPUS \
             /net/deimos/scratch/jsarrato/MNIST_Diffusion/src/run_experiment.py
else
    echo "Launching with standard python (Single GPU Mode)..."
    python -u /net/deimos/scratch/jsarrato/MNIST_Diffusion/src/run_experiment.py
fi
