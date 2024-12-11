#!/bin/bash

#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --partition=g2-standard-48
#SBATCH --job-name=hpml_project_dis
#SBATCH --output=hpml_project_dis.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:l4:4

singularity exec --nv --overlay /scratch/jja435/python-env/python_env.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
"source /ext3/env.sh; cd /scratch/$USER/project;\
pip install torch transformers torch_pruning evaluate peft accelerate;\
pip install -U bitsandbytes;\
pip install kagglehub seaborn wandb;\
torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 16 --model bert;\
torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 8 --model gpt2;\
torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 8 --model llama;"