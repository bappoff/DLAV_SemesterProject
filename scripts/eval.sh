#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --begin=now
#SBATCH --mem 40G
#SBATCH --partition gpu
#SBATCH --gres gpu:1

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$/work/scitas-share/voxformer/VoxFormer":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $/work/scitas-share/voxformer/VoxFormer/tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
