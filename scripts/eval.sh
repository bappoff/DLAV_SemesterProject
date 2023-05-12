#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --begin=now
#SBATCH --mem 40G
#SBATCH --partition gpu
#SBATCH --gres gpu:1

sbatch ./tools/dist_test.sh ./projects/configs/voxformer/qpn.py ./ckpts/resnet50-19c8e357.pth 4
