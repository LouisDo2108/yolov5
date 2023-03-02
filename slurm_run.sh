#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab3
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=999:00:00

eval "$(conda shell.bash hook)"
conda activate vofo

cd /home/htluc/yolov5

#Train
# python train.py \
# --img 480 --batch 32 --epochs 300 \
# --data /home/htluc/datasets/aim_folds/fold_0/annotations/aim_fold_0.yaml \
# --weights yolov5s.pt \
# --name 'yolov5s_fold_0_2_stages'

# Validation
# fold=4
# python val.py \
# --img 480 --batch 1 \
# --data /home/htluc/datasets/aim_folds/fold_$fold/annotations/aim_fold_$fold.yaml \
# --weights /home/htluc/yolov5/runs/train/yolov5s_fold_$fold/weights/last.pt \
# --verbose --save-json \
# --name yolov5s_fold_$fold\_val_last \

# Subnet Train
python /home/htluc/yolov5/my_scripts/subnet.py

# Subnet Validation
# python /home/htluc/yolov5/my_scripts/val_new.py \
# --img 480 --batch 1 \
# --data /home/htluc/datasets/aim_folds/fold_0/annotations/aim_fold_0.yaml \
# --weights /home/htluc/yolov5/runs/train/yolov5s_fold_0/weights/best.pt \
# --verbose \
# --project "/home/htluc/yolov5/my_scripts/runs/val" \
# --name 'subnetv1_new_test'