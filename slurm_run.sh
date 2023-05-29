#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=phoenix3
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=999:00:00

eval "$(conda shell.bash hook)"
conda activate vofo

cd /home/dtpthao/workspace/yolov5

# Train
# python train.py \
# --img 480 --batch 64 --epochs 1 \
# --data /home/dtpthao/workspace/yolov5/aim_folds_bigger_box/fold_0/annotations/aim_fold_0.yaml \
# --weights yolov5s.pt \
# --name 'yolov5s_fold_0_bigger' \
# --patience 300

# Validation
# fold=0
# python val.py \
# --img 480 --batch 1 \
# --data /home/dtpthao/workspace/yolov5/my_scripts/aim_esr/annotations/aim_fold_0.yaml \
# --weights /home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_$fold/weights/best.pt \
# --verbose --save-json \
# --name yolov5s_fold_$fold\_esr_blur_20_val_test \

# Subnet Train
# python /home/dtpthao/workspace/yolov5/my_scripts/train_subnet.py

# Subnet Validation
# fold=0
# python /home/dtpthao/workspace/yolov5/my_scripts/val_subnet.py \
# --img 480 --batch 1 \
# --data /home/dtpthao/workspace/yolov5/my_scripts/aim_esr/annotations/aim_fold_0.yaml \
# --weights /home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_$fold/weights/best.pt \
# --verbose \
# --save-json \
# --project "/home/dtpthao/workspace/yolov5/my_scripts/runs/val" \
# --name test

fold=0
python /home/dtpthao/workspace/yolov5/my_scripts/val_subnet.py \
--img 480 --batch 1 \
--data /home/dtpthao/workspace/yolov5/aim_folds/fold_0/annotations/aim_fold_0.yaml \
--weights /home/dtpthao/workspace/yolov5/best.pt \
--verbose \
--save-json \
--project "/home/dtpthao/workspace/yolov5/my_scripts/runs/val" \
--name test