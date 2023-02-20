#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab3
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=999:00:00

eval "$(conda shell.bash hook)"
conda activate vofo

cd /home/htluc/yolov5

# python train.py \
# --img 480 --batch 32 --epochs 300 \
# --data /home/htluc/datasets/aim_folds/fold_0/annotations/aim_fold_0.yaml \
# --weights yolov5s.pt \
# --name 'yolov5s_fold_0_no_augmentation' \
# --hyp "/home/htluc/yolov5/data/hyps/hyp.no-augmentation.yaml"

# Train YOLOv5s on COCO128 for 3 epochs
# python train.py \
# --img 480 --batch 32 --epochs 300 \
# --data /home/htluc/datasets/aim_benign_malignant_lesions/fold_0/annotations/aim_fold_0.yaml \
# --weights yolov5s.pt \
# --name 'yolov5s_benign_malignant_fold_0' \

# python train.py \
# --img 480 --batch 32 --epochs 300 \
# --data /home/htluc/datasets/aim_lesions/fold_0/annotations/aim_fold_0.yaml \
# --weights yolov5s.pt \
# --name 'yolov5s_lesions_fold_0' \

# python val.py \
# --img 480 --batch 32 \
# --data /home/htluc/datasets/aim_lesions/fold_0/annotations/aim_fold_0.yaml \
# --weights /home/htluc/yolov5/runs/train/yolov5s_lesions_fold_0/weights/best.pt \
# --verbose --save-json \
# --name 'yolov5s_lesions_fold_0_val' \

# python val.py \
# --img 480 --batch 32 \
# --data /home/htluc/datasets/aim_benign_malignant_lesions/fold_0/annotations/aim_fold_0.yaml \
# --weights /home/htluc/yolov5/runs/train/yolov5s_benign_malignant_fold_0/weights/best.pt \
# --verbose \
# --name 'yolov5s_benign_malignant_fold_0_val' \

# python val.py \
# --img 480 --batch 1 \
# --data /home/htluc/datasets/aim_folds/fold_0/annotations/aim_fold_0.yaml \
# --weights /home/htluc/yolov5/runs/train/yolov5s_fold_0/weights/best.pt \
# --verbose \
# --name 'nosubnet' \

python /home/htluc/yolov5/my_scripts/subnet.py