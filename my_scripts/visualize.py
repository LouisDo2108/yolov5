import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
from pprint import pprint
from natsort import natsorted

DATA_PATH = Path("/home/htluc/datasets/aim/images/Train_4classes")

BGRforLabel = { 
               1: (205, 117, 149),
               2: (142, 207, 159), 
               3: (0, 255, 255),   
               4: (225, 208, 77),
               5: (107, 121, 0),
               6: (0, 0, 255),
            }

label = { 
               1: "L_Vocal Fold",
               2: "L_Arytenoid cartilage", 
               3: "Benign lesion",   
               4: "Malignant lesion",
               5: "R_Vocal Fold",
               6: "R_Arytenoid cartilage",
            }

gt_json_path = "/home/htluc/yolov5/runs/val/yolov5s_fold_1_val_last/better_annotation_1_val.json"
pred_json_path = "/home/htluc/yolov5/runs/val/yolov5s_fold_1_val_last/better_last_predictions.json"

with open(gt_json_path, "r") as f:
    gt_json = json.load(f)
    
with open(pred_json_path, "r") as f:
    pred_json = json.load(f)

output_path = Path("/home/htluc/yolov5")
filename = 20191010151023

input_name = DATA_PATH / (str(filename)+".jpg")
img = cv2.imread(str(input_name))

for mask in gt_json['annotations']:
    if mask["image_id"] == filename:
        cat = mask['category_id']
        xmin, ymin, w, h = list(map(int, mask['bbox']))
        xmax = w +xmin
        ymax = h +ymin
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), BGRforLabel[int(cat)], 2)
        cv2.putText(img, label[int(cat)], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BGRforLabel[int(cat)], 2)
        
           
output_name = output_path / (str(filename)+"_gt.jpg")
cv2.imwrite(str(output_name), img)
        
input_name = DATA_PATH / (str(filename)+".jpg")
img = cv2.imread(str(input_name))

for mask in pred_json:
    if mask["image_id"] == filename:
        if mask["score"] < 0.25:
            continue
        cat = mask['category_id']
        xmin, ymin, w, h = list(map(int, mask['bbox']))
        # xmin = int(xmin / 480.0 * 480.0)
        # ymin = int(ymin / 480.0 * 360.0)
        # w = int(w / 480.0 * 480.0)
        # h = int(h / 480.0 * 360.0)
        xmax = w +xmin
        ymax = h +ymin
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), BGRforLabel[int(cat)+1], 2)
        cv2.putText(img, label[int(cat)+1], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BGRforLabel[int(cat)+1], 2)
        
output_name = output_path / (str(filename)+"_pred.jpg")
cv2.imwrite(str(output_name), img)
