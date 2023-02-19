from pathlib import Path
import os
import numpy as np
from natsort import natsorted
import json

folder = "lesions"

gt_json_path = "/home/htluc/datasets/aim/annotations/annotation_0_val.json"
pred_json_path = "/home/htluc/yolov5/runs/val/yolov5s_{}_fold_0_val/best_predictions.json".format(folder)
with open(gt_json_path, "r") as f:
    gt_json = json.load(f)
with open(pred_json_path, "r") as f:
    pred_json = json.load(f)

better_pred_json = {}

for img in gt_json["images"]:
    image_id = img["file_name"].split("_")[-1]
    better_pred_json[image_id] = {
            "image_name": image_id,
            "bbox": [],
            "category_id": [],
            "score": []
        }
for item in pred_json:
    
    image_id, cls, bbox, score = item.values()
    image_id = "{}.jpg".format(image_id)

    if score < 0.001:
        continue
    better_pred_json[image_id]["bbox"].append(bbox)
    better_pred_json[image_id]["category_id"].append(cls)
    better_pred_json[image_id]["score"].append(score)

with open('/home/htluc/yolov5/runs/val/yolov5s_{}_fold_0_val/better.json'.format(folder), 'w') as f:
    json.dump(list(better_pred_json.values()), f)