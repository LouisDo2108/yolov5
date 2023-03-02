from pathlib import Path
import os
import numpy as np
from natsort import natsorted
import json

"""
annotation{
"id": int, "image_id": int, "category_id": int, 
"segmentation": RLE or [polygon], "area": float, 
"bbox": [x,y,width,height], "iscrowd": 0 or 1,
}

categories[{
"id": int, "name": str, "supercategory": str,
}]
"""

fold=4
gt_json_path = "/home/htluc/datasets/aim/annotations/annotation_{}_val.json".format(fold)

with open(gt_json_path, "r") as f:
    gt_json = json.load(f)

for ix, annot in enumerate(gt_json["annotations"]):
    gt_json["annotations"][ix]["iscrowd"] = 0
    gt_json["annotations"][ix]["area"] = annot["bbox"][2] * annot["bbox"][3] if len(annot["bbox"]) > 0 else 0.0

with open('/home/htluc/yolov5/runs/val/yolov5s_fold_{}_val_last/better_annotation_{}_val.json'.format(fold, fold),
          'w', encoding='utf-8') as f:
    json.dump(gt_json, f)