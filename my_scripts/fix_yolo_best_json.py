from pathlib import Path
import os
import numpy as np
from natsort import natsorted
import json

# folder = "lesions"

# gt_json_path = "/home/htluc/datasets/aim/annotations/annotation_0_val.json"
# pred_json_path = "/home/htluc/yolov5/runs/val/yolov5s_{}_fold_0_val/best_predictions.json".format(folder)
# with open(gt_json_path, "r") as f:
#     gt_json = json.load(f)
# with open(pred_json_path, "r") as f:
#     pred_json = json.load(f)

# better_pred_json = {}

# for img in gt_json["images"]:
#     image_id = img["file_name"].split("_")[-1]
#     better_pred_json[image_id] = {
#             "image_name": image_id,
#             "bbox": [],
#             "category_id": [],
#             "score": []
#         }
# for item in pred_json:
    
#     image_id, cls, bbox, score = item.values()
#     image_id = "{}.jpg".format(image_id)

#     if score < 0.001:
#         continue
#     better_pred_json[image_id]["bbox"].append(bbox)
#     better_pred_json[image_id]["category_id"].append(cls)
#     better_pred_json[image_id]["score"].append(score)

# with open('/home/htluc/yolov5/runs/val/yolov5s_{}_fold_0_val/better.json'.format(folder), 'w') as f:
#     json.dump(list(better_pred_json.values()), f)

fold=4
gt_json_path = "/home/htluc/datasets/aim/annotations/annotation_{}_val.json".format(fold)
pred_json_path = "/home/htluc/yolov5/runs/val/yolov5s_fold_{}_val_last/last_predictions.json".format(fold)
with open(gt_json_path, "r") as f:
    gt_json = json.load(f)
with open(pred_json_path, "r") as f:
    pred_json = json.load(f)

result = []
id_dict = {}

for img in gt_json["images"]:
    file_name = int(img["file_name"].split("_")[-1][:-4])
    image_id = int(img["id"])
    id_dict[file_name] = image_id
    # print(file_name)
    # if image_id == 921:
    #     print(file_name)
    
for ix, item in enumerate(pred_json):
    image_id, cls, bbox, score = item.values()
    if score < 0.001:
        continue
    image_id = id_dict[int(image_id)]
    area = bbox[2] * bbox[3] if len(bbox) > 0 else 0.0
    result.append({
        "image_id": image_id,
        "category_id": cls+1,
        "bbox": bbox,
        "score": score,
    })

with open('/home/htluc/yolov5/runs/val/yolov5s_fold_{}_val_last/better_last_predictions.json'.format(fold), 
          'w', encoding='utf-8') as f:
    json.dump(result, f)