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

classes = {
  0: "L_Vocal Fold",
  1: "L_Arytenoid cartilage",
  2: "Benign lesion",
  3: "Malignant lesion",
  4: "R_Vocal Fold",
  5: "R_Arytenoid cartilage",
}

import webcolors

STANDARD_COLORS = [
    "DarkGrey", # background
    "LawnGreen", # L_Vocal Fold
    "Crimson", # L_Arytenoid cartilage
    "LightBlue", # Benign lesion
    "Gold", # Malignant lesion
    "DarkViolet", # R_Vocal Fold
    "DarkGreen", # R_Arytenoid cartilage
]

def puttext(img, text):
    return cv2.putText(img, text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2)


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    # result = (
    #     rgb_color.blue / 255.0,
    #     rgb_color.green / 255.0,
    #     rgb_color.red / 255.0,
    # )
    result = (
        rgb_color.blue,
        rgb_color.green,
        rgb_color.red,
    )
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i, _ in enumerate(list_color_name):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
        # standard.append(list_color_name[i])
    return standard
color_list = standard_to_bgr(STANDARD_COLORS)


def parse_pred_from_json(file_name):
    new_dict = {}
    with open(pred_json_path, "r") as f:
        pred_json = json.load(f)
    
    for d in pred_json:
        if not d['image_id'] in new_dict.keys():
            new_dict[d['image_id']] = {
                'category_id': [d['category_id']],
                'bbox': [d['bbox']],
                'score': [d['score']]
            }
        else:
            new_dict[d['image_id']]['category_id'].append(d['category_id'])
            new_dict[d['image_id']]['bbox'].append(d['bbox'])
            new_dict[d['image_id']]['score'].append(d['score'])
    
    return new_dict


def parse_from_pred_dict(file_name):
    boxes, colors, names, scores = [], [], [], []

    if file_name in pred_dict.keys():
        pred = pred_dict[file_name]
    else:
        return False
    
    for i in range(len(pred['category_id'])):
        category = int(pred['category_id'][i])
        score = pred['score'][i]
        if score < 0.25:
            continue
        xmin, ymin, w, h = pred['bbox'][i]
        # xmin = int(float(xmin) / 1920.0 * 480.0)
        # ymin = int(float(ymin) / 1440.0 * 360.0)
        # w = int(float(w) / 1920.0 * 480.0)
        # h = int(float(h) / 1440.0 * 360.0)
        xmax = xmin + w
        ymax = ymin + h
        color = color_list[category+1]
        
        boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        colors.append(color)
        names.append(classes[category])
        scores.append(score)
    return boxes, colors, names, scores


def parse_gt_from_txt(file_name):
    try:
        path = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/labels/"
        txt_path = os.path.join(path, '{}.txt'.format(file_name))
        with open(txt_path, 'r') as f:
            gt = f.read().split('\n')
    except:
        path = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/train/labels/"
        txt_path = os.path.join(path, '{}.txt'.format(file_name))
        with open(txt_path, 'r') as f:
            gt = f.read().split('\n')
    boxes, colors, names = [], [], []
    for box in gt:
        if box == '':
            continue
        else:
            category, x, y, w, h = box.split(' ')
            category = int(category)
            w = int(float(w)*480)
            h = int(float(h)*360)
            xmin = int(float(x)*480 - w/2) 
            ymin = int(float(y)*360 - h/2)
            xmax = xmin + w
            ymax = ymin + h
            color = color_list[category+1]

            boxes.append((xmin, ymin, xmax, ymax))
            colors.append(color)
            names.append(classes[category])
    return boxes, colors, names


def draw_detections(boxes, colors, names, img, scores=None):
    if scores != None:
        for box, color, name, score in zip(boxes, colors, names, scores):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                color, 
                2)
            #new
            tl = int(round(0.001 * max(img.shape[:2]))) if int(round(0.001 * max(img.shape[:2]))) > 1 else 1
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(
                "{} {:.2f}".format(name, score), 0, fontScale=float(tl) / 3, thickness=tf
            )[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + s_size[0], ymin - s_size[1] - 3), color, -1) 
            
            cv2.putText(
                img,
                "{} {:.2f}".format(name, score),
                (xmin, ymin + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(tl) / 3,
                [0, 0, 0],
                thickness=tf,
                lineType=cv2.FONT_HERSHEY_SIMPLEX,
            )
    else:
        for box, color, name in zip(boxes, colors, names):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                color, 
                2)
            #new
            tl = int(round(0.001 * max(img.shape[:2]))) if int(round(0.001 * max(img.shape[:2]))) > 1 else 1
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(
                "{}".format(name), 0, fontScale=float(tl) / 3, thickness=tf
            )[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + s_size[0], ymin - s_size[1] - 3), color, -1) 
            
            cv2.putText(
                img,
                "{}".format(name),
                (xmin, ymin + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(tl) / 3,
                [0, 0, 0],
                thickness=tf,
                lineType=cv2.FONT_HERSHEY_SIMPLEX,
            )
    return img


# pred_json_path = "/home/dtpthao/workspace/yolov5/my_scripts/runs/val/real_fold_0_original_val/best_predictions.json"
# pred_dict = parse_pred_from_json(pred_json_path)

# img_root = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/images"
# output_path = '/home/dtpthao/workspace/yolov5/my_scripts/aim_fold'

# for image in os.listdir(img_root):
#     file_name = int(image[:-4])
#     image_path = os.path.join(img_root, '{}.jpg'.format(file_name))

#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)

#     if parse_from_pred_dict(file_name) != False:
#         boxes, colors, names, scores = parse_from_pred_dict(file_name)
#         pred_img = draw_detections(boxes, colors, names, img.copy(), scores)
#     else:
#         pred_img = img.copy()
#     Image.fromarray(pred_img[...,::-1]).save(os.path.join(output_path, '{}.jpg'.format(file_name)))
    
img_root = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/images"
output_path = '/home/dtpthao/workspace/yolov5/my_scripts/utils_code/blur'

gt_path = '/home/dtpthao/workspace/yolov5/my_scripts/aim_fold/'
blur_10_path = "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/pred/blur_10"
blur_20_path = "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/pred/blur_20"
blur_30_path = "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/pred/blur_30"
blur_40_path = "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/pred/blur_40"

for image in os.listdir(img_root):
    file_name = int(image[:-4])
    image_path = os.path.join(img_root, '{}.jpg'.format(file_name))
    
    img= cv2.imread(image_path)

    meal_result = cv2.imread(os.path.join(gt_path, '{}.jpg'.format(file_name)))
    meal_result = puttext(meal_result, 'MEAL')[...,::-1]
        
    # Row 1.2
    boxes, colors, names = parse_gt_from_txt(file_name)
    gt_img = draw_detections(boxes, colors, names, img.copy())
    gt_result = cv2.resize(gt_img, (480, 360))
    gt_result = puttext(gt_result, "gt")[...,::-1]
    
    blur_10 = cv2.imread(os.path.join(blur_10_path, '{}.jpg'.format(file_name)))
    blur_10 = puttext(blur_10, 'blur 10%')[...,::-1]
    blur_20 = cv2.imread(os.path.join(blur_20_path, '{}.jpg'.format(file_name)))
    blur_20 = puttext(blur_20, 'blur 20%')[...,::-1]
    blur_30 = cv2.imread(os.path.join(blur_30_path, '{}.jpg'.format(file_name)))
    blur_30 = puttext(blur_30, 'blur 30%')[...,::-1]
    blur_40 = cv2.imread(os.path.join(blur_40_path, '{}.jpg'.format(file_name)))
    blur_40 = puttext(blur_40, 'blur 40%')[...,::-1]
        
    # first_row = np.vstack((
    #     np.hstack((img[...,::-1], gt_result[...,::-1], blur_10)),
    #     np.hstack((blur_20, blur_30, blur_40)),
    # ))
    first_row = np.vstack((
        np.hstack((gt_result, meal_result, blur_10)),
        np.hstack((blur_20, blur_30, blur_40)),
    ))
    Image.fromarray(first_row).save(os.path.join(output_path, '{}.jpg'.format(file_name)))
    # break
print(len(os.listdir(output_path)))

# img_root = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/images"
# output_path = '/home/dtpthao/workspace/yolov5/my_scripts/utils_code/esrgan'
# meal_result_path = '/home/dtpthao/workspace/yolov5/my_scripts/aim_fold/'
# esrgan_original_path = '/home/dtpthao/workspace/yolov5/my_scripts/aim_esr/pred/esrgan_original/'
# blur_30_path = "/home/dtpthao/workspace/yolov5/my_scripts/aim_blur/pred/blur_30"
# esrgan_blur_30_path = '/home/dtpthao/workspace/yolov5/my_scripts/aim_esr/pred/esrgan_blur_30'

# for image in os.listdir(img_root):
#     file_name = int(image[:-4])
#     image_path = os.path.join(img_root, '{}.jpg'.format(file_name))

#     img= cv2.imread(image_path)
    
#     meal_result = cv2.imread(os.path.join(meal_result_path, '{}.jpg'.format(file_name)))
#     meal_result = puttext(meal_result, 'MEAL')[...,::-1]
        
#     # Row 1.2
#     boxes, colors, names = parse_gt_from_txt(file_name)
#     gt_img = draw_detections(boxes, colors, names, img.copy())
#     gt_result = cv2.resize(gt_img, (480, 360))
#     gt_result = puttext(gt_result, "gt")[...,::-1]
    
#     esrgan_original_result = cv2.imread(os.path.join(esrgan_original_path, '{}.jpg'.format(file_name)))
#     esrgan_original_result = puttext(esrgan_original_result, 'superres original')[...,::-1]
    
#     blur_30_result = cv2.imread(os.path.join(blur_30_path, '{}.jpg'.format(file_name)))
#     blur_30_result = puttext(blur_30_result, 'blur 30%')[...,::-1]
    
#     esrgan_blur_30_result = cv2.imread(os.path.join(esrgan_blur_30_path, '{}.jpg'.format(file_name)))
#     esrgan_blur_30_result = puttext(esrgan_blur_30_result, 'superres blur 30%')[...,::-1]

#     first_row = np.vstack((
#         np.hstack((img[...,::-1], gt_result, meal_result)),
#         np.hstack((esrgan_original_result, blur_30_result, esrgan_blur_30_result)),
#     ))
#     Image.fromarray(first_row).save(os.path.join(output_path, '{}.jpg'.format(file_name)))

# print(len(os.listdir(output_path)))