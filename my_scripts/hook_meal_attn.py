import os
from pathlib import Path
from copy import deepcopy
from natsort import natsorted
import json
from ranger21 import Ranger21
from einops import rearrange
from PIL import Image

import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torchvision.ops import roi_pool

import numpy as np
import cv2
from sklearn.metrics import f1_score
import albumentations as A
from tqdm import tqdm 
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, LOGGER
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from transformer import *
from train_subnet import *

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


def parse_gt_from_txt(file_name):
    try:
        path = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/val/labels/"
        txt_path = os.path.join(path, file_name+'.txt')
        with open(txt_path, 'r') as f:
            gt = f.read().split('\n')
    except:
        path = "/home/dtpthao/workspace/yolov5/aim_folds/fold_0/train/labels/"
        txt_path = os.path.join(path, file_name+'.txt')
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


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)
        #new
        tl = 1 #int(round(0.001 * max(img.shape[:2])))
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(
            f"{name}", 0, fontScale=float(tl) / 3, thickness=tf
        )[0]
        cv2.rectangle(img, (xmin, ymin), (xmin + s_size[0] + 15, ymin - s_size[1] - 3), color, -1) 
        
        cv2.putText(
            img,
            f"{name}",
            (xmin, ymin + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(tl) / 3,
            [0, 0, 0],
            thickness=tf,
            lineType=cv2.FONT_HERSHEY_SIMPLEX,
        )
        #new
        # cv2.putText(img, name, (xmin, ymin - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        #             lineType=cv2.LINE_AA)
    return img


def overwrite_key(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


def get_cls_model(fold):
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        "/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/cls/tf{}.pth".format(fold), map_location=torch.device('cpu')
    )["model"]
    model.load_state_dict(overwrite_key(checkpoint))
    # model.cuda()
    return model


def get_yolo_model(fold):
    dnn = False
    half = False
    # device = ""
    # device = select_device(device)
    model = DetectMultiBackend(
        weights="/home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_{}/weights/best.pt".format(fold),  # "/home/htluc/yolov5/runs/train/yolov5s_lesions_fold_0/weights/best.pt",
        device=torch.device('cpu'),
        dnn=dnn,
        data=None,
        fp16=half,
    )
    imgsz = [480, 480]
    model.warmup(imgsz=(1 if model.pt or model.triton else 4, 3, *imgsz))
    return model


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

# a dict to store the activations
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output[1].detach()
    return hook


def norm(x):
    AA = x.view(x.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(*x.shape)
    AA = AA*255.0
    AA = AA.numpy().astype(np.uint8)
    return AA


def superimpose_attn_map(img, attn_map):
    heatmap_img = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.4, img, 0.6, 0)
    return super_imposed_img


if __name__ == "__main__":
    # fold = 0
    # model_name="MEAL"
    # if model_name == "REAL":
    #     path2weights = "/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/subnet_v1_gap_ce_fold_{}_fix.pt".format(fold)
    # else:
    #     path2weights = "/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/subnet_v41_gap_ce_fold_{}_fix.pt".format(fold)


    # model, loss_func, opt, lr_scheduler = init_model(model_name=model_name)
    # print(model)
    
    yolo_model = get_yolo_model(0)
    yolo_model.eval()
    yolo_model.cpu()

    subnet = MEAL(device='cpu')
    subnet.load_state_dict(torch.load("/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/subnet_v41_gap_ce_100epochs_fold_0_fix.pt"))
    subnet.eval()
    subnet.cpu()

    cls_model = get_cls_model(0)
    cls_model.eval()
    cls_model.cpu()
    
    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    root_path = "/home/dtpthao/workspace/vocal-folds/data/aim/images/Train_4classes"
    
    for ix, file_name in tqdm(enumerate(os.listdir(root_path))):
        file_name = file_name[:-4]
        if file_name != '20200713101902':
            continue
        image_path = "/home/dtpthao/workspace/vocal-folds/data/aim/images/Train_4classes/{}.jpg".format(file_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).float().div(255.0)
        with torch.no_grad():
            small, medium, large, preds = yolo_model(x, feature_map=True)

        # Convert image to classification format and extract classification features
        x = cls_transform(img.copy()).unsqueeze(0)
        with torch.no_grad():
            global_feature = cls_model.forward_features(x)
            
        preds = non_max_suppression(preds,
                                    conf_thres=0.001,
                                    iou_thres=0.6,
                                    labels=(),
                                    multi_label=True,
                                    agnostic=False,
                                    max_det=300)
        
        bbox_preds = preds[0].clone()
        # Convert predictions to absolute coordinates
        bbox_preds[:, 2], bbox_preds[:, 3] = bbox_preds[:, 0] + bbox_preds[:, 2], bbox_preds[:, 1] + bbox_preds[:, 3]
        bboxes = [bbox_preds[:, :4]]
        
        output = subnet((small, medium, large, bboxes, global_feature))
    
        h1 = subnet.mce0.multihead_attn.register_forward_hook(getActivation("small"))
        h2 = subnet.mce1.multihead_attn.register_forward_hook(getActivation("medium"))
        h3 = subnet.mce2.multihead_attn.register_forward_hook(getActivation("large"))
        h4 = subnet.mce3.multihead_attn.register_forward_hook(getActivation("global"))
        # h5 = subnet.tf.attn.register_forward_hook(getActivation("eff_tf"))

        output = subnet((small, medium, large, bboxes, global_feature))
        
        activation['small'] = norm(activation['small'][0].reshape((64, 60, -1)))
        activation['medium'] = norm(activation['medium'][0].reshape(( 64, 30, -1)))
        activation['large'] = norm(activation['large'][0].reshape((64, 15, -1)))
        activation['global'] = norm(activation['global'][0].reshape((64, 8, -1)))
        # activation['eff_tf'] = norm(activation['eff_tf'][0])
        
        # Row 1.1
        img = cv2.imread(image_path)
        
        # Row 1.2
        boxes, colors, names = parse_gt_from_txt(file_name)
        gt_img = draw_detections(boxes, colors, names, img.copy())
        gt_result = cv2.resize(gt_img, (480, 360))
        gt_result = puttext(gt_result, "gt")
        
        first_row = np.hstack((img[...,::-1], gt_result[...,::-1], np.zeros((360, 480, 3), dtype=np.uint8), np.zeros((360, 480, 3), dtype=np.uint8)))
        
        output_path = "/home/dtpthao/workspace/yolov5/my_scripts/meal_attn"
        for idx in tqdm(range(64)):
            if idx not in [0, 14]:
                continue
            # output_path = os.path.join("/home/dtpthao/workspace/yolov5/my_scripts/meal_attn", "combine_idx_"+str(idx))
            # Path(output_path).mkdir(parents=True, exist_ok=True)
            activation_small = cv2.resize(activation['small'][idx], (480 ,360), interpolation=cv2.INTER_AREA)
            activation_medium = cv2.resize(activation['medium'][idx], (480 ,360), interpolation=cv2.INTER_AREA)
            activation_large = cv2.resize(activation['large'][idx], (480 ,360), interpolation=cv2.INTER_AREA)
            activation_global = cv2.resize(activation['global'][idx], (480 ,360), interpolation=cv2.INTER_AREA)
        
            # activation['eff_tf'] = cv2.resize(activation['eff_tf'], (480 ,360), interpolation=cv2.INTER_AREA)
            
            # img = cv2.imread(image_path)
            result_small = superimpose_attn_map(img.copy(), activation_small.copy())
            result_small = puttext(result_small, "small-{}".format(idx))
            result_medium = superimpose_attn_map(img.copy(), activation_medium.copy())
            result_medium = puttext(result_medium, "medium-{}".format(idx))
            result_large = superimpose_attn_map(img.copy(), activation_large.copy())
            result_large = puttext(result_large, "large-{}".format(idx))
            result_global = superimpose_attn_map(img.copy(), activation_global.copy())
            result_global = puttext(result_global, "global-{}".format(idx))
            # result_eff_tf = superimpose_attn_map(img.copy(), activation['eff_tf'].copy())
            # result_eff_tf = puttext(result_eff_tf, "eff_tf")
            
            # boxes, colors, names = parse_gt_from_txt(file_name)
            # gt_img = draw_detections(boxes, colors, names, img.copy())
            # gt_result = cv2.resize(gt_img, (480, 360))
            # gt_result = puttext(gt_result, "gt")
            
            result = np.vstack(
                (
                    np.hstack((img[...,::-1], gt_result[...,::-1], result_global[...,::-1])),
                    np.hstack((result_small[...,::-1], result_medium[...,::-1], result_large[...,::-1]))
                    
                )
            )
            Image.fromarray(result).save(os.path.join(output_path, file_name+'idx_{}.jpg'.format(idx)))
            # first_row = np.vstack(
            #     (
            #         first_row,
            #         np.hstack((result_global[...,::-1], result_small[...,::-1], result_medium[...,::-1], result_large[...,::-1]))                    
            #     )
            # )
        
        # Image.fromarray(first_row).save(os.path.join(output_path, file_name+'.jpg'))
            
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()