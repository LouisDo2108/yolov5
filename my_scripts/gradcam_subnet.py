import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import timm 
import sys
import os
from pathlib import Path
from sklearn.metrics import f1_score
import albumentations as A
from tqdm import tqdm 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, LOGGER
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from my_scripts.train_subnet import REAL, MEAL

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
    "DarkGrey",
    "LawnGreen",
    "Crimson",
    "LightBlue",
    "Gold",
    "DarkViolet",
    "Lavender",
]

def overwrite_key(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    # result = (
    #     rgb_color.blue / 255.0,
    #     rgb_color.green / 255.0,
    #     rgb_color.red / 255.0,
    # )
    result = (
        rgb_color.red,
        rgb_color.green,
        rgb_color.blue,
    )
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i, _ in enumerate(list_color_name):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
        # standard.append(list_color_name[i])
    return standard
color_list = standard_to_bgr(STANDARD_COLORS)
def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = color_list[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names

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
            h = int(float(h)*480)
            xmin = int(float(x)*480 - w/2) 
            ymin = int(float(y)*480 - h/2)
            xmax = xmin + w
            ymax = ymin + h
            color = color_list[category]

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

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

def get_cls_model(fold):
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        "/home/dtpthao/workspace/yolov5/tf{}.pth".format(fold), map_location=torch.device('cpu')
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

with open("/home/dtpthao/workspace/eigencam_visualize.txt", 'r') as f:
    files = f.read().split('\n')
    
# cls_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((256, 256)),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

    
# yolo_model = get_yolo_model(0)
# yolo_model.eval()
# yolo_model.cpu()

# subnet = SubnetV1(roip_output_size=(8, 8), dim=896, device='cpu')
# # subnet = SubnetV41(dim=896, device='cpu')
# subnet.load_state_dict(torch.load("/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/subnet_v1_gap_100epochs_fold_0_fix.pt"))
# subnet.eval()
# subnet.cpu()
# target_layers = [subnet.conv11_conv11_global_pooling]

# cls_model = get_cls_model(0)
# cls_model.eval()
# cls_model.cpu()

# model = torch.hub.load('ultralytics/yolov5', 'custom', \
#                     path='/home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_0/weights/best.pt', device='cpu')
# model.eval()
# model.cpu()

gt_path = "/home/dtpthao/workspace/gradcam_real_global_v/gt"
cam_path = "/home/dtpthao/workspace/gradcam_real_global_v/cam"
result_path = "/home/dtpthao/workspace/gradcam_real_global_v/result"


# for file_name in files:
root_path = "/home/dtpthao/workspace/vocal-folds/data/aim/images/Train_4classes"
for file_name in tqdm(os.listdir(root_path)):
    file_name = file_name[:-4]
    image_path = "/home/dtpthao/workspace/vocal-folds/data/aim/images/Train_4classes/{}.jpg".format(file_name)
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
    # x = x.transpose((2, 0, 1))
    # x = torch.from_numpy(x).unsqueeze(0).float().div(255.0)
    # with torch.no_grad():
    #     small, medium, large, preds = yolo_model(x, feature_map=True)

    # # Convert image to classification format and extract classification features
    # x = cls_transform(img.copy()).unsqueeze(0)
    # with torch.no_grad():
    #     global_feature = cls_model.forward_features(x)
        
    # preds = non_max_suppression(preds,
    #                             conf_thres=0.001,
    #                             iou_thres=0.6,
    #                             labels=(),
    #                             multi_label=True,
    #                             agnostic=False,
    #                             max_det=300)
    
    # bbox_preds = preds[0].clone()
    # # Convert predictions to absolute coordinates
    # bbox_preds[:, 2], bbox_preds[:, 3] = bbox_preds[:, 0] + bbox_preds[:, 2], bbox_preds[:, 1] + bbox_preds[:, 3]
    # bboxes = [bbox_preds[:, :4]]

    # ----------------------------------------------------------------------------------------------------------#
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (480, 480))
    rgb_img = img.copy()
    # img = np.float32(img) / 255
    # transform = transforms.ToTensor()
    # tensor = transform(img).unsqueeze(0)

    # results = model([rgb_img])

    # boxes, colors, names = parse_detections(results)
    # detections = draw_detections(boxes, colors, names, rgb_img.copy())

    # cam = EigenCAM(subnet, target_layers, use_cuda=False)
    # grayscale_cam = cam((small, medium, large, bboxes, global_feature))[0, :, :]
    # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # cam_result = draw_detections(boxes, colors, names, cam_image.copy())
    # renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)

    boxes, colors, names = parse_gt_from_txt(file_name)
    gt_img = draw_detections(boxes, colors, names, rgb_img.copy())
    
    #   resize to original
    # rgb_img = cv2.resize(rgb_img, (480, 360))
    gt_img = cv2.resize(gt_img, (480, 360))
    # cam_image = cv2.resize(cam_image, (480, 360))
    # cam_result = cv2.resize(cam_result, (480, 360))
    # renormalized_cam_image = cv2.resize(renormalized_cam_image, (480, 360))
    
    gt_result = Image.fromarray(gt_img)
    gt_result.save(os.path.join(gt_path, file_name+'.jpg'))
    
    # cam_result = Image.fromarray(cam_result)
    # cam_result.save(os.path.join(cam_path, file_name+'.jpg'))
    
    # result = Image.fromarray(np.vstack(
    #     (
    #         np.hstack((rgb_img, gt_img)),
    #         np.hstack((cam_image, renormalized_cam_image))
    #     )
    # ))
    # result.save(os.path.join(result_path, file_name+'.jpg'))
    # break
