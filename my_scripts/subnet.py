import os
from pathlib import Path
from copy import deepcopy
from natsort import natsorted
import json
from ranger21 import Ranger21
from einops import rearrange

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


import numpy as np
import cv2
from sklearn.metrics import f1_score
import albumentations as A

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, LOGGER
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from transformer import * #TransformerBlockV1, TransformerBlockV2, TransformerBlockV3


def overwrite_key(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


def get_cls_model(fold):
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        "/home/dtpthao/workspace/yolov5/tf{}.pth".format(fold), map_location=torch.device('cpu')
    )["model"]
    model.load_state_dict(overwrite_key(checkpoint))
    model.cuda()
    return model


def get_yolo_model(fold):
    dnn = False
    half = False
    device = ""
    device = select_device(device)
    model = DetectMultiBackend(
        weights="/home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_{}/weights/best.pt".format(fold),  # "/home/htluc/yolov5/runs/train/yolov5s_lesions_fold_0/weights/best.pt",
        device=device,
        dnn=dnn,
        data=None,
        fp16=half,
    )
    imgsz = [480, 480]
    model.warmup(imgsz=(1 if model.pt or model.triton else 4, 3, *imgsz))
    return model


class SubnetDataset(Dataset):
    def __init__(self, root_dir, train_val, transform=None, target_transform=None, fold=0):
        self.root_dir = root_dir
        self.train_val = train_val
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(self.root_dir, "images", "Train_4classes")
        self.label_dict = {"Nor-VF": 0, "Non-VF": 1, "Ben-VF": 2, "Mag-VF": 3}
        self.data = {
            "img_path": [],
            "cls": [],
        }
        self.yolo_model = get_yolo_model(fold=fold)
        self.cls_model = get_cls_model(fold=fold)
        self.yolo_model.eval()
        self.cls_model.eval()
        self.yolo_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.cls_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        with open(
            "/home/dtpthao/workspace/vocal-folds/data/aim/annotations/annotation_0_{}.json".format(
                train_val
            )
        ) as f:
            js = json.load(f)

        self.cls_dict = {}
        for image in js["images"]:
            cls = image["file_name"].split("_")[0]
            filename = image["file_name"].split("_")[-1]
            self.cls_dict[filename] = self.label_dict[cls]

        for ix, img in enumerate(natsorted(os.listdir(self.img_dir))):
            if img not in self.cls_dict.keys():
                continue
            self.data["img_path"].append(os.path.join(self.img_dir, img))
            self.data["cls"].append(self.cls_dict[img])

    def __len__(self):
        return len(self.data["cls"])

    def __getitem__(self, idx):
        img_path, cls = self.data["img_path"][idx], self.data["cls"][idx]
        # Get yolo features and bbox
        img = cv2.imread(img_path)
        x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
        # x = letterbox(img.copy(), (480, 480), auto=True, stride=32)[0]
        x = x.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()
        x = x.half() if self.yolo_model.fp16 else x.float()  # uint8 to fp16/32
        x /= 255
        with torch.no_grad():
            small, medium, large, pred = self.yolo_model(x, feature_map=True)
        pred = non_max_suppression(
            pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False
        )

        # Filter bbox with lesions
        prediction_list = []
        for elem in pred[0].cpu().numpy().tolist():
            prediction_list = []
            cls = int(elem[-1])
            if cls not in [2, 3]:
                continue
            prediction_list.append(elem)
        pred = torch.tensor([prediction_list])

        if pred[0].shape[0] <= 0:
            return None, 1, 1, 1

        x = img.copy()[::-1]
        x = self.cls_transform(x.copy()).unsqueeze(0).cuda()

        # Get classification label and global feature
        with torch.no_grad():
            global_feature = self.cls_model.forward_features(x)

        return (
            pred[0],
            [small[0], medium[0], large[0]],
            global_feature[0],
            cls,  # torch.tensor([cls] * pred[0].shape[0], dtype=torch.long),
        )


class SubnetDataset_new(Dataset):
    
    def __init__(self, root_dir, train_val, fold=0, transform=None, target_transform=None):
        super(SubnetDataset_new, self).__init__()
        
        self.root_dir = root_dir
        self.train_val = train_val
        self.fold = fold
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_dir = os.path.join(self.root_dir, "images", "Train_4classes")
        self.label_dict = {
            "Nor-VF": 0,
            "Non-VF": 1,
            "Ben-VF": 2,
            "Mag-VF": 3,
        }  # set the labels' order

        self.yolo_model = get_yolo_model(fold)
        self.cls_model = get_cls_model(fold)
        self.yolo_model.eval()
        self.cls_model.eval()
        self.yolo_transform = transforms.ToTensor()
        self.cls_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        with open("/home/dtpthao/workspace/vocal-folds/data/aim/annotations/annotation_{}_{}.json".format(fold, train_val)) as f:
            js = json.load(f)
            
        self.data = []
        self.image_id_to_index = {}
        self.categories = js["categories"]

        for i, image in enumerate(js["images"]):
            new_dict = {}
            # Extract the class id from the orignal image name
            cls = image["file_name"].split("_")[0]
            # Get the image name in the image folder
            filename = image["file_name"].split("_")[-1]

            new_dict["id"] = image["id"]
            new_dict["file_name"] = filename
            new_dict["img_path"] = os.path.join(self.img_dir, filename)
            new_dict["bbox"] = []
            new_dict["bbox_categories"] = []
            new_dict["cls"] = self.label_dict[cls]
            self.data.append(new_dict)
            self.image_id_to_index[image["id"]] = i

        # Second loop -> Add bbox
        for annot in js["annotations"]:
            img_index = self.image_id_to_index[annot["image_id"]]
            img_data = self.data[img_index]
            img_data["bbox"].append(annot["bbox"])
            img_data["bbox_categories"].append(annot["category_id"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve data from the dataset at the given index
        data = self.data[idx]

        # Extract image path, bounding boxes, bbox categories, and class ID
        img_path = data["img_path"]
        bboxes = data.get("bbox", [])
        bbox_categories = data.get("bbox_categories", [])
        cls = data["cls"]

        # Load image from path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # If no bounding boxes are found, add a default bounding box
        if not bboxes:
            bboxes.append([0, 0, img.shape[1], img.shape[0]])
            bbox_categories.append(7)

        # LOGGER.info("BBOX before: {}".format(bboxes))
        # Apply image transforms if available
        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes, bbox_categories=bbox_categories)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            bbox_categories = transformed["bbox_categories"]

        # Apply target transforms if available
        if self.target_transform:
            transformed = self.target_transform(image=img, bboxes=bboxes, bbox_categories=bbox_categories)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            bbox_categories = transformed["bbox_categories"]
        # LOGGER.info("BBOX after: {}".format(bboxes))
        # Convert image to YOLO format and extract YOLO features and bounding boxes
        x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).cuda().float().div(255.0)
        with torch.no_grad():
             small, medium, large, pred = self.yolo_model(x, feature_map=True)

        # Convert image to classification format and extract classification features
        x = self.cls_transform(img.copy()).unsqueeze(0).cuda()
        with torch.no_grad():
            global_feature = self.cls_model.forward_features(x)
    
        return (small[0], medium[0], large[0], bboxes, global_feature[0]), cls


def collate_fn(batch):
    list_bbox_tensor = []
    list_local_feature_tuple = []
    list_global_feature_tensor = []
    list_cls = []

    for bbox, local_feature, global_feature, cls in batch:
        if bbox == None:
            continue
        _box = bbox[:, :4].clone()
        _box[:, 2], _box[:, 3] = _box[:, 0] + _box[:, 2], _box[:, 1] + _box[:, 3]
        list_bbox_tensor.append(_box)
        list_local_feature_tuple.append(
            (local_feature[0], local_feature[1], local_feature[2])
        )
        list_global_feature_tensor.append(global_feature)
        list_cls.append(cls)
    if len(list_bbox_tensor) <= 0:
        return None
    return (
        list_bbox_tensor,
        list_local_feature_tuple,
        torch.stack(list_global_feature_tensor),
    ), torch.tensor(list_cls, dtype=torch.long)


def collate_fn_new(batch):
    small_list, medium_list, large_list, bbox_list, global_feature_list, cls_list = [], [], [], [], [], []
    for x, y in batch:
        small_list.append(x[0])
        medium_list.append(x[1])
        large_list.append(x[2])
        _box = torch.tensor(x[3])
        _box[:, 2], _box[:, 3] = _box[:, 0] + _box[:, 2], _box[:, 1] + _box[:, 3]
        bbox_list.append(_box)
        global_feature_list.append(x[4])
        cls_list.append(y)
        
    # Collate tensors within each list
    small_list = default_collate(small_list)
    medium_list = default_collate(medium_list)
    large_list = default_collate(large_list)
    global_feature_list = default_collate(global_feature_list)
    cls_list = default_collate(cls_list)
    return (
        small_list,
        medium_list,
        large_list,
        bbox_list,
        global_feature_list,
    ), cls_list


class SubnetV1(nn.Module):
    def __init__(
        self,
        roip_output_size=(36, 36),
        dim=48,
        num_heads=8,
        bias=False,
        device="cuda:0",
    ):
        super().__init__()
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        
        # Define the network layers
        self.tf = TransformerBlockV1(dim)
        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, 1, 1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, 1, 1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, 1, 1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4)
        )
        
        # self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
        # self.fc = nn.Sequential(
        #     nn.Linear(4096, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        # )

        

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        bbox = [x.to(self.device).float() for x in bboxes]
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()
        # LOGGER.info("BBOX", bbox)
        # Apply ROI pooling
        x0 = roi_pool(
            small_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 8.0,
        )
        x1 = roi_pool(
            medium_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 16.0,
        )
        x2 = roi_pool(
            large_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 32.0,
        )
        global_feature = roi_pool(
            global_embed,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 45.0,
        )
        
        # Stack all local features together
        local_feature = self.conv11_local(torch.cat((x0, x1, x2), dim=1))
        
        # Apply global and local feature fusion
        global_local_feature = torch.cat(
            (local_feature, self.conv11_global_pooling(global_feature)), dim=1
        )

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))
            LOGGER.info("Local+Global: {}".format(global_local_feature.shape))

        global_feature = torch.cat(
            [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
            dim=0
        ).to(self.device)
        
        global_feature = self.conv11_attn_feature(torch.tensor(global_feature))
        x = self.tf(global_local_feature, global_feature, debug=debug)
        # x = self.conv11_fc(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SubnetV2(nn.Module):
    def __init__(
        self,
        roip_output_size=(36, 36),
        dim=48,
        num_heads=8,
        bias=False,
        device="cuda:0",
    ):
        super().__init__()
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        
        # Define the network layers
        self.tf = TransformerBlockV2(dim)
        self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
        self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        bbox = [x.to(self.device).float() for x in bboxes]
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()

        # Apply ROI pooling
        x0 = roi_pool(
            small_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 6.0,
        )
        x1 = roi_pool(
            medium_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 12.0,
        )
        x2 = roi_pool(
            large_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 24.0,
        )

        # Stack all local features together
        local_feature = torch.cat((x0, x1, x2), dim=1)

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))

        # global_feature = []
        # for b, embedding in zip(bbox, global_embed):
        #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
        #     global_feature.append(_global_feature)
        # global_feature = torch.cat(global_feature, dim=0).to(self.device)
        global_feature = torch.cat(
            [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
            dim=0
        ).to(self.device)
        global_feature = self.conv11_global(global_feature)

        x = self.tf(local_feature, global_feature, debug=debug)
        x = self.conv11_fc(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SubnetV3(nn.Module):
    def __init__(
        self,
        roip_output_size=(36, 36),
        dim=48,
        num_heads=8,
        bias=False,
        device="cuda:0",
    ):
        super().__init__()
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        
        # Define the network layers
        self.tf = TransformerBlockV3(dim)
        self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
        self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        bbox = [x.to(self.device).float() for x in bboxes]
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()

        # Apply ROI pooling
        x0 = roi_pool(
            small_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 6.0,
        )
        x1 = roi_pool(
            medium_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 12.0,
        )
        x2 = roi_pool(
            large_conv,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 24.0,
        )

        # Stack all local features together
        local_feature = torch.cat((x0, x1, x2), dim=1)

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))

        # global_feature = []
        # for b, embedding in zip(bbox, global_embed):
        #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
        #     global_feature.append(_global_feature)
        # global_feature = torch.cat(global_feature, dim=0).to(self.device)
        global_feature = torch.cat(
            [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
            dim=0
        ).to(self.device)
        global_feature = self.conv11_global(global_feature)
        
        # local_global_fusion_feature = torch.matmul(local_feature, global_feature) # v1
        local_global_fusion_feature = torch.mul(local_feature, global_feature) # v2

        x = self.tf(local_global_fusion_feature, debug=debug)
        x = self.conv11_fc(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def get_binary_mask(shape, boxes, fill_value=1):
    """
    Generates a binary mask tensor of size `shape` with values set to `fill_value` inside
    the bounding boxes specified by `boxes`, and 0 outside the boxes.
    
    Args:
    - shape (tuple[int, int]): The shape of the output binary mask tensor.
    - boxes (list[list[int]]): A list of bounding box coordinates in the format [x, y, w, h].
    - fill_value (float): The value to fill inside the bounding boxes. Default is 1.
    
    Returns:
    - binary_mask (torch.Tensor): A binary mask tensor of size `shape`.
    """
    binary_mask = torch.zeros(shape)
    for box in boxes:
        x1, y1, x2, y2 = box
        y1 = int(y1 / 360.0 * 480.0)
        y2 = int(y2 / 360.0 * 480.0)
        binary_mask[y1:y2, x1:x2] = fill_value
    return binary_mask


# Replace ROI pooling with Mask SE from Mask2Former
class SubnetV43(nn.Module):
    def __init__(
        self,
        roip_output_size=(36, 36),
        dim=48,
        num_heads=8,
        num_queries=64,
        bias=False,
        device="cuda:0",
    ):
        super().__init__()
        # self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.bias = bias
        self.device = device
        
        # Define the network layers
        self.tf = TransformerBlockV3(dim)
        # self.conv11_small = nn.Conv2d(128, self.dim, kernel_size=1)
        # c2_xavier_fill(self.conv11_small)
        # self.conv11_medium = nn.Conv2d(256, self.dim, kernel_size=1)
        # c2_xavier_fill(self.conv11_medium)
        # self.conv11_large = nn.Conv2d(512, self.dim, kernel_size=1)
        # c2_xavier_fill(self.conv11_large)
        
        self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
        c2_xavier_fill(self.conv11_global)
        self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
        c2_xavier_fill(self.conv11_fc)
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.mce0 = Mask2Former_CA(
            128,
            self.num_heads,
        )
        self.mce1 = Mask2Former_CA(
            256,
            self.num_heads,
        )
        self.mce2 = Mask2Former_CA(
            512,
            self.num_heads,
        )
        N_steps = self.dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.query_feat0 = nn.Embedding(self.num_queries, 128)
        self.query_feat1 = nn.Embedding(self.num_queries, 256)
        self.query_feat2 = nn.Embedding(self.num_queries, 512)
        self.query_embed0 = nn.Embedding(self.num_queries, 128)
        self.query_embed1 = nn.Embedding(self.num_queries, 256)
        self.query_embed2 = nn.Embedding(self.num_queries, 512)
        # self.level_embed = nn.Embedding(1, self.dim)
        # self.resize0 = transforms.Resize((60, 60))
        # self.resize1 = transforms.Resize((30, 30))
        # self.resize2 = transforms.Resize((15, 15))
        # self.decoder_norm = nn.LayerNorm(self.dim)

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        # bbox = [x.to(self.device).float() for x in bboxes]
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()
        
        bs = len(bboxes)
        
        tgt0 = self.query_embed0.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        tgt1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        tgt2 = self.query_embed2.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        
        memory0 = small_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
        memory1 = medium_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
        memory2 = large_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
        
        attn_mask_tensor0 = self.get_attn_mask_tensor(small_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor1 = self.get_attn_mask_tensor(medium_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor2 = self.get_attn_mask_tensor(large_conv.shape[-2:], bboxes).to(self.device).float().detach()

        # pos0 = self.pe_layer(small_conv, None).flatten(2).permute(2, 0, 1)
        # query_pos0 = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
        # print("tgt0", tgt0.shape)
        # print("memory0", small_conv.flatten(2).permute(2, 0, 1).shape)
        # print("attn_mask_tensor0", attn_mask_tensor0.shape)
         
        x0 = self.mce0(
            tgt=tgt0,
            memory=memory0,
            memory_mask=attn_mask_tensor0,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            # pos=pos0, 
            # query_pos=query_pos0
        )
        
        x1 = self.mce1(
            tgt=tgt1,
            memory=memory1,
            memory_mask=attn_mask_tensor1,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            # pos=pos0, 
            # query_pos=query_pos0
        )
        
        x2 = self.mce2(
            tgt=tgt2,
            memory=memory2,
            memory_mask=attn_mask_tensor2,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            # pos=pos0, 
            # query_pos=query_pos0
        )
        x0 = x0.permute(1, 2, 0)
        x1 = x1.permute(1, 2, 0)
        x2 = x2.permute(1, 2, 0)

        # Stack all local features together
        local_feature = torch.cat((x0, x1, x2), dim=1).reshape(-1, 896, 8, 8)

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))
            LOGGER.info("Global embedding: {}".format(global_embed.shape))

        # global_feature = []
        # for b, embedding in zip(bbox, global_embed):
        #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
        #     global_feature.append(_global_feature)
        # global_feature = torch.cat(global_feature, dim=0).to(self.device)
        # global_feature = torch.cat(
        #     [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bboxes, global_embed)], 
        #     dim=0
        # ).to(self.device)
        global_feature = self.conv11_global(global_embed)
        # LOGGER.info("global feature: {}".format(global_feature.shape))
        local_global_fusion_feature = torch.matmul(local_feature, global_feature) # v1
        # local_global_fusion_feature = torch.mul(local_feature, global_feature) # v2

        x = self.tf(local_global_fusion_feature, debug=debug)
        x = self.conv11_fc(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def get_attn_mask_tensor(self, resize, bbox):
        mask_list = []
        for box in bbox:
            box = box.int().tolist()
            mask = get_binary_mask((360, 480), box).unsqueeze(0)
            mask_list.append(mask)
        mask_tensor = torch.stack(mask_list)#.to(self.device).float().detach()
        mask_tensor = F.interpolate(mask_tensor, size=resize, mode="bilinear", align_corners=False)
        mask_tensor = mask_tensor.flatten(2).repeat(1, self.num_heads, self.num_queries, 1).flatten(0, 1)
        return mask_tensor


class SubnetV41(nn.Module):
    def __init__(
        self,
        dim=48,
        num_heads=8,
        num_queries=64,
        bias=False,
        device="cuda:0",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.bias = bias
        self.device = device
        
        # Define the network layers
        self.tf = TransformerBlockV1(dim)
        
        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, 1, 1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, 1, 1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, 1, 1)
        # self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
        
        
        # c2_xavier_fill(self.conv11_fc)
        c2_xavier_fill(self.conv11_local)
        c2_xavier_fill(self.conv11_global_pooling)
        c2_xavier_fill(self.conv11_attn_feature)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4)
        )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(4096, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        # )
        
        self.mce0 = Mask2Former_CA(
            128,
            self.num_heads,
        )
        self.mce1 = Mask2Former_CA(
            256,
            self.num_heads,
        )
        self.mce2 = Mask2Former_CA(
            512,
            self.num_heads,
        )
        self.mce3 = Mask2Former_CA(
            1280,
            self.num_heads,
        )
        N_steps = self.dim // 2
        self.pe_layer0 = PositionEmbeddingSine(128 // 2, normalize=True)
        self.pe_layer1 = PositionEmbeddingSine(256 // 2, normalize=True)
        self.pe_layer2 = PositionEmbeddingSine(512 // 2, normalize=True)
        self.pe_layer3 = PositionEmbeddingSine(1280 // 2, normalize=True)
        self.query_feat0 = nn.Embedding(num_queries, 128)
        self.query_feat1 = nn.Embedding(num_queries, 256)
        self.query_feat2 = nn.Embedding(num_queries, 512)
        self.query_feat3 = nn.Embedding(num_queries, 1280)
        self.query_embed0 = nn.Embedding(num_queries, 128)
        self.query_embed1 = nn.Embedding(num_queries, 256)
        self.query_embed2 = nn.Embedding(num_queries, 512)
        self.query_embed3 = nn.Embedding(num_queries, 1280)

    def get_attn_mask_tensor(self, resize, bbox):
        mask_list = []
        for box in bbox:
            box = box.int().tolist()
            mask = get_binary_mask((480, 480), box).unsqueeze(0)
            mask_list.append(mask)
        mask_tensor = torch.stack(mask_list)#.to(self.device).float().detach()
        mask_tensor = F.interpolate(mask_tensor, size=resize, mode="bilinear", align_corners=False)
        mask_tensor = mask_tensor.flatten(2).repeat(1, self.num_heads, self.num_queries, 1).flatten(0, 1)
        return mask_tensor

    def forward(self, x, debug=False):
        # Unpack the input
        small_conv, medium_conv, large_conv, bboxes, global_embed = x

        # Move the data to the GPU
        large_conv = large_conv.to(self.device).float()
        medium_conv = medium_conv.to(self.device).float()
        small_conv = small_conv.to(self.device).float()
        global_embed = global_embed.to(self.device).float()
        
        bs = len(bboxes)
        
        tgt0 = self.query_feat0.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        tgt1 = self.query_feat1.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        tgt2 = self.query_feat2.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        tgt3 = self.query_feat3.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        # LOGGER.info("tgt: {}".format(tgt0.shape))
        memory0 = small_conv.flatten(2).permute(2, 0, 1)
        memory1 = medium_conv.flatten(2).permute(2, 0, 1)
        memory2 = large_conv.flatten(2).permute(2, 0, 1)
        memory3 = global_embed.flatten(2).permute(2, 0, 1)
        # LOGGER.info("memory: {}".format(memory0.shape))
        attn_mask_tensor0 = self.get_attn_mask_tensor(small_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor1 = self.get_attn_mask_tensor(medium_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor2 = self.get_attn_mask_tensor(large_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor3 = self.get_attn_mask_tensor(global_embed.shape[-2:], bboxes).to(self.device).float().detach()
        # LOGGER.info("attn_mask: {}".format(attn_mask_tensor0.shape))
        query_embed0 = self.query_embed0.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed2 = self.query_embed2.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed3 = self.query_embed3.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        # LOGGER.info("query_embed: {}".format(query_embed0.shape))
        pos0 = self.pe_layer0(small_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos1 = self.pe_layer1(medium_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos2 = self.pe_layer2(large_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos3 = self.pe_layer3(global_embed, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        # LOGGER.info("pos: {}".format(pos0.shape))
        x0 = self.mce0(
            tgt=tgt0,
            memory=memory0,
            memory_mask=attn_mask_tensor0,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos0, 
            query_pos=query_embed0
        )
        
        x1 = self.mce1(
            tgt=tgt1,
            memory=memory1,
            memory_mask=attn_mask_tensor1,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos1, 
            query_pos=query_embed1
        )
        
        x2 = self.mce2(
            tgt=tgt2,
            memory=memory2,
            memory_mask=attn_mask_tensor2,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos2, 
            query_pos=query_embed2
        )
        
        global_feature = self.mce3(
            tgt=tgt3,
            memory=memory3,
            memory_mask=attn_mask_tensor3,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos3, 
            query_pos=query_embed3
        )
        
        x0 = x0.permute(1, 2, 0)
        x1 = x1.permute(1, 2, 0)
        x2 = x2.permute(1, 2, 0)
        global_feature = global_feature.permute(1, 2, 0).reshape(-1, 1280, 8, 8)

        local_feature = self.conv11_local(torch.cat((x0, x1, x2), dim=1).reshape(-1, 896, 8, 8))
        
        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))
            LOGGER.info("Global embedding: {}".format(global_feature.shape))
        
        # Apply global and local feature fusion
        local_global_fusion_feature = torch.cat(
            (local_feature, self.conv11_global_pooling(global_feature)), dim=1
        )
        global_feature = self.conv11_attn_feature(global_feature)

        x = self.tf(local_global_fusion_feature, global_feature)
        # x = self.conv11_fc(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def train_subnet_old(model, params):

    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    # history of loss values in each epoch
    loss_history = {
        "train": [],
        "val": [],
    }
    # history of metric values in each epoch
    metric_history = {
        "train": {"micro": [], "weighted": []},
        "val": {"micro": [], "weighted": []},
    }

    # a deep copy of weights for the best performing model
    best_model_wts = deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float("inf")

    model = model.to(model.device)
    
    def metrics_batch(output, target):
        output = torch.argmax(output, dim=1).cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        weighted_score = f1_score(output, target, average="weighted")
        micro_score = f1_score(output, target, average="micro")
        return [weighted_score, micro_score]


    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            weighted_score, score_score = metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), (weighted_score, score_score)


    def loss_epoch(model, loss_func, dataset_dl, opt=None):
        running_loss = 0.0
        micro_score = 0.0
        weighted_score = 0.0
        count = 0
        for ix, (x, y) in enumerate(dataset_dl):
            bbox, _, _ = x
            ylist = []
            for b, cls in zip(bbox, y):
                ylist.extend([cls] * b.shape[0])
            y = torch.tensor(ylist, dtype=torch.long)
            output = model(x, debug=False)
            y = y.to(model.device)
            # get loss per batch
            loss_b, metric_b = loss_batch(loss_func, output, y, opt)
            # update running loss
            running_loss += loss_b
            # update running metric
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1
        # average loss value
        loss = running_loss / count  # float(len_data)
        # average metric value
        micro_metric = micro_score / count  # float(len_data)
        weighted_metric = weighted_score / count  # float(len_data)
        return loss, (micro_metric, weighted_metric)


    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group["lr"]

    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = get_lr(opt)
        LOGGER.info(
            "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
        )
        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"]["micro"].append(train_metric[0])
        metric_history["train"]["weighted"].append(train_metric[1])

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"]["micro"].append(val_metric[0])
            metric_history["val"]["weighted"].append(val_metric[1])

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            # store weights into a local file
            torch.save(model.state_dict(), path2weights)
            LOGGER.info("Copied best model weights!")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            LOGGER.info("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        LOGGER.info(
            "train loss: {:.6f}, val loss: {:.6f}, f1-score-micro: {:.2f}, f1-score-weighted: {:.2f}".format(
                train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
            )
        )
        LOGGER.info("-" * 10)

    return model, loss_history, metric_history


def train_subnet(model, params):

    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    # history of loss values in each epoch
    loss_history = {
        "train": [],
        "val": [],
    }
    # history of metric values in each epoch
    metric_history = {
        "train": {"micro": [], "weighted": []},
        "val": {"micro": [], "weighted": []},
    }

    # a deep copy of weights for the best performing model
    best_model_wts = deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float("inf")

    model = model.to(model.device)
    
    def metrics_batch(output, target):
        output = torch.argmax(output, dim=1).cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        weighted_score = f1_score(output, target, average="weighted")
        micro_score = f1_score(output, target, average="micro")
        return [weighted_score, micro_score]


    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            weighted_score, score_score = metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), (weighted_score, score_score)


    def loss_epoch(model, loss_func, dataset_dl, opt=None):
        running_loss = 0.0
        micro_score = 0.0
        weighted_score = 0.0
        count = 0
        for ix, (x, y) in enumerate(dataset_dl):
            _, _, _, bbox, _ = x
            ylist = []
            for b, cls in zip(bbox, y):
                ylist.extend([cls] * b.shape[0])
            output = model(x, debug=False)
            y = torch.tensor(ylist, dtype=torch.long).to(model.device)
            # get loss per batch
            loss_b, metric_b = loss_batch(loss_func, output, y, opt)
            # update running loss
            running_loss += loss_b
            # update running metric
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1

        # average loss value
        loss = running_loss / count  # float(len_data)
        # average metric value
        micro_metric = micro_score / count  # float(len_data)
        weighted_metric = weighted_score / count  # float(len_data)
        return loss, (micro_metric, weighted_metric)


    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group["lr"]

    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = get_lr(opt)
        LOGGER.info(
            "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
        )
        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"]["micro"].append(train_metric[0])
        metric_history["train"]["weighted"].append(train_metric[1])

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"]["micro"].append(val_metric[0])
            metric_history["val"]["weighted"].append(val_metric[1])

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            # store weights into a local file
            torch.save(model.state_dict(), path2weights)
            LOGGER.info("Copied best model weights!")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            LOGGER.info("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        LOGGER.info(
            "train loss: {:.6f}, val loss: {:.6f}, f1-score-micro: {:.2f}, f1-score-weighted: {:.2f}".format(
                train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
            )
        )
        LOGGER.info("-" * 10)

    return model, loss_history, metric_history


def train_subnet_v4(model, params):

    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    # history of loss values in each epoch
    loss_history = {
        "train": [],
        "val": [],
    }
    # history of metric values in each epoch
    metric_history = {
        "train": {"micro": [], "weighted": []},
        "val": {"micro": [], "weighted": []},
    }

    # a deep copy of weights for the best performing model
    best_model_wts = deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float("inf")

    model = model.to(model.device)
    
    def metrics_batch(output, target):
        output = torch.argmax(output, dim=1).cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        weighted_score = f1_score(output, target, average="weighted")
        micro_score = f1_score(output, target, average="micro")
        return [weighted_score, micro_score]


    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            weighted_score, score_score = metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), (weighted_score, score_score)


    def loss_epoch(model, loss_func, dataset_dl, opt=None):
        running_loss = 0.0
        micro_score = 0.0
        weighted_score = 0.0
        count = 0
        for ix, (x, y) in enumerate(dataset_dl):
            _, _, _, bbox, _ = x
            output = model(x, debug=False)
            y = y.to(model.device)
            # get loss per batch
            loss_b, metric_b = loss_batch(loss_func, output, y, opt)
            # update running loss
            running_loss += loss_b
            # update running metric
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1
        # average loss value
        loss = running_loss / count  # float(len_data)
        # average metric value
        micro_metric = micro_score / count  # float(len_data)
        weighted_metric = weighted_score / count  # float(len_data)
        return loss, (micro_metric, weighted_metric)


    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group["lr"]

    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = get_lr(opt)
        LOGGER.info(
            "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
        )
        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"]["micro"].append(train_metric[0])
        metric_history["train"]["weighted"].append(train_metric[1])

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"]["micro"].append(val_metric[0])
            metric_history["val"]["weighted"].append(val_metric[1])

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            # store weights into a local file
            torch.save(model.state_dict(), path2weights)
            LOGGER.info("Copied best model weights!")
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            LOGGER.info("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        LOGGER.info(
            "train loss: {:.6f}, val loss: {:.6f}, f1-score-micro: {:.2f}, f1-score-weighted: {:.2f}".format(
                train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
            )
        )
        LOGGER.info("-" * 10)

    return model, loss_history, metric_history


if __name__ == "__main__":
    
    # # opt = Ranger21(model.parameters(), lr=1e-4, num_epochs=100, num_batches_per_epoch=len(train_dl))
    # # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5)
    fold = 0
    transform = A.Compose(
        [
            A.augmentations.crops.transforms.RandomSizedBBoxSafeCrop(360, 480, p=0.3),
            A.augmentations.geometric.transforms.Affine(scale=0.5, translate_percent=0.1, p=0.3),
            A.augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.7, hue=0.015, p=0.3)
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_categories"]),
    )
    
    # train_ds = SubnetDataset_new("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="train", transform=transform, fold=fold)
    # train_dl = DataLoader(
    #     train_ds, batch_size=1, drop_last=False, collate_fn=collate_fn_new, shuffle=True,
    # )
    
    val_ds = SubnetDataset_new("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="val", fold=fold)
    val_dl = DataLoader(
        val_ds, batch_size=1, drop_last=False, collate_fn=collate_fn_new, shuffle=True
    )
    

    # # model = SubnetV1(roip_output_size=(8, 8), dim=896)
    # model = SubnetV41(dim=896)
    # loss_func = nn.CrossEntropyLoss()
    # # loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.5, 1.5]).cuda(), label_smoothing=0.1)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=10, verbose=1
    # )

    # params_train = {
    #     "num_epochs": 100,
    #     "optimizer": opt,
    #     "loss_func": loss_func,
    #     "train_dl": train_dl,
    #     "val_dl": val_dl,
    #     "lr_scheduler": lr_scheduler,
    #     "path2weights": "/home/dtpthao/workspace/yolov5/my_scripts/subnet_v41_no_kaiming_init_100epochs_fold_{}_fix.pt".format(fold),
    # }

    # # model, loss_hist, metric_hist = train_subnet(model, params_train)
    # model, loss_hist, metric_hist = train_subnet_v4(model, params_train)

    # # with open("/home/dtpthao/workspace/yolov5/my_scripts/subnet_v41_gap_ce_fold_{}_loss.json".format(fold), "w") as fp:
    # #     json.dump(loss_hist, fp)

    # # with open("/home/dtpthao/workspace/yolov5/my_scripts/subnet_v41_gap_ce_fold_{}_metric.json".format(fold), "w") as fp:
    # #     json.dump(metric_hist, fp)
    def get_subnet(model_path, v=1):
        if v == 1:
            subnet = SubnetV1(roip_output_size=(8, 8), dim=896)
        elif v == 2:
            subnet = SubnetV2(roip_output_size=(8, 8), dim=896)
        elif v == 3:
            subnet = SubnetV3(roip_output_size=(8, 8), dim=896)
        elif v == 41:
            subnet = SubnetV41(dim=896)
        elif v == 43:
            subnet = SubnetV43(roip_output_size=(8, 8), dim=896)
        subnet.load_state_dict(torch.load(model_path))
        subnet.cuda()
        return subnet
    # from fvcore.nn import FlopCountAnalysis
    model = get_subnet("/home/dtpthao/workspace/yolov5/my_scripts/subnet_v41_gap_100epochs_fold_0_fix.pt", 41)
    # model(next(iter(val_dl))[0])
    from torchsummary import summary
    model = Mask2Former_CA(128, 8)
    # small, medium ,large, bbox, global_embed = next(iter(val_dl))[0]
    # small = small.squeeze(0)
    # medium = medium.squeeze(0)
    # large = large.squeeze(0)
    # bbox = bbox[0][0].unsqueeze(0)
    # global_embed = global_embed.squeeze(0)
    summary(model, [(64, 128), (128, 60, 60), (64, 60, 60), None, (64, 128), (128, 60, 60)])

