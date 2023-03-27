import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from natsort import natsorted
import albumentations as A
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torchvision.ops import roi_pool

import timm
from einops import rearrange
from ranger21 import Ranger21

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from transformer import *
from utils.general import non_max_suppression, LOGGER
from utils.torch_utils import select_device
from utils.augmentations import letterbox

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
print("RANK: ", RANK)
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

from train_copy import train as train_yolov5
from train_copy import parse_opt

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
        

def get_cls_model(fold):
    
    def overwrite_key(state_dict):
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
        return state_dict
    
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        "/home/dtpthao/workspace/yolov5/my_scripts/checkpoints/cls/tf{}.pth".format(fold), map_location=torch.device('cpu')
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
        weights="/home/dtpthao/workspace/yolov5/runs/train/yolov5s_fold_{}/weights/best.pt".format(fold),
        device=device,
        dnn=dnn,
        data=None,
        fp16=half,
    )
    imgsz = [480, 480]
    model.warmup(imgsz=(1 if model.pt or model.triton else 4, 3, *imgsz))
    return model


class SubnetDataset(Dataset):	
    	
    def __init__(self, root_dir, train_val, fold=0, transform=None, target_transform=None):	
        super().__init__()	
        self.root_dir = Path("/home/dtpthao/workspace/vocal-folds/data/aim")	
        self.train_val = train_val	
        self.fold = fold	
        self.transform = transform	
        self.target_transform = target_transform	
        self.img_dir = self.root_dir / "images" / "Train_4classes"	
        self.label_dict = {	
            "Nor-VF": 0,	
            "Non-VF": 1,	
            "Ben-VF": 2,	
            "Mag-VF": 3,	
        }	
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
        with open(self.root_dir / "annotations" / f"annotation_{fold}_{train_val}.json", "r") as f:	
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
            
        # Convert image to YOLO format and extract YOLO features and bounding boxes
        x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).cuda().float().div(255.0)
        with torch.no_grad():
             small, medium, large, pred = self.yolo_model(x, feature_map=True)

        # Convert image to classification format and extract classification features
        x = self.cls_transform(img.copy()).unsqueeze(0).cuda()
        with torch.no_grad():
            global_embedding = self.cls_model.forward_features(x)
    
        return (small[0], medium[0], large[0], bboxes, global_embedding[0]), cls
        # return (bboxes, global_embedding[0]), cls
    
    
def collate_fn(batch):
    small_list, medium_list, large_list, bbox_list, global_embedding_list, cls_list = [], [], [], [], [], []
    for x, y in batch:
        small_list.append(x[0])
        medium_list.append(x[1])
        large_list.append(x[2])
        _box = torch.tensor(x[3])
        _box[:, 2], _box[:, 3] = _box[:, 0] + _box[:, 2], _box[:, 1] + _box[:, 3]
        bbox_list.append(_box)
        global_embedding_list.append(x[4])
        cls_list.append(y)
        
    # Collate tensors within each list
    small_list = default_collate(small_list)
    medium_list = default_collate(medium_list)
    large_list = default_collate(large_list)
    global_embedding_list = default_collate(global_embedding_list)
    cls_list = default_collate(cls_list)
    # return (
    #     small_list,
    #     medium_list,
    #     large_list,
    #     bbox_list,
    #     global_embedding_list,
    # ), cls_list
    return (
        bbox_list,
        global_embedding_list,
    ), cls_list


# Roi_Efficient_Transformer_Laryngoscopy
class REAL(nn.Module):
    def __init__(
        self,
        roip_output_size=(8, 8),
        dim=896,
        num_heads=8,
        bias=False,
        device="cuda:0",
        yolo_model=None,
    ):
        super().__init__()
        
        # Define instance variables
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        self.yolo_model = yolo_model
        
        # Define the network layers
        self.tf = TransformerBlockV1(dim)
        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, kernel_size=1, stride=1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, kernel_size=1, stride=1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, kernel_size=1, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4)
        )
        
    def forward(self, x, yolo_imgs, amp, debug=False):
        # Unpack the input
        with torch.cuda.amp.autocast(amp):
            small_conv, medium_conv, large_conv, yolo_pred = self.yolo_model(yolo_imgs, feature_map=True)  # forward
        
        bboxes, global_embed = x
        # small_conv, medium_conv, large_conv, bboxes, global_embed = x

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
        global_embedding = roi_pool(
            global_embed,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 45.0,
        )
        
        # Stack all local embedding together
        local_embedding = self.conv11_local(torch.cat((x0, x1, x2), dim=1))
        
        # Apply global and local embedding fusion (Q, K)
        global_local_embedding = torch.cat(
            (local_embedding, self.conv11_global_pooling(global_embedding)), dim=1
        )

        # Duplicate global embedding for each bounding box to the corresponding image => V
        global_embedding = torch.cat(
            [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
            dim=0
        ).to(self.device)
        global_embedding = self.conv11_attn_feature(torch.tensor(global_embedding))
        
        # Feed Q, K, V into transformer
        x = self.tf(global_local_embedding, global_embedding)
        
        # GAP + FFN
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        if debug:
            LOGGER.info("In debug mode, printing out shapes")
            LOGGER.info("Small Conv:  {}".format(small_conv.shape))
            LOGGER.info("Medium Conv: {}".format(medium_conv.shape))
            LOGGER.info("Large Conv:  {}".format(large_conv.shape))
            LOGGER.info("ROI small:   {}".format(x0.shape))
            LOGGER.info("ROI medium:  {}".format(x1.shape))
            LOGGER.info("ROI large:   {}".format(x2.shape))
            LOGGER.info("Local embed: {}".format(local_embedding.shape))
            LOGGER.info("Global embed:{}".format(global_embedding.shape))
            LOGGER.info("Q, K:        {}".format(global_local_embedding.shape))
            LOGGER.info("V:           {}".format(global_embedding.shape))
            LOGGER.info("Output:      {}".format(x.shape))
        return x, yolo_pred
 

# Masked_Attention_Efficient_Transformer_Laryngoscopy
class MEAL(nn.Module):
    def __init__(
        self,
        dim=896,
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
        c2_xavier_fill(self.conv11_local)
        c2_xavier_fill(self.conv11_global_pooling)
        c2_xavier_fill(self.conv11_attn_feature)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(896, 448),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(448, 4),
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
        self.mce3 = Mask2Former_CA(
            1280,
            self.num_heads,
        )

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
        # Create a list to store the masks for each bounding box
        mask_list = []

        # Iterate over each bounding box
        for box in bbox:
            # Convert the bounding box coordinates to integers and convert to a binary mask
            box = box.int().tolist()
            mask = get_binary_mask((480, 480), box).unsqueeze(0)

            # Add the mask to the list
            mask_list.append(mask)

        # Stack the masks into a tensor and resize
        mask_tensor = torch.stack(mask_list)
        mask_tensor = F.interpolate(
            mask_tensor, size=resize, mode="bilinear", align_corners=False
        )
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
        
        memory0 = small_conv.flatten(2).permute(2, 0, 1)
        memory1 = medium_conv.flatten(2).permute(2, 0, 1)
        memory2 = large_conv.flatten(2).permute(2, 0, 1)
        memory3 = global_embed.flatten(2).permute(2, 0, 1)
        
        attn_mask_tensor0 = self.get_attn_mask_tensor(small_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor1 = self.get_attn_mask_tensor(medium_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor2 = self.get_attn_mask_tensor(large_conv.shape[-2:], bboxes).to(self.device).float().detach()
        attn_mask_tensor3 = self.get_attn_mask_tensor(global_embed.shape[-2:], bboxes).to(self.device).float().detach()

        query_embed0 = self.query_embed0.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed2 = self.query_embed2.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        query_embed3 = self.query_embed3.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        
        pos0 = self.pe_layer0(small_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos1 = self.pe_layer1(medium_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos2 = self.pe_layer2(large_conv, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        pos3 = self.pe_layer3(global_embed, None).flatten(2).permute(2, 0, 1).to(self.device).float()
        
        # Apply masked attention
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
        
        global_embedding = self.mce3(
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
        local_embedding = self.conv11_local(torch.cat((x0, x1, x2), dim=1).reshape(-1, 896, 8, 8))            
        global_embedding = global_embedding.permute(1, 2, 0).reshape(-1, 1280, 8, 8)

        # Apply global and local feature fusion (Q, K)
        global_local_embedding = torch.cat(
            (local_embedding, self.conv11_global_pooling(global_embedding)), dim=1
        )
        global_embedding = self.conv11_attn_feature(global_embedding) # V

        # Feed Q, K, V into transformer
        x = self.tf(global_local_embedding, global_embedding)

        # GAP + FFN
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
    
        if debug:
            LOGGER.info("In debug mode, printing out shapes")
            LOGGER.info("Small Conv:   {}".format(small_conv.shape))
            LOGGER.info("Medium Conv:  {}".format(medium_conv.shape))
            LOGGER.info("Large Conv:   {}".format(large_conv.shape))
            LOGGER.info("M-attn small: {}".format(x0.shape))
            LOGGER.info("M-attn medium:{}".format(x1.shape))
            LOGGER.info("M-attn large: {}".format(x2.shape))
            LOGGER.info("Local embed:  {}".format(local_embedding.shape))
            LOGGER.info("Global embed: {}".format(global_embedding.shape))
            LOGGER.info("Q, K:         {}".format(global_local_embedding.shape))
            LOGGER.info("V:            {}".format(global_embedding.shape))
            LOGGER.info("Output:       {}".format(x.shape))
        return x


def train(model, params, model_name):

    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    optimizer = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    yolo_train_loader = params["yolo_train_dl"]
    yolo_val_loader = params["yolo_val_dl"]
    yolo_compute_loss = params["yolo_loss_func"]
    yolo_scaler = params["yolo_scaler"]
    yolo_opt = params["yolo_opt"]
    yolo_optimizer = params["yolo_optimizer"]
    yolo_hyp = params["yolo_hyp"]
    yolo_scheduler = params["yolo_scheduler"]
    amp = params["amp"]
    yolo_callbacks = params['yolo_callbacks']
    yolo_start_epoch = params['yolo_start_epoch']
    ema = params['yolo_ema']
    best_fitness = params['yolo_best_fitness'],
    nc = params['nc']
    ni = 0
    accumulate = params['accumulate']
    yolo_data_dict = params['yolo_data_dict']
    imgsz = params['yolo_imgsz']
    batch_size = params['yolo_batch_size']

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
    
    t0 = time.time()
    nb = len(yolo_train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = yolo_start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=yolo_opt.patience), False
    yolo_callbacks.run('on_train_start')
    # LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
    #             f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
    #             f"Logging results to {colorstr('bold', save_dir)}\n"
    #             f'Starting training for {epochs} epochs...')
    
    def metrics_batch(output, target):
        output = torch.argmax(output, dim=1).cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        weighted_score = f1_score(output, target, average="weighted")
        micro_score = f1_score(output, target, average="micro")
        return [weighted_score, micro_score]


    def loss_batch(
        loss_func, output, target, 
        yolo_pred, yolo_input,
        last_opt_step, mloss,
        optimizer=None
    ):
        yolo_imgs, yolo_targets, yolo_paths, _ = yolo_input
        # yolov5
        with torch.cuda.amp.autocast(amp):
            yolo_loss, yolo_loss_items = compute_loss(yolo_pred, yolo_targets.to(model.device))  # loss scaled by batch_size
        # yolov5
        
        # REAL
        loss = loss_func(output, target) + scaler.scale(yolo_loss)
        loss.backward()
        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html

        if ni - last_opt_step >= accumulate:
            scaler.unscale_(yolo_optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.yolo_model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(yolo_optimizer)  # optimizer.step
            scaler.update()
            yolo_optimizer.zero_grad()
            if ema:
                ema.update(model.yolo_model)
            last_opt_step = ni

        # Log
        if RANK in {-1, 0}:
            epoch = 0
            epochs = 300
            i = 1
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            # pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
            #                         (f'{epoch}/{epochs - 1}', mem, *mloss, yolo_targets.shape[0], yolo_imgs.shape[-1]))
            yolo_callbacks.run('on_train_batch_end', model.yolo_model, ni, yolo_imgs, yolo_targets, yolo_paths, list(mloss))
            if yolo_callbacks.stop_training:
                return
        with torch.no_grad():
            weighted_score, score_score = metrics_batch(output, target)
        # if optimizer is not None:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        return loss.item(), (weighted_score, score_score)


    def loss_epoch(model, loss_func, dataset_dl, yolo_dl, last_opt_step, optimizer):
        running_loss = 0.0
        micro_score = 0.0
        weighted_score = 0.0
        count = 0
        mloss = torch.zeros(3, device=model.device)

        for ix, ((x, y), z) in enumerate(zip(dataset_dl, yolo_dl)):
            if model_name == "REAL":
                bbox, _ = x
                ylist = []
                for b, cls in zip(bbox, y):
                    ylist.extend([cls] * b.shape[0])
                y = torch.tensor(ylist, dtype=torch.long)
            elif model_name == "MEAL":
                pass
            else:
                print("FATAL ERROR: NO MODEL NAMED {}".format(model_name))
                return
            y = y.to(model.device)
            # Train yolov5 #
            imgs, targets, paths, _ = z
            ni = ix + nb * epoch
            imgs = imgs.to(model.device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            # Train yolov5 #
            output, yolo_pred = model(x, imgs, amp, debug=True)
            # Calculate loss and metric
            loss_b, metric_b = loss_batch(loss_func, output, y, yolo_pred, z, last_opt_step, mloss, optimizer)
            running_loss += loss_b
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1
            break

        # average loss value
        loss = running_loss / count
        # average metric value
        micro_metric = micro_score / count
        weighted_metric = weighted_score / count
        return loss, (micro_metric, weighted_metric)


    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]
    
    # main loop
    for epoch in range(num_epochs):
        yolo_callbacks.run('on_train_batch_start')
        # get current learning rate
        current_lr = get_lr(optimizer)
        LOGGER.info(
            "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
        )
        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, yolo_train_loader, last_opt_step, optimizer)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"]["micro"].append(train_metric[0])
        metric_history["train"]["weighted"].append(train_metric[1])
        
        # YoloV5 Scheduler
        lr = [x['lr'] for x in yolo_optimizer.param_groups]  # for loggers
        yolo_scheduler.step()
        
        if RANK in {-1, 0}:
            # mAP
            yolo_callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model.yolo_model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == num_epochs) or stopper.possible_stop
            if not yolo_opt.noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=yolo_opt,
                                                dataloader=yolo_val_loader,
                                                save_dir=yolo_opt.save_dir,
                                                plots=False,
                                                callbacks=yolo_callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            yolo_callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks
        print('success')
        return
        # model.eval()
        # with torch.no_grad():
        #     val_loss, val_metric = loss_epoch(model, loss_func, val_dl, yolo_val_loader, optimizer)
        #     # collect loss and metric for validation dataset
        #     loss_history["val"].append(val_loss)
        #     metric_history["val"]["micro"].append(val_metric[0])
        #     metric_history["val"]["weighted"].append(val_metric[1])

        # # store best model
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model_wts = deepcopy(model.state_dict())
        #     # store weights into a local file
        #     torch.save(model.state_dict(), path2weights)
        #     LOGGER.info("Copied best model weights!")

        # lr_scheduler.step(val_loss)
        # if current_lr != get_lr(opt):
        #     LOGGER.info("Loading best model weights!")
        #     model.load_state_dict(best_model_wts)

        # LOGGER.info(
        #     "train loss: {:.6f}, val loss: {:.6f}, f1-micro: {:.2f}, f1-weighted: {:.2f}".format(
        #         train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
        #     )
        # )
        # LOGGER.info("-" * 10)

    return model, loss_history, metric_history


def init_data_loaders(train_transforms, val_transforms, batch_size, fold):
    train_ds = SubnetDataset("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="train",
                                 transform=train_transforms, fold=fold)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    val_ds = SubnetDataset("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="val", fold=fold,
                               transform=val_transforms)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    return train_dl, val_dl


def init_model(model_name, yolo_model):
    if model_name == "REAL":
        model = REAL(yolo_model=yolo_model)
    elif model_name == "MEAL":
        model = MEAL()
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.5, 1.5]).cuda(), label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, verbose=1
    )
    return model, loss_func, opt, lr_scheduler


def train_yolov5_main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    return opt, device, callbacks


def get_train_yolo_model(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    return model, train_loader, val_loader, dataset, optimizer, scheduler, hyp,\
        start_epoch, ema, best_fitness, nc, accumulate, data_dict, imgsz, batch_size


if __name__ == "__main__":
    
    opt = parse_opt()
    callbacks=Callbacks()
    opt, device, callbacks = train_yolov5_main(opt, callbacks)
    yolo_model, train_loader, val_loader, dataset, optimizer, scheduler, \
        hyp, start_epoch, ema, best_fitness, nc, accumulate, data_dict, imgsz, batch_size = \
        get_train_yolo_model(opt.hyp, opt, device, callbacks)
    yolo_model.train()
    amp = check_amp(yolo_model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    compute_loss = ComputeLoss(yolo_model)
    fold = 0
    model_name="REAL"
    if model_name == "REAL":
        path2weights = "/home/dtpthao/workspace/yolov5/my_scripts/subnet_v1_gap_ce_fold_{}_fix.pt".format(fold)
    else:
        path2weights = "/home/dtpthao/workspace/yolov5/my_scripts/subnet_v41_gap_ce_fold_{}_fix.pt".format(fold)
    
    train_transforms = A.Compose(
        [
            A.augmentations.crops.transforms.RandomSizedBBoxSafeCrop(360, 480, p=0.3),
            A.augmentations.geometric.transforms.Affine(scale=0.5, translate_percent=0.1, p=0.3),
            A.augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.7, hue=0.015, p=0.3)
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_categories"]),
    )
    val_transforms = None
    train_dl, val_dl = init_data_loaders(
        train_transforms, val_transforms,
        batch_size=2,
        fold=fold,
    )
    
    model, loss_func, optimizer, lr_scheduler = init_model(model_name=model_name, yolo_model=yolo_model)
    params = {
        "num_epochs": 1,
        "optimizer": optimizer,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2weights,
        "yolo_train_dl": train_loader,
        "yolo_val_dl": val_loader,
        "yolo_loss_func": compute_loss,
        "yolo_scaler": scaler,
        "yolo_opt": opt,
        "yolo_optimizer": optimizer,
        "yolo_scheduler": scheduler,
        "yolo_hyp": hyp,
        "amp": amp,
        "yolo_callbacks": callbacks,
        "yolo_start_epoch": start_epoch,
        "yolo_ema": ema,
        "yolo_best_fitness": best_fitness,
        "nc": nc,
        "accumulate": accumulate,
        "yolo_data_dict": data_dict,
        "yolo_imgsz": imgsz,
        "yolo_batch_size": batch_size,
    }
    
    model, loss_hist, metric_hist = train(
        model=model,
        params=params,
        model_name=model_name,
    )