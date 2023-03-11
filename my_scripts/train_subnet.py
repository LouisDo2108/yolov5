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
    return (
        small_list,
        medium_list,
        large_list,
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
    ):
        super().__init__()
        
        # Define instance variables
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        
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
        return x
 
   
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
            if model_name == "REAL":
                _, _, _, bbox, _ = x
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
            output = model(x, debug=True)
            loss_b, metric_b = loss_batch(loss_func, output, y, opt)
            running_loss += loss_b
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1

        # average loss value
        loss = running_loss / count
        # average metric value
        micro_metric = micro_score / count
        weighted_metric = weighted_score / count
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
            "train loss: {:.6f}, val loss: {:.6f}, f1-micro: {:.2f}, f1-weighted: {:.2f}".format(
                train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
            )
        )
        LOGGER.info("-" * 10)

    return model, loss_history, metric_history


def init_data_loaders(train_transforms, val_transforms, batch_size, fold):
    train_ds = SubnetDataset("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="train",
                                 transform=train_transforms, fold=fold)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    val_ds = SubnetDataset("/home/dtpthao/workspace/vocal-folds/data/aim", train_val="val", fold=fold,
                               transform=val_transforms)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    return train_dl, val_dl


def init_model(model_name):
    if model_name == "REAL":
        model = REAL()
    elif model_name == "MEAL":
        model = MEAL()
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.5, 1.5]).cuda(), label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, verbose=1
    )
    return model, loss_func, opt, lr_scheduler


if __name__ == "__main__":
    
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
        batch_size=256,
        fold=fold,
    )

    model, loss_func, opt, lr_scheduler = init_model(model_name=model_name)
    params = {
        "num_epochs": 70,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "lr_scheduler": lr_scheduler,
        "path2weights": path2weights,
    }
    
    model, loss_hist, metric_hist = train(
        model=model,
        params=params,
        model_name=model_name,
    )
    
    
    