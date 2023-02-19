import os
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from natsort import natsorted
import json

from detect import run

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import roi_pool


import numpy as np
import cv2
from sklearn.metrics import f1_score

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, LOGGER
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def overwrite_key(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


def get_cls_model():
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        "/home/htluc/vocal-folds/checkpoints/cls_tf_effnet_b0_fold_0_best.pth"
    )["model"]
    model.load_state_dict(overwrite_key(checkpoint))
    model.cuda()
    return model


def get_yolo_model():
    dnn = False
    half = False
    device = ""
    device = select_device(device)
    model = DetectMultiBackend(
        weights="/home/htluc/yolov5/runs/train/yolov5s_fold_0/weights/best.pt", #"/home/htluc/yolov5/runs/train/yolov5s_lesions_fold_0/weights/best.pt",
        device=device,
        dnn=dnn,
        data=None,
        fp16=half,
    )
    imgsz = [480, 480]
    model.warmup(imgsz=(1 if model.pt or model.triton else 4, 3, *imgsz))
    return model


class SubnetDataset(Dataset):
    def __init__(self, root_dir, train_val, transform=None, target_transform=None):
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
        self.yolo_model = get_yolo_model()
        self.cls_model = get_cls_model()
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
            "/home/htluc/datasets/aim/annotations/annotation_0_{}.json".format(
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
        x = letterbox(img.copy(), (480, 480), auto=True, stride=32)[0]
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
    return (list_bbox_tensor, list_local_feature_tuple, torch.stack(list_global_feature_tensor)), torch.tensor(list_cls, dtype=torch.long)


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
        from transformer import Attention, TransformerBlock
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        self.tf = TransformerBlock(dim)
        self.conv11_local = nn.Conv2d(512 + 256 + 128, 448, 1, 1)
        self.conv11_global_pooling = nn.Conv2d(1280, 448, 1, 1)
        self.conv11_attn_feature = nn.Conv2d(1280, 896, 1, 1)
        self.conv11_fc = nn.Conv2d(896, 64, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x, debug=False):
        bbox, local_embed, global_embed = x

        bbox = [x.to(self.device) for x in bbox]
        large_conv = torch.stack([x[2].to(self.device) for x in local_embed])
        medium_conv = torch.stack([x[1].to(self.device) for x in local_embed])
        small_conv = torch.stack([x[0].to(self.device) for x in local_embed])
        global_embed = global_embed.to(self.device)

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

        global_feature = roi_pool(
            global_embed,
            boxes=bbox,
            output_size=self.roip_output_size,
            spatial_scale=1 / 45.0,
        )
        
        local_feature = self.conv11_local(torch.cat((x0, x1, x2), dim=1))
        global_local_feature = torch.cat((local_feature, self.conv11_global_pooling(global_feature)), dim=1)

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))
            LOGGER.info("Local+Global: {}".format(global_local_feature.shape))
            
        global_feature = []
        for b, embedding in zip(bbox, global_embed):
            _global_feature = embedding.expand(
                b.shape[0], -1, -1, -1
            )
            global_feature.append(_global_feature)
        global_feature = torch.cat(global_feature, dim=0).to(self.device)
        global_feature = self.conv11_attn_feature(torch.tensor(global_feature))
        x = self.tf(global_local_feature, global_feature, debug=debug)
        x = self.conv11_fc(x)
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
        from transformerv2 import Attention, TransformerBlock
        self.roip_output_size = roip_output_size
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.device = device
        self.tf = TransformerBlock(dim)
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
        bbox, local_embed, global_embed = x

        bbox = [x.to(self.device) for x in bbox]
        small_conv = torch.stack([x[0].to(self.device) for x in local_embed])
        medium_conv = torch.stack([x[1].to(self.device) for x in local_embed])
        large_conv = torch.stack([x[2].to(self.device) for x in local_embed])

        global_embed = global_embed.to(self.device)

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

        local_feature = torch.cat((x0, x1, x2), dim=1)

        if debug:
            LOGGER.info("LargeConv: {}".format(large_conv.shape))
            LOGGER.info("MediumConv: {}".format(medium_conv.shape))
            LOGGER.info("SmallConv: {}".format(small_conv.shape))
            LOGGER.info("X0: {}".format(x0.shape))
            LOGGER.info("X1: {}".format(x1.shape))
            LOGGER.info("X2: {}".format(x2.shape))
            LOGGER.info("Local feature: {}".format(local_feature.shape))

        global_feature = []
        for b, embedding in zip(bbox, global_embed):
            _global_feature = embedding.expand(
                b.shape[0], -1, -1, -1
            )
            global_feature.append(_global_feature)
        global_feature = torch.cat(global_feature, dim=0).to(self.device)
        global_feature = self.conv11_global(global_feature)
        
        x = self.tf(local_feature, global_feature, debug=debug)
        x = self.conv11_fc(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


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
    loss = running_loss / count #float(len_data)
    # average metric value
    micro_metric = micro_score / count #float(len_data)
    weighted_metric = weighted_score / count #float(len_data)
    return loss, (micro_metric, weighted_metric)


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


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
        "train": {
            "micro":    [],
            "weighted": []
        },
        "val": {
            "micro":    [],
            "weighted": []
        }
    }

    # a deep copy of weights for the best performing model
    best_model_wts = deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float("inf")

    model = model.to(model.device)

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
    train_ds = SubnetDataset("/home/htluc/datasets/aim/", train_val="train")
    val_ds = SubnetDataset("/home/htluc/datasets/aim", train_val="val")

    train_dl = DataLoader(
        train_ds, batch_size=128, drop_last=True, collate_fn=collate_fn, shuffle=True
    )

    val_dl = DataLoader(
        val_ds, batch_size=128, drop_last=True, collate_fn=collate_fn, shuffle=True
    )

    model = SubnetV1(roip_output_size=(8, 8), dim=896)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=1
    )

    params_train = {
        "num_epochs": 100,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "lr_scheduler": lr_scheduler,
        "path2weights": "/home/htluc/yolov5/subnet_v1.pt",
    }

    model, loss_hist, metric_hist = train_subnet(model, params_train)

    with open('subnet_v1_loss.json', 'w') as fp:
        json.dump(loss_hist, fp)
        
    with open('subnet_v1_metric.json', 'w') as fp:
        json.dump(metric_hist, fp)
