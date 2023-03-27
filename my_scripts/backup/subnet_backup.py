# class SubnetV2(nn.Module):
#     def __init__(
#         self,
#         roip_output_size=(36, 36),
#         dim=48,
#         num_heads=8,
#         bias=False,
#         device="cuda:0",
#     ):
#         super().__init__()
#         self.roip_output_size = roip_output_size
#         self.dim = dim
#         self.num_heads = num_heads
#         self.bias = bias
#         self.device = device
        
#         # Define the network layers
#         self.tf = TransformerBlockV2(dim)
#         self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
#         self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.Linear(512, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4),
#         )

#     def forward(self, x, debug=False):
#         # Unpack the input
#         small_conv, medium_conv, large_conv, bboxes, global_embed = x

#         # Move the data to the GPU
#         bbox = [x.to(self.device).float() for x in bboxes]
#         large_conv = large_conv.to(self.device).float()
#         medium_conv = medium_conv.to(self.device).float()
#         small_conv = small_conv.to(self.device).float()
#         global_embed = global_embed.to(self.device).float()

#         # Apply ROI pooling
#         x0 = roi_pool(
#             small_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 6.0,
#         )
#         x1 = roi_pool(
#             medium_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 12.0,
#         )
#         x2 = roi_pool(
#             large_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 24.0,
#         )

#         # Stack all local features together
#         local_feature = torch.cat((x0, x1, x2), dim=1)

#         if debug:
#             LOGGER.info("LargeConv: {}".format(large_conv.shape))
#             LOGGER.info("MediumConv: {}".format(medium_conv.shape))
#             LOGGER.info("SmallConv: {}".format(small_conv.shape))
#             LOGGER.info("X0: {}".format(x0.shape))
#             LOGGER.info("X1: {}".format(x1.shape))
#             LOGGER.info("X2: {}".format(x2.shape))
#             LOGGER.info("Local feature: {}".format(local_feature.shape))

#         # global_feature = []
#         # for b, embedding in zip(bbox, global_embed):
#         #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
#         #     global_feature.append(_global_feature)
#         # global_feature = torch.cat(global_feature, dim=0).to(self.device)
#         global_feature = torch.cat(
#             [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
#             dim=0
#         ).to(self.device)
#         global_feature = self.conv11_global(global_feature)

#         x = self.tf(local_feature, global_feature, debug=debug)
#         x = self.conv11_fc(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x


# class SubnetV3(nn.Module):
#     def __init__(
#         self,
#         roip_output_size=(36, 36),
#         dim=48,
#         num_heads=8,
#         bias=False,
#         device="cuda:0",
#     ):
#         super().__init__()
#         self.roip_output_size = roip_output_size
#         self.dim = dim
#         self.num_heads = num_heads
#         self.bias = bias
#         self.device = device
        
#         # Define the network layers
#         self.tf = TransformerBlockV3(dim)
#         self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
#         self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.Linear(512, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4),
#         )

#     def forward(self, x, debug=False):
#         # Unpack the input
#         small_conv, medium_conv, large_conv, bboxes, global_embed = x

#         # Move the data to the GPU
#         bbox = [x.to(self.device).float() for x in bboxes]
#         large_conv = large_conv.to(self.device).float()
#         medium_conv = medium_conv.to(self.device).float()
#         small_conv = small_conv.to(self.device).float()
#         global_embed = global_embed.to(self.device).float()

#         # Apply ROI pooling
#         x0 = roi_pool(
#             small_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 6.0,
#         )
#         x1 = roi_pool(
#             medium_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 12.0,
#         )
#         x2 = roi_pool(
#             large_conv,
#             boxes=bbox,
#             output_size=self.roip_output_size,
#             spatial_scale=1 / 24.0,
#         )

#         # Stack all local features together
#         local_feature = torch.cat((x0, x1, x2), dim=1)

#         if debug:
#             LOGGER.info("LargeConv: {}".format(large_conv.shape))
#             LOGGER.info("MediumConv: {}".format(medium_conv.shape))
#             LOGGER.info("SmallConv: {}".format(small_conv.shape))
#             LOGGER.info("X0: {}".format(x0.shape))
#             LOGGER.info("X1: {}".format(x1.shape))
#             LOGGER.info("X2: {}".format(x2.shape))
#             LOGGER.info("Local feature: {}".format(local_feature.shape))

#         # global_feature = []
#         # for b, embedding in zip(bbox, global_embed):
#         #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
#         #     global_feature.append(_global_feature)
#         # global_feature = torch.cat(global_feature, dim=0).to(self.device)
#         global_feature = torch.cat(
#             [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bbox, global_embed)], 
#             dim=0
#         ).to(self.device)
#         global_feature = self.conv11_global(global_feature)
        
#         # local_global_fusion_feature = torch.matmul(local_feature, global_feature) # v1
#         local_global_fusion_feature = torch.mul(local_feature, global_feature) # v2

#         x = self.tf(local_global_fusion_feature, debug=debug)
#         x = self.conv11_fc(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x

# Replace ROI pooling with Mask SE from Mask2Former
# class SubnetV43(nn.Module):
#     def __init__(
#         self,
#         roip_output_size=(36, 36),
#         dim=48,
#         num_heads=8,
#         num_queries=64,
#         bias=False,
#         device="cuda:0",
#     ):
#         super().__init__()
#         # self.roip_output_size = roip_output_size
#         self.dim = dim
#         self.num_heads = num_heads
#         self.num_queries = num_queries
#         self.bias = bias
#         self.device = device
        
#         # Define the network layers
#         self.tf = TransformerBlockV3(dim)
#         # self.conv11_small = nn.Conv2d(128, self.dim, kernel_size=1)
#         # c2_xavier_fill(self.conv11_small)
#         # self.conv11_medium = nn.Conv2d(256, self.dim, kernel_size=1)
#         # c2_xavier_fill(self.conv11_medium)
#         # self.conv11_large = nn.Conv2d(512, self.dim, kernel_size=1)
#         # c2_xavier_fill(self.conv11_large)
        
#         self.conv11_global = nn.Conv2d(1280, 896, kernel_size=1)
#         c2_xavier_fill(self.conv11_global)
#         self.conv11_fc = nn.Conv2d(896, 64, kernel_size=1)
#         c2_xavier_fill(self.conv11_fc)
#         self.fc = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.Linear(512, 32),
#             nn.ReLU(),
#             nn.Linear(32, 4),
#         )
#         self.mce0 = Mask2Former_CA(
#             128,
#             self.num_heads,
#         )
#         self.mce1 = Mask2Former_CA(
#             256,
#             self.num_heads,
#         )
#         self.mce2 = Mask2Former_CA(
#             512,
#             self.num_heads,
#         )
#         N_steps = self.dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
#         self.query_feat0 = nn.Embedding(self.num_queries, 128)
#         self.query_feat1 = nn.Embedding(self.num_queries, 256)
#         self.query_feat2 = nn.Embedding(self.num_queries, 512)
#         self.query_embed0 = nn.Embedding(self.num_queries, 128)
#         self.query_embed1 = nn.Embedding(self.num_queries, 256)
#         self.query_embed2 = nn.Embedding(self.num_queries, 512)
#         # self.level_embed = nn.Embedding(1, self.dim)
#         # self.resize0 = transforms.Resize((60, 60))
#         # self.resize1 = transforms.Resize((30, 30))
#         # self.resize2 = transforms.Resize((15, 15))
#         # self.decoder_norm = nn.LayerNorm(self.dim)

#     def forward(self, x, debug=False):
#         # Unpack the input
#         small_conv, medium_conv, large_conv, bboxes, global_embed = x

#         # Move the data to the GPU
#         # bbox = [x.to(self.device).float() for x in bboxes]
#         large_conv = large_conv.to(self.device).float()
#         medium_conv = medium_conv.to(self.device).float()
#         small_conv = small_conv.to(self.device).float()
#         global_embed = global_embed.to(self.device).float()
        
#         bs = len(bboxes)
        
#         tgt0 = self.query_embed0.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
#         tgt1 = self.query_embed1.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
#         tgt2 = self.query_embed2.weight.unsqueeze(1).repeat(1, bs, 1).to(self.device).float()
        
#         memory0 = small_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
#         memory1 = medium_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
#         memory2 = large_conv.flatten(2).permute(2, 0, 1) # + self.level_embed.weight[0][None, :, None]).permute(2, 0, 1)        
        
#         attn_mask_tensor0 = self.get_attn_mask_tensor(small_conv.shape[-2:], bboxes).to(self.device).float().detach()
#         attn_mask_tensor1 = self.get_attn_mask_tensor(medium_conv.shape[-2:], bboxes).to(self.device).float().detach()
#         attn_mask_tensor2 = self.get_attn_mask_tensor(large_conv.shape[-2:], bboxes).to(self.device).float().detach()

#         # pos0 = self.pe_layer(small_conv, None).flatten(2).permute(2, 0, 1)
#         # query_pos0 = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
#         # print("tgt0", tgt0.shape)
#         # print("memory0", small_conv.flatten(2).permute(2, 0, 1).shape)
#         # print("attn_mask_tensor0", attn_mask_tensor0.shape)
         
#         x0 = self.mce0(
#             tgt=tgt0,
#             memory=memory0,
#             memory_mask=attn_mask_tensor0,
#             memory_key_padding_mask=None,  # here we do not apply masking on padded region
#             # pos=pos0, 
#             # query_pos=query_pos0
#         )
        
#         x1 = self.mce1(
#             tgt=tgt1,
#             memory=memory1,
#             memory_mask=attn_mask_tensor1,
#             memory_key_padding_mask=None,  # here we do not apply masking on padded region
#             # pos=pos0, 
#             # query_pos=query_pos0
#         )
        
#         x2 = self.mce2(
#             tgt=tgt2,
#             memory=memory2,
#             memory_mask=attn_mask_tensor2,
#             memory_key_padding_mask=None,  # here we do not apply masking on padded region
#             # pos=pos0, 
#             # query_pos=query_pos0
#         )
#         x0 = x0.permute(1, 2, 0)
#         x1 = x1.permute(1, 2, 0)
#         x2 = x2.permute(1, 2, 0)

#         # Stack all local features together
#         local_feature = torch.cat((x0, x1, x2), dim=1).reshape(-1, 896, 8, 8)

#         if debug:
#             LOGGER.info("LargeConv: {}".format(large_conv.shape))
#             LOGGER.info("MediumConv: {}".format(medium_conv.shape))
#             LOGGER.info("SmallConv: {}".format(small_conv.shape))
#             LOGGER.info("X0: {}".format(x0.shape))
#             LOGGER.info("X1: {}".format(x1.shape))
#             LOGGER.info("X2: {}".format(x2.shape))
#             LOGGER.info("Local feature: {}".format(local_feature.shape))
#             LOGGER.info("Global embedding: {}".format(global_embed.shape))

#         # global_feature = []
#         # for b, embedding in zip(bbox, global_embed):
#         #     _global_feature = embedding.expand(b.shape[0], -1, -1, -1)
#         #     global_feature.append(_global_feature)
#         # global_feature = torch.cat(global_feature, dim=0).to(self.device)
#         # global_feature = torch.cat(
#         #     [embedding.expand(len(b), -1, -1, -1) for b, embedding in zip(bboxes, global_embed)], 
#         #     dim=0
#         # ).to(self.device)
#         global_feature = self.conv11_global(global_embed)
#         # LOGGER.info("global feature: {}".format(global_feature.shape))
#         local_global_fusion_feature = torch.matmul(local_feature, global_feature) # v1
#         # local_global_fusion_feature = torch.mul(local_feature, global_feature) # v2

#         x = self.tf(local_global_fusion_feature, debug=debug)
#         x = self.conv11_fc(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#         return x

#     def get_attn_mask_tensor(self, resize, bbox):
#         mask_list = []
#         for box in bbox:
#             box = box.int().tolist()
#             mask = get_binary_mask((360, 480), box).unsqueeze(0)
#             mask_list.append(mask)
#         mask_tensor = torch.stack(mask_list)#.to(self.device).float().detach()
#         mask_tensor = F.interpolate(mask_tensor, size=resize, mode="bilinear", align_corners=False)
#         mask_tensor = mask_tensor.flatten(2).repeat(1, self.num_heads, self.num_queries, 1).flatten(0, 1)
#         return mask_tensor

# def train_subnet_old(model, params):

#     num_epochs = params["num_epochs"]
#     loss_func = params["loss_func"]
#     opt = params["optimizer"]
#     train_dl = params["train_dl"]
#     val_dl = params["val_dl"]
#     lr_scheduler = params["lr_scheduler"]
#     path2weights = params["path2weights"]

#     # history of loss values in each epoch
#     loss_history = {
#         "train": [],
#         "val": [],
#     }
#     # history of metric values in each epoch
#     metric_history = {
#         "train": {"micro": [], "weighted": []},
#         "val": {"micro": [], "weighted": []},
#     }

#     # a deep copy of weights for the best performing model
#     best_model_wts = deepcopy(model.state_dict())

#     # initialize best loss to a large value
#     best_loss = float("inf")

#     model = model.to(model.device)
    
#     def metrics_batch(output, target):
#         output = torch.argmax(output, dim=1).cpu().numpy().tolist()
#         target = target.cpu().numpy().tolist()
#         weighted_score = f1_score(output, target, average="weighted")
#         micro_score = f1_score(output, target, average="micro")
#         return [weighted_score, micro_score]


#     def loss_batch(loss_func, output, target, opt=None):
#         loss = loss_func(output, target)
#         with torch.no_grad():
#             weighted_score, score_score = metrics_batch(output, target)
#         if opt is not None:
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         return loss.item(), (weighted_score, score_score)


#     def loss_epoch(model, loss_func, dataset_dl, opt=None):
#         running_loss = 0.0
#         micro_score = 0.0
#         weighted_score = 0.0
#         count = 0
#         for ix, (x, y) in enumerate(dataset_dl):
#             bbox, _, _ = x
#             ylist = []
#             for b, cls in zip(bbox, y):
#                 ylist.extend([cls] * b.shape[0])
#             y = torch.tensor(ylist, dtype=torch.long)
#             output = model(x, debug=False)
#             y = y.to(model.device)
#             # get loss per batch
#             loss_b, metric_b = loss_batch(loss_func, output, y, opt)
#             # update running loss
#             running_loss += loss_b
#             # update running metric
#             if metric_b is not None:
#                 micro_score += metric_b[0]
#                 weighted_score += metric_b[1]
#             count += 1
#         # average loss value
#         loss = running_loss / count  # float(len_data)
#         # average metric value
#         micro_metric = micro_score / count  # float(len_data)
#         weighted_metric = weighted_score / count  # float(len_data)
#         return loss, (micro_metric, weighted_metric)


#     def get_lr(opt):
#         for param_group in opt.param_groups:
#             return param_group["lr"]

#     # main loop
#     for epoch in range(num_epochs):
#         # get current learning rate
#         current_lr = get_lr(opt)
#         LOGGER.info(
#             "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
#         )
#         # train model on training dataset
#         model.train()
#         train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
#         # collect loss and metric for training dataset
#         loss_history["train"].append(train_loss)
#         metric_history["train"]["micro"].append(train_metric[0])
#         metric_history["train"]["weighted"].append(train_metric[1])

#         model.eval()
#         with torch.no_grad():
#             val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
#             # collect loss and metric for validation dataset
#             loss_history["val"].append(val_loss)
#             metric_history["val"]["micro"].append(val_metric[0])
#             metric_history["val"]["weighted"].append(val_metric[1])

#         # store best model
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_wts = deepcopy(model.state_dict())
#             # store weights into a local file
#             torch.save(model.state_dict(), path2weights)
#             LOGGER.info("Copied best model weights!")

#         lr_scheduler.step(val_loss)
#         if current_lr != get_lr(opt):
#             LOGGER.info("Loading best model weights!")
#             model.load_state_dict(best_model_wts)

#         LOGGER.info(
#             "train loss: {:.6f}, val loss: {:.6f}, f1-score-micro: {:.2f}, f1-score-weighted: {:.2f}".format(
#                 train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
#             )
#         )
#         LOGGER.info("-" * 10)

#     return model, loss_history, metric_history

# class SubnetDataset(Dataset):
#     def __init__(self, root_dir, train_val, transform=None, target_transform=None, fold=0):
#         self.root_dir = root_dir
#         self.train_val = train_val
#         self.transform = transform
#         self.target_transform = target_transform
#         self.img_dir = os.path.join(self.root_dir, "images", "Train_4classes")
#         self.label_dict = {"Nor-VF": 0, "Non-VF": 1, "Ben-VF": 2, "Mag-VF": 3}
#         self.data = {
#             "img_path": [],
#             "cls": [],
#         }
#         self.yolo_model = get_yolo_model(fold=fold)
#         self.cls_model = get_cls_model(fold=fold)
#         self.yolo_model.eval()
#         self.cls_model.eval()
#         self.yolo_transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#             ]
#         )
#         self.cls_transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Resize((256, 256)),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )

#         with open(
#             "/home/dtpthao/workspace/vocal-folds/data/aim/annotations/annotation_0_{}.json".format(
#                 train_val
#             )
#         ) as f:
#             js = json.load(f)

#         self.cls_dict = {}
#         for image in js["images"]:
#             cls = image["file_name"].split("_")[0]
#             filename = image["file_name"].split("_")[-1]
#             self.cls_dict[filename] = self.label_dict[cls]

#         for ix, img in enumerate(natsorted(os.listdir(self.img_dir))):
#             if img not in self.cls_dict.keys():
#                 continue
#             self.data["img_path"].append(os.path.join(self.img_dir, img))
#             self.data["cls"].append(self.cls_dict[img])

#     def __len__(self):
#         return len(self.data["cls"])

#     def __getitem__(self, idx):
#         img_path, cls = self.data["img_path"][idx], self.data["cls"][idx]
#         # Get yolo features and bbox
#         img = cv2.imread(img_path)
#         x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
#         # x = letterbox(img.copy(), (480, 480), auto=True, stride=32)[0]
#         x = x.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()
#         x = x.half() if self.yolo_model.fp16 else x.float()  # uint8 to fp16/32
#         x /= 255
#         with torch.no_grad():
#             small, medium, large, pred = self.yolo_model(x, feature_map=True)
#         pred = non_max_suppression(
#             pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False
#         )

#         # Filter bbox with lesions
#         prediction_list = []
#         for elem in pred[0].cpu().numpy().tolist():
#             prediction_list = []
#             cls = int(elem[-1])
#             if cls not in [2, 3]:
#                 continue
#             prediction_list.append(elem)
#         pred = torch.tensor([prediction_list])

#         if pred[0].shape[0] <= 0:
#             return None, 1, 1, 1

#         x = img.copy()[::-1]
#         x = self.cls_transform(x.copy()).unsqueeze(0).cuda()

#         # Get classification label and global feature
#         with torch.no_grad():
#             global_feature = self.cls_model.forward_features(x)

#         return (
#             pred[0],
#             [small[0], medium[0], large[0]],
#             global_feature[0],
#             cls,  # torch.tensor([cls] * pred[0].shape[0], dtype=torch.long),
#         )

# def collate_fn(batch):
#     list_bbox_tensor = []
#     list_local_feature_tuple = []
#     list_global_feature_tensor = []
#     list_cls = []

#     for bbox, local_feature, global_feature, cls in batch:
#         if bbox == None:
#             continue
#         _box = bbox[:, :4].clone()
#         _box[:, 2], _box[:, 3] = _box[:, 0] + _box[:, 2], _box[:, 1] + _box[:, 3]
#         list_bbox_tensor.append(_box)
#         list_local_feature_tuple.append(
#             (local_feature[0], local_feature[1], local_feature[2])
#         )
#         list_global_feature_tensor.append(global_feature)
#         list_cls.append(cls)
#     if len(list_bbox_tensor) <= 0:
#         return None
#     return (
#         list_bbox_tensor,
#         list_local_feature_tuple,
#         torch.stack(list_global_feature_tensor),
#     ), torch.tensor(list_cls, dtype=torch.long)