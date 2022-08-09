import os
import datetime
import glob
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage


# %cd parts-classify-learn
from utils.utils import (
    non_max_suppression, xywh2xyxy,
    get_batch_statistics, ap_per_class, load_classe_names, weights_init_normal, Logger, bbox_on_img)
from models.yolo_nano import YOLONano
from modules.datasets import Car_dataset, batch_idx_fn
from modules.transforms import train_transform, val_transform
from visualize import visualize_bbox
from matplotlib import pyplot as plt
import cv2
from PIL import Image

from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("val_result_path", type=str)
    args = parser.parse_args()

    val_result_path = args.val_result_path
    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)

    dataset_path = config["dataset_path"]
    val_dataset = Car_dataset(dataset_path, val_transform, mode="val")

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=4, collate_fn=batch_idx_fn)

    device = config["device"]
    model = YOLONano(num_classes=config["model"]["num_classes"], image_size=config["model"]["image_size"]).to(device)
    checkpoint = torch.load(os.path.join(val_result_path, "checkpoints", "best_model.pth"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"], weight_decay=config["optimizer"]["weight_decaty"])
    accumulation_batch_size = config["train"]["accumulation_batch_size"]
    accumulation_steps = accumulation_batch_size // config["train"]["batch_size"]
    epochs = config["epoch"]
    no_result = config["val"]["no_result"]
    log = Logger()
    if not no_result:
        log.open(val_result_path+'/log.val.txt',mode='a')


    # val f1 and map
    labels = []
    sample_metrics = []
    start = datetime.datetime.now()
    for batch_idx, (val_images, val_target) in enumerate(tqdm(val_data_loader)):
        model.eval()
        with torch.no_grad():
            val_images = val_images.to(device)
            val_target = val_target.to(device)
            val_loss, detections = model.forward(val_images, val_target)
        val_target[:, 2:] = xywh2xyxy(val_target[:, 2:])
        val_target[:, 2:] *= config["model"]["image_size"]
        nms_detections = non_max_suppression(detections, conf_thres=config["conf_thres"], nms_thres=config["nms_thres"])
        if not config["val"]["no_map"]:
            sample_metrics += get_batch_statistics(nms_detections, val_target.cpu(), iou_threshold=0.5)
            labels += val_target[:, 1].tolist()

        with open(config["label_txt_path"]) as f:
            category_id_to_name = {i:j.strip() for i,j in enumerate(f.readlines())}
        transform = torchvision.transforms.ToPILImage()
        imgs = [transform(img) for img in val_images.cpu()] # N*H*W*C
        # target = target.cpu()
        board_imgs = []
        for i in range(len(imgs)): # batch_size
            img = np.array(imgs[i])[:, :, ::-1] # RGB
            detection = nms_detections[i]
            board_img = img.copy()
            if detection is not None:
                board_img = bbox_on_img(board_img, detection, category_id_to_name)
            # plt.figure(figsize=(12, 12))
            # plt.imshow(board_img)
            # plt.show()
            board_imgs.append(board_img.transpose(2, 0, 1))
        total_time = str(datetime.datetime.now()-start).split(".")[0]
        log.write("images:{}|total time:{}\n".fotmat(len(val_dataset), total_time))

    if not config["val"]["no_map"]:
        if len(sample_metrics) == 0:
            true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
        else:
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        lavel_count = [labels.count(i) for i in ap_class]
        with open(config["label_txt_path"]) as f:
            category_name = [i.strip() for i in f.readlines()]

        headers = ["class", "precision","recall", "AP", "count"]
        category_name.append("all")
        precision = np.append(precision, np.mean(precision))
        recall = np.append(recall, np.mean(recall))
        AP = np.append(AP, np.mean(AP))
        lavel_count.append(np.sum(lavel_count))

        table = zip(category_name, precision, recall, AP, lavel_count)
        log.write("conf_thres:{}\t|\tnms_thres:{}\n".format(config["conf_thres"], config["nms_thres"]))
        log.write(tabulate(table, headers=headers))