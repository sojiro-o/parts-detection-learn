import os
import datetime
import glob
import yaml
import shutil
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
# from models.yolov3 import YOLOv3
# from models.sin_yolo_v3 import Darknet
from modules.datasets import Car_dataset, batch_idx_fn
from modules.transforms import train_transform, val_transform, get_transforms
from visualize import visualize_bbox
from matplotlib import pyplot as plt
import cv2
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print('Error: apex cannot import')

if __name__ == "__main__":
    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)

    dataset_path = config["dataset_path"]
    exc_img_text_paths = config["exc_img_text_paths"]
    exc_img_list = []
    for path in exc_img_text_paths:
        with open(path) as f:
            exc_img_list.extend([i.strip() for i in f.readlines()])
    train_dataset = Car_dataset(dataset_path, get_transforms(config, "train"), mode="train", exc_img_list=exc_img_list)
    val_dataset = Car_dataset(dataset_path, get_transforms(config, "val"), mode="val", exc_img_list=exc_img_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=4, collate_fn=batch_idx_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=4, collate_fn=batch_idx_fn)

    device = config["device"]
    model = YOLONano(num_classes=config["model"]["num_classes"], image_size=config["model"]["image_size"]).to(device)
    # checkpoint = torch.load("./results/20191217_27_YOLONano/checkpoints/best_model.pth")
    # model.load_state_dict(checkpoint["state_dict"], strict=False)
    # model.apply(weights_init_normal)
    # model = YOLOv3(cfg['MODEL'], ignore_thre=cfg['TRAIN']['IGNORETHRE'])
    # model = Darknet("/PyTorch-YOLOv3/config/yolov3.cfg").to(device)
    # model.load_darknet_weights("/PyTorch-YOLOv3/weights/yolov3.weights")

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"], weight_decay=config["optimizer"]["weight_decaty"])
    if config["fp16"]:
        # https://nvidia.github.io/apex/amp.html
        model, optimizer = amp.initialize(model, optimizer, opt_level="01")
    accumulation_batch_size = config["train"]["accumulation_batch_size"]
    accumulation_steps = accumulation_batch_size // config["train"]["batch_size"]
    epochs = config["train"]["epoch"]

    result_dir = "../results"
    dt = datetime.datetime.now()
    model_id = len(glob.glob(os.path.join(result_dir, "{}{:02}{:02}*".format(dt.year, dt.month, dt.day))))
    result_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
    result_path = os.path.join(result_dir, result_name)

    no_result = config["train"]["no_result"]

    start = datetime.datetime.now()
    log = Logger()
    if not no_result:
        writer = SummaryWriter(log_dir=os.path.join(result_path, "tensorboard"))
        log.open(result_path+'/log.train.txt',mode='a')
        shutil.copy('../config.yaml', result_path)
        shutil.copytree("../src", os.path.join(result_path, "code"))
    best_mAP = 0.0
    for epoch in range(1, epochs+1):
        for batch_idx, (images, target) in enumerate(train_data_loader):

            model.train()
            images = images.to(device)
            target = target.to(device)
            loss, detections = model.forward(images, target)
            loss = loss / accumulation_steps
            if config["fp16"]:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch_idx + 1 ) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if not no_result:
                writer.add_scalar("loss/train_loss", loss.item(), (len(train_data_loader)*(epoch-1)+batch_idx)) #

            # log
            if batch_idx % config["step"]["log"] == 0:
                log.write(
                    '\nTrain Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain_loss: {:>2.4f}'.format(
                        epoch,
                        batch_idx * len(images), len(train_data_loader.dataset), 100. * batch_idx / len(train_data_loader),
                        loss.item())
                    )

                # batch f1 and map
                if not config["train"]["no_batch_metrics"]:
                    target[:, 2:] = xywh2xyxy(target[:, 2:])
                    target[:, 2:] *= config["model"]["image_size"]
                    detections = non_max_suppression(detections, conf_thres=config["conf_thres"], nms_thres=config["nms_thres"]) # [array([[x1, y1, x2, y2, object_conf, class_score, class_pred], ..]), ..]
                    sample_metrics = get_batch_statistics(detections, target.cpu(), iou_threshold=0.5)

                    if len(sample_metrics) == 0:
                        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
                    else:
                        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
                    labels = target[:, 1].tolist()
                    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
                    log.write(
                        "\t|  train_f1:{:.4f}  |  train_mAP:{:.4f}".format(
                            f1.mean(), AP.mean()
                    ))
                    if not no_result:
                        writer.add_scalar("mAP/train_mAP", AP.mean(), (len(train_data_loader)*(epoch-1)+batch_idx))
                log.write('\t|  time: {}  |'.format(str(datetime.datetime.now()-start).split(".")[0]))

            # save
            if batch_idx % config["step"]["save"] == 0:
                checkpoint_path = os.path.join(result_path, "checkpoints")
                save_file = "newest_model.pth"
                states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                }
                # if config["fp16"]:
                #     states["amp"] = amp.state_dict
                if not no_result:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save(states, os.path.join(checkpoint_path, save_file))


            # tensorboard image
            if batch_idx % config["step"]["tensorbaord_img"] == 0:
                with open(config["label_txt_path"]) as f:
                    category_id_to_name = {i:j.strip() for i,j in enumerate(f.readlines())}
                transform = torchvision.transforms.ToPILImage()
                imgs = [transform(img) for img in images.cpu()] # N * H * W * C
                # target = target.cpu()
                board_imgs = []
                for i in range(len(imgs)): # batch_size
                    img = np.array(imgs[i])[:, :, ::-1] #RGB
                    detection = detections[i]
                    board_img = img.copy()
                    if detection is not None:
                        board_img = bbox_on_img(board_img, detection, category_id_to_name)
                    board_imgs.append(board_img)
                board_imgs = cv2.hconcat(board_imgs).transpose(2, 0, 1)
                if not no_result:
                    writer.add_image('train_imgs', board_imgs, (epoch-1)*len(train_data_loader) + batch_idx)
                # https://tensorboardx.readthedocs.io/en/latest/tutorial.html#add-image
                # https://pytorch.org/docs/stable/tensorboard.html

        # val f1 and map
        labels = []
        sample_metrics = []
        for batch_idx, (val_images, val_target) in enumerate(val_data_loader):
            model.eval()
            with torch.no_grad():
                val_images = val_images.to(device)
                val_target = val_target.to(device)
                val_loss, detections = model.forward(val_images, val_target)
                val_target[:, 2:] = xywh2xyxy(val_target[:, 2:])
                val_target[:, 2:] *= config["model"]["image_size"]
                detections = non_max_suppression(detections, conf_thres=config["conf_thres"], nms_thres=config["nms_thres"])
                sample_metrics += get_batch_statistics(detections, val_target.cpu(), iou_threshold=0.5)
                labels += val_target[:, 1].tolist()


        if len(sample_metrics) == 0:
            true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
        else:
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        log.write("\n|  Epoch{}  |  val_loss:{:.4f}  |  val_f1:{:.4f}  |  val_mAP:{:.4f}  |".format(
            epoch, val_loss, f1.mean(), AP.mean()
        ))
        if not no_result:
            writer.add_scalar("loss/val_loss", loss.item(), epoch)
            writer.add_scalar("mAP/val_mAP", f1.mean(), epoch)
            # writer.add_scalar("val_f1", AP.mean(), epoch)
        mAP = AP.mean()
        if best_mAP < mAP:
            log.write("save model")
            best_mAP = mAP
            best_save_file = "best_model.pth"
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            if not no_result:
                torch.save(states, os.path.join(checkpoint_path, best_save_file))
