from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    ShiftScaleRotate,
    HueSaturationValue,
    RandomContrast,
    RandomBrightness,
    Compose
)
# from src.modules.datasets import Car_dataset, batch_idx_fn


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    # x_min, y_min, x_max, _ = bbox
    x_min, y_min, x_max, y_max = [int(i) for i in bbox]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def img_path2anno(image_path, transform):
    image = cv2.imread(image_path)
    base_path = image_path.split("Images")[0]
    bbox_dir_path = base_path + "Labels"
    text_file = image_path.split("/")[-1].split(".")[0] + ".txt"
    bbox_text_path = os.path.join(bbox_dir_path, text_file)
    category_id = []
    bboxes = []
    with open(bbox_text_path) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if i == 0:
            continue
        line = lines[i]
        label_bbox = [round(float(i), 5) for i in line.split("\n")[0].split(" ")]
        label = label_bbox[4]
        bbox = label_bbox[:4]
        print(bbox)
        category_id.append(label)
        bboxes.append(bbox)

    annotations = {'image':image, 'bboxes':bboxes, 'category_id':category_id}
    augmented = transform(**annotations)
    return augmented

def img_bbox_show(image_path, transform, label_txt_path):
    annotations = img_path2anno(image_path, transform)
    with open(label_txt_path) as f:
        category_id_to_name = {i:j.strip() for i,j in enumerate(f.readlines())}
    visualize(annotations, category_id_to_name)

if __name__ =="__main__":
    image_path = "../datasets/Images/hogehoge.jpg"
    label_txt_path = "../datasets/label.txt"
    transform = Compose([
        # HorizontalFlip(p=0.5),
        # ShiftScaleRotate(rotate_limit=(-30,30), p=0.5),
        # HueSaturationValue(p=0.5),
        # RandomContrast(p=0.5),
        # RandomBrightness(p=0.5),
        Resize(416, 416),
        # ToTensor(),
    ], bbox_params=BboxParams(format='pascal_voc', label_fields=['category_id']))
    img_bbox_show(image_path, transform, label_txt_path)