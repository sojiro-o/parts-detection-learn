import cv2
import os
import os.path
import glob
import re
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from albumentations import BboxParams, Compose


def xywh_clip(bboxes):
    """
    画像サイズを超えたとき整形する
    input
        type:np.array
        normalized
        [[xc, yc, w, h],]
    output
        type:np.array
        normalized
        [[xc, yc, w, h],]
    """
    bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
    bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
    bboxes = np.clip(bboxes, 0., 1.)
    bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
    bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
    return bboxes


def xyxy2xywh(x, w, h):
    """
    input
        type:np.array
        [[xmin, ymin, xmax, ymax], ...]
    return
        正規化された
        [[x, y, w, h], ...]
    """
    y = x.copy()
    # print(y, w, h)
    y[:, 0] = ((x[:, 2] + x[:, 0]) / 2) / w
    y[:, 1] = ((x[:, 3] + x[:, 1]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def get_aug(aug, min_area=0., min_visibility=0.):
    """https://albumentations.readthedocs.io/en/latest/_modules/albumentations/core/composition.html
    The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
    The `pascal_voc` format
        `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
    The `albumentations` format
        is like `pascal_voc`, but normalized,
        in other words: [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
    The `yolo` format
        `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
        `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
    "Labels"ならpascal_voc
    "BBoxes"ならyolo
    だけど np.clip(y, 0 ,1) するならalbumentationsもあり
    """
    return Compose(aug, bbox_params=BboxParams(format='yolo', min_area=min_area,min_visibility=min_visibility, label_fields=['category_id']))


class Car_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, transforms, mode, exc_img_list=None):
        """
        args
        dataset_path:
            データセットパス
            例 "./datasets"
        transforms:
            albumenationのCompose済みtransformリスト
        mode:
            "train" or "val"
        exc_img_list:
            dataset_pathのデータセットから除外する画像のリスト
            例 ["006613.jpg", "006349.jpg", ...]

        """
        self.dataset_path = dataset_path
        image_dir_path = os.path.join(dataset_path, "Images")
        bbox_dir_path = os.path.join(dataset_path, "BBoxes")
        # bbox_dir_path = os.path.join(dataset_path, "Labels")
        self.image_dir_path = image_dir_path
        self.bbox_dir_path = bbox_dir_path

        image_path_list = glob.glob(os.path.join(image_dir_path, "*.jpg"))
        val_path_list = glob.glob(os.path.join(image_dir_path, "val*.jpg"))
        
        if exc_img_list is not None:
            for i in exc_img_list:
                image_path_list = [s for s in image_path_list if not re.match('.*{}.*'.format(i.split(".")[0]), s)]
        
        train_path_list = list(set(image_path_list) - set(val_path_list))
        self.train_path_list = train_path_list 
        self.val_path_list = val_path_list
        self.transforms = transforms # albumentaionのComposeのリスト
        self.mode = mode


    def __getitem__(self, index):
        if self.mode=="train":
            image_path = self.train_path_list[index]
        elif self.mode=="val":
            image_path = self.val_path_list[index]
        image = cv2.imread(image_path)
        h, w = image.shape[0], image.shape[1] 
        text_file = image_path.split("/")[-1].split(".")[0] + ".txt"
        bbox_text_path = os.path.join(self.bbox_dir_path, text_file)
        
        category_id = []
        bboxes = []
        with open(bbox_text_path) as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            label_bbox = [round(float(i), 5) for i in line.split("\n")[0].split(" ")]
            label = label_bbox[0]
            bbox = label_bbox[1:5]
            category_id.append(label)
            bboxes.append(bbox)
        bboxes = xywh_clip(np.array(bboxes))

        # albumenation
        # https://github.com/albumentations-team/albumentations/blob/master/notebooks/example_bboxes.ipynb
        annotations = {'image':image, 'bboxes':bboxes, 'category_id':category_id}
        aug = get_aug(self.transforms)
        augmented = aug(**annotations)
        aug_image = augmented["image"]
        aug_bboxes = augmented["bboxes"]
       
        targets = np.concatenate([
            np.array(augmented['category_id']).reshape([-1, 1]),
            np.array(aug_bboxes)
            ], axis=1)
        
        return aug_image, targets


    def __len__(self):
        if self.mode=="train":
            return len(self.train_path_list)
        elif self.mode=="val":
            return len(self.val_path_list)


def batch_idx_fn(batch):
    images, bboxes = list(zip(*batch))
    targets = []
    for idx, bbox in enumerate(bboxes):
        target = np.zeros((len(bbox), 6))
        target[:, 1:] = bbox
        target[:, 0] = idx
        targets.append(target)
    images = torch.stack(images)
    targets = torch.Tensor(np.concatenate(targets)) # [[batch_idx, label, xc, yx, w, h], ...]
    return images, targets


def voc_xml2list(xml, classes):
    # pytroch発展から
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Parameters
    ----------
    xml_path : str
        xmlファイルへのパス。
    classes : list
        labelのリスト
    Returns
    -------
    ret : [[label_ind, xmin, ymin, xmax, ymax], ... ]
    """
    xml = xml["annotation"]
    height, width = int(xml['size']["height"]), int(xml['size']["width"])
    # 画像内の全ての物体のアノテーションをこのリストに格納します
    ret = []
    # 物体一つの場合はリスト化されてない
    if type(xml["object"]) is dict:
        xml["object"] = [xml["object"]]
    # 画像内にある物体（object）の数だけループする
    for obj in xml['object']:
        # アノテーションで検知がdifficultに設定されているものは除外
        # print(obj)
        difficult = int(obj['difficult'])
        if difficult == 1:
            continue
        # 1つの物体に対するアノテーションを格納するリスト
        bndbox = []
        name = obj['name'].lower().strip()  # 物体名
        bbox = obj['bndbox']  # バウンディングボックスの情報
        # アノテーションのクラス名のindexを取得して追加
        label_idx = classes.index(name)
        bndbox.append(label_idx)
        # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        xyxy = [int(bbox[i])-1 for i in pts] # VOCは原点が(1,1)なので1を引き算して（0, 0）
        # 大体のyoloのinputはこれらしい結局中で戻しているけど
        x = ((xyxy[2]+xyxy[0])/2) / width # center_x
        y = ((xyxy[3]+xyxy[1])/2) / height
        w = (xyxy[2]-xyxy[0]) / width
        h = (xyxy[3]-xyxy[1]) / height
        bndbox.extend([x,y,w,h])
        ret.append(bndbox)
    return np.array(ret)  # [[label_ind, xmin, ymin, xmax, ymax], ... ]

if __name__ =="__main__":
    # %cd parts-classify-learn
    import yaml
    from transforms import train_transform, val_transform, get_transforms
    from tqdm import tqdm

    config_path = "../config.yaml"
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)
    dataset_path = config["dataset_path"]
    exc_img_text_paths = config["exc_img_text_paths"]
    exc_img_list = []
    for path in exc_img_text_paths:
        with open(path) as f:
            exc_img_list.extend([i.strip() for i in f.readlines()])

    test_dataset = Car_dataset(dataset_path, get_transforms(config, "train"), mode="train", exc_img_list=exc_img_list)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=batch_idx_fn)
    for batch_idx, (images, target) in enumerate(test_data_loader):
        pass
