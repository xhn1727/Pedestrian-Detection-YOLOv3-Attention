import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import ops
from sklearn.model_selection import train_test_split
import albumentations as A

_SPLITS_CACHE = None
IMG_SIZE = 416

def _find_roots(root_dir):
    img_root, ann_root = None, None
    for d, dirs, files in os.walk(root_dir):
        if os.path.basename(d) == "leftImg8bit":
            img_root = d
        if os.path.basename(d) == "gtBboxCityPersons":
            ann_root = d
    if img_root is None or ann_root is None:
        raise RuntimeError("未找到 leftImg8bit / gtBboxCityPersons 目录")
    return img_root, ann_root

def _collect_all_images(img_root):
    imgs = []
    for subset in ["train", "val"]:
        sub = os.path.join(img_root, subset)
        if not os.path.exists(sub):
            continue
        for city in os.listdir(sub):
            city_dir = os.path.join(sub, city)
            if not os.path.isdir(city_dir):
                continue
            for f in os.listdir(city_dir):
                if f.endswith("_leftImg8bit.png"):
                    imgs.append(os.path.join(city_dir, f))
    return imgs

def _has_pedestrian(img_path, ann_root):
    city = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path).replace("_leftImg8bit.png", "")

    for subset in ["train", "val"]:
        p = os.path.join(ann_root, subset, city, f"{base}_gtBboxCityPersons.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                ann = json.load(f)
            return any(o["label"] == "pedestrian" for o in ann["objects"])
    return False

def _filter_pedestrian_images(all_imgs, ann_root):
    return [p for p in all_imgs if _has_pedestrian(p, ann_root)]

def _load_filtered_boxes(img_path, ann_root, vis_thresh=0.75, min_height=120):
    city = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path).replace("_leftImg8bit.png", "")

    # 查找对应 JSON
    ann_path = None
    for subset in ["train", "val"]:
        p = os.path.join(ann_root, subset, city, f"{base}_gtBboxCityPersons.json")
        if os.path.exists(p):
            ann_path = p
            break
    if ann_path is None:
        return []

    with open(ann_path, "r") as f:
        data = json.load(f)

    raw_boxes = []

    # 处理行人标注
    for obj in data["objects"]:
        if obj["label"] != "pedestrian":
            continue

        x, y, w, h = obj["bbox"]

        if "bboxVis" in obj:
            _, _, _, h_vis = obj["bboxVis"]
            vis = h_vis / h
        else:
            vis = 1.0

        if vis <= vis_thresh:
            continue
        if h < min_height:
            continue

        raw_boxes.append([x, y, x + w, y + h])

    if len(raw_boxes) == 0:
        return []

    boxes = torch.tensor(raw_boxes, dtype=torch.float32)
    keep = ops.nms(boxes, torch.ones(len(boxes)), 0.4)
    return boxes[keep].tolist()

def _build_splits(root_dir, val_size=0.1, test_size=0.1, random_state=42):
    img_root, ann_root = _find_roots(root_dir)
    all_imgs = _collect_all_images(img_root)
    valid_imgs = _filter_pedestrian_images(all_imgs, ann_root)

    cleaned = [p for p in valid_imgs if len(_load_filtered_boxes(p, ann_root)) > 0]

    train_val, test = train_test_split(cleaned, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)

    return {
        "train": train,
        "val": val,
        "test": test,
        "ann_root": ann_root
    }

def _get_splits(root_dir):
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        _SPLITS_CACHE = _build_splits(root_dir)
    return _SPLITS_CACHE

class CityPersonsBase(Dataset):
    def __init__(self, root, img_size=416, split="train"):
        splits = _get_splits(root)
        self.img_files = splits[split]
        self.ann_root = splits["ann_root"]
        self.img_size = img_size
        self.split = split

        if split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.4),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                )
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                )
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        gt_boxes = _load_filtered_boxes(img_path, self.ann_root)

        # 裁剪超界框
        clipped_boxes = []
        labels = []
        for xmin, ymin, xmax, ymax in gt_boxes:
            xmin = max(0, min(xmin, W-1))
            xmax = max(0, min(xmax, W-1))
            ymin = max(0, min(ymin, H-1))
            ymax = max(0, min(ymax, H-1))
            if xmax <= xmin or ymax <= ymin:
                continue
            clipped_boxes.append([xmin, ymin, xmax, ymax])
            labels.append(0)

        bboxes = clipped_boxes

        # Albumentations transform
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img_aug = transformed["image"]
        bboxes_aug = transformed["bboxes"]

        # 转为 YOLO 格式
        targets = []
        for xmin, ymin, xmax, ymax in bboxes_aug:
            xc = (xmin + xmax) / 2 / self.img_size
            yc = (ymin + ymax) / 2 / self.img_size
            bw = (xmax - xmin) / self.img_size
            bh = (ymax - ymin) / self.img_size
            targets.append([0, xc, yc, bw, bh])

        return (
            torch.from_numpy(img_aug).permute(2, 0, 1).float(),
            torch.tensor(targets, dtype=torch.float32)
        )

class CityPersonsTrain(CityPersonsBase):
    def __init__(self, root, img_size=416):
        super().__init__(root, img_size, split="train")

class CityPersonsVal(CityPersonsBase):
    def __init__(self, root, img_size=416):
        super().__init__(root, img_size, split="val")

class CityPersonsTest(CityPersonsBase):
    def __init__(self, root, img_size=416):
        super().__init__(root, img_size, split="test")