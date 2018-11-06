import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import random
import settings


# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
def draw_cv2(raw_strokes, size=256, lw=4, time_color=False):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img = cv2.copyMakeBorder(img,4,4,4,4,cv2.BORDER_CONSTANT)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

class ImageDataset(data.Dataset):
    def __init__(self, train_mode, img_ids, img_dir, classes, stoi, val_index=None, label_names=None, tta_index=0):
        self.input_size = settings.IMG_SZ
        self.train_mode = train_mode
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num = len(img_ids)
        self.classes = classes 
        self.stoi = stoi
        self.label_names = label_names
        if val_index is not None:
            #self.df_class_counts = df_class_counts.set_index('label_code')
            self.val_index = val_index
        self.tta_index = tta_index

    def __getitem__(self, index):
        fn = os.path.join(self.img_dir, '{}.jpg'.format(self.img_ids[index]))
        img = Image.open(fn, 'r')
        img = img.convert('RGB')
        
        if self.train_mode:
            img = train_transforms(img)
        else:
            tta_transform = get_tta_transform(self.tta_index)
            img = tta_transform(img)

        if self.label_names is None:
            return img
        else:
            labels = self.get_label(index)
            return img, labels[0], labels[1]

    def __len__(self):
        return self.num