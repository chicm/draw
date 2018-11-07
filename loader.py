import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import random
from utils import get_classes, get_img, get_train_meta
import settings

class ImageDataset(data.Dataset):
    def __init__(self, train_mode, train_meta, dfs, stoi, has_label=True):
        self.train_mode = train_mode
        self.stoi = stoi
        self.train_meta = train_meta
        self.dfs = dfs
        self.has_label = has_label
        
    def __getitem__(self, index):
        df_row = self.train_meta.iloc[index]
        key_id = df_row.key_id
        filename = df_row.filename
        img = get_img(key_id, filename, self.dfs)
        if self.has_label:
            word = df_row.word
            target_name = word.replace(' ', '_')
            return img, self.stoi[target_name]
        else:
            return img


    def __len__(self):
        return len(self.train_meta)

def get_train_loader(batch_size=4):
    _, stoi = get_classes()
    train_meta, dfs = get_train_meta()
    dset = ImageDataset(True, train_meta, dfs, stoi)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader.num = len(dset)
    return dloader

def test_train_loader():
    loader = get_train_loader()
    for img, target in loader:
        print(img.size(), target)

if __name__ == '__main__':
    test_train_loader()