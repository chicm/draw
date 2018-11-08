import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import random
from utils import get_classes, get_train_meta, get_val_meta
import settings

train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])

test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])

class ImageDataset(data.Dataset):
    def __init__(self, df, img_dir, stoi, has_label=True, img_transform=None):
        self.stoi = stoi
        self.img_dir = img_dir
        self.df = df
        self.has_label = has_label
        self.img_transform = img_transform
        
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        key_id = df_row.key_id
        fn = os.path.join(self.img_dir, '{}.jpg'.format(key_id))
        #img = cv2.imread(fn)
        img = Image.open(fn, 'r')
        img = img.convert('RGB')
        img = self.img_transform(img)  
        if self.has_label:
            word = df_row.word
            target_name = word.replace(' ', '_')
            return img, self.stoi[target_name]
        else:
            return img
    def __len__(self):
        return len(self.df)

def get_train_loader(train_index, batch_size=4, dev_mode=False):
    _, stoi = get_classes()
    df, img_dir = get_train_meta(index=train_index)
    dset = ImageDataset(df, img_dir, stoi, img_transform=train_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader.num = len(dset)
    return dloader

def get_val_loader(val_num=50, batch_size=4, dev_mode=False):
    _, stoi = get_classes()
    df, img_dir = get_val_meta(val_num=val_num)
    dset = ImageDataset(df, img_dir, stoi, img_transform=test_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = len(dset)
    return dloader


def test_train_loader():
    loader = get_train_loader(0)
    for img, target in loader:
        print(img.size(), target)

def test_val_loader():
    loader = get_val_loader()
    for img, target in loader:
        print(img.size(), target)
        print(torch.max(img), torch.min(img))

if __name__ == '__main__':
    #test_train_loader()
    test_val_loader()