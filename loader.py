import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import random
from utils import get_classes, get_train_meta, get_val_meta, draw_cv2
import settings

'''
train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])

test_transforms = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])
'''


class Resize(object):
    def __init__(self, img_sz=(128,128)):
        self.img_sz = img_sz
    def __call__(self, img):
        return cv2.resize(img, self.img_sz)

class HFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.flip(img, 1)
        else:
            return img

class ToTensor(object):
    def __call__(self, img):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img = (img / 255.).astype(np.float32)
        img = np.stack([(img-mean[0])/std[0], (img-mean[1])/std[1], (img-mean[2])/std[2]])
        return img

train_transforms = transforms.Compose([
            HFlip(),
            Resize((128, 128)),
            ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])

test_transforms = transforms.Compose([
            Resize((128, 128)),
            ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # open images mean and std
        ])

class ImageDataset(data.Dataset):
    def __init__(self, df, has_label=True, img_transform=None):
        self.df = df
        self.has_label = has_label
        self.img_transform = img_transform
        
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        img = draw_cv2(df_row.drawing)
        img = self.img_transform(img)  
        if self.has_label:
            return img, df_row.y
        else:
            return img
    def __len__(self):
        return len(self.df)

def get_train_loader(train_index, batch_size=4, dev_mode=False):
    _, stoi = get_classes()
    df = get_train_meta(index=train_index, dev_mode=dev_mode)

    if dev_mode:
        df = df.iloc[:10]

    dset = ImageDataset(df, img_transform=train_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader.num = len(dset)
    return dloader

def get_val_loader(val_num=50, batch_size=4, dev_mode=False):
    _, stoi = get_classes()
    df = get_val_meta(val_num=val_num)

    if dev_mode:
        df = df.iloc[:10]

    dset = ImageDataset(df, img_transform=test_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = len(dset)
    return dloader

def get_test_loader(batch_size=256, dev_mode=False):
    _, stoi = get_classes()
    test_df = pd.read_csv(settings.SAMPLE_SUBMISSION, dtype={'key_id': np.str})

    if dev_mode:
        test_df = test_df.iloc[:10]

    img_dir = settings.TEST_SIMPLIFIED_IMG_DIR
    #print(test_df.head())
    dset = ImageDataset(test_df, img_dir, stoi, has_label=False, img_transform=test_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = len(dset)
    dloader.meta = test_df
    return dloader

def test_train_loader():
    loader = get_train_loader(0, dev_mode=True)
    for img, target in loader:
        print(img.size(), target)
        print(img)

def test_val_loader():
    loader = get_val_loader()
    for img, target in loader:
        print(img.size(), target)
        print(torch.max(img), torch.min(img))

def test_test_loader():
    loader = get_test_loader(dev_mode=True)
    print(loader.num)
    for img in loader:
        print(img.size())

if __name__ == '__main__':
    test_train_loader()
    #test_val_loader()
    #test_test_loader()
