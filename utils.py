import os
import cv2
from PIL import Image
import glob
import numpy as np
import pandas as pd
import time
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
    #img = cv2.copyMakeBorder(img,4,4,4,4,cv2.BORDER_CONSTANT)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

def get_classes():
    df = pd.read_csv('classes.csv')
    classes = df.classes.values.tolist()
    #print(len(classes))
    #print(classes)
    stoi = {classes[i]: i for i in range(len(classes))}
    return classes, stoi


def get_sub_df(df, index):
    start_index = len(df) * index // 100
    end_index = len(df) * (index+1) // 100
    return df.iloc[start_index:end_index]

def get_train_meta(index=0, img_sz=256):
    df_file = os.path.join(settings.DATA_DIR, 'train-{}'.format(img_sz), 'train_{}.csv'.format(index))
    print(df_file)
    df = pd.read_csv(df_file, dtype={'key_id': np.str})
    img_dir = os.path.join(settings.DATA_DIR, 'train-{}'.format(img_sz), 'train_{}'.format(index))
    
    return df, img_dir

def get_val_meta(val_num=50, img_sz=256):
    df_val_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'val_ids_{}.csv'.format(val_num)), dtype={'key_id': np.str})
    img_dir = os.path.join(settings.DATA_DIR, 'val-50-{}'.format(img_sz))

    return df_val_ids, img_dir

def get_img(key_id, filename, dfs):
    #print(type(key_id))
    if filename in dfs:
        df = dfs[filename]
    else:
        df = pd.read_csv(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, filename), dtype={'key_id': np.str}).set_index('key_id')
        dfs[filename] = df
    #print(df.head())
    #print(key_id)

    #print('>>', df.loc[key_id].drawing)
    stroke = eval(df.loc[key_id].drawing)

    #print(stroke)
    return draw_cv2(stroke)

def test_draw():
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    df_test = pd.read_csv(df_files[0])
    print(df_test.shape)
    df_test = df_test[df_test.recognized==True]
    print(df_test.shape)

    bg = time.time()
    test = None
    for i, stroke in enumerate(df_test.drawing.values):
        img = draw_cv2(eval(stroke))
        #print(img.shape)
        test = img

        if i % 10000 == 0:
            print(i, time.time() - bg)

        #cv2.imshow("Image", img)
        #cv2.waitKey(0)

def test_train_meta():
    df, img_dir = get_train_meta(0)
    print(df.head())
    for row in df.iloc[:10].values:
        print(row)
        fn = os.path.join(img_dir, '{}.jpg'.format(row[0]))
        img = cv2.imread(fn)
        cv2.imshow('image', img)
        cv2.waitKey(0)

def test_iloc():
    df = pd.read_csv(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, 'flying saucer.csv'), dtype={'key_id': np.str}).set_index('key_id')
    print(df.head())
    print(df.loc['4596155960786944'])

if __name__ == '__main__':
    #test_draw()
    #generate_classes()
    #get_classes()
    #get_train_val_meta()
    test_train_meta()
    #test_iloc()