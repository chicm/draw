import os
import cv2
from PIL import Image
import glob
import numpy as np
import pandas as pd
import time
import settings

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
    print(len(classes))
    print(classes)
    return classes

def generate_classes():
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    classes = [os.path.basename(x).split('.')[0].replace(' ', '_') for x in df_files]
    print(classes)
    df_classes = pd.DataFrame(data=classes, columns=['classes'])
    print(df_classes.head())
    df_classes.to_csv('classes.csv', index=False)

def get_sub_df(df, index):
    start_index = len(df) * index // 100
    end_index = len(df) * (index+1) // 100
    return df.iloc[start_index:end_index]

def get_train_val_meta(index=0):
    val_percent = 0.05
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    df_train = None
    df_val = None
    for df_file in df_files:
        print(df_file)
        df = pd.read_csv(df_file)
        #print('unique key', df.key_id.nunique())
        print(df.shape)
        split_index = int(len(df) * (1-val_percent))
        print(split_index)
        df_t = df.iloc[:split_index] #get_sub_df(df.iloc[:split_index], index)
        print(df_t.shape)
        #print(df_t.head())
        df_v = df.iloc[split_index:]
        if df_train is None:
            df_train = df_t
        else:
            df_train = pd.concat([df_train, df_t])

        print('unique key', df_train.shape, df_train.key_id.nunique())
        #if df_val is None:
        #    df_val = df_v
        #else:
        #    df_val = pd.concat([df_val, df_v])
    print(df_train.shape)
    #print(df_val.shape)

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

if __name__ == '__main__':
    #test_draw()
    #generate_classes()
    #get_classes()
    get_train_val_meta()