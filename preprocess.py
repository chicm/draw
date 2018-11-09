import os
import glob
import numpy as np
import pandas as pd
import time
import cv2
from sklearn.utils import shuffle
import pickle
from utils import draw_cv2, get_train_meta
import settings


def generate_classes():
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    classes = [os.path.basename(x).split('.')[0].replace(' ', '_') for x in df_files]
    print(classes)
    df_classes = pd.DataFrame(data=classes, columns=['classes'])
    print(df_classes.head())
    df_classes.to_csv('classes.csv', index=False)

def generate_train_ids():
    val_percent = 0.05
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    train_dfs = []
    val_dfs = []
    val_dfs_100 = []
    val_dfs_20 = []
    val_dfs_50 = []

    for i, df_file in enumerate(df_files):
        print(df_file)
        df = pd.read_csv(df_file)
        df = shuffle(df, random_state=1234)
        #print('unique key', df.key_id.nunique())
        filename = os.path.basename(df_file)
        df['filename'] = filename
        #print(df.head())
        df = df[['key_id', 'word', 'filename', 'recognized', 'countrycode']]
        #print(df.head())
        split_index = int(len(df) * (1-val_percent))

        df_t = df.iloc[:split_index]
        df_v = df.iloc[split_index:]
        print(df_v.shape)
        #df_v = shuffle(df_v[df_v['recognized']==True].sort_values('key_id'), random_state=1234)
        
        train_dfs.append(df_t)
        val_dfs.append(df_v)
        val_dfs_100.append(df_v.iloc[:100])
        val_dfs_20.append(df_v.iloc[:20])
        val_dfs_50.append(df_v.iloc[:50])
        
        #if i == 2:
        #    break
    print('sorting...')
    df_train = shuffle(pd.concat(train_dfs).sort_values('key_id'), random_state=1234)
    df_val = shuffle(pd.concat(val_dfs).sort_values('key_id'), random_state=1234)
    df_val_100 = shuffle(pd.concat(val_dfs_100).sort_values('key_id'), random_state=1234)
    df_val_20 = shuffle(pd.concat(val_dfs_20).sort_values('key_id'), random_state=1234)
    df_val_50 = shuffle(pd.concat(val_dfs_50).sort_values('key_id'), random_state=1234)
    
    print('saving...')
    df_train.to_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'), index=False)
    df_val.to_csv(os.path.join(settings.DATA_DIR, 'val_ids.csv'), index=False)
    df_val_100.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_100.csv'), index=False)
    df_val_20.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_20.csv'), index=False)
    df_val_50.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_50.csv'), index=False)
'''
def generate_recognized_train_ids():
    bg = time.time()
    df_train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'), dtype={'key_id': np.str})
    print(df_train_ids.shape, time.time() - bg)
    df_train_ids = df_train_ids[df_train_ids['recognized'] == True]
    print(df_train_ids.shape, time.time() - bg)

    df_train_ids.to_csv(os.path.join(settings.DATA_DIR, 'train_ids_recognized.csv'), index=False)
'''

def generate_train_images(index, img_sz=256):
    bunch_size = 1000000
    #df_train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids_recognized.csv'), dtype={'key_id': np.str}).iloc[bunch_size*index:bunch_size*(index+1)]
    df_train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'), dtype={'key_id': np.str}).iloc[bunch_size*index:bunch_size*(index+1)]
    print(df_train_ids.shape)
    csv_file = os.path.join(settings.DATA_DIR, 'train-{}'.format(img_sz), 'train_{}.csv'.format(index))
    img_dir = os.path.join(settings.DATA_DIR, 'train-{}'.format(img_sz), 'train_{}'.format(index))

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    for df_file in df_files:
        print(df_file)
        df = pd.read_csv(df_file, dtype={'key_id': np.str}).set_index('key_id')
        df_cls_ids = df_train_ids[df_train_ids['filename']==os.path.basename(df_file)]
        print(df_cls_ids.shape)
        for key_id in df_cls_ids['key_id'].values:
            stroke = eval(df.loc[key_id].drawing)
            img = draw_cv2(stroke)
            fn = os.path.join(img_dir, '{}.jpg'.format(key_id))
            cv2.imwrite(fn, img)

    df_train_ids.to_csv(csv_file, index=False)


def generate_val_100_images(img_sz=256):
    df_val_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'val_ids_100.csv'), dtype={'key_id': np.str})
    print(df_val_ids.shape)
    img_dir = os.path.join(settings.DATA_DIR, 'val-100-{}'.format(img_sz))

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    for df_file in df_files:
        print(df_file)
        df = pd.read_csv(df_file, dtype={'key_id': np.str}).set_index('key_id')
        df_cls_ids = df_val_ids[df_val_ids['filename']==os.path.basename(df_file)]
        print(df_cls_ids.shape)
        for key_id in df_cls_ids['key_id'].values:
            stroke = eval(df.loc[key_id].drawing)
            img = draw_cv2(stroke)
            fn = os.path.join(img_dir, '{}.jpg'.format(key_id))
            cv2.imwrite(fn, img)

def generate_test_images(img_sz=256):
    df = pd.read_csv(settings.TEST_SIMPLIFIED, dtype={'key_id': np.str})
    img_dir = settings.TEST_SIMPLIFIED_IMG_DIR
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for row in df.values:
        key_id = row[0]
        stroke = eval(row[2])
        img = draw_cv2(stroke)
        fn = os.path.join(img_dir, '{}.jpg'.format(key_id))
        cv2.imwrite(fn, img)

def test_train_meta():
    bg = time.time()
    df_train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'))
    print(df_train_ids.shape, time.time() - bg)
    df_train_ids = df_train_ids[df_train_ids['recognized'] == True]
    print(df_train_ids.shape, time.time() - bg)

    return df_train_ids, {}

def get_country_codes():
    df = get_train_meta(0)
    codes = sorted(list(set(df['countrycode'].values.tolist())))
    print(codes)

if __name__ == '__main__':
    #generate_train_ids()
    #generate_val_100_images()
    #test_train_meta()
    #for i in range(9,30):
    #print('index:', i)
    #generate_train_images(0)
    #generate_val_50_images()
    #test_train_imgs()
    get_country_codes()
    
