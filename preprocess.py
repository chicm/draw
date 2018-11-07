import os
import glob
import numpy as np
import pandas as pd
import time
from sklearn.utils import shuffle
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
    val_dfs_10 = []
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
        df_v = shuffle(df_v[df_v['recognized']==True].sort_values('key_id'), random_state=1234)
        print(df_v.shape)
        
        train_dfs.append(df_t)
        val_dfs.append(df_v)
        val_dfs_10.append(df_v.iloc[:10])
        val_dfs_20.append(df_v.iloc[:20])
        val_dfs_50.append(df_v.iloc[:50])
        
        #if i == 2:
        #    break

    df_train = shuffle(pd.concat(train_dfs), random_state=1234)
    df_val = shuffle(pd.concat(val_dfs), random_state=1234)
    df_val_10 = shuffle(pd.concat(val_dfs_10), random_state=1234)
    df_val_20 = shuffle(pd.concat(val_dfs_20), random_state=1234)
    df_val_50 = shuffle(pd.concat(val_dfs_50), random_state=1234)

    df_train.to_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'), index=False)
    df_val.to_csv(os.path.join(settings.DATA_DIR, 'val_ids.csv'), index=False)
    df_val_10.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_10.csv'), index=False)
    df_val_20.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_20.csv'), index=False)
    df_val_50.to_csv(os.path.join(settings.DATA_DIR, 'val_ids_50.csv'), index=False)

def test_train_meta():
    bg = time.time()
    df_train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'))
    print(df_train_ids.shape, time.time() - bg)
    df_train_ids = df_train_ids[df_train_ids['recognized'] == True]
    print(df_train_ids.shape, time.time() - bg)

    return df_train_ids, {}
    '''
    dfs = {}
    df_files = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    for df_file in df_files:
        print(df_file)
        df = pd.read_csv(df_file)
        filename = os.path.basename(df_file)
        dfs[filename] = df

    print('done')
    '''

if __name__ == '__main__':
    #generate_train_ids()
    #test_train_meta()
    x = {}
    print(x['abc'])