import ast
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from utils import get_classes
import settings

DATA_DIR = settings.DATA_DIR
NCSVS = 100

#def f2cat(filename: str) -> str:
#    return filename.split('.')[0]


#class Simplified(object):
#    def __init__(self, input_path=DATA_DIR):
#        self.input_path = input_path

    #def list_all_categories(self):
    #    files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
    #    return sorted([f2cat(f) for f in files], key=str.lower)

def read_training_csv(category, nrows=None, usecols=None, drawing_transform=False):
    base_file_name = category.replace('_', ' ')

    df = pd.read_csv(os.path.join(DATA_DIR, 'train_simplified', base_file_name + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
    if drawing_transform:
        df['drawing'] = df['drawing'].apply(ast.literal_eval)
    #print(df.head())
    return df


def generate_shards(categories, shuffle_dir='train_shuffle'):
    target_dir = os.path.join(DATA_DIR, shuffle_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for y, cat in tqdm(enumerate(categories)):
        #df = read_training_csv(cat, nrows=30000)
        df = read_training_csv(cat)
        df['y'] = y
        df['cv'] = (df.key_id // 10 ** 7) % NCSVS
        for k in range(NCSVS):
            filename = os.path.join(target_dir, 'train_k{}.csv'.format(k))
            chunk = df[df.cv == k]
            #chunk = chunk.drop(['key_id'], axis=1)
            if y == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

def shuffle_csvs():
    for k in tqdm(range(NCSVS)):
        filename = os.path.join(DATA_DIR, 'train_shuffle', 'train_k{}.csv'.format(k))
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = shuffle(df.sort_values('key_id'), random_state=1234)
            df.to_csv(filename, index=False)
            #df['rnd'] = np.random.rand(len(df))
            #df = df.sort_values(by='rnd').drop('rnd', axis=1)
            #df.to_csv(filename + '.gz', compression='gzip', index=False)
            #os.remove(filename)

def main():
    #start = dt.datetime.now()
    
    categories, _ = get_classes()
    print(len(categories))
    generate_shards(categories)

if __name__ == '__main__':
    main()
    shuffle_csvs()

