import os
import glob
import torch
import pandas as pd
import numpy as np
import json

import settings
from utils import get_classes


def create_submission(predictions, outfile):
    meta = pd.read_csv(settings.SAMPLE_SUBMISSION)
    meta['word'] = predictions
    meta.to_csv(outfile, index=False)

def get_gjx_classes():
    df = pd.read_csv(r'F:\BaiduYunDownload\model_scores\label_index.csv', names=['classes'])
    gjx_classes = df['classes'].values.tolist()
    gjx_classes = [x.replace(' ', '_') for x in gjx_classes]
    return gjx_classes

def check_classes():
    classes, _ = get_classes()
    df = pd.read_csv(r'F:\BaiduYunDownload\model_scores\label_index.csv', names=['classes'])
    gjx_classes = df['classes'].values.tolist()
    gjx_classes = [x.replace(' ', '_') for x in gjx_classes]
    #print(gjx_classes[:10])
    #print(classes)
    #print('>>>')
    #print(gjx_classes)
    for i in range(len(classes)):
        if classes[i] != gjx_classes[i]:
            print(i, classes[i], gjx_classes[i])

def ensemble_gjx_csv(csv_dir=r'F:\BaiduYunDownload\model_scores'):
    pass
    # df_files = 

def convert_gjx_csvs(csv_dir=r'F:\BaiduYunDownload\model_scores'):
    gjx_classes = get_gjx_classes()
    print(len(gjx_classes))
    convert_dir = os.path.join(csv_dir, 'converted')
    for file_name in glob.glob(os.path.join(csv_dir, 'scores', '*')):
        print(file_name)
        df = pd.read_csv(file_name)
        print('eval...')
        df['score'] = df['score'].map(lambda x: eval(','.join(x.replace('\n', '').split())))
        print(df.head())
        for i, c in enumerate(gjx_classes):
            #print(i)
            df[c] = df['score'].map(lambda x: x[i])
            #df[c] = df['score'].map(lambda x: ','.join(x.replace('\n', '').split()))
            
        col_names = ['key_id', *gjx_classes]
        df.to_csv(os.path.join(convert_dir, os.path.basename(file_name)), index=False, columns=col_names)

def save_gjx_to_npy(csv_dir=r'F:\BaiduYunDownload\model_scores'):
    #gjx_classes = get_gjx_classes()
    classes, _ = get_classes()
    convert_dir = os.path.join(csv_dir, 'converted')
    target_np_file = os.path.join(csv_dir, 'gjx.npy')
    results = []
    for file_name in glob.glob(os.path.join(convert_dir, '*')):
        print(file_name)
        df = pd.read_csv(file_name)
        df = df[classes]
        print(df.head())
        print(df.values.shape)
        results.append(df.values)
    mean_results = np.mean(results, 0)
    np.save(target_np_file, mean_results)

def ensemble_np(np_files, sub_file='sub/gjx_ensemble_test.csv'):
    print(np_files)
    outputs_all = []
    for np_file in np_files:
        outputs_all.append(np.load(np_file))
    outputs = np.mean(outputs_all, 0)
    print(outputs.shape)
    outputs = torch.from_numpy(outputs)
    _, preds = outputs.topk(3, 1, True, True)

    classes, _ = get_classes()
    label_names = []
    preds = preds.numpy()
    print(preds.shape)
    for row in preds:
        label_names.append(' '.join([classes[i] for i in row]))
    create_submission(label_names, sub_file)

if __name__ == '__main__':
    #check_classes()
    #convert_gjx_csvs()
    #save_gjx_to_npy()
    ensemble_np([r'F:\BaiduYunDownload\model_scores\gjx.npy'])