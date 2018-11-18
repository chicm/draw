import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import settings
from loader import get_test_loader
import cv2
from models import DrawNet, create_model
from utils import get_classes


def create_submission(args, predictions, outfile):
    meta = pd.read_csv(settings.SAMPLE_SUBMISSION)
    if args.dev_mode:
        meta = meta.iloc[:len(predictions)]  # for dev mode
    meta['word'] = predictions
    meta.to_csv(outfile, index=False)

def model_predict(args, model, model_file, check=False, tta_num=2):
    model.eval()

    preds = []
    for flip_index in range(tta_num):
        print('tta index:', flip_index)
        test_loader = get_test_loader(batch_size=args.batch_size, img_sz=args.img_sz, dev_mode=args.dev_mode, tta_index=flip_index)

        outputs = None
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.cuda()
                output = model(x)
                output = F.softmax(output, dim=1)
                if outputs is None:
                    outputs = output.cpu()
                else:
                    outputs = torch.cat([outputs, output.cpu()], 0)
                print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
                if check and i == 0:
                    break

        preds.append(outputs)
        #return outputs
    results = torch.mean(torch.stack(preds), 0)

    parent_dir = model_file+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred.npy')
    np.save(np_file, results.numpy())

    return results

def predict_top3(args):
    model, model_file = create_model(args.backbone, args.img_sz)

    if not os.path.exists(model_file):
        raise AssertionError('model file not exist: {}'.format(model_file))

    model.eval()
    test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode, img_sz=args.img_sz)

    outputs = model_predict(args, model, model_file)
    _, preds = outputs.topk(3, 1, True, True)

    classes, _ = get_classes()
    label_names = []
    preds = preds.numpy()
    print(preds.shape)
    for row in preds:
        label_names.append(' '.join([classes[i] for i in row]))
    if args.dev_mode:
        print(len(label_names))
        print(label_names)
        key_id = test_loader.meta['key_id'].values.tolist()[0]
        show_test_img(key_id)

    create_submission(args, label_names, args.sub_file)

def ensemble_np(np_files):
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
    create_submission(args, label_names, args.sub_file)


def show_test_img(key_id):
    fn = os.path.join(settings.TEST_SIMPLIFIED_IMG_DIR, '{}.jpg'.format(key_id))
    img = cv2.imread(fn)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Quick Draw')
    parser.add_argument('--backbone', default='resnet18', type=str, help='backbone')
    parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--sub_file', default='sub/sub1.csv', help='submission file')
    parser.add_argument('--img_sz', default=256, type=int, help='alway save')
    parser.add_argument('--ensemble_np', default=None, type=str, help='np files')
    
    args = parser.parse_args()
    print(args)

    if args.ensemble_np:
        np_files = args.ensemble_np.split(',')
        ensemble_np(np_files)
    else:
        predict_top3(args)
