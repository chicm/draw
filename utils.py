import os
import cv2
from PIL import Image
import glob
import numpy as np
import pandas as pd
import json
import time
import settings

country_code_names = ['OTHER', 'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AN', 'AO', 'AR', 'AS', 'AT', 'AU', 'AW', 'AX',
    'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BM', 'BN', 'BO', 'BR', 'BS', 'BT', 'BU',
    'BW', 'BY', 'BZ', 'CA', 'CD', 'CF', 'CG', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR', 'CV', 'CW', 'CX',
    'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'ET', 'FI', 'FJ', 'FM', 'FO',
    'FR', 'GA', 'GB', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP', 'GR', 'GT', 'GU', 'GY',
    'HK', 'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IQ', 'IR', 'IS', 'IT', 'JE', 'JM', 'JO',
    'JP', 'KE', 'KG', 'KH', 'KN', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS', 'LT',
    'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ',
    'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NC', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ',
    'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL', 'PM', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO',
    'RS', 'RU', 'RW', 'SA', 'SC', 'SE', 'SG', 'SI', 'SJ', 'SK', 'SM', 'SN', 'SO', 'SR', 'SS', 'ST', 'SV',
    'SX', 'SZ', 'TC', 'TG', 'TH', 'TJ', 'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TW', 'TZ', 'UA', 'UG', 'US',
    'UY', 'UZ', 'VC', 'VE', 'VG', 'VI', 'VN', 'VU', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW', 'ZZ']

country_codes = {country_code_names[i]: i for i in range(len(country_code_names))} 
country_codes_len = len(country_code_names)

def get_country_code(df_row):
    #print(df_row)
    if pd.isna(df_row.countrycode):
        #print('NA country code')
        return 0.
    else:
        return country_codes[df_row.countrycode] / country_codes_len

# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
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

def get_train_meta(index=0, dev_mode=False):
    df_file = os.path.join(settings.DATA_DIR, 'train_shuffle', 'train_k{}.csv'.format(index))
    print(df_file)

    df = pd.read_csv(df_file)
    if dev_mode:
        df = df.iloc[:10]
    df['drawing'] = df['drawing'].apply(json.loads)
    #print(df.head())
    
    return df

def get_val_meta(val_num=20000):
    df_file = os.path.join(settings.DATA_DIR, 'train_shuffle', 'train_k99.csv')
    
    df = pd.read_csv(df_file).iloc[:val_num]
    df['drawing'] = df['drawing'].apply(json.loads)
    return df

def test_train_meta():
    df = get_train_meta(0)
    print(df.head())
    for stroke, word in df.iloc[:10][['drawing', 'word']].values:
        #print(stroke)
        img = draw_cv2(stroke)
        cv2.imshow(word, img)
        cv2.waitKey(0)

def test_val_meta():
    df = get_val_meta()
    print(df.head())
    for stroke, word in df.iloc[:10][['drawing', 'word']].values:
        #print(stroke)
        img = draw_cv2(stroke)
        cv2.imshow(word, img)
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
    #test_train_meta()
    #test_val_meta()
    #test_iloc()
    print(sorted(country_codes), len(country_codes))