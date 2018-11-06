import os
import tqdm
import numpy as np
import pandas as pd
import glob
from PIL import Image, ImageDraw
from dask import bag
import ast

import settings


#%% set label dictionary and params
#classfiles = os.listdir('../input/train_simplified/')
#numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #adds underscores

num_classes = 340    #340 max 
imheight, imwidth = 32, 32  
ims_per_class = 2000  #max?

img_rows, img_cols = 32, 32

def draw_it(strokes):
    image = Image.new("P", (255,255), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                            stroke[1][i],
                            stroke[0][i+1], 
                            stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((img_rows, img_cols))
    return np.array(image)/255.

def strokes_to_img(in_strokes):
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12.) #  marker='.',
    ax.axis('off')
    fig.canvas.draw()
    
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return (cv2.resize(X, (96, 96)) / 255.)[::-1]


if __name__ == '__main__':
    #%% get train arrays
    train_grand = []
    class_paths = glob.glob(os.path.join(settings.TRAIN_SIMPLIFIED_DIR, '*.csv'))
    for i,c in enumerate(class_paths[0: num_classes]):
        train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)
        train = train[train.recognized == True].head(ims_per_class)
        imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
        trainarray = np.array(imagebag.compute())  # PARALLELIZE
        trainarray = np.reshape(trainarray, (ims_per_class, -1))    
        labelarray = np.full((train.shape[0], 1), i)
        trainarray = np.concatenate((labelarray, trainarray), axis=1)
        train_grand.append(trainarray)
        
    train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
    train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

    del trainarray
    del train