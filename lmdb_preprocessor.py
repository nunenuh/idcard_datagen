import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import json

import matplotlib.pyplot as plt

from collections import OrderedDict

from datagen.imgen.ops import boxes_ops
from datagen.imgen import transforms
from datagen.imgen.content import utils as content_utils
from datagen.config import data_config
from datagen.imgen.io import fop

from tqdm import tqdm

from datagen.imgen.idcard import combiner 

# !pip install StringGenerator
import strgen
import fire


def open_json_file(path):
    with open(str(path), 'r') as js_file:
        json_data = json.load(js_file)
    return json_data

def append_new_line(text_file, line):
    with open(text_file, "a") as a_file:
        a_file.write(line)
        a_file.write("\n")
        
def deep_text_preprocessor(src, dst, val_split=0.2):
    train_dpath =  Path(dst).joinpath("train")
    valid_dpath =  Path(dst).joinpath("valid")
    
    json_files = list(Path(src).glob("*.json"))
    img_files = list(Path(src).glob("*image.jpg"))
    json_files = sorted(json_files)
    img_files = sorted(img_files)
    
    random.seed(1261)
    val_count = int(val_split * len(json_files))
    list_index = [i for i in range(len(json_files))]
    val_index = sorted(random.sample(list_index, k=val_count))
    trn_index = set(list_index) - set(val_index)
    
    json_trn_files = [json_files[i] for i in trn_index]
    img_trn_files = [img_files[i] for i in trn_index]

    json_val_files = [json_files[i] for i in val_index]
    img_val_files = [img_files[i] for i in val_index]
    
    for f_idx in tqdm(range(len(json_trn_files))):
        js = str(json_trn_files[f_idx])
        im = str(img_trn_files[f_idx])
        img = cv.imread(im)
        js_dict = open_json_file(js)
        
        for js_obj in js_dict['objects']:
            text, point = js_obj['text'], np.array(js_obj['points'], dtype=np.int)
            if len(text)>1 and text!=":":
                ymin, xmin, ymax, xmax = box = boxes_ops.to_xyminmax(point)
                crop = img[ymin:ymax, xmin:xmax]
                
                rand_str = strgen.StringGenerator("[\d\w]{21}").render()
                fname = f'{rand_str}.jpg'
                fname_gt = f'data/{rand_str}.jpg'
                fname = str(train_dpath.joinpath('data').joinpath(fname))

                cv.imwrite(fname, crop)

                text_file = str(train_dpath.joinpath('gt.txt'))
                text_line = f'{fname_gt}\t{text}'
                append_new_line(text_file, text_line)

        
    for f_idx in tqdm(range(len(json_val_files))):
        js = str(json_val_files[f_idx])
        im = str(img_val_files[f_idx])
        img = cv.imread(im)
        js_dict = open_json_file(js)

        for js_obj in js_dict['objects']:
            text, point = js_obj['text'], np.array(js_obj['points'], dtype=np.int)
            if len(text)>1 and text!=":":
                ymin, xmin, ymax, xmax = box = boxes_ops.to_xyminmax(point)
                crop = img[ymin:ymax, xmin:xmax]
                
                rand_str = strgen.StringGenerator("[\d\w]{21}").render()
                fname = f'{rand_str}.jpg'
                fname_gt = f'data/{rand_str}.jpg'
                fname = str(valid_dpath.joinpath('data').joinpath(fname))

                cv.imwrite(fname, crop)

                text_file = str(valid_dpath.joinpath('gt.txt'))
                text_line = f'{fname_gt}\t{text}'
                append_new_line(text_file, text_line)

            

if __name__ == '__main__':
    fire.Fire(deep_text_preprocessor)