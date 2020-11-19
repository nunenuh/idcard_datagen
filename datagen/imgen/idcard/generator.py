
import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import pandas as pd

from ..content import data_filler
from tqdm import trange


def generate(csv_path, dst_path, image_path, json_path, photo_path):
    cleaned_params = clean_parameters(csv_path, dst_path, image_path, json_path, photo_path)
    dframe, dst_path, photo_data, image_path, json_path = cleaned_params
    
    bar_range = trange(len(dframe))

    for idx in bar_range:
        data = dframe.iloc[idx].to_dict()
        rnd_idx = random.choice([i for i in range(len(photo_data))])
        data['face'] = str(photo_data[rnd_idx])

        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        
        out_image, out_data = data_filler.fill_data(image, data, file_path=str(json_path))

        rstr = str(random.randint(0, 100000))
        image_filename = str(data['nik']) + f"_{rstr}.png"
        json_filename = str(data['nik']) + f"_{rstr}.json"

        fpath = dst_path.joinpath(image_filename)
        cv.imwrite(str(fpath), out_image)

        json_fpath = dst_path.joinpath(json_filename)
        obj = {
            "image_filename": image_filename,
            "json_filename": json_filename,
            "size": [w, h],
            "points": [[0, 0], [w, 0], [w, h], [0, h]],
            'objects': out_data
        }
        
        with open(str(json_fpath), 'w') as file:
            json.dump(obj, file, indent=4) 


def clean_parameters(csv_path, dst_path, image_path, json_path, photo_path):
    csv_path = Path(csv_path)
    dst_path = Path(dst_path)
    image_path = Path(image_path)
    json_path = Path(json_path)
    photo_path = Path(photo_path)
    
    # print(f'{str(csv_path)}')
    # print(f'{str(dst_path)}')
    # print(f'{str(image_path)}')
    # print(f'{str(json_path)}')
    # print(f'{str(photo_path)}')
    
    
    if not (csv_path.exists() and csv_path.is_file()):
        raise ValueError(f"Directory path to csv_path at {str(csv_path)} is not exist!")
    
    if not dst_path.exists():
        raise ValueError(f"Directory path to dst_path at {str(dst_path)} is not exist!")

    if not (image_path.exists() and image_path.is_file()):
        raise ValueError(f"Directory path to image_path at {str(image_path)} is not exist!")

    if not photo_path.exists():
        raise ValueError(f"Directory path to photo_path at {str(photo_path)} is not exist!")

    if not (json_path.exists() and json_path.is_file()):
        raise ValueError(f"Directory path to json_path at {str(json_path)} is not exist!")
    
    dataframe = pd.read_csv(csv_path)
    photo_data = list(photo_path.glob('*.png'))
    
    return dataframe, dst_path, photo_data, image_path, json_path
    
    