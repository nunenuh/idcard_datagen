import argparse
import json
import random
from pathlib import Path

import cv2 as cv
import pandas as pd

from datagen.utils import idcard
from tqdm import tqdm, trange

parser = argparse.ArgumentParser(description='IDCard base data generator tools')
parser.add_argument('--csv_path', type=str, help='path to csv path for generating data', required=True)
parser.add_argument('--idcard_path', type=str, help='source directory for background image', required=True)
parser.add_argument('--idcard_json_path', default='data/idcard/base3.json', type=str,
                    help='source directory for background image', required=False)
parser.add_argument('--photo_path', type=str, help='source directory for 3x4 photo image', required=True)
parser.add_argument('--dst_path', type=str, help='destination path for generating result data', required=True)

arial_ttf_path = 'data/fonts/arial.ttf'
ocra_ttf_path = 'data/fonts/ocr_a_ext.ttf'

args = parser.parse_args()

if __name__ == "__main__":
    csv_path = Path(args.csv_path)
    if not (csv_path.exists() and csv_path.is_file()):
        raise ValueError(f"Directory path to {str(csv_path)} is not exist!")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.iloc[0].to_dict())

    idcard_path = Path(args.idcard_path)
    if not (idcard_path.exists() and idcard_path.is_file()):
        raise ValueError(f"Directory path to {str(idcard_path)} is not exist!")

    photo_path = Path(args.photo_path)
    if not photo_path.exists():
        raise ValueError(f"Directory path to {str(photo_path)} is not exist!")
    photo_data = list(photo_path.glob('*.png'))

    dst_path = Path(args.dst_path)
    if not dst_path.exists():
        raise ValueError(f"Directory path to {str(dst_path)} is not exist!")

    idcard_json_path = Path(args.idcard_json_path)

    bar_range = trange(len(dataframe))

    for idx in bar_range:
        data = dataframe.iloc[idx].to_dict()
        rnd_idx = random.choice([i for i in range(len(photo_data))])
        data['face'] = str(photo_data[rnd_idx])

        # print(data['face'])
        # cv.imread(data['face'])

        image = cv.imread(str(idcard_path), cv.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        out_image, out_data = idcard.build_content(image, data, file_path=str(idcard_json_path))

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
