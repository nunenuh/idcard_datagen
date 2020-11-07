import argparse
import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from datagen.utils import fop
from datagen.utils import functions as F
from datagen.utils import transform, idcard
from tqdm import tqdm, trange
import json

parser = argparse.ArgumentParser(description='IDCard data generator tools')
parser.add_argument('--bg_path', type=str, help='source directory for background image', required=True)
parser.add_argument('--bg_path_ext', type=str, default='jpg|png', help='extension for background image', required=False)
parser.add_argument('--id_path', type=str, help='source directory for id card image', required=True)
parser.add_argument('--id_path_ext', type=str, default='png',
                    help='extension for idcard image, works only with png!', required=False)
parser.add_argument('--dst_path', type=str, help='destination directory for generated data', required=True)
parser.add_argument('--angle', default=30, type=int, help='random rotation angle')
parser.add_argument('--shear', default=0.5, type=float, help='random shear factor')
parser.add_argument('--scale_ratio', default=(0.3, 0.8), type=float, help="scale ratio between idcard and background")
parser.add_argument('--num_generated', default=6, type=int,
                    help='number of combined generated data from same idcard and background image')

args = parser.parse_args()

if __name__ == "__main__":

    bg_path = Path(args.bg_path)
    if not bg_path.exists(): raise ValueError(f"Directory path to {str(bg_path)} is not exist!")
    bg_path_ext = args.bg_path_ext.split("|")
    bg_data = []
    for ext in bg_path_ext:
        bg_data = bg_data + list(bg_path.glob(f"*.{ext}"))
    bg_data.sort()
    print(f'Logs: Loading {len(bg_data)} data from {str(bg_path)} as background')

    id_path = Path(args.id_path)
    if not id_path.exists(): raise ValueError(f"Directory path to {str(id_path)} is not exist!")
    id_path_ext = args.id_path_ext
    id_data = list(id_path.glob(f"*.{id_path_ext}"))
    id_data_json = list(id_path.glob(f"*.json"))

    id_data.sort()
    id_data_json.sort()
    print(f'Logs: Loading {len(id_data)} data from {str(id_path)} as IDCard')

    dst_path = Path(args.dst_path)
    if not dst_path.exists(): raise ValueError(f"Directory path to {str(dst_path)} is not exist!")
    print(f'Logs: Preparing destination directory at {str(dst_path)}')
    tnum = int(time.time())
    base_path = dst_path.joinpath(str(tnum))
    base_path.mkdir(parents=True, exist_ok=True)
    print(f'Logs: Creating directory recursively')

    angle = int(args.angle)
    shear = float(args.shear)

    scale_ratio = args.scale_ratio
    sfactor =  0.1
    scale_ratio: list = [i for i in np.arange(scale_ratio[0], scale_ratio[1], sfactor)]

    num_generated = int(args.num_generated)
    print(f'Info: angle={str(angle)} shear_factor={str(shear)} '
          f'scale_ratio={str(scale_ratio)}')

    bg_bar = tqdm(bg_data)
    c = 0
    tc = len(bg_data) * len(id_data) * num_generated
    for bgfile in bg_bar:
        bg_bar.set_description(f"Progress All Data ({str(c)}/{str(tc)})")
        id_bar = tqdm(zip(id_data, id_data_json))
        for (idfile, jsfile) in id_bar:
            for n in range(num_generated):

                bg_bar.set_description(f"Progress All Data ({str(c)}/{str(tc)})")
                id_bar.set_description(f"Processing augmented ({str(n)}/{str(num_generated)}) saved to {str(base_path)}")

                id_img = cv.imread(str(idfile), cv.IMREAD_UNCHANGED)
                bg_img = cv.imread(str(bgfile), cv.IMREAD_COLOR)

                with open(str(jsfile), 'r') as js_file:
                    json_data = json.load(js_file)

                # print(jsfile, idfile)
                boxes, cnames, scnames, sequence, texts = idcard.convert_json_boxes_to_numpy(json_data)
                boxes = boxes.reshape(-1, 8)

                ratio = random.choice(scale_ratio)
                # print(f'Choiced Ratio: {ratio}')
                augment = transform.AugmentGenerator(scale_ratio=ratio, angle=angle, shear_factor=shear)
                seg_img, cmp_img, boxes = augment(bg_img, id_img, boxes)
                seg_img = (seg_img * 255).astype(np.uint8)

                main_boxes = boxes[0].copy()
                # print(f'main_boxes shape: {main_boxes.shape}')

                main_boxes = F.order_points(main_boxes).tolist()

                child_boxes = boxes[1:len(boxes)].copy()
                # print(child_boxes.shape)
                child_boxes = F.order_points_batch(child_boxes).tolist()
                objects = []

                for (cbox, cn, scn, seq, txt) in zip(child_boxes, cnames, scnames, sequence, texts):
                    dt = {
                        'text': txt, 'points': cbox,
                        'classname': idcard.classname_list[cn],
                        'subclass': idcard.subclassname_list[scn],
                        'sequence': seq
                    }

                    # print(idcard.classname_list[cn])

                    objects.append(dt)


                rnum = str(random.randrange(0, 999999))
                image_fpath = base_path.joinpath(f'{rnum}_image.jpg')
                mask_fpath = base_path.joinpath(f'{rnum}_mask.jpg')
                json_fpath = base_path.joinpath(f'{rnum}_json.json')

                cv.imwrite(str(image_fpath), cmp_img)
                cv.imwrite(str(mask_fpath), seg_img)

                json_dict = {
                    'image': {'filename': str(image_fpath.name), 'dim': cmp_img.shape},
                    'mask': {'filename': str(mask_fpath.name), 'dim': seg_img.shape},
                    'scale_ratio': ratio,
                    'angle': augment.actual_angle,
                    'shear_factor': augment.actual_shear,
                    'box': main_boxes,
                    'objects': objects,
                }
                fop.save_json_file(str(json_fpath), json_dict)
                c = c + 1
