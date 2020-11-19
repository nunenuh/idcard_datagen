import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import json


from datagen.imgen.ops import boxes_ops
from datagen.imgen import transforms
from datagen.imgen.content import utils as content_utils
from datagen.config import data_config
from datagen.imgen.io import fop

from tqdm import tqdm


def combine(bg_path, idcard_path, dst_path,
            idcard_ext="png", bg_ext="jpg|png",
            angle: int = 30, shear: float = 0.5,
            scale_ratio: tuple = (0.3, 0.8),
            num_generated: int = 6):

    bg_data, bg_path = clean_background_data(bg_path, bg_ext)
    idcard_image_data, idcard_json_data = clean_idcard_data(
        idcard_path, image_ext=idcard_ext)
    dst_path, base_path = clean_destination_path(dst_path)
    angle, shear, scale_ratio, num_generated = clean_augment_param(
        angle, shear, scale_ratio, num_generated
    )

    c = 0
    tc = len(bg_data) * len(idcard_image_data) * num_generated
    bg_bar = tqdm(bg_data)
    for bgfile in bg_bar:
        bg_bar.set_description(f"Progress All Data ({str(c)}/{str(tc)})")
        idcard_bar = tqdm(zip(idcard_image_data, idcard_json_data))
        for (idfile, jsfile) in idcard_bar:
            for n in range(num_generated):

                bg_bar.set_description(
                    f"Progress All Data ({str(c)}/{str(tc)})")
                idcard_bar.set_description(
                    f"Processing augmented ({str(n)}/{str(num_generated)}) saved to {str(base_path)}")

                id_img = cv.imread(str(idfile), cv.IMREAD_UNCHANGED)
                bg_img = cv.imread(str(bgfile), cv.IMREAD_COLOR)

                with open(str(jsfile), 'r') as js_file:
                    json_data = json.load(js_file)

                # print(jsfile, idfile)
                boxes, cnames, scnames, sequence, texts = content_utils.convert_json_boxes_to_numpy(
                    json_data)
                boxes = boxes.reshape(-1, 8)

                ratio = random.choice(scale_ratio)
                # print(f'Choiced Ratio: {ratio}')
                augment = transforms.AugmentGenerator(
                    scale_ratio=ratio, angle=angle, shear_factor=shear)
                seg_img, cmp_img, boxes = augment(bg_img, id_img, boxes)
                seg_img = (seg_img * 255).astype(np.uint8)

                main_boxes = boxes[0].copy()
                # print(f'main_boxes shape: {main_boxes.shape}')

                main_boxes = boxes_ops.order_points(main_boxes).tolist()

                child_boxes = boxes[1:len(boxes)].copy()
                # print(child_boxes.shape)
                child_boxes = boxes_ops.order_points_batch(
                    child_boxes).tolist()
                objects = []

                for (cbox, cn, scn, seq, txt) in zip(child_boxes, cnames, scnames, sequence, texts):
                    dt = {
                        'text': txt, 'points': cbox,
                        'classname': data_config.classname_list[cn],
                        'subclass': data_config.subclassname_list[scn],
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


def clean_background_data(bg_path, bg_path_ext="jpg|png"):
    bg_path = Path(bg_path)
    if not bg_path.exists():
        raise ValueError(f"Directory path to {str(bg_path)} is not exist!")
    bg_path_ext = bg_path_ext.split("|")
    bg_data = []
    for ext in bg_path_ext:
        bg_data = bg_data + list(bg_path.glob(f"*.{ext}"))
    bg_data.sort()
    print(
        f'Logs: Loading {len(bg_data)} data from {str(bg_path)} as background')
    return bg_data, bg_path


def clean_idcard_data(idcard_path, image_ext="png"):
    idcard_path = Path(idcard_path)
    if not idcard_path.exists():
        raise ValueError(f"Directory path to {str(idcard_path)} is not exist!")
    image_data = list(idcard_path.glob(f"*.{image_ext}"))
    json_data = list(idcard_path.glob(f"*.json"))

    image_data.sort()
    json_data.sort()
    print(
        f'Logs: Loading {len(image_data)} data from {str(idcard_path)} as IDCard')

    return image_data, json_data


def clean_destination_path(dst_path):
    dst_path = Path(dst_path)
    if not dst_path.exists():
        raise ValueError(f"Directory path to {str(dst_path)} is not exist!")
    print(f'Logs: Preparing destination directory at {str(dst_path)}')
    time_num = int(time.time())
    base_path = dst_path.joinpath(str(time_num))
    base_path.mkdir(parents=True, exist_ok=True)
    print(f'Logs: Creating directory recursively')

    return dst_path, base_path


def clean_augment_param(angle, shear, scale_ratio, num_generated):
    angle = int(angle)
    shear = float(shear)

    sfactor = 0.1
    scale_ratio: list = [i for i in np.arange(
        scale_ratio[0], scale_ratio[1], sfactor)]

    num_generated = int(num_generated)
    print(f'Info: angle={str(angle)} shear_factor={str(shear)} '
          f'scale_ratio={str(scale_ratio)}')

    return angle, shear, scale_ratio, num_generated
