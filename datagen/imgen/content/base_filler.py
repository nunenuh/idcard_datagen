import json

import cv2 as cv
import numpy as np

from ...config import idcard_config
from ..ops import imtext_ops
from . import draw


def fill_content(image, key, text):
    param = idcard_config.idcard_value_template[key]
    data = []
    if param['type'] == 'text':
        img, data = fill_text(image, key, text)
    elif param['type'] == 'picture':
        img = fill_photo(image, text)
        data = []
    else:
        raise ValueError(f"Only 'text' and 'picture' "
                         f"are accepted as value of adjust parameter!")

    return img, data


def fill_text(image, key, text):
    text = str(text)
    param: dict = idcard_config.idcard_value_template[key]
    if param['upper']:
        text = text.upper()
    if param['adjust'] == 'normal':
        img, data = draw.text(image, text, classname=key, subclass=param['subclass'],
                              pos=param['pos'], adjust=param['adjust'],
                              font_name=param['font_name'], font_size=param['font_size'])
    elif param['adjust'] == 'center':
        x_center = param.get('x_center', 0)
        x_min, x_max = param.get('x_min', 0), param.get('x_max', 0)
        img, data = draw.text(image, text, classname=key, subclass=param['subclass'],
                              pos=param['pos'], adjust=param['adjust'],
                              x_center=x_center, x_min=x_min, x_max=x_max,
                              font_name=param['font_name'], font_size=param['font_size'])
    else:
        raise ValueError(
            "Only 'normal' and 'center' are accepted as value of adjust parameter!")
    return img, data


def fill_photo(image, photo_path, face_position=idcard_config.fpos):
    img = image.copy()
    xmin, ymin, xmax, ymax = face_position
    w, h = xmax - xmin, ymax - ymin

    face_img = cv.imread(photo_path, cv.IMREAD_UNCHANGED)
    shape = face_img.shape
    if len(shape) > 2:
        hs, ws, wd = shape
        if wd == 3:
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2BGRA)
    face_resize = cv.resize(face_img, (w, h), interpolation=cv.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax] = face_resize

    return img


def fill_text_below_photo(image, key, text, face_position=idcard_config.fpos):
    txt = text.upper()
    param = idcard_config.idcard_template[key]
    wpos, hpos = get_text_position(text)
    xmin, ymin, xmax, ymax = face_position
    out_img = imtext_ops.draw_text(image, txt, font_variant="Bold",
                                   font_name=param['font_name'], 
                                   xymin=(xmin + wpos - 10, param['pos']),
                                   font_size=param['font_size'])
    return out_img


def get_text_position(text, font_size=19, font_name="arial", face_position=idcard_config.fpos):
    xmin, ymin, xmax, ymax = face_position
    w, h = xmax - xmin, ymax - ymin

    img_picture = np.zeros((h, w, 3))
    wpos, hpos = imtext_ops.get_text_position(
        img_picture, text.upper(), font_name=font_name, font_size=font_size)
    return wpos, hpos

