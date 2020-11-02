from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import functions as F

arial_ttf_path = 'data/fonts/arial.ttf'
ocra_ttf_path = 'data/fonts/ocr_a_ext.ttf'


def datalog_drawtext(image: np.ndarray, text: str, classname: str, subclass: str,
                     pos: tuple or int, adjust: str = "normal",
                     use_pad = False, pad_factor=0.01,
                     x_center: int = 0, x_min: int = 0, x_max: int = 0,
                     delimiter: str = " ", font_size: int = 25, font_name: str = "arial",
                     img_mode: str = "RGBA", color: tuple = (0, 0, 0)):
    if adjust == "normal":
        img, data = datalog_drawtext_normal(image, text, classname=classname, subclass=subclass,
                                            xy_pos=pos,
                                            use_pad=use_pad, pad_factor=pad_factor, delimiter=delimiter,
                                            font_name=font_name, font_size=font_size,
                                            img_mode=img_mode, color=color)
    elif adjust == "center":
        img, data = datalog_drawtext_center(image, text, classname=classname, subclass=subclass,
                                            y_pos=pos, x_center=x_center, x_min=x_min, x_max=x_max,
                                            use_pad=use_pad, pad_factor=pad_factor,
                                            delimiter=delimiter, font_name=font_name, font_size=font_size,
                                            img_mode=img_mode, color=color)
    else:
        raise ValueError("Only 'normal' and 'center' are accepted as value of adjust parameter!")

    return img, data


def datalog_drawtext_normal(image: np.ndarray, text: str, classname: str, subclass: str, xy_pos: tuple,
                            use_pad=False, pad_factor:float=0.01,
                            delimiter: str = " ", font_name: str = "arial", font_size=14,
                            img_mode="RGBA", color=(0, 0, 0)):
    np_img = image.copy()

    img, draw = get_image_draw(np_img, img_mode=img_mode)
    font = get_image_font(font_name=font_name, font_size=font_size)
    dlm_w, dlm_h = find_textsize(np_img, delimiter, font_name=font_name, font_size=font_size)
    x, y = xy_pos

    data = []
    text_split = text.split(" ")
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = find_textsize(np_img, txt, font_name=font_name, font_size=font_size)
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = F.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx})
        data.append(odt)

        draw.text(xymin, txt, font=font, fill=color)
        x = x + txt_w + dlm_w

    return np.array(img), data


def datalog_drawtext_center(image: np.ndarray, text: str, classname: str, subclass: str,
                            y_pos: int, x_center: int = 0, x_min: int = 0, x_max: int = 0,
                            use_pad=False, pad_factor = 0.01,
                            delimiter: str = " ",
                            font_name: str = "arial", font_size=14, img_mode="RGBA", color=(0, 0, 0)):
    np_img = image.copy()
    img, draw = get_image_draw(np_img, img_mode=img_mode)
    font = get_image_font(font_name=font_name, font_size=font_size)
    dlm_w, dlm_h = find_textsize(np_img, delimiter, font_name=font_name, font_size=font_size)

    if x_center != 0:
        ft_w, ft_h = find_textsize(np_img, text, font_name=font_name, font_size=font_size)
        if x_min != 0 and x_max != 0:
            x_center = ((x_max - x_min) / 2) + x_min
        x_pos = (x_center - (ft_w / 2))
    else:
        ft_cw, ft_ch = find_center_textsize(np_img, text, font_name=font_name, font_size=font_size)
        x_pos = ft_cw

    x, y = x_pos, y_pos

    data = []
    text_split = text.split(" ")
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = find_textsize(np_img, txt, font_name=font_name, font_size=font_size)
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = F.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx})
        data.append(odt)

        draw.text(xymin, txt, font=font, fill=color)
        x = x + txt_w + dlm_w

    return np.array(img), data


def draw_text(np_img: np.ndarray, text: str, xymin: tuple, font_name: str = "ocra",
              font_size=14, img_mode="RGBA", color=(0, 0, 0), font_variant="Normal"):
    img, draw = get_image_draw(np_img, img_mode)
    font = get_image_font(font_name, font_size)
    draw.text(xymin, text, font=font, fill=color)
    return np.array(img)


def draw_text_center(np_img: np.ndarray, text: str, ypos=0, font_name: str = "ocra",
                     font_size=14, img_mode="RGBA", color=(0, 0, 0)):
    img, draw = get_image_draw(np_img, img_mode)
    font = get_image_font(font_name, font_size)

    txt_cw, txt_ch = find_center_textsize(np_img, text, font_name=font_name, font_size=font_size, img_mode=img_mode)
    position = txt_cw, ypos

    draw.text(position, text, font=font, fill=color)
    return np.array(img)

def split_text_by_max_width(image, text, max_width, font_name, font_size):
    text_split = text.split(" ")
    last_tw, twa = 0, 0
    texts, text = [], []
    for txt in text_split:
        tw, th = find_textsize(image, txt, font_name=font_name, font_size=font_size)
        twa = twa + tw
        if twa > max_width:
            # print(f'{twa}>{max_width}')
            last = text.pop()
            texts.append(text)
            text, twa = [], tw + last_tw
            last_tw = tw
            text.append(last)
            text.append(txt)
        else:
            last_tw = tw
            text.append(txt)

    texts.append(text)
    joined_text = [" ".join(txt) for txt in texts]

    return joined_text


def font_path(name="arial"):
    if name == "arial":
        return arial_ttf_path
    elif name == "ocra":
        return ocra_ttf_path
    else:
        return arial_ttf_path


def find_center_textsize(np_img: np.ndarray, text: str, font_name: str = "ocra", font_size=14, img_mode="RGBA"):
    im_h, im_w = np_img.shape[:2]
    txt_w, txt_h = find_textsize(np_img, text, font_name=font_name, font_size=font_size, img_mode=img_mode)
    w, h = (im_w - txt_w) / 2, (im_h - txt_h) / 2
    return w, h


def find_center_textsize_width(np_img: np.ndarray, text: str, font_name: str = "ocra", font_size=14, img_mode="RGBA"):
    w, h = find_center_textsize(np_img, text, font_name=font_name, font_size=font_size, img_mode=img_mode)
    return w


def find_center_textsize_height(np_img: np.ndarray, text: str, font_name: str = "ocra", font_size=14, img_mode="RGBA"):
    w, h = find_center_textsize(np_img, text, font_name=font_name, font_size=font_size, img_mode=img_mode)
    return h


def find_textsize(np_img: np.ndarray, text: str, font_name: str = "ocra", font_size=14, img_mode="RGBA"):
    img, draw = get_image_draw(np_img, img_mode=img_mode)
    font = get_image_font(font_name=font_name, font_size=font_size)
    w, h = draw.textsize(text, font)
    return w, h


def get_image_draw(np_img: np.ndarray, img_mode="RGBA"):
    img = Image.fromarray(np_img, img_mode)
    draw = ImageDraw.Draw(img)
    return img, draw


def get_image_font(font_name: str = "ocra", font_size=14):
    font = ImageFont.truetype(font_path(font_name), size=font_size)
    return font
