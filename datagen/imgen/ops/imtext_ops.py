from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ...config import font_config



def draw_text_normal(np_img: np.ndarray, text: str, xymin: tuple, font_name: str = "ocra",
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


def get_text_position(np_img: np.ndarray, text: str, font_name: str = "ocra",
                      font_size=14, img_mode="RGBA", color=(0, 0, 0), font_variant="Normal"):
    img, draw = get_image_draw(np_img, img_mode)
    font = ImageFont.truetype(font_config.font_path(font_name), size=font_size)

    im_height, im_width = np_img.shape[:2]
    txt_width, txt_height = draw.textsize(text, font)
    position = (im_width - txt_width) / 2, (im_height - txt_height) / 2
    return position


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
    font = ImageFont.truetype(font_config.font_path(font_name), size=font_size)
    return font
