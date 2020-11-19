from collections import OrderedDict

import numpy as np

# from . import functions as F
from ..ops import imtext_ops
from ..ops import boxes_ops


def text(
    image: np.ndarray,
    text: str,
    classname: str,
    subclass: str,
    pos: tuple or int,
    adjust: str = "normal",
    use_pad: bool = False,
    pad_factor: float = 0.01,
    x_center: int = 0,
    x_min: int = 0,
    x_max: int = 0,
    delimiter: str = " ",
    font_size: int = 25,
    font_name: str = "arial",
    img_mode: str = "RGBA",
    color: tuple = (0, 0, 0)
):
    if adjust == "normal":
        img, data = text_normal(image, text, 
                                classname=classname, 
                                subclass=subclass,
                                xy_pos=pos,
                                use_pad=use_pad, 
                                pad_factor=pad_factor, 
                                delimiter=delimiter,
                                font_name=font_name, 
                                font_size=font_size,
                                img_mode=img_mode, 
                                color=color)
    elif adjust == "center":
        img, data = text_center(image, text, 
                                classname=classname, 
                                subclass=subclass,
                                y_pos=pos, 
                                x_center=x_center, 
                                x_min=x_min, 
                                x_max=x_max,
                                use_pad=use_pad, 
                                pad_factor=pad_factor,
                                delimiter=delimiter, 
                                font_name=font_name, 
                                font_size=font_size,
                                img_mode=img_mode, 
                                color=color)
    else:
        raise ValueError(f"Only 'normal' and 'center' "
                         f"are accepted as value of adjust parameter!")

    return img, data


def text_normal(
    image: np.ndarray, 
    text: str, 
    classname: str, 
    subclass: str, 
    xy_pos: tuple,
    use_pad: bool = False, 
    pad_factor: float = 0.01,
    delimiter: str = " ", 
    font_name: str = "arial", 
    font_size: int =14,
    img_mode: str = "RGBA", 
    color: tuple = (0, 0, 0)
):
    np_img = image.copy()

    img, draw = imtext_ops.get_image_draw(np_img, img_mode=img_mode)
    font = imtext_ops.get_image_font(font_name=font_name, font_size=font_size)
    dlm_w, dlm_h = imtext_ops.find_textsize(
        np_img, delimiter, font_name=font_name, font_size=font_size)
    x, y = xy_pos

    data = []
    text_split = text.split(" ")
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = imtext_ops.find_textsize(
            np_img, txt, 
            font_name=font_name, 
            font_size=font_size
        )
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = boxes_ops.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx})
        data.append(odt)

        draw.text(xymin, txt, font=font, fill=color)
        x = x + txt_w + dlm_w

    return np.array(img), data


def text_center(
    image: np.ndarray, 
    text: str, 
    classname: str, 
    subclass: str,
    y_pos: int, 
    x_center: int = 0, 
    x_min: int = 0, 
    x_max: int = 0,
    use_pad: bool = False, 
    pad_factor: float=0.01,
    delimiter: str = " ",
    font_name: str = "arial", 
    font_size: int = 14, 
    img_mode: str = "RGBA", 
    color: tuple = (0, 0, 0)
):
    np_img = image.copy()
    img, draw = imtext_ops.get_image_draw(np_img, img_mode=img_mode)
    font = imtext_ops.get_image_font(font_name=font_name, font_size=font_size)
    dlm_w, dlm_h = imtext_ops.find_textsize(
        np_img, delimiter, 
        font_name=font_name, 
        font_size=font_size
    )

    if x_center != 0:
        ft_w, ft_h = imtext_ops.find_textsize(
            np_img, text, 
            font_name=font_name, 
            font_size=font_size
        )
        if x_min != 0 and x_max != 0:
            x_center = ((x_max - x_min) / 2) + x_min
        x_pos = (x_center - (ft_w / 2))
    else:
        ft_cw, ft_ch = imtext_ops.find_center_textsize(
            np_img, text, 
            font_name=font_name, 
            font_size=font_size
        )
        x_pos = ft_cw

    x, y = x_pos, y_pos

    data = []
    text_split = text.split(" ")
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = imtext_ops.find_textsize(
            np_img, txt, font_name=font_name, font_size=font_size)
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = boxes_ops.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx})
        data.append(odt)

        draw.text(xymin, txt, font=font, fill=color)
        x = x + txt_w + dlm_w

    return np.array(img), data
