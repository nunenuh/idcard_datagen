from collections import OrderedDict

import numpy as np

# from . import functions as F
from ..ops import imtext_ops
from ..ops import boxes_ops
import cv2 as cv


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
    color: tuple = (0, 0, 0),
    line: int = 0
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
                                color=color,
                                line=line)
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
                                color=color,
                                line=line)
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
    color: tuple = (0, 0, 0),
    line: int = 0
):
    np_img = image.copy()

    img, draw = imtext_ops.get_image_draw(np_img, img_mode=img_mode)
    font = imtext_ops.get_image_font(font_name=font_name, font_size=font_size)
    dlm_w, dlm_h = imtext_ops.find_textsize(
        np_img, delimiter, font_name=font_name, font_size=font_size)
    x, y = xy_pos

    data = []
    text_split = text.split(" ")
    cleaned_text = []
    for txt in text_split:
        txt = txt.strip()
        if len(txt)>0:
            cleaned_text.append(txt)
    text_split = cleaned_text
    
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = imtext_ops.find_textsize(
            np_img, txt, 
            font_name=font_name, 
            font_size=font_size
        )
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = boxes_ops.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        char_data = []
        out_img, charbox_list = char_bbox(np_img, txt, xy_pos=xymin, font_name=font_name, font_size=font_size)
        for bxt in charbox_list:
            cpoints, char = bxt
            # bpoints = boxes_ops.xywh_to_point(bbox, use_pad=False)
            
            char_dict = OrderedDict({"char": char, "points": cpoints.tolist()})
            char_data.append(char_dict) 


        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           'chardata': char_data,
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx, "line": line})
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
    color: tuple = (0, 0, 0),
    line: int = 0
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
    cleaned_text = []
    for txt in text_split:
        txt = txt.strip()
        if len(txt)>0:
            cleaned_text.append(txt)
    text_split = cleaned_text
    
    for idx, txt in enumerate(text_split):
        # you need to add space for every text
        txt_w, txt_h = imtext_ops.find_textsize(
            np_img, txt, font_name=font_name, font_size=font_size)
        xymin, xywh = (x, y), (x, y, txt_w, txt_h)
        points = boxes_ops.xywh_to_point(xywh, use_pad=use_pad, pad_factor=pad_factor)

        char_data = []
        out_img, charbox_list = char_bbox(np_img, txt, xy_pos=xymin, font_name=font_name, font_size=font_size)
        for bxt in charbox_list:
            cpoints, char = bxt
            # bpoints = boxes_ops.xywh_to_point(bbox, use_pad=False)
            
            char_dict = OrderedDict({"char": char, "points": cpoints.tolist()})
            char_data.append(char_dict)


        odt = OrderedDict({"text": txt, "points": points.tolist(),
                           'chardata': char_data,
                           "classname": classname, 'subclass': subclass,
                           "sequence": idx, "line": line})
        data.append(odt)

        draw.text(xymin, txt, font=font, fill=color)
        x = x + txt_w + dlm_w

    return np.array(img), data


def char_bbox(
    image: np.ndarray, 
    text: str,  
    xy_pos: tuple, 
    font_name: str = "arial", 
    font_size: int = 14, 
    img_mode: str = "RGBA",
    color: tuple = (0,0,0),
    debug_draw: bool = False
):
    np_img = image.copy()
    
    data_tuple = []
    xmin, ymin = xy_pos
    for i in range(len(text)):
        if len(text[i])>0:
            img, draw = imtext_ops.get_image_draw(np_img, img_mode=img_mode)
            font = imtext_ops.get_image_font(font_name=font_name, font_size=font_size)
            tw, th = font.getsize(text[i])
            ox, oy = font.getoffset(text[i])
            xmax, ymax = xmin + tw, ymin + th
            if debug_draw:  
                draw.text((xmin,ymin), text[i], font=font, fill=color)
            xminr, yminr = xmin + ox, ymin + oy
            if debug_draw:
                np_img = cv.rectangle(np.array(img), (xminr, yminr), (xmax, ymax), (0, 255, 0), 3)
            xymm = [xminr, yminr, xmax, ymax]
            xywh = boxes_ops.xymm2xywh(xymm)
            points = boxes_ops.xywh_to_point(xywh, use_pad=False)
            xmin = xmax
            data_tuple.append((points, text[i]))
    
    return np_img, data_tuple