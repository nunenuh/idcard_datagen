from datagen.imgen.transforms.functional import coin_toss
import json
import cv2 as cv
import numpy as np

from ..ops import imtext_ops
from . import base_filler, draw
from . import utils
from . import ner_utils

import random

def fill_data(image, data_value, file_path: str = 'data/idcard/base3.json', pad_factor=0.01,
              randomize=False, rand_prob=0.8, xrange_pos=(-20,10), yrange_pos=(0, 15)):
    config = utils.inject_config(data_value, file_path=file_path)
    if randomize and utils.coin_toss(p=rand_prob):
        # print("randomize true")
        config = randomize_text_position(config, xrpos=xrange_pos, yrpos=yrange_pos)
    default_setting: dict = config.get('default_setting')
    line_height: int = default_setting.get("line_height")
    last_added_line = 0
    datas = []
    for k, v in config['classname'].items():
        obj = config['classname'][k]
        if obj.get("type") == "text":
            
            adjust = obj.get("adjust")
            font_name = obj.get("font_name")
            font_size = obj.get("font_size")
            
            field = obj.get('field', {})
            deli = obj.get('delimiter', {})
            value = obj.get('value', {})

            if field.get('is_used', False):
                ftext = field.get("text")
                if field.get("is_capital", False):
                    ftext = ftext.upper()

                fpos = field.get('position')
                # print(f'============================ {fpos}')
                
                fpos[1] = fpos[1] + last_added_line
                image, data = draw.text(image, ftext,
                                        classname=k, subclass='field', 
                                        pos=fpos,pad_factor=pad_factor,
                                        adjust=adjust, font_name=font_name, 
                                        font_size=font_size)
                
                # print(data)
                datas = datas + data

            if deli.get('is_used', False):
                dtext = deli.get("text", "")
                if deli.get("is_capital", False):
                    dtext = dtext.upper()

                dpos = deli.get('position')
                dpos[1] = dpos[1] + last_added_line
                image, data = draw.text(image, dtext,
                                        classname=k, subclass='delimiter', 
                                        pos=dpos, pad_factor=pad_factor,
                                        adjust=adjust, font_name=font_name, 
                                        font_size=font_size)
                datas = datas + data

            if value.get('is_used', False):
                vtext = value.get("text")
                vtext = str(vtext)
                
                if value.get("is_capital", False): vtext = vtext.upper()
                
                if value.get("is_width_limited"):
                    w, h = imtext_ops.find_textsize(image, vtext, font_name=font_name, font_size=font_size)
                    max_width = config.get('default_setting').get("value").get("max_width")

                    if w > max_width:
                        joined_text = imtext_ops.split_text_by_max_width(
                            image, vtext,
                            max_width=max_width,
                            font_name=font_name, 
                            font_size=font_size
                        )
                        
                        vpos = value.get('position')
                        if last_added_line != 0:
                            vpos[1] = vpos[1] + last_added_line
                        for idx, jtxt in enumerate(joined_text):
                            image, data = draw.text(image, jtxt,
                                                    classname=k, subclass="value", 
                                                    pos=vpos, pad_factor=pad_factor,
                                                    x_center=value.get("x_center", 0),
                                                    x_min=value.get("x_min", 0),
                                                    x_max=value.get("x_max", 0),
                                                    adjust=adjust, font_name=font_name,
                                                    font_size=font_size,
                                                    line=idx)
                            
                            vpos[1] = vpos[1] + line_height
                            datas = datas + data
                        last_added_line = last_added_line + (len(joined_text) - 1) * line_height

                    else:
                        vpos = value.get('position')
                        vpos[1] = vpos[1] + last_added_line
                        image, data = draw.text(image, vtext,
                                                classname=k, subclass='value', 
                                                pos=vpos, pad_factor=pad_factor,
                                                x_center=value.get("x_center", 0),
                                                x_min=value.get("x_min", 0),
                                                x_max=value.get("x_max", 0),
                                                adjust=adjust, font_name=font_name, 
                                                font_size=font_size)
                        datas = datas + data

                else:
                    vpos = value.get('position')
                    image, data = draw.text(image, vtext,
                                            classname=k, subclass='value', pos=vpos,
                                            pad_factor=pad_factor,
                                            x_center=value.get("x_center", 0),
                                            x_min=value.get("x_min", 0),
                                            x_max=value.get("x_max", 0),
                                            adjust=adjust, font_name=font_name, font_size=font_size)
                    
                    data = utils.inject_subclass_and_sequence(data)
                    datas = datas + data

        elif obj.get("type") == "image":
            pos = obj.get('value').get('position')
            fpath = obj.get('value').get('path')
            image = base_filler.fill_photo(image, fpath, face_position=pos)
    
            
    datas = ner_utils.label_genseq_injector(datas)
    # print(datas)
    return image, datas


    
def randomize_text_position(config, xrpos=(-20,10), yrpos=(0, 15)):
    xval = random.randint(xrpos[0], xrpos[1])
    yval = random.randint(yrpos[0], yrpos[1])

    cnames = config['classname']
    for k in cnames.keys():
        child = cnames[k]
        if child['adjust'] == "normal":
            fpos = child['field']['position']
            fpos = [fpos[0] + xval, fpos[1]+yval]
            child['field']['position'] = fpos

            dpos = child['delimiter']['position']
            dpos = [dpos[0] + xval, dpos[1]+yval]
            child['delimiter']['position'] = dpos

            vpos = child['value']['position']
            vpos = [vpos[0] + xval, vpos[1]+yval]
            child['value']['position'] = vpos

        else:
#             print(child)
#             print()
            pass
    
    config['classname'] = cnames
    
    return config