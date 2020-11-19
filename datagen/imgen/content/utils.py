
import json
import numpy as np
from ...config import data_config


def inject_config(data_value: dict, file_path: str = 'data/idcard/base3.json'):
    with open(file_path) as json_file:
        data = json.load(json_file)
    # print(data_value)
    for idx, (k, v) in enumerate(data['classname'].items()):
        if data['classname'][k]['type'] == "text":
            data['classname'][k]['value']['text'] = data_value[k]
        elif data['classname'][k]['type'] == "image":
            data['classname'][k]['value']['path'] = data_value[k]
        else:
            pass

    return data

def convert_json_boxes_to_numpy(data_dict: dict):
    boxes = []
    box_pts = data_dict.get('points')
    boxes.append(box_pts)
    objects: list = data_dict.get('objects')
    for obj in objects:
        # print(obj.get('classname'), obj.get('text'))
        pts = obj.get('points')
        boxes.append(pts)

    cnames, scnames, sequence, texts = create_class_number(data_dict)

    return np.array(boxes), cnames, scnames, sequence, texts


def create_class_number(data_dict: dict):
    objects = data_dict.get('objects').copy()
    csname, scname, sequence, text = [], [], [], []

    for obj in objects:
        cn = obj.get('classname')
        scn = obj.get('subclass')
        seq = obj.get('sequence')
        txt = obj.get("text")
        csname.append(data_config.classname[cn])
        scname.append(data_config.subclassname[scn])
        sequence.append(seq)
        text.append(txt)

    return csname, scname, sequence, text