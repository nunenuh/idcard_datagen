
import json
import numpy as np
from ...config import data_config
import random


def coin_toss(p=0.5, n=1):    
    result = np.random.binomial(n,p) 
    return bool(result)

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


def inject_subclass_and_sequence(data):
    seq = 0
    for idx in range(len(data)):
        if idx==0:
            cname = data[idx]['classname'] 
            text = data[idx]['text']
            
            if cname  == 'provinsi':
                data[idx]['subclass'] = 'field'

            if  cname == 'kabupaten' and ( text == "KOTA" or text == 'KABUPATEN'):
                    data[idx]['subclass'] = 'field'
        else:
            data[idx]['sequence'] = seq
            seq += 1
            
    return data

def reformat_json_data(data_dict: dict):
    boxes = []
    box_pts = data_dict.get('points')
    boxes.append(box_pts)
    objects: list = data_dict.get('objects')
    cboxes, clist = [], []
    for obj in objects:
        # print(obj.get('classname'), obj.get('text'))
        pts = obj.get('points')
        boxes.append(pts)
        
        cchar, cbox = [], []
        char_data = obj.get('chardata')
        # if char_data != None:
        for obj_char in char_data:
            char_pts = obj_char.get('points')
            char_letter = obj_char.get('char')
            
            cbox.append(char_pts)
            cchar.append(char_letter)
            
        cboxes.append(np.array(cbox))
        clist.append(cchar)
        
            

    cnames, scnames, texts, labels, line, sequence, genseq, char_data = create_class_number(data_dict)

    return np.array(boxes), cnames, scnames, texts, labels, line, sequence, genseq, cboxes, clist


def create_class_number(data_dict: dict):
    objects = data_dict.get('objects').copy()
    csname, scname, sequence, text, label, line, genseq = [], [], [], [], [], [], []
    char_data = []
    
    for obj in objects:
        cn = obj.get('classname')
        scn = obj.get('subclass')
        seq = obj.get('sequence')
        txt = obj.get("text")
        lbl = obj.get('label')
        ln = obj.get("line")
        gs = obj.get('genseq')
        cdata = obj.get('chardata')
        
        csname.append(data_config.classname[cn])
        scname.append(data_config.subclassname[scn])
        sequence.append(seq)
        label.append(lbl)
        line.append(ln)
        text.append(txt)
        genseq.append(gs)
        char_data.append(cdata)
        

    return csname, scname, text, label, line, sequence, genseq, char_data