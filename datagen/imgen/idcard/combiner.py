import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import json


from collections import OrderedDict

from datagen.imgen.ops import boxes_ops
from datagen.imgen import transforms
from datagen.imgen.content import utils as content_utils
from datagen.config import data_config
from datagen.imgen.io import fop

from tqdm import tqdm

      
def bg_data_balancer(bg_data, idcard_image_data):
    if len(idcard_image_data) > len(bg_data):
        mn_factor = len(idcard_image_data) // len(bg_data)
        bg_data = bg_data * mn_factor * 2
        bg_data = random.choices(bg_data, k=len(idcard_image_data))
    else:
        bg_data = random.choices(bg_data, k=len(idcard_image_data))
    return bg_data

def open_json_file(path):
    with open(str(path), 'r') as js_file:
        json_data = json.load(js_file)
    return json_data

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

def clean_background_size(bg_size):
    if bg_size != None:
        if 'x' in bg_size:
            w , h = bg_size.split('x')
            bg_size = (int(w),int(h))
        else:
            raise ValueError(f"--bg_size format error fill with WxH format, e.g 1200x1000")
    return bg_size

def add_white_gray_files(bg_path, bg_data, repeat_factor=None,
                         wg_file = [ 'w1.jpg','w2.jpg','w3.jpg','w4.jpg',
                                    'gray1.jpg','gray2.jpg','gray3.jpg','gray4.jpg']):
    bg_wg = []
    if repeat_factor==None:
        repeat = int(len(bg_data)/len(wg_file))
    else:
        repeat = int((len(bg_data)*repeat_factor)/len(wg_file))
    for i in range(repeat):
        for fname in wg_file:
            bgp  = bg_path.joinpath(fname)
            bg_wg.append(bgp)

    new_bg_data = bg_data + bg_wg
    random.shuffle(new_bg_data)
    return new_bg_data

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
    scale_ratio = scale_ratio.split(",")
    scale_ratio = [float(ratio.strip()) for ratio in scale_ratio]
    scale_ratio: list = [i for i in np.arange(scale_ratio[0], scale_ratio[1], sfactor)]

    num_generated = int(num_generated)
    print(f'Info: angle={str(angle)} shear_factor={str(shear)} '
          f'scale_ratio={str(scale_ratio)}')

    return angle, shear, scale_ratio, num_generated


def combine_single(bgfile, idfile, jsfile, base_path: Path,
                   bg_size, scale_ratio, angle, shear, 
                   force_resize: bool = False,
                   use_basic_effect: bool = True,
                   basic_effect_mode: str = "simple",
                   use_adv_effect: bool = True,
                   adv_effect_mode: str = "simple"):
    
    id_img = cv.imread(str(idfile), cv.IMREAD_UNCHANGED)
    bg_img = cv.imread(str(bgfile), cv.IMREAD_COLOR)
    json_data = open_json_file(jsfile)
    
    if bg_size != None:
        bsw, bsh = bg_size
        bgh, bgw = bg_img.shape[:2]
        if force_resize==False:
            if bgh>bgw: # potrait
                bg_size = (bsh, bsw)
            if bgw>=bgh: #landscape
                bg_size = (bsw, bsh)
        
        bg_img = cv.resize(bg_img, bg_size, interpolation=cv.INTER_LINEAR)
        

    # reformat json data
    data_record  = content_utils.reformat_json_data(json_data)
    mwboxes, cnames, scnames, texts, labels, sequence, genseq, cboxes, clist = data_record
    
    
    #prepare for augment
    mwboxes = mwboxes.reshape(-1, 8)
    cboxes_list = []
    for cbox in cboxes:
        cbox = cbox.reshape(-1, 8)
        cboxes_list.append(cbox)
    cboxes = cboxes_list
    
    #Augment Generator
    ratio = random.choice(scale_ratio)
    augment = transforms.AugmentGenerator(scale_ratio=ratio, angle=angle, shear_factor=shear,
                                          use_basic_effect=use_basic_effect, 
                                          basic_effect_mode=basic_effect_mode,
                                          use_adv_effect=use_adv_effect,
                                          adv_effect_mode=adv_effect_mode)
    
    seg_img, cmp_img, mwboxes, cboxes = augment(bg_img, id_img, mwboxes, cboxes)
    seg_img = (seg_img * 255).astype(np.uint8)
    
    #prepare for creating child_boxes
    main_boxes = mwboxes[0].copy()
    main_boxes = boxes_ops.order_points(main_boxes).tolist()
    word_boxes = mwboxes[1:len(mwboxes)].copy()
    word_boxes = boxes_ops.order_points_batch(word_boxes).tolist()
    
    #prepare chardata
    chardata_list = []
    for (cbox, clist) in zip(cboxes, clist):
        cbox = boxes_ops.order_points_batch(cbox).tolist()
        cdata_list = []
        for (cb, cl) in zip(cbox, clist):
            cdict = OrderedDict({"char": cl, "points": cb})
            cdata_list.append(cdict)
            
        chardata_list.append(cdata_list)
        
    #build and append every text
    objects = []
    zipped_iter = [word_boxes, chardata_list, cnames, scnames, texts, labels, sequence, genseq]
    for (wbox, cdata, cn, scn, txt, lbl, seq, gs) in zip(*zipped_iter):
        dt = OrderedDict({
            'text': txt, 
            'points': wbox,
            'classname': data_config.classname_list[cn],
            'subclass': data_config.subclassname_list[scn],
            'chardata': cdata,
            'label': lbl,
            'sequence': seq,
            'genseq': gs,
        })
        objects.append(dt)
        
    #prepare for savinf data
    rnum = str(random.randrange(0, 999999))
    image_fpath = base_path.joinpath(f'{rnum}_image.jpg')
    mask_fpath = base_path.joinpath(f'{rnum}_mask.jpg')
    json_fpath = base_path.joinpath(f'{rnum}_json.json')
    
    json_dict = {
        'image': {'filename': str(image_fpath.name), 'dim': cmp_img.shape},
        'mask': {'filename': str(mask_fpath.name), 'dim': seg_img.shape},
        'scale_ratio': ratio,
        'angle': augment.actual_angle,
        'shear_factor': augment.actual_shear,
        'box': main_boxes,
        'objects': objects,
    }
    
    data = {
        'image_path': str(image_fpath),
        'image_data': cmp_img,
        'mask_path': str(mask_fpath),
        'mask_data': seg_img,
        'json_path': str(json_fpath),
        'json_data': json_dict
    }
    
    return data


def combine(bg_path, idcard_path, dst_path,
            idcard_ext="png", bg_ext="jpg|png", bg_size=None,
            balance_white_bg=False, balance_bg=False, sampled_bg=True,
            white_bg_factor=1.0,
            angle: int = 30, shear: float = 0.5,
            scale_ratio: str = "0.3,0.8",
            num_generated: int = 6,
            force_resize: bool = False,
            use_basic_effect: bool = True,
            basic_effect_mode: str = "simple",
            use_adv_effect: bool = True,
            adv_effect_mode: str = "simple"):

    
    bg_data, bg_path = clean_background_data(bg_path, bg_ext)
    if balance_white_bg:
        bg_data = add_white_gray_files(bg_path, bg_data, repeat_factor=white_bg_factor)
    
    idcard_image_data, idcard_json_data = clean_idcard_data(
        idcard_path, image_ext=idcard_ext
    )
    bg_size = clean_background_size(bg_size)
    dst_path, base_path = clean_destination_path(dst_path)
    
    augment_param = clean_augment_param(angle, shear, scale_ratio, num_generated)
    angle, shear, scale_ratio, num_generated  = augment_param
    
    if balance_bg:
        bg_data = bg_data_balancer(bg_data, idcard_image_data)
        

    c = 0
    tc = len(idcard_image_data) * num_generated
    print(f'Length of idcard data is {tc}')
    if balance_bg:
        idcard_bar = tqdm(zip(idcard_image_data, idcard_json_data, bg_data))
    else:
        idcard_bar = tqdm(zip(idcard_image_data, idcard_json_data))
        
    for idcard_zipped in idcard_bar:
        if balance_bg:
            idfile, jsfile, bgfile = idcard_zipped
        else:
            idfile, jsfile = idcard_zipped
            
        if sampled_bg:
            bgfile = random.sample(bg_data, k=1)[0]
            
        for n in range(num_generated):

            info =  f"Progress ({str(c)}/{str(tc)}): "
            info += f"Processing augmented ({str(n)}/{str(num_generated)}) saved to {str(base_path)}"
            idcard_bar.set_description(info)

            data = combine_single(bgfile, idfile, jsfile, base_path, 
                                  bg_size, scale_ratio, angle, shear,
                                  force_resize=force_resize,
                                  use_basic_effect=use_basic_effect,
                                  basic_effect_mode=basic_effect_mode,
                                  use_adv_effect=use_adv_effect,
                                  adv_effect_mode=adv_effect_mode)
            
            image_fpath, cmp_img = data['image_path'], data['image_data']
            mask_fpath, seg_img = data['mask_path'], data['mask_data']
            json_fpath, json_dict = data['json_path'], data['json_data']
            
            
            cv.imwrite(str(image_fpath), cmp_img)
            cv.imwrite(str(mask_fpath), seg_img)
            fop.save_json_file(str(json_fpath), json_dict)
            c = c + 1
                
