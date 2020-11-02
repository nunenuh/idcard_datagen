import json

import cv2 as cv
import numpy as np

from . import idcard_config, imtext
from . import imops

__all__ = ['put_content', 'build_content']


def put_content(image, key, text):
    param = idcard_config.idcard_value_template[key]
    data = []
    if param['type'] == 'text':
        img, data = put_text(image, key, text)
    elif param['type'] == 'picture':
        path = text
        img = put_photo(image, text)
        data = []

    return img, data


def put_text(image, key, text):
    text = str(text)
    param: dict = idcard_config.idcard_value_template[key]
    if param['upper']: text = text.upper()
    if param['adjust'] == 'normal':
        img, data = imtext.datalog_drawtext(image, text, classname=key, subclass=param['subclass'],
                                            pos=param['pos'], adjust=param['adjust'],
                                            font_name=param['font_name'], font_size=param['font_size'])
    elif param['adjust'] == 'center':
        x_center = param.get('x_center', 0)
        x_min = param.get('x_min', 0)
        x_max = param.get('x_max', 0)
        img, data = imtext.datalog_drawtext(image, text, classname=key, subclass=param['subclass'],
                                            pos=param['pos'], adjust=param['adjust'],
                                            x_center=x_center, x_min=x_min, x_max=x_max,
                                            font_name=param['font_name'], font_size=param['font_size'])

    return img, data


def put_photo(image, photo_path, face_position=idcard_config.fpos):
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


def get_wh_text(text, font_size=19, font_name="arial", face_position=idcard_config.fpos):
    xmin, ymin, xmax, ymax = face_position
    w, h = xmax - xmin, ymax - ymin

    img_picture = np.zeros((h, w, 3))
    wpos, hpos = imops.get_text_position(img_picture, text.upper(), font_name=font_name, font_size=font_size)
    return wpos, hpos


def put_text_below_photo(image, key, text, face_position=idcard_config.fpos):
    txt = text.upper()
    param = idcard_config.idcard_template[key]
    wpos, hpos = get_wh_text(text)
    xmin, ymin, xmax, ymax = face_position
    out_img = imops.draw_text(image, txt, font_variant="Bold",
                              font_name=param['font_name'], xymin=(xmin + wpos - 10, param['pos']),
                              font_size=param['font_size'])
    return out_img


def inject_config(data_value: dict, file_path: str = 'data/idcard/base3.json'):
    with open(file_path) as json_file:
        data = json.load(json_file)
    for idx, (k, v) in enumerate(data['classname'].items()):
        if data['classname'][k]['type'] == "text":
            data['classname'][k]['value']['text'] = data_value[k]
        elif data['classname'][k]['type'] == "image":
            data['classname'][k]['value']['path'] = data_value[k]
        else:
            pass

    return data


def build_content(image, data_value, file_path: str = 'data/idcard/base3.json', pad_factor=0.01):
    config = inject_config(data_value, file_path=file_path)
    default_setting: dict = config.get('default_setting')
    line_height: int = default_setting.get("line_height")
    last_added_line = 0
    datas = []
    for k, v in config['classname'].items():
        obj = config['classname'][k]
        if obj.get("type") == "text":
            adjust, font_name, font_size = obj.get("adjust"), obj.get("font_name"), obj.get("font_size")
            field, deli, value = obj.get('field', {}), obj.get('delimiter', {}), obj.get('value', {})

            if field.get('is_used', False):
                ftext = field.get("text")
                if field.get("is_capital", False):
                    ftext = ftext.upper()

                fpos = field.get('position')
                fpos[1] = fpos[1] + last_added_line
                image, data = imtext.datalog_drawtext(image, ftext,
                                                      classname=k, subclass='field', pos=fpos,
                                                      pad_factor=pad_factor,
                                                      adjust=adjust, font_name=font_name, font_size=font_size)
                datas = datas + data

            if deli.get('is_used', False):
                dtext = deli.get("text", "")
                if deli.get("is_capital", False):
                    dtext = dtext.upper()

                dpos = deli.get('position')
                dpos[1] = dpos[1] + last_added_line
                image, data = imtext.datalog_drawtext(image, dtext,
                                                      classname=k, subclass='delimiter', pos=dpos,
                                                      pad_factor=pad_factor,
                                                      adjust=adjust, font_name=font_name, font_size=font_size)
                datas = datas + data

            if value.get('is_used', False):
                vtext = value.get("text")
                vtext = str(vtext)
                if value.get("is_capital", False):
                    vtext = vtext.upper()
                if value.get("is_width_limited"):
                    w, h = imtext.find_textsize(image, vtext, font_name=font_name, font_size=font_size)
                    max_width = config.get('default_setting').get("value").get("max_width")

                    if w > max_width:
                        joined_text = imtext.split_text_by_max_width(image, vtext,
                                                                     max_width=max_width,
                                                                     font_name=font_name, font_size=font_size)
                        vpos = value.get('position')
                        if last_added_line != 0:
                            vpos[1] = vpos[1] + last_added_line
                        for jtxt in joined_text:
                            image, data = imtext.datalog_drawtext(image, jtxt,
                                                                  classname=k, subclass='value', pos=vpos,
                                                                  pad_factor=pad_factor,
                                                                  x_center=value.get("x_center", 0),
                                                                  x_min=value.get("x_min", 0),
                                                                  x_max=value.get("x_max", 0),
                                                                  adjust=adjust, font_name=font_name,
                                                                  font_size=font_size)
                            vpos[1] = vpos[1] + line_height
                            datas = datas + data
                        last_added_line = last_added_line + (len(joined_text) - 1) * line_height

                    else:
                        vpos = value.get('position')
                        vpos[1] = vpos[1] + last_added_line
                        image, data = imtext.datalog_drawtext(image, vtext,
                                                              classname=k, subclass='value', pos=vpos,
                                                              pad_factor=pad_factor,
                                                              x_center=value.get("x_center", 0),
                                                              x_min=value.get("x_min", 0),
                                                              x_max=value.get("x_max", 0),
                                                              adjust=adjust, font_name=font_name, font_size=font_size)
                        datas = datas + data

                else:
                    # print(texts)
                    vpos = value.get('position')
                    image, data = imtext.datalog_drawtext(image, vtext,
                                                          classname=k, subclass='value', pos=vpos,
                                                          pad_factor=pad_factor,
                                                          x_center=value.get("x_center", 0),
                                                          x_min=value.get("x_min", 0),
                                                          x_max=value.get("x_max", 0),
                                                          adjust=adjust, font_name=font_name, font_size=font_size)
                    datas = datas + data

        elif obj.get("type") == "image":
            pos = obj.get('value').get('position')
            fpath = obj.get('value').get('path')
            image = put_photo(image, fpath, face_position=pos)

    return image, datas


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
        csname.append(classname[cn])
        scname.append(subclassname[scn])
        sequence.append(seq)
        text.append(txt)

    return csname, scname, sequence, text


def revert_to_dict():
    pass


classname = {
    'provinsi': 0,
    'kabkota': 1,
    'nik': 2,
    'nama': 3,
    'ttl': 4,
    'gender': 5,
    'goldar': 6,
    'alamat': 7,
    'rtrw': 8,
    'keldesa': 9,
    'kecamatan': 10,
    'agama': 11,
    'status': 12,
    'pekerjaan': 13,
    'warga': 14,
    'berlaku': 15,
    'sign_kabkota': 16,
    'sign_tgl': 17,
}

classname_list = [
    'provinsi', 'kabkota',
    'nik', 'nama', 'ttl', 'gender', 'goldar',
    'alamat', 'rtrw', 'keldesa', 'kecamatan',
    'agama', 'status', 'pekerjaan', 'warga', 'berlaku',
    'sign_kabkota', 'sign_tgl',
]

subclassname = {'field': 0, 'value': 1, 'delimiter': 2}
subclassname_list = ['field', 'value', 'delimiter']