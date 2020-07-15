import cv2 as cv
import numpy as np

from idcard_datagen.utils import imops

__all__ = ['idcard_template', 'put', 'build_content']

idcard_template = {
    'provinsi': {'pos': 20, 'dpos': 'top', 'font_size': 38, 'font_name': 'arial', 'type': 'text'},
    'kabkota': {'pos': 60, 'dpos': 'top', 'font_size': 38, 'font_name': 'arial', 'type': 'text'},
    'nik': {'pos': (260, 140), 'dpos': 'left', 'font_size': 42, 'font_name': 'ocra', 'type': 'text'},
    'nama': {'pos': (275, 194), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'ttl': {'pos': (275, 223), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'gender': {'pos': (275, 253), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'goldar': {'pos': (640, 253), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'alamat': {'pos': (275, 283), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'rtrw': {'pos': (275, 313), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'keldes': {'pos': (275, 343), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'kecamatan': {'pos': (275, 370), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'agama': {'pos': (275, 399), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'status': {'pos': (275, 429), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'pekerjaan': {'pos': (275, 456), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'warga': {'pos': (275, 486), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'berlaku': {'pos': (275, 514), 'dpos': 'left', 'font_size': 23, 'font_name': 'arial', 'type': 'text'},
    'kabkota_bface': {'pos': 453, 'dpos': 'bottom', 'font_size': 19, 'font_name': 'arial', 'type': 'text_below'},
    'tgl_bface': {'pos': 473, 'dpos': 'bottom', 'font_size': 19, 'font_name': 'arial', 'type': 'text_below'},
    'face': {'dpos': 'bottom', 'type': 'picture'}
}

fpos = [720, 170, 965, 450]
face_info = {
    'position': fpos,
    'width': fpos[2] - fpos[0],
    'height': fpos[3] - fpos[1]
}


def put(image, key, text):
    param = idcard_template[key]
    if param['type'] == 'text':
        return put_text(image, key, text)
    elif param['type'] == 'text_below':
        return put_text_below_photo(image, key, text)
    elif param['type'] == 'picture':
        path = text
        return put_photo(image, text)


def put_text(image, key, text):
    param = idcard_template[key]
    if param['dpos'] == 'top':
        return imops.draw_text_center(image, text.upper(), ypos=param['pos'],
                                      font_variant="Bold", font_name=param['font_name'],
                                      font_size=param['font_size'])
    elif param['dpos'] == 'left':
        return imops.draw_text(image, text.upper(), xymin=param['pos'],
                               font_name=param['font_name'], font_size=param['font_size'])
    elif param['dpos'] == 'bottom':
        return imops.draw_text(image, text.upper(), xymin=param['pos'],
                               font_name=param['font_name'], font_size=param['font_size'])
    else:
        pass


def put_photo(image, photo_path, face_position=fpos):
    img = image.copy()
    xmin, ymin, xmax, ymax = face_position
    w, h = xmax - xmin, ymax - ymin

    face_img = cv.imread(photo_path, cv.IMREAD_UNCHANGED)
    face_resize = cv.resize(face_img, (w, h), interpolation=cv.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax] = face_resize

    return img


def get_wh_text(text, font_size=19, font_name="arial", face_position=fpos):
    xmin, ymin, xmax, ymax = face_position
    w, h = xmax - xmin, ymax - ymin

    img_picture = np.zeros((h, w, 3))
    wpos, hpos = imops.get_text_position(img_picture, text.upper(), font_name=font_name, font_size=font_size)
    return wpos, hpos


def put_text_below_photo(image, key, text, face_position=fpos):
    txt = text.upper()
    param = idcard_template[key]
    wpos, hpos = get_wh_text(text)
    xmin, ymin, xmax, ymax = face_position
    out_img = imops.draw_text(image, txt, font_variant="Bold",
                              font_name=param['font_name'], xymin=(xmin + wpos - 10, param['pos']),
                              font_size=param['font_size'])
    return out_img


def build_content(data, image):
    base_image = image.copy()
    for k, v in data.items():
        base_image = put(base_image, k, v)
    return base_image
