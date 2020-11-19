fpos = [730, 130, 945, 400]

face_info = {
    'position': fpos,
    'width': fpos[2] - fpos[0],
    'height': fpos[3] - fpos[1]
}

idcard_value_template = {
    'provinsi': {'pos': 30, 'adjust': 'center',
                 'font_size': 30, 'font_name': 'arial', 'upper': True,
                 'type': 'text', 'subclass': 'value'},
    'kabkota': {'pos': 60, 'adjust': 'center',
                'font_size': 30, 'font_name': 'arial', 'upper': True,
                'type': 'text', 'subclass': 'value'},
    'nik': {'pos': (260, 100), 'adjust': 'normal',
            'font_size': 42, 'font_name': 'ocra', 'upper': True,
            'type': 'text', 'subclass': 'value'},
    'nama': {'pos': (275, 164), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': True,
             'type': 'text', 'subclass': 'value'},
    'ttl': {'pos': (275, 193), 'adjust': 'normal',
            'font_size': 23, 'font_name': 'arial', 'upper': True,
            'type': 'text', 'subclass': 'value'},
    'gender': {'pos': (275, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': True,
               'type': 'text', 'subclass': 'value'},
    'goldar': {'pos': (640, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': True,
               'type': 'text', 'subclass': 'value'},
    'alamat': {'pos': (275, 253), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': True,
               'type': 'text', 'subclass': 'value'},
    'rtrw': {'pos': (275, 283), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': True,
             'type': 'text', 'subclass': 'value'},
    'keldes': {'pos': (275, 313), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': True,
               'type': 'text', 'subclass': 'value'},
    'kecamatan': {'pos': (275, 340), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': True,
                  'type': 'text', 'subclass': 'value'},
    'agama': {'pos': (275, 369), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': True,
              'type': 'text', 'subclass': 'value'},
    'status': {'pos': (275, 399), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': True,
               'type': 'text', 'subclass': 'value'},
    'pekerjaan': {'pos': (275, 426), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': True,
                  'type': 'text', 'subclass': 'value'},
    'warga': {'pos': (275, 456), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': True,
              'type': 'text', 'subclass': 'value'},
    'berlaku': {'pos': (275, 484), 'adjust': 'normal',
                'font_size': 23, 'font_name': 'arial', 'upper': True,
                'type': 'text', 'subclass': 'value'},
    'kabkota_bface': {'pos': 413, 'adjust': 'center',
                      "x_center": 842, "x_min": fpos[0], 'x_max': fpos[2],
                      'font_size': 19, 'font_name': 'arial', 'upper': True,
                      'type': 'text', 'subclass': 'value'},
    'tgl_bface': {'pos': 433, 'adjust': 'center',
                  "x_center": 842, 'x_min': fpos[0], 'x_max': fpos[2],
                  'font_size': 19, 'font_name': 'arial', 'upper': True,
                  'type': 'text', 'subclass': 'value'},
    'face': {'adjust': 'bottom', 'type': 'picture'}
}

idcard_field_template = {
    'nik': {'text': 'NIK', 'pos': (30, 100), 'adjust': 'normal',
            'font_size': 42, 'font_name': 'ocra', 'upper': False,
            'type': 'text', 'subclass': 'field'},
    'nama': {'text': 'Nama', 'pos': (30, 164), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': False,
             'type': 'text', 'subclass': 'field'},
    'ttl': {'text': 'Tempat/Tgl Lahir', 'pos': (30, 193), 'adjust': 'normal',
            'font_size': 23, 'font_name': 'arial', 'upper': False,
            'type': 'text', 'subclass': 'field'},
    'gender': {'text': 'Jenis Kelamin', 'pos': (30, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'field'},
    'goldar': {'text': 'Gol. Darah', 'pos': (520, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'field'},
    'alamat': {'text': 'Alamat', 'pos': (30, 253), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'field'},
    'rtrw': {'text': 'RT/RW', 'pos': (70, 283), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': False,
             'type': 'text', 'subclass': 'field'},
    'keldes': {'text': 'Kel/Desa', 'pos': (70, 313), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'field'},
    'kecamatan': {'text': 'Kecamatan', 'pos': (70, 340), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': False,
                  'type': 'text', 'subclass': 'field'},
    'agama': {'text': 'Agama', 'pos': (30, 369), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': False,
              'type': 'text', 'subclass': 'field'},
    'status': {'text': 'Status Perkawinan', 'pos': (30, 399), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'field'},
    'pekerjaan': {'text': 'Pekerjaaan', 'pos': (30, 426), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': False,
                  'type': 'text', 'subclass': 'field'},
    'warga': {'text': 'Kewarganegaraan', 'pos': (30, 456), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': False,
              'type': 'text', 'subclass': 'field'},
    'berlaku': {'text': 'Berlaku Hingga', 'pos': (30, 484), 'adjust': 'normal',
                'font_size': 23, 'font_name': 'arial', 'upper': False,
                'type': 'text', 'subclass': 'field'},
}

idcard_delimiter_template = {
    'nik': {'text': ':', 'pos': (220, 100), 'adjust': 'normal',
            'font_size': 42, 'font_name': 'ocra', 'upper': False,
            'type': 'text', 'subclass': 'delimiter'},
    'nama': {'text': ':', 'pos': (260, 164), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': False,
             'type': 'text', 'subclass': 'delimiter'},
    'ttl': {'text': ':', 'pos': (260, 193), 'adjust': 'normal',
            'font_size': 23, 'font_name': 'arial', 'upper': False,
            'type': 'text', 'subclass': 'delimiter'},
    'gender': {'text': ':', 'pos': (260, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'delimiter'},
    'goldar': {'text': ':', 'pos': (635, 223), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'delimiter'},
    'alamat': {'text': ':', 'pos': (260, 253), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'delimiter'},
    'rtrw': {'text': ':', 'pos': (260, 283), 'adjust': 'normal',
             'font_size': 23, 'font_name': 'arial', 'upper': False,
             'type': 'text', 'subclass': 'delimiter'},
    'keldesa': {'text': ':', 'pos': (260, 313), 'adjust': 'normal',
                'font_size': 23, 'font_name': 'arial', 'upper': False,
                'type': 'text', 'subclass': 'delimiter'},
    'kecamatan': {'text': ':', 'pos': (260, 340), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': False,
                  'type': 'text', 'subclass': 'delimiter'},
    'agama': {'text': ':', 'pos': (260, 369), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': False,
              'type': 'text', 'subclass': 'delimiter'},
    'status': {'text': ':', 'pos': (260, 399), 'adjust': 'normal',
               'font_size': 23, 'font_name': 'arial', 'upper': False,
               'type': 'text', 'subclass': 'delimiter'},
    'pekerjaan': {'text': ':', 'pos': (260, 426), 'adjust': 'normal',
                  'font_size': 23, 'font_name': 'arial', 'upper': False,
                  'type': 'text', 'subclass': 'delimiter'},
    'warga': {'text': ':', 'pos': (260, 456), 'adjust': 'normal',
              'font_size': 23, 'font_name': 'arial', 'upper': False,
              'type': 'text', 'subclass': 'delimiter'},
    'berlaku': {'text': ':', 'pos': (260, 484), 'adjust': 'normal',
                'font_size': 23, 'font_name': 'arial', 'upper': False,
                'type': 'text', 'subclass': 'delimiter'},
}


import json

def inject_config(fpath: str, data_value: dict):
    with open(fpath) as json_file:
        data = json.load(json_file)
    for idx, (k, v) in enumerate(data['classname'].items()):
        if data['classname'][k]['type'] == "text":
            data['classname'][k]['value']['text'] = data_value[k]
        elif data['classname'][k]['type'] == "image":
            data['classname'][k]['path'] = data_value[k]
        else:
            pass

    return data


if __name__ == "__main__":
    data_value = {
        'provinsi': "Provinsi Nusa Tenggara Barat",
        'kabkota': "Kota Mataram",
        'nik': "1050245708900002",
        'nama': "Lalu Erfandi Maula Yusnu",
        'ttl': "Mataram, 24-10-1988",
        'goldar': 'O',
        'gender': 'Laki-Laki',
        'alamat': 'Jl Musium No 19 A',
        'rtrw': '003/005',
        'keldesa': "Taman Sari",
        'kecamatan': 'Ampenan',
        'agama': 'Islam',
        'status': 'Kawin',
        'pekerjaan': 'Wiraswasta',
        'warga': 'WNI',
        'berlaku': 'Seumur Hidup',
        'sign_kabkota': 'Kota Mataram',
        'sign_tgl': '22-09-2016',
        'face': '../data/face/fandi.png',
    }

    fpath = "../../data/idcard/base3.json"
    data = inject_config(fpath, data_value)

    print(data['classname']["face"])
