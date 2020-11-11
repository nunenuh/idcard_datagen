import random
import pandas as pd
from tqdm import tqdm
from . import randgen
from . import config


def ktp_data():
    wilayah = randgen.wilayah()
    provinsi = "PROVINSI " + wilayah['prov']['nama'].upper()
    kabupaten = wilayah['kab']['nama'].upper()
    kecamatan = wilayah['kec']['nama'].upper()
    kelurahan = wilayah['kel']['nama'].upper()
    
    
    birthdate_string, birthdate_list = randgen.birth_date()
    places_data = [wilayah['kab']['nama'],wilayah['kec']['nama']]
    birth_place = randgen.birth_place(places_data).upper()
    ttl = f'{birth_place}, {birthdate_string}'
    
    nik = randgen.nik(
        wilayah['prov']['kode'], wilayah['kab']['kode'], wilayah['kec']['kode'], 
        birthdate_list[0], birthdate_list[1], birthdate_list[2]
    )
    
    
    data = {    
        'provinsi': provinsi,
        'kabupaten': kabupaten,
        'nik': nik,
        'nama': randgen.name().upper(),
        'ttl': ttl,
        'gender': randgen.gender(),
        'goldar': randgen.blood_type(),
        'alamat': randgen.address().upper(),
        'rtrw': randgen.rtrw().upper(),
        'kelurahan': kelurahan,
        'kecamatan': kecamatan,
        'agama': randgen.agama().upper(),
        'perkawinan': randgen.perkawinan().upper(),
        'pekerjaan': randgen.pekerjaan().upper(),
        'kewarganegaraan': randgen.kewarganegaraan(),
        'berlaku': randgen.berlaku(),
        'sign_place': randgen.sign_place(kabupaten),
        'sign_date':  randgen.sign_date().upper()
    }
    
    return data


def ktp_generator(num_data:int, seed=None):
    if seed==None:
        random.seed(config.random_seed)
        
    data = {    
        'provinsi': [],
        'kabupaten': [],
        'nik': [],
        'nama': [],
        'ttl': [],
        'gender': [],
        'goldar': [],
        'alamat': [],
        'rtrw': [],
        'kelurahan': [],
        'kecamatan': [],
        'agama': [],
        'perkawinan':[],
        'pekerjaan': [],
        'kewarganegaraan': [],
        'berlaku': [],
        'sign_place': [],
        'sign_date':  []
    }
       
    for idx in tqdm(range(num_data)):
        kdata = ktp_data()
        for k,v in kdata.items():
            data[k].append(v)
            
    return pd.DataFrame(data)

