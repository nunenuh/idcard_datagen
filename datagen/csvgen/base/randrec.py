import random
import pandas as pd
from tqdm import tqdm
from . import randgen
from . import config


def ktp_data(kode_wilayah=None, 
             name_degree_prob=0.1, 
             name_abbrev_prob=0.2,
             berlaku_date_prob=0.2):
    wilayah = randgen.wilayah(kode_wilayah)
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
    
    gender = randgen.gender()
    gname = "male"
    if gender=="PEREMPUAN":
        gname = "female" 
    
    
    nama = randgen.name(gender=gname, 
                        degree_prob=name_degree_prob, 
                        abbrev_prob=name_abbrev_prob).upper()
    
    data = {    
        'provinsi': provinsi,
        'kabupaten': kabupaten,
        'nik': nik,
        'nama': nama,
        'ttl': ttl,
        'gender': gender,
        'goldar': randgen.blood_type(),
        'alamat': randgen.address().upper(),
        'rtrw': randgen.rtrw().upper(),
        'kelurahan': kelurahan,
        'kecamatan': kecamatan,
        'agama': randgen.agama().upper(),
        'perkawinan': randgen.perkawinan().upper(),
        'pekerjaan': randgen.pekerjaan().upper(),
        'kewarganegaraan': randgen.kewarganegaraan(),
        'berlaku': randgen.berlaku(date_prob=berlaku_date_prob),
        'sign_place': randgen.sign_place(kabupaten),
        'sign_date':  randgen.sign_date().upper()
    }
    
    return data


def ktp_generator(num_data:int, 
                  kode_wilayah=None,
                  name_degree_prob=0.1, 
                  name_abbrev_prob=0.2,
                  berlaku_date_prob=0.2 ,
                  seed=None):
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
        kdata = ktp_data(kode_wilayah=kode_wilayah,
                         name_degree_prob=name_degree_prob, 
                         name_abbrev_prob=name_abbrev_prob,
                         berlaku_date_prob=berlaku_date_prob,)
        for k,v in kdata.items():
            data[k].append(v)
            
    return pd.DataFrame(data)

