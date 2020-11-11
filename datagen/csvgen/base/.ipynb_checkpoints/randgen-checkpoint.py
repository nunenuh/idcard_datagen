import random

import pandas as pd
from datetime import datetime

from . import config



def random_choice(data, weight, k=1):
    choice = random.choices(data, weights=weight,k=k)
    return choice[0]


def birth_date(age_range=[17,63], date_format="%d-%m-%Y"):
    curr_age = random.randint(age_range[0],age_range[1])
    curr_year = datetime.now().year
    ryear = curr_year - curr_age
    
    rmonth = random.randint(1,12)
    if rmonth % 2 == 0 and rmonth != 2:
        rday = random.randint(1,30)
    elif rmonth % 2 == 0 and rmonth==2:
        rday = random.randint(1,27)
    else: # still not working
        rday = random.randint(1,31)
        
    try:
        bdt = datetime(ryear, rmonth, rday)
    except:
        bdt = datetime(ryear, rmonth, 27)
        
    date_str = bdt.strftime(date_format)
    date_comp_str = bdt.strftime('%d-%m-%y').split('-')
    
    return date_str, date_comp_str

def birth_place(places:list):
    tplaces = []
    for place in places:
        if place.startswith('KOTA'):
            place = place.replace('KOTA ','')
        if place.startswith('KABUPATEN'):
            place = place.replace('KABUPATEN ','')
        
        
        tplaces.append(place)
    places = tplaces
    
    data = [place.capitalize() for place in places]
    weight = [1/len(data) for idx in range(len(data))]
    choice = random_choice(data, weight)
    return choice


def blood_type(csv_path=None):
    if csv_path == None:
        csv_path = config.csv_goldarah
    dframe = pd.read_csv(csv_path)
    data = dframe['texts'].tolist()
    weight = dframe['weights'].tolist()
    choice = random_choice(data, weight)
    return choice
    
def name(csv_path = None):
    if csv_path == None:
            csv_path = config.csv_nama
    dframe = pd.read_csv(csv_path)
    sample = dframe.sample().reset_index(drop=True)
    sample = sample['name'][0]
    return sample


def address(csv_path = None):
    if csv_path == None:
            csv_path = config.csv_address
    dframe = pd.read_csv(csv_path)
    sample = dframe.sample().reset_index(drop=True)
    sample = sample['address'][0].strip()
    return sample
    

def sign_date(year_ago=10, date_format="%d-%m-%Y"):
    curr_year_ago = random.randint(0, year_ago)
    curr_year = datetime.now().year
    ryear = curr_year - curr_year_ago
    
    rmonth = random.randint(1,12)
    if rmonth % 2 == 0 and rmonth != 2:
        rday = random.randint(1,30)
    elif rmonth % 2 == 0 and rmonth==2:
        rday = random.randint(1,27)
    else: # still not working
        rday = random.randint(1,31)
    
    try:
        bdt = datetime(ryear, rmonth, rday)
    except:
        bdt = datetime(ryear, rmonth, 27)
    date_str = bdt.strftime(date_format)
    return date_str

def sign_place(text):
    if "KABUPATEN" in text:
        text = text.split("KABUPATEN ")[-1]
    
    return text


def gender(data=None, weight=None):
    if data==None:
        data=["LAKI-LAKI", "PEREMPUAN"]
    if weight==None:
        weight=[0.4, 0.55]
    choice = random_choice(data, weight)
    return choice


def perkawinan(csv_path = None):
    if csv_path == None:
        csv_path = config.csv_perkawinan
    dframe = pd.read_csv(csv_path)
    data = dframe['texts'].tolist()
    weight = dframe['weights'].tolist()
    choice = random_choice(data, weight)
    return choice


def agama(csv_path=None):
    if csv_path == None:
        csv_path = config.csv_agama
    dframe = pd.read_csv(csv_path)
    data = dframe['texts'].tolist()
    weight = dframe['weights'].tolist()
    choice = random_choice(data, weight)
    return choice

def kewarganegaraan():
    return "WNI"

def berlaku():
    return "SEUMUR HIDUP"

def rtrw():
    st, ed, zf = 1, 30, 3
    rt = str(random.randint(st, ed)).zfill(zf)
    rw = str(random.randint(st, ed)).zfill(zf)
    rtrw_str = f'{rt:3}/{rw:3}'
    return rtrw_str


def pekerjaan(csv_path=None):
    if csv_path == None:
        csv_path = config.csv_pekerjaan
    dframe = pd.read_csv(csv_path)
    data = dframe['texts'].tolist()
    weight = dframe['weights'].tolist()
    choice = random_choice(data, weight)
    return choice


def clean_kab_text(text):
    if text.startswith('KOTA ADM.'):
        text = text.replace('KOTA ADM. ','KOTA')
    if text.startswith("KAB."):
        text = text.replace('KAB. ','KABUPATEN ')
    if text.startswith("KAB"):
        text = text.replace('KAB ','KABUPATEN ')
    return text


def wilayah(kode_wilayah=None, csv_path=None):
    if csv_path == None:
        csv_path = config.csv_wilayah
    
    if kode_wilayah !=None:
        pass
    
    df = pd.read_csv(csv_path)
    dfkel = df[df['kode'].str.len()>8]
    kel_sample = dfkel.sample().reset_index(drop=True)
    
    kode_kel, nama_kel = kel_sample['kode'][0], kel_sample['nama'][0]
    kode_prov, kode_kab, kode_kec, kode_kel = kode_kel.split('.')
    nama_prov = df[df['kode']==f'{kode_prov}'].iloc[0]['nama']
    nama_kab = df[df['kode']==f'{kode_prov}.{kode_kab}'].iloc[0]['nama']
    nama_kec = df[df['kode']==f'{kode_prov}.{kode_kab}.{kode_kec}'].iloc[0]['nama']

    nama_kab = clean_kab_text(nama_kab)

    dout = {
        'prov':{'kode':kode_prov, 'nama': nama_prov},
        'kab': {'kode': kode_kab, 'nama': nama_kab},
        'kec': {'kode': kode_kec, 'nama': nama_kec},
        'kel': {'kode': kode_kel, 'nama': nama_kel},
    }

    
    return dout


def nik(kp, kb, kl, kd, km, ky):
    no_komp = str(random.randint(0, 9999)).zfill(4)
    nik_gen = f'{kp}{kb}{kl}{kd}{km}{ky}{no_komp}'
    return nik_gen