import time
from pathlib import Path
from si_prefix import si_format

from . import randrec


def num_format(num, precision=0):
    out = si_format(num, precision=precision)
    out = out.split(" ")
    out = "".join(out)
    return out


def clean_kode(kode):
    if kode != None and '|' in kode:
            kode = kode.split("|")
    return kode

def clean_dst_path(dst_path):
    dst_path = Path(dst_path)
    if not (dst_path.exists() and dst_path.is_dir()):
        raise ValueError(f"Directory path to dst_path at {str(dst_path)} is not exist!")
    return dst_path

def build_filename(num):
    time_number = str(f'{time.time():.0f}')
    numk = num_format(num)
    fname = f'idcard_{numk}_{time_number}.csv'
    return fname
    

def generate(num, dst_path, kode):
    kode = clean_kode(kode)
    dst_path = clean_dst_path(dst_path)
    
    fname = build_filename(num)
    fpath = str(dst_path.joinpath(fname))
    
    print(f'Prepare to generate data, please wait!')
    ktp_data = randrec.ktp_generator(num, kode_wilayah=kode)
    
    print(f'Saving data to {fpath}!')
    ktp_data.to_csv(fpath, index=False, index_label=False)