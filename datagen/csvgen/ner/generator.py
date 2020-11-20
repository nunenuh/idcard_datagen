
import pandas as pd
import numpy as np
from datagen.csvgen.ner import converter
from pathlib import Path
import time
from si_prefix import si_format


def convert_to_ner(csv_path, dst_path):
    csv_path, dst_path = clean_parameter(csv_path, dst_path)
    
    idcard_data = pd.read_csv(str(csv_path))
    print(f'Logs:\t Length of csv data is {len(idcard_data)} record')
    
    print(f'Logs:\t Prepare to convert dataframe from csv data to ner format')
    ktp_ner = converter.to_ner_dataframe(idcard_data)
    
    
    print(f'Logs:\t Prepare to split converted ner format data to train, valid and test dataframe')
    trainframe, validframe, testframe = converter.split_ner_dataframe(ktp_ner)
    
    print(f'Logs:\t Reset sentence index of train dataframe')
    trainframe = converter.reset_sentence_index(trainframe)
    
    print(f'Logs:\t Reset sentence index of valid dataframe')
    validframe = converter.reset_sentence_index(validframe)
    
    print(f'Logs:\t Reset sentence index of test dataframe')
    testframe = converter.reset_sentence_index(testframe)
    
    time_number = str(f'{time.time():.0f}')
    base_path = dst_path.joinpath(str(time_number))
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Logs:\t Save all data to {str(dst_path.joinpath(time_number))}')
    save_dataframe(trainframe, prefix_filename="trainset", dst_path=dst_path, prefix_dirname=time_number)
    save_dataframe(validframe, prefix_filename="validset", dst_path=dst_path, prefix_dirname=time_number)
    save_dataframe(testframe, prefix_filename="testset", dst_path=dst_path, prefix_dirname=time_number)



def num_format(num, precision=0):
    out = si_format(num, precision=precision)
    out = out.split(" ")
    out = "".join(out)
    return out


def build_filename(prefix, num):
    numk = num_format(num)
    fname = f'{prefix}_{numk}.csv'
    return fname

def clean_parameter(csv_path, dst_path):
    csv_path = Path(csv_path)
    dst_path = Path(dst_path)
    
    if not (csv_path.exists() and csv_path.is_file()):
        raise ValueError(f"Directory path to csv_path at {str(csv_path)} is not exist!")

    if not dst_path.exists():
        raise ValueError(f"Directory path to dst_path at {str(dst_path)} is not exist!")
    
    return csv_path, dst_path


def save_dataframe(dframe, prefix_filename, dst_path, prefix_dirname):
    fname = build_filename(prefix_filename, len(dframe))
    fpath = dst_path.joinpath(prefix_dirname).joinpath(fname)
    fpath = str(fpath)
    
    dframe.to_csv(fpath, index=False, index_label=False)
