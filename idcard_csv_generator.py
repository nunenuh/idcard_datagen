import argparse
from datagen.csvgen.base import generator as csv_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ID Card csv data generator')

    parser.add_argument('--num', type=int, required=True,
                        help='num generated data')

    parser.add_argument('--dst_path', type=str, required=True, 
                        help='destination path for generating result data')

    parser.add_argument('--kode', type=str, default=None, 
                        help='kode wilayah with separator | , example 31|32|52 ')

    args = parser.parse_args()
    
    csv_generator.generate(
        num = args.num,
        dst_path = args.dst_path,
        kode = args.kode
    )