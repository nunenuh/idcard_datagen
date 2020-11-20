import argparse
from datagen.csvgen.ner import generator as ner_converter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ID Card csv data generator')

    parser.add_argument('--csv_path', type=str, required=True,
                        help='path to csv file that generated from idcard_csv_generator')

    parser.add_argument('--dst_path', type=str, required=True, 
                        help='destination path for generating result data')

    args = parser.parse_args()
    
    ner_converter.convert_to_ner(
        csv_path = args.csv_path,
        dst_path = args.dst_path,
    )