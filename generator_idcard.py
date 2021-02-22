import argparse
from datagen.imgen.idcard import generator


parser = argparse.ArgumentParser(description='IDCard base data generator tools')

parser.add_argument('--csv_path', type=str, required=True,
                    help='path to csv path for generating data')

parser.add_argument('--dst_path', type=str, required=True, 
                    help='destination path for generating result data')

parser.add_argument('--image_path', type=str, required=True,
                    default='data/idcard/base3.png', 
                    help='source directory for background image')

parser.add_argument('--json_path', type=str, required=False,
                    default='data/idcard/base3.json', 
                    help='source directory for background image')

parser.add_argument('--photo_path', type=str, required=True,
                    default='data/face/', 
                    help='source directory for 3x4 photo image')

parser.add_argument('--randomize', type=bool, required=False,
                    default=False, 
                    help='randomize text position')


args = parser.parse_args()

if __name__ == "__main__":
    
    generator.generate(
        csv_path = args.csv_path,
        dst_path = args.dst_path,
        photo_path = args.photo_path,
        image_path = args.image_path,
        json_path = args.json_path,
        randomize = args.randomize
    )
    