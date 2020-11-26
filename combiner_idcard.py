import argparse

from datagen.imgen.idcard import combiner

parser = argparse.ArgumentParser(description='IDCard data generator tools')
parser.add_argument('--bg_path', type=str,
                    help='source directory for background image', required=True)
parser.add_argument('--bg_ext', type=str, default='jpg|png',
                    help='extension for background image', required=False)

parser.add_argument('--bg_size', type=str, default=None,
                    help='resize background image to choosed size WxH format', required=False)

parser.add_argument('--idcard_path', type=str,
                    help='source directory for id card image', required=True)
parser.add_argument('--idcard_ext', type=str, default='png',
                    help='extension for idcard image, works only with png!', required=False)
parser.add_argument('--dst_path', type=str,
                    help='destination directory for generated data', required=True)
parser.add_argument('--angle', default=30, type=int,
                    help='random rotation angle')
parser.add_argument('--shear', default=0.5, type=float,
                    help='random shear factor')
parser.add_argument('--scale_ratio', default="0.3,0.8", type=str, 
                    help="scale ratio between idcard and background")
parser.add_argument('--num_generated', default=6, type=int,
                    help='number of combined generated data from same idcard and background image')

args = parser.parse_args()


if __name__ == "__main__":
    
    combiner.combine(
        bg_path=args.bg_path,
        idcard_path=args.idcard_path,
        dst_path=args.dst_path,
        idcard_ext=args.idcard_ext,
        bg_ext=args.bg_ext,
        bg_size=args.bg_size,
        angle=args.angle,
        shear=args.shear,
        scale_ratio=args.scale_ratio,
        num_generated=args.num_generated
    )
