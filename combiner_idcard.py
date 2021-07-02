import argparse

from datagen.imgen.idcard import combiner

parser = argparse.ArgumentParser(description='IDCard data generator tools')
parser.add_argument('--bg_path', type=str,
                    help='source directory for background image', required=True)
parser.add_argument('--bg_ext', type=str, default='jpg|png',
                    help='extension for background image', required=False)

parser.add_argument('--bg_size', type=str, default=None,
                    help='resize background image to choosed size WxH format', required=False)

parser.add_argument('--force_resize', default=False, type=bool,
                    help='force resize landscape or portrait background image to meet --bg_size',
                    required=False)

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

parser.add_argument('--foreground_fx', default="simple", type=str, required=False,
                    help='fill idcard effect with simple, medium, or complex')

parser.add_argument('--background_fx', default=None, type=str, required=False,
                    help='fill background effect with simple, medium, or complex')

parser.add_argument('--composite_bfx', default=None, type=str, required=False,
                    help='fill composite image with basic simple, medium, or complex')

parser.add_argument('--composite_afx', default="simple", type=str, required=False,
                    help='fill composite image with advance simple, medium, or complex')

parser.add_argument('--balance_white_background', default=True, type=bool,
                    help='balance background with white background, this is useful especially in scanner data',
                    required=False)

parser.add_argument('--white_background_factor', default=1.0, type=float,
                    help='the factor of white background inserted to background',
                    required=False)

parser.add_argument('--balance_idcard_background', default=False, type=bool,
                    help='balance the number idcard data and background data using random choice method',
                    required=False)

parser.add_argument('--sampled_background', default=True, type=bool,
                    help='use random sampling to take background',
                    required=False)


# balance_white_bg=False, balance_bg=False, sampled_bg=True,

args = parser.parse_args()


if __name__ == "__main__":
    
    combiner.combine(
        bg_path=args.bg_path,
        idcard_path=args.idcard_path,
        dst_path=args.dst_path,
        idcard_ext=args.idcard_ext,
        bg_ext=args.bg_ext,
        bg_size=args.bg_size,
        balance_white_bg=args.balance_white_background,
        white_bg_factor =args.white_background_factor,
        balance_bg=args.balance_idcard_background,
        sampled_bg=args.sampled_background,
        force_resize=args.force_resize,
        angle=args.angle,
        shear=args.shear,
        scale_ratio=args.scale_ratio,
        num_generated=args.num_generated,
        foreground_fx=args.foreground_fx,
        background_fx=args.background_fx,
        composite_bfx=args.composite_bfx,
        composite_afx=args.composite_afx
    )
