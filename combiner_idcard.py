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

parser.add_argument('--use_basic_effect', default=True, type=bool,
                    help='use basic effect',
                    required=False)

parser.add_argument('--basic_effect_mode', default="simple", type=str,
                    help='fill basic_effect with simple, medium, or complex',
                    required=False)

parser.add_argument('--use_adv_effect', default=True, type=bool,
                    help='use advance effect',
                    required=False)

parser.add_argument('--adv_effect_mode', default="simple", type=str,
                    help='fill adv_effect with simple, medium, or complex',
                    required=False)



args = parser.parse_args()


if __name__ == "__main__":
    
    combiner.combine(
        bg_path=args.bg_path,
        idcard_path=args.idcard_path,
        dst_path=args.dst_path,
        idcard_ext=args.idcard_ext,
        bg_ext=args.bg_ext,
        bg_size=args.bg_size,
        force_resize=args.force_resize,
        angle=args.angle,
        shear=args.shear,
        scale_ratio=args.scale_ratio,
        num_generated=args.num_generated,
        use_basic_effect=args.use_basic_effect,
        basic_effect_mode=args.basic_effect_mode,
        use_adv_effect=args.use_adv_effect,
        adv_effect_mode=args.adv_effect_mode
    )
