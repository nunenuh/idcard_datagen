# idcard_datagen
IDCard Data Generator for creating synthetic dataset generator for segmentation network


python generate_idcard.py --csv_path data/text/dataset_ktp.csv --idcard_path data/idcard/base3.png --idcard_json_path data/idcard/base3.json --photo_path data/face --dst_path data/fake_ktp/

python run.py --bg_path data/background/ --id_path data/fake_ktp/ --num_generated 12 --dst_path result
