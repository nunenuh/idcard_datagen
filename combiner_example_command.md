python combiner_idcard.py --bg_path /data/idcard/source/background/clean --bg_size "1000x750" \
--idcard_path /data/idcard/results/base/randomize/base20k --dst_path /data/idcard/results/combined/segmentv3/20kv1 \
--angle 45 --scale_ratio "0.5, 1.0" --shear 0.2 \
--num_generated 2


python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base20k \
--dst_path /data/idcard/results/combined/crnn/ \
--bg_size "1000x750" --force_resize True \
--angle 2 --scale_ratio "0.80, 1.0" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1