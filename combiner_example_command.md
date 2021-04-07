python combiner_idcard.py --bg_path /data/idcard/source/background/clean --bg_size "1000x750" \
--idcard_path /data/idcard/results/base/randomize/base20k --dst_path /data/idcard/results/combined/segmentv3/20kv1 \
--angle 45 --scale_ratio "0.5, 1.0" --shear 0.2 \
--num_generated 2



python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base10k \
--dst_path /data/idcard/results/combined/layoutlm/v2/10k \
--bg_size "1000x750" --force_resize True \
--angle 30 --scale_ratio "0.60, 1.001" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1

python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base20k \
--dst_path /data/idcard/results/combined/layoutlm/v1/20k \
--bg_size "1000x750" --force_resize True \
--angle 30 --scale_ratio "0.60, 1.001" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1

python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base50k \
--dst_path /data/idcard/results/combined/layoutlm/v1/50k \
--bg_size "1000x750" --force_resize True \
--angle 30 --scale_ratio "0.60, 1.001" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1

python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base100k \
--dst_path /data/idcard/results/combined/layoutlm/v1/100k \
--bg_size "1000x750" --force_resize True \
--angle 30 --scale_ratio "0.60, 1.001" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1


python combiner_idcard.py --bg_path /data/idcard/source/background/clean \
--idcard_path /data/idcard/results/base/randomize/base20k \
--dst_path /data/idcard/results/combined/layoutlm \
--bg_size "1000x750" --force_resize True \
--angle 15 --scale_ratio "0.70, 1.0" --shear 0.2 \
--use_basic_effect True --basic_effect_mode "simple" \
--use_adv_effect True --adv_effect_mode "simple" \
--num_generated 1