python src/train_ours.py \
        --EPN_input_radius 0.4 \
        --EPN_layer_num 2 \
        --aug_type no \
        --batch_size 2 \
        --epochs 400 \
        --gt_part_seg auto \
        --gt-flag true \
        --aug-flag false \
        --kinematic_cond yes \
        --num_point 50000
# 50000 if H100