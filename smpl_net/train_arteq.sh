python src/train_arteq.py \
    --EPN_input_radius 0.4 \
    --EPN_layer_num 2 \
    --aug_type so3 \
    --batch_size 2 \
    --epochs 15 \
    --gt_part_seg auto \
    --i 0 \
    --kinematic_cond yes \
    --num_point 5000