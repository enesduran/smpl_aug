#!/bin/bash 
module load cuda/11.8 
/home/eduran2/miniconda3/envs/arteq/bin/python src/train_ours.py \
--EPN_input_radius 0.4 \
--EPN_layer_num 2 \
--aug_type no \
--epochs 15 \
--batch_size 2 \
--gt_part_seg auto \
--garment-flag false \
--gt-flag false \
--aug-flag true \
--kinematic_cond yes \
--num_point 50000  
