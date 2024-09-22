#!/bin/bash 
module load cuda/11.8 
/home/eduran2/miniconda3/envs/arteq/bin/python src/eval_ours.py  \
--EPN_input_radius 0.4  \
--EPN_layer_num 2  \
--aug_type no  \
--epoch 15  \
--batch_size 1  \
--train-gt-flag false  \
--train-aug-flag true  \
--gt_part_seg auto  \
--garment-flag true  \
--test-gt-flag true  \
--kinematic_cond yes  \
--num_point 50000