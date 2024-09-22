

TRAIN_TEMPLATE = f'#!/bin/bash \n' \
        'module load cuda/11.8 \n' \
        '/home/eduran2/miniconda3/envs/arteq/bin/python src/train_ours.py \\\n' \
        '--EPN_input_radius 0.4 \\\n' \
        '--EPN_layer_num 2 \\\n' \
        '--aug_type no \\\n' \
        '--epochs 15 \\\n' \
        '--batch_size 2 \\\n' \
        '--gt_part_seg auto \\\n' \
        '--garment-flag TRAINGARMENTFLAG \\\n' \
        '--gt-flag TRAINGTFLAG \\\n' \
        '--aug-flag TRAINAUGFLAG \\\n' \
        '--kinematic_cond yes \\\n' \
        '--num_point 50000  \n' \

EVAL_TEMPLATE =  f'#!/bin/bash \n' \
        'module load cuda/11.8 \n' \
        '/home/eduran2/miniconda3/envs/arteq/bin/python src/eval_ours.py  \\\n' \
        '--EPN_input_radius 0.4  \\\n' \
        '--EPN_layer_num 2  \\\n' \
        '--aug_type no  \\\n' \
        '--epoch 15  \\\n' \
        '--batch_size 1  \\\n' \
        '--train-gt-flag TRAINGTFLAG  \\\n' \
        '--train-aug-flag TRAINAUGFLAG  \\\n' \
        '--gt_part_seg auto  \\\n' \
        '--garment-flag TESTGARMENTFLAG  \\\n' \
        '--test-gt-flag TESTGTFLAG  \\\n' \
        '--kinematic_cond yes  \\\n' \
        '--num_point 50000'


SUBMISSION_TEMPLATE = f'executable = RUN_SCRIPT\n' \
                       'arguments = $(Process) $(Cluster)\n' \
                       'error = ./cluster_logs/$(Cluster).$(Process).err\n' \
                       'output = ./cluster_logs/$(Cluster).$(Process).out\n' \
                       'log = ./cluster_logs/$(Cluster).$(Process).log\n' \
                       'request_memory = 120000\n' \
                       'request_cpus=4\n' \
                       'request_gpus=1\n' \
                       'request_disk=100000\n' \
                       'requirements = TARGET.CUDACapability == 9.0 \n' \
                       'queue 1 \n' \
                       '+BypassLXCfs="true"'

import os 
import stat 
import time
import joblib
import shutil
import subprocess
from tabulate import tabulate

TRAIN_FILENAME = 'sh_scripts/train_ours_GARMENTFLAGGTFLAGAUGFLAG.sh'
EVAL_FILENAME = 'sh_scripts/eval_ours_GARMENTFLAGGTFLAGAUGFLAG_TESTNOISEFLAG.sh'

 
def spawn_train():
    i = 0
    shutil.rmtree('cluster_logs')
    os.mkdir('cluster_logs')


    for garment_flag in [True, False]:
        for train_gt_noise_flag in [True, False]:
            for train_pose_aug_flag in [True, False]: 

                garment_flag_capital = str(garment_flag).capitalize()[:1]
                train_gt_noise_flag_capital = str(train_gt_noise_flag).capitalize()[:1]
                pose_aug_flag_capital = str(train_pose_aug_flag).capitalize()[:1]

                filename_temp = TRAIN_FILENAME.replace('GARMENTFLAG', garment_flag_capital).replace('GTFLAG', train_gt_noise_flag_capital).replace('AUGFLAG', pose_aug_flag_capital)

                TRAIN_CHANGED = TRAIN_TEMPLATE.replace('TRAINGARMENTFLAG', str(garment_flag).lower()) \
                    .replace('TRAINGTFLAG', str(train_gt_noise_flag).lower())\
                    .replace('TRAINAUGFLAG', str(train_pose_aug_flag).lower())
                
                with open(filename_temp, 'w') as f:
                    f.write(TRAIN_CHANGED)
                os.chmod(filename_temp, stat.S_IRWXU)

                SUBMISSION_TEMPLATE_CHANGED = SUBMISSION_TEMPLATE.replace('RUN_SCRIPT', filename_temp)

                submission_path = f'sub_files/{i:04d}.sub'

                with open(submission_path, 'w') as f:
                    f.write(SUBMISSION_TEMPLATE_CHANGED)

                # create and write the bash run script.
                cmd = ['condor_submit_bid', f'300', str(submission_path)]
                print('Executing ' + ' '.join(cmd))

                i += 1

                subprocess.run(cmd)
                time.sleep(0.5)

def spawn_test():

    i = 100

    shutil.rmtree('cluster_logs')
    os.mkdir('cluster_logs')

    for garment_flag in [True, False]:
        for train_gt_noise_flag in [True, False]:
            for train_pose_aug_flag in [True, False]: 
                for test_gt_noise_flag in [True, False]:

                    garment_flag_capital = str(garment_flag).capitalize()[:1]
                    train_gt_noise_flag_capital = str(train_gt_noise_flag).capitalize()[:1]
                    test_gt_noise_flag_capital = str(test_gt_noise_flag).capitalize()[:1]
                    pose_aug_flag_capital = str(train_pose_aug_flag).capitalize()[:1]

                    filename_temp = EVAL_FILENAME.replace('GARMENTFLAG', garment_flag_capital).replace('GTFLAG', train_gt_noise_flag_capital).replace('AUGFLAG', pose_aug_flag_capital)
                    filename_temp = filename_temp.replace('TESTNOISEFLAG', test_gt_noise_flag_capital)
 
                    EVAL_CHANGED = EVAL_TEMPLATE.replace('TESTGARMENTFLAG', str(garment_flag).lower())\
                        .replace('TESTGTFLAG', str(test_gt_noise_flag).lower()) \
                        .replace('TRAINAUGFLAG', str(train_pose_aug_flag).lower())\
                        .replace('TRAINGTFLAG', str(train_gt_noise_flag).lower())
                     
                    with open(filename_temp, 'w') as f:
                        f.write(EVAL_CHANGED)
                    os.chmod(filename_temp, stat.S_IRWXU)

                    SUBMISSION_TEMPLATE_CHANGED = SUBMISSION_TEMPLATE.replace('RUN_SCRIPT', filename_temp)

                    submission_path = f'sub_files/{i:04d}.sub'

                    with open(submission_path, 'w') as f:
                        f.write(SUBMISSION_TEMPLATE_CHANGED)

                    # create and write the bash run script.
                    cmd = ['condor_submit_bid', f'300', str(submission_path)]
                    print('Executing ' + ' '.join(cmd))

                    i += 1

                    subprocess.run(cmd)
                    time.sleep(0.5)
    return 


def make_table():

    metric_names = ['setting', 'j2j (cm)', 'v2v (cm)']
    table = []

    for garment_flag in [True, False]:
        for test_gt_noise_flag in [True, False]:
            for train_gt_noise_flag in [True, False]:
                for train_pose_aug_flag in [True, False]: 

                    setting_name = f'GARMENT_{garment_flag}_TESTGT_{test_gt_noise_flag}_TRAINGT_{train_gt_noise_flag}_TRAINAUG_{train_pose_aug_flag}'

                    try:
                        perf_dict = joblib.load(f'experiments_test/{setting_name}/metrics.pkl')
                        table.append([setting_name, perf_dict['j2j'], perf_dict['v2v']])
                    except:
                        print(f'Could not find {setting_name}')
                        continue

                        

    open('results.txt', 'w').write(tabulate(table, headers=metric_names, tablefmt="grid", missingval='N/A'))

# spawn_train()
# spawn_test()
make_table()     