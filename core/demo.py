import os
import glob 
import argparse
import itertools
from smpl_wrapper import SMPL_WRAPPER
import yaml
 
 
def demo():

    DFaust = glob.glob('motion_data/DFaust/DFaust_67/*/*.npz')
    MPI_Limits = glob.glob('motion_data/PosePrior/MPI_Limits/*/*.npz')

    data_list = list(itertools.chain(MPI_Limits, DFaust))

    if len(data_list) == 0:
        data_list = [motion_path_npz]
 
    for motion_path in sorted(data_list):
        if 'shape.npz' in motion_path:
            continue
        else:
            wrapper_obj.load_data(motion_path)
            split_list = motion_path.strip('.npz').split('/')
            outdir = os.path.join("outdir", f"{split_list[1]}_{args.clothing_option}", "/".join(split_list[2:]))
             
            wrapper_obj.augment_loop(outdir=outdir, augment_pose_flag=args.augment_pose_flag, render_flag=args.render_flag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation Framework Demo')

    parser.add_argument('--body-model-type', choices=['smplx', 'smplh', 'smpl'], type=str,
                        help='Body Model to be used.')
    parser.add_argument('--clothing-option', choices=['clothed', 'minimal'], type=str,
                        default='minimal', help='Flag for garmenting minimal body.')
    parser.add_argument('--augment-pose-flag', default=True, type=bool,
                        help='Augment the pose through reflection')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--use-layer-instance', type=bool, default=False,
                        help='Flag to use the layer instance')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--config-file', default='configs/config.yaml', type=str,
                        help='The path to configuration yaml file')

    args = parser.parse_args()
    args.render_flag = True

    config_path = args.config_file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_folder = config['model_folder']
    motion_path_npz = config['motion_path']
    camera_config = config['camera_config']

    wrapper_obj = SMPL_WRAPPER(model_folder,
                               args.body_model_type,
                               gender=args.gender,
                               num_betas=10,
                               camera_config=camera_config,
                               use_face_contour=args.use_face_contour, 
                               clothing_option=args.clothing_option,
                               use_layer=args.use_layer_instance, 
                               render_flag=args.render_flag) 
    demo()

    