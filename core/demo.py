import os
import glob 
import argparse
from smpl_wrapper import SMPL_WRAPPER
 
 
def demo():
 
    for motion_path in glob.glob('motion_data/DFaust/DFaust_67/*/*.npz'):
        wrapper_obj.load_data(motion_path)

        if 'shape.npz' in motion_path:
            continue
        else:
            outdir = os.path.join("outdir", "/".join(motion_path.strip('.npz').split('/')[1:]))
            wrapper_obj.augment_loop(outdir=outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation Framework Demo')

    parser.add_argument('--body-model-type', choices=['smplx', 'smplh', 'smpl'], type=str,
                        help='Body Model to be used.')
    parser.add_argument('--clothing-option', choices=['clothed', 'minimal'], type=str,
                        default='clothed', help='Flag for garmenting minimal body.')
    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--motion-path', required=True, type=str,
                        help='The path to motion data')
    parser.add_argument('--camera-config', default='', type=str,
                        help='The path to the camera configuration')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--use-layer-instance', type=bool, default=False,
                        help='Flag to use the layer instance')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    wrapper_obj = SMPL_WRAPPER(args.model_folder,
                               args.body_model_type,
                               gender=args.gender,
                               num_betas=10,
                               camera_config=args.camera_config,
                               use_face_contour=args.use_face_contour, 
                               clothing_option=args.clothing_option,
                               use_layer=args.use_layer_instance) 
    demo()

    