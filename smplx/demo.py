# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os 
import torch
import argparse
import numpy as np
import os.path as osp
from smplx.body_models import SMPL 

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from simkinect.add_noise_smpl_no_discrete import mesh_2_kinectpcd

### compatibility with python 2.7
np.str = np.str_
np.int = np.int_
np.bool = np.bool_
np.float = np.float_
np.object = np.object_
np.unicode = np.unicode_
np.complex = np.complex_


def main(model_folder,
         ext='npz',
         gender='neutral',
         num_betas=10,
         camera_config="",
         sample_expression=True,
         use_face_contour=False):
    
    model_folder = os.path.join(os.path.dirname(__file__), model_folder)

    model = SMPL(model_folder, 
                gender=gender, 
                use_face_contour=use_face_contour,
                num_betas=num_betas,
                ext=ext,
                clothing_option='clothed',
                render_depth=False)

    expression = None

    if sample_expression:
        expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

    cloth_types = np.ones((120, 6), dtype=np.int64) * 3
    cloth_types[:, 3] = 1
    kwargs_dict = {'cloth_types': cloth_types}

    motion_dict = np.load("motion_data/sample_motion_data.npz")
    body_pose = torch.tensor(motion_dict["poses"][1200:1320, 3:72], dtype=torch.float32)
    global_orient = torch.tensor(motion_dict["poses"][1200:1320, :3], dtype=torch.float32)
    betas = torch.tensor(motion_dict["betas"][None, :10], dtype=torch.float32)

    output = model(betas=betas, 
                   expression=expression,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   return_verts=True,
                   **kwargs_dict)
    
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    mesh_2_kinectpcd(vertices, model.faces, camera_config_file=camera_config)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation Framework Demo')

    parser.add_argument('--body-model', choices=['smplx', 'smplh', 'smpl'], type=int,
                        help='Body Model to be used.')
    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--camera-config', default='', type=str,
                        help='The path to the camera configuration')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = args.model_folder
    ext = args.ext
    gender = args.gender
    camera_config = args.camera_config
    use_face_contour = args.use_face_contour

    main(model_folder, 
         ext=ext,
         gender=gender, 
         camera_config=camera_config,
         use_face_contour=use_face_contour)
