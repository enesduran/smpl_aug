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

import trimesh
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
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):
    
    model_folder = os.path.join(os.path.dirname(__file__), model_folder)

    model = SMPL(model_folder, 
                gender=gender, 
                use_face_contour=use_face_contour,
                num_betas=num_betas,
                num_expression_coeffs=num_expression_coeffs,
                ext=ext,
                clothing_option='clothed',
                render_depth=False)

    betas, expression = None, None

    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

    cloth_types = np.ones((120, 6), dtype=np.int64) * 3
    cloth_types[:, 3] = 1
    kwargs_dict = {'cloth_types': cloth_types}

    motion_dict = np.load("motion_data/sample_motion_data.npz")
    body_pose = torch.tensor(motion_dict["poses"][1200:1320, 3:72], dtype=torch.float32)
    global_orient = torch.tensor(motion_dict["poses"][1200:1320, :3], dtype=torch.float32)

    output = model(betas=betas, 
                   expression=expression,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   return_verts=True,
                   **kwargs_dict)
    
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    mesh_2_kinectpcd(vertices, model.faces)
    
   
    if plotting_module == 'trimesh':   
        print(f"Trimesh Rendering")

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        for _i_, _v_ in enumerate(vertices):
            tri_mesh = trimesh.Trimesh(_v_, model.faces,
                                   vertex_colors=vertex_colors)
            tri_mesh.export(f"smplx/sculpt/outdir/tri_mesh/{_i_:04d}.obj")

      
    elif plotting_module == 'matplotlib':
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)

        plt.savefig('smplx_demo.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='trimesh',
                        dest='plotting_module',
                        choices=['matplotlib', 'trimesh'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = args.model_folder
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(model_folder, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
