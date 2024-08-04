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

import torch
import trimesh 
torch.set_warn_always(False)
import smpl_aug 
import argparse
import numpy as np
import torch.nn as nn
from pytorch3d import transforms
from simkinect.add_noise_smpl_no_discrete import mesh_2_kinectpcd

### compatibility with python 2.7
np.str = np.str_
np.int = np.int_
np.bool = np.bool_
np.float = np.float_
np.object = np.object_
np.unicode = np.unicode_
np.complex = np.complex_


class SMPL_WRAPPER(nn.Module):
    def __init__(self,
                 model_folder,
                 body_model_type,
                 clothing_option,
                 ext='npz',
                 gender='neutral',
                 num_betas=10,
                 camera_config="",
                 use_face_contour=False,
                 use_layer=False):
        
        super(SMPL_WRAPPER, self).__init__()

        self.use_layer = use_layer  
        self.body_model_type = body_model_type

        if use_layer:
            self.model = smpl_aug.build_layer(model_path=model_folder,
                                                    model_type=body_model_type, 
                                                    clothing_option=clothing_option)
        else:
            self.model = smpl_aug.create(model_path=model_folder,
                            model_type=body_model_type,
                            gender=gender, 
                            use_face_contour=use_face_contour,
                            num_betas=num_betas,
                            ext=ext,
                            use_pca=False,
                            clothing_option=clothing_option)


        self.camera_config = camera_config


    def forward(self, betas, expression, global_orient, transl, reye_pose, leye_pose, 
                jaw_pose, left_hand_pose, right_hand_pose, body_pose, **kwargs_dict):
        """
        Args:
            betas: torch.Tensor, shape [N, 10]
            expression: torch.Tensor, shape [N, 10]
            global_orient: torch.Tensor, shape [N, 3]
            transl: torch.Tensor, shape [N, 3]
            reye_pose: torch.Tensor, shape [N, 3]
            leye_pose: torch.Tensor, shape [N, 3]
            jaw_pose: torch.Tensor, shape [N, 3]
            left_hand_pose: torch.Tensor, shape [N, 45]
            right_hand_pose: torch.Tensor, shape [N, 45]
            body_pose: torch.Tensor, shape [N, 72]
            kwargs_dict: dict, additional arguments to be passed to the model"""

       
        if self.use_layer:
            
            T = body_pose.shape[0]
            
            # Layers accept rotmat only. This is a legacy of smplx library.
            body_pose_rotmat = transforms.axis_angle_to_matrix(body_pose.reshape(T, -1, 3))
            global_orient_rotmat = transforms.axis_angle_to_matrix(global_orient.reshape(T, -1, 3))

            output = self.model_layer(betas=betas, 
                        expression=expression,
                        global_orient=global_orient_rotmat,
                        transl=transl,
                        reye_pose=reye_pose,
                        leye_pose=leye_pose,
                        jaw_pose=jaw_pose,
                        left_hand_pose=left_hand_pose,
                        right_hand_pose=right_hand_pose,
                        body_pose=body_pose_rotmat,
                        **kwargs_dict)
            
        else:
            output = self.model(betas=betas, 
                        expression=expression,
                        global_orient=global_orient,
                        transl=transl,
                        reye_pose=reye_pose,
                        leye_pose=leye_pose,
                        jaw_pose=jaw_pose,
                        left_hand_pose=left_hand_pose,
                        right_hand_pose=right_hand_pose,
                        body_pose=body_pose,
                        **kwargs_dict)
           
        vertices = output.vertices.detach().cpu().numpy().squeeze()
 
        trimesh.Trimesh(vertices[0], self.model.faces).export('test.obj')
        

        mesh_2_kinectpcd(vertices, self.model.faces, camera_config_file=self.camera_config)

    
    def augment(self, motion_path):
        motion_dict = np.load(motion_path)

        motion_T = motion_dict["poses"].shape[0]
        motion_T = min(motion_T, 120)


        cloth_types = np.ones((motion_T, 6), dtype=np.int64) * 3
        cloth_types[:, 3] = 1
        kwargs_dict = {'cloth_types': cloth_types}

        
        transl = jaw_pose = reye_pose = leye_pose = jaw_pose = torch.zeros((motion_T, 3), dtype=torch.float32)
        right_hand_pose = left_hand_pose = torch.zeros((motion_T, 45), dtype=torch.float32)
        
        
        if self.body_model_type == 'smpl':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:72], dtype=torch.float32)[:motion_T]
        elif self.body_model_type == 'smplh':
            import ipdb; ipdb.set_trace()
        elif self.body_model_type == 'smplx':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:66], dtype=torch.float32)[:motion_T]
            jaw_pose = torch.tensor(motion_dict["poses"][:, 66:69], dtype=torch.float32)[:motion_T]
            leye_pose = torch.tensor(motion_dict["poses"][:, 69:72], dtype=torch.float32)[:motion_T]
            reye_pose = torch.tensor(motion_dict["poses"][:, 72:75], dtype=torch.float32)[:motion_T]
            left_hand_pose = torch.tensor(motion_dict["poses"][:, 75:120], dtype=torch.float32)[:motion_T]
            right_hand_pose = torch.tensor(motion_dict["poses"][:, 120:165], dtype=torch.float32)[:motion_T]

        elif self.body_model_type == 'mano':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:66], dtype=torch.float32)[:motion_T]
        elif self.body_model_type == 'flame':
            raise NotImplementedError
        else:
            raise ValueError('Unknown body model type: {}'.format(self.body_model_type))

        global_orient = torch.tensor(motion_dict["poses"][:, :3], dtype=torch.float32)[:motion_T]
        betas = torch.tensor(motion_dict["betas"][None, :10], dtype=torch.float32).repeat(motion_T, 1)
        expression = torch.zeros_like(betas)
         
        self.forward(betas, expression, global_orient, transl, reye_pose, leye_pose,
                    jaw_pose, left_hand_pose, right_hand_pose, body_pose, **kwargs_dict)

        
    

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
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    wrapper_obj = SMPL_WRAPPER(args.model_folder,
                               args.body_model_type,
                               ext=args.ext,
                               gender=args.gender,
                               num_betas=10,
                               camera_config=args.camera_config,
                               use_face_contour=args.use_face_contour, 
                               clothing_option=args.clothing_option) 
    
    wrapper_obj.augment(args.motion_path)
