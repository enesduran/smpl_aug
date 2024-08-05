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

import cv2
import json
import torch
import trimesh 
torch.set_warn_always(False)
import smpl_aug 
import argparse
import numpy as np
import torch.nn as nn
from pytorch3d import transforms
from simkinect.add_noise_smpl_no_discrete import mesh_2_kinectpcd
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras

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
                            # ext=ext,
                            use_pca=False,
                            clothing_option=clothing_option)


        self.camera_config = camera_config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # reading the image directly in gray with 0 as input 
        self.kinect_dot_pattern = cv2.imread("core/simkinect/data/sample_pattern.png", cv2.IMREAD_GRAYSCALE)


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
            body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(T, -1, 3))
            global_orient = transforms.axis_angle_to_matrix(global_orient.reshape(T, -1, 3))

            # weirdly, SMPLHLayer and SMPLXLayer have different hand pose representations
            if self.body_model_type == 'smplh':	
                left_hand_pose = transforms.axis_angle_to_matrix(left_hand_pose.reshape(T, -1, 3))
                right_hand_pose = transforms.axis_angle_to_matrix(right_hand_pose.reshape(T, -1, 3))

            elif self.body_model_type == 'smplx':
                left_hand_pose = transforms.axis_angle_to_matrix(left_hand_pose.reshape(T, -1, 3))
                right_hand_pose = transforms.axis_angle_to_matrix(right_hand_pose.reshape(T, -1, 3))
                reye_pose = transforms.axis_angle_to_matrix(reye_pose.reshape(T, -1, 3))
                leye_pose = transforms.axis_angle_to_matrix(leye_pose.reshape(T, -1, 3))
                jaw_pose = transforms.axis_angle_to_matrix(jaw_pose.reshape(T, -1, 3))


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
 
        trimesh.Trimesh(vertices[30], self.model.faces).export('test.obj')
        

        self.load_cameras(self.camera_config)
        camera_config_dict = json.load(open(self.camera_config, 'r'))

 
        mesh_2_kinectpcd(vertices, self.model.faces, camera_config_dict, self.cameras_batch, self.kinect_dot_pattern)

    
    def load_data(self, motion_path):
        motion_dict = np.load(motion_path)

        motion_T = min(motion_dict["poses"].shape[0], 120)

        cloth_types = np.ones((motion_T, 6), dtype=np.int64) * 3
        cloth_types[:, 3] = 1
        kwargs_dict = {'cloth_types': cloth_types}

        transl = jaw_pose = reye_pose = leye_pose = jaw_pose = torch.zeros((motion_T, 3), dtype=torch.float32)
        right_hand_pose = left_hand_pose = torch.zeros((motion_T, 45), dtype=torch.float32)
        
        if self.body_model_type == 'smpl':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:72], dtype=torch.float32)[:motion_T]
            # body_pose = torch.zeros_like(body_pose)
        elif self.body_model_type == 'smplh':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:66], dtype=torch.float32)[:motion_T]
            left_hand_pose = torch.tensor(motion_dict["poses"][:, 66:111], dtype=torch.float32)[:motion_T]
            right_hand_pose = torch.tensor(motion_dict["poses"][:, 111:], dtype=torch.float32)[:motion_T]
        elif self.body_model_type == 'smplx':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:66], dtype=torch.float32)[:motion_T]
            jaw_pose = torch.tensor(motion_dict["poses"][:, 66:69], dtype=torch.float32)[:motion_T]
            leye_pose = torch.tensor(motion_dict["poses"][:, 69:72], dtype=torch.float32)[:motion_T]
            reye_pose = torch.tensor(motion_dict["poses"][:, 72:75], dtype=torch.float32)[:motion_T]
            left_hand_pose = torch.tensor(motion_dict["poses"][:, 75:120], dtype=torch.float32)[:motion_T]
            right_hand_pose = torch.tensor(motion_dict["poses"][:, 120:165], dtype=torch.float32)[:motion_T]

        elif self.body_model_type == 'mano':
            raise NotImplementedError
        elif self.body_model_type == 'flame':
            raise NotImplementedError
        else:
            raise ValueError('Unknown body model type: {}'.format(self.body_model_type))

        global_orient = torch.tensor(motion_dict["poses"][:, :3], dtype=torch.float32)[:motion_T]
        betas = torch.tensor(motion_dict["betas"][None, :10], dtype=torch.float32).repeat(motion_T, 1)
        expression = torch.zeros_like(betas)
         
        self.forward(betas=betas, 
                    expression=expression, 
                    global_orient=global_orient, 
                    transl=transl, 
                    reye_pose=reye_pose, 
                    leye_pose=leye_pose,
                    jaw_pose=jaw_pose, 
                    left_hand_pose=left_hand_pose, 
                    right_hand_pose=right_hand_pose, 
                    body_pose=body_pose, **kwargs_dict)
        

    def augment_loop(self):


        # try changing camera pose along the way 

        pass


    def load_cameras(self, camera_config_filepath):
        with open(camera_config_filepath, 'r') as f:
            camera_config_dict = json.load(f)

        camera_ext_dict = {_cam_['camera_id']: {'R': _cam_['cam_R'], 'T': _cam_['cam_T']} for _cam_ in camera_config_dict["cameras"]}
        num_cameras = len(camera_ext_dict)

        cam_batch_R = torch.tensor([elem['R'] for elem in camera_ext_dict.values()])
        cam_batch_T = torch.tensor([elem['T'] for elem in camera_ext_dict.values()])
        image_size = (camera_config_dict['height'], camera_config_dict['width'])
        fx, fy = camera_config_dict['focal_length_x'], camera_config_dict['focal_length_y']

        focal_length = torch.tensor([fx, fy]*num_cameras).reshape(-1, 2)
        image_size_mult = torch.tensor([image_size]*num_cameras).reshape(-1, 2)

        self.cameras_batch = PerspectiveCameras(focal_length=focal_length, device=self.device, R=cam_batch_R, T=cam_batch_T, image_size=image_size_mult)
 

    def update_camera(self, camera_config_filepath):
        
        # make sure load_cameras is called before this function
        assert hasattr(self, 'cameras_batch'), 'Please load the cameras first using load_cameras function'


        with open(camera_config_filepath, 'r') as f:
            camera_config_dict = json.load(f)

        camera_ext_dict = {_cam_['camera_id']: {'R': _cam_['cam_R'], 'T': _cam_['cam_T']} for _cam_ in camera_config_dict["cameras"]}
        num_cameras = len(camera_ext_dict)

        cam_batch_R = torch.tensor([elem['R'] for elem in camera_ext_dict.values()])
        cam_batch_T = torch.tensor([elem['T'] for elem in camera_ext_dict.values()])
        image_size = (camera_config_dict['height'], camera_config_dict['width'])
        fx, fy = camera_config_dict['focal_length_x'], camera_config_dict['focal_length_y']

        focal_length = torch.tensor([fx, fy]*num_cameras).reshape(-1, 2)
        image_size_mult = torch.tensor([image_size]*num_cameras).reshape(-1, 2)

        import ipdb; ipdb.set_trace()
        

        
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
    parser.add_argument('--use-layer-instance', type=bool, default=True,
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
    
    wrapper_obj.load_data(args.motion_path)
