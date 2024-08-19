import os
import cv2
import json
import torch
import smpl_aug 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from loguru import logger
import pytorch3d
from pytorch3d.io import IO
from pytorch3d import transforms
from simkinect.add_noise_smpl_no_discrete import mesh2pcd
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
                 render_flag=False,
                 camera_config="",
                 use_face_contour=False,
                 use_layer=False):
        
        super(SMPL_WRAPPER, self).__init__()

        self.use_layer = use_layer  
        self.render_flag = render_flag
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
                            use_pca=False,
                            clothing_option=clothing_option)

        if render_flag:
            self.camera_config = camera_config
            self.camera_config1 = 'camera_configs/kinect_batch_update.json'
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            # reading the image directly in gray with 0 as input 
            self.kinect_dot_pattern = cv2.imread("core/simkinect/data/sample_pattern.png", cv2.IMREAD_GRAYSCALE)
            self.camera_config_dict = json.load(open(self.camera_config, 'r'))

            self.vertex2pcd_rot = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)

            for cam_id in range(len(self.camera_config_dict["cameras"])):
                self.camera_config_dict["cameras"][cam_id]["cam_T"] = \
                        torch.tensor(self.camera_config_dict["cameras"][cam_id]["cam_T"])
                self.camera_config_dict["cameras"][cam_id]["cam_R"] = \
                        torch.tensor(self.camera_config_dict["cameras"][cam_id]["cam_R"])
            
            self.load_cameras()


    def forward(self, betas, expression, global_orient, transl, reye_pose, leye_pose, 
                jaw_pose, left_hand_pose, right_hand_pose, body_pose, 
                augment_pose_flag=False, **kwargs_dict):
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


        return self.model(betas=betas, 
                    expression=expression,
                    global_orient=global_orient,
                    transl=transl,
                    reye_pose=reye_pose,
                    leye_pose=leye_pose,
                    jaw_pose=jaw_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    body_pose=body_pose,
                    augment_pose=augment_pose_flag,
                    **kwargs_dict)
               
        
    
    def load_data(self, motion_path):
        motion_dict = np.load(motion_path)

        try: 
            motion_T = motion_dict["poses"].shape[0]
        except:
            import ipdb; ipdb.set_trace()

        cloth_types = np.ones((motion_T, 6), dtype=np.int64) * 3
        cloth_types[:, 3] = 1
        kwargs_dict = {'cloth_types': cloth_types}

        jaw_pose = reye_pose = leye_pose = jaw_pose = torch.zeros((motion_T, 3), dtype=torch.float32)
        right_hand_pose = left_hand_pose = torch.zeros((motion_T, 45), dtype=torch.float32)
        
        transl = torch.tensor(motion_dict["trans"], dtype=torch.float32)[:motion_T]
        
        
        if self.body_model_type == 'smpl':
            body_pose = torch.tensor(motion_dict["poses"][:, 3:72], dtype=torch.float32)[:motion_T]
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
  
        self.motion_data =  {'betas':betas, 
                            'expression':expression, 
                            'global_orient':global_orient, 
                            'transl':transl,  
                            'reye_pose':reye_pose, 
                            'leye_pose':leye_pose,
                            'jaw_pose':jaw_pose, 
                            'left_hand_pose':left_hand_pose, 
                            'right_hand_pose':right_hand_pose, 
                            'body_pose':body_pose,
                            'motion_T':motion_T,
                            **kwargs_dict}
        

    def augment_loop(self, outdir, augment_pose_flag=False, render_flag=False):


        # load cameras from camera config file
        augment_list = [False] if augment_pose_flag else [False, True]

        io_object = IO()
        print(outdir)
         
        # create output directories
        if render_flag or self.render_flag:
            setting_name = self.camera_config_dict['setting_name']
            os.makedirs(f'{outdir}/{setting_name}', exist_ok=True)
            os.makedirs(f'{outdir}/{setting_name}/point_cloud', exist_ok=True)
            os.makedirs(f'{outdir}/{setting_name}/body_meshes', exist_ok=True)
            os.makedirs(f'{outdir}/{setting_name}/perfect_depth', exist_ok=True)
            os.makedirs(f'{outdir}/{setting_name}/noised_depth', exist_ok=True)
            os.makedirs(f'{outdir}/{setting_name}/processed_depth', exist_ok=True)


        for i in tqdm(range(self.motion_data['motion_T'])):

            motion_data_i = {k: v[i][None] for k,v in self.motion_data.items() if k != 'motion_T'}
            
            if i == 0:
                delta_cam_trans = torch.zeros_like(self.motion_data['transl'][0])   
                delta_cam_rot = torch.eye(3)
            else:   
                delta_cam_trans = motion_data_i["transl"][0] - prev_transl
                delta_cam_rot = transforms.axis_angle_to_matrix(motion_data_i['global_orient'][0]) @ prev_rot.T 
                
            prev_transl = self.motion_data['transl'][0].clone()
            prev_rot = transforms.axis_angle_to_matrix(motion_data_i['global_orient'][0])

            for flag in augment_list:

                append_str = '_aug' if flag else ''
                output = self(**motion_data_i, augment_pose_flag=flag)
                vertices = output.vertices.detach().cpu().numpy().squeeze()

                if render_flag or self.render_flag:

                    _, depth_gt, depth, noisy_depth, projected_pcd_noised = mesh2pcd(vertices, 
                                                                                    self.model.faces, 
                                                                                    self.camera_config_dict, 
                                                                                    self.cameras_batch, 
                                                                                    self.kinect_dot_pattern, 
                                                                                    single_pc_flag=True)

                    # save depth for each camera
                    for cam_id in range(len(self.cameras_batch)):
                        cv2.imwrite(f'{outdir}/{setting_name}/perfect_depth/{i:04d}_{cam_id}{append_str}.png', depth_gt[cam_id] * 255)
                        cv2.imwrite(f'{outdir}/{setting_name}/noised_depth/{i:04d}_{cam_id}{append_str}.png', noisy_depth[cam_id] * 255)
                        cv2.imwrite(f'{outdir}/{setting_name}/processed_depth/{i:04d}_{cam_id}{append_str}.png', depth[cam_id] * 255)
                
                    io_object.save_pointcloud(projected_pcd_noised, f'{outdir}/{setting_name}/point_cloud/{i:04d}{append_str}.ply')

 
                verts = torch.tensor(vertices) @ self.vertex2pcd_rot

                # save point cloud
                pytorch3d.io.save_obj(f'{outdir}/{setting_name}/body_meshes/{i:04d}{append_str}.obj', 
                            verts=verts, faces=torch.tensor(self.model.faces.astype(np.int32)))
             
            # update camera extrinsics between each frame
            for cam_id in range(len(self.cameras_batch)): 
             
                self.camera_config_dict["cameras"][cam_id]["cam_R"] = torch.matmul(delta_cam_rot, self.camera_config_dict["cameras"][cam_id]["cam_R"])
                
                rotated_delta_cam_trans = self.camera_config_dict["cameras"][cam_id]["cam_R"] @ delta_cam_trans
                # self.camera_config_dict["cameras"][cam_id]["cam_T"] += rotated_delta_cam_trans
                
            # update camera extrinsics between each frame
            self.update_camera_extrinsics()
    
        
 
    def load_cameras(self):
    
        camera_ext_dict = {_cam_['camera_id']: {'R': _cam_['cam_R'], 'T':_cam_['cam_T']} 
                           for _cam_ in self.camera_config_dict["cameras"]}
        num_cameras = len(camera_ext_dict)

        cam_batch_R = torch.vstack([elem['R'] for elem in camera_ext_dict.values()]).reshape(-1, 3, 3)
        cam_batch_T = torch.vstack([elem['T'] for elem in camera_ext_dict.values()])
        image_size = (self.camera_config_dict['height'], self.camera_config_dict['width'])
        fx, fy = self.camera_config_dict['focal_length_x'], self.camera_config_dict['focal_length_y']

        focal_length = torch.tensor([fx, fy]*num_cameras).reshape(-1, 2)
        image_size_mult = torch.tensor([image_size]*num_cameras).reshape(-1, 2)

        # principal_point
        if "cx" in self.camera_config_dict and "cy" in self.camera_config_dict:
            cx, cy = self.camera_config_dict['cx'], self.camera_config_dict['cy']
        else:
            cx, cy = image_size[1]//2, image_size[0]//2

        principal_point = torch.tensor([cx, cy]*num_cameras).reshape(-1, 2)
    
        self.cameras_batch = PerspectiveCameras(focal_length=focal_length,
                                                principal_point=principal_point,
                                                device=self.device,
                                                R=cam_batch_R, 
                                                T=cam_batch_T, 
                                                image_size=image_size_mult, 
                                                in_ndc=False)
 

    def update_camera_extrinsics(self):
        
        # make sure load_cameras is called before this function
        assert hasattr(self, 'cameras_batch'), 'Please load the cameras first using load_cameras function'

        camera_ext_dict = {_cam_['camera_id']: {'R': _cam_['cam_R'], 'T':_cam_['cam_T']} 
                           for _cam_ in self.camera_config_dict["cameras"]}

        
        cam_batch_R = torch.vstack([elem['R'] for elem in camera_ext_dict.values()]).reshape(-1, 3, 3)
        cam_batch_T = torch.vstack([elem['T'] for elem in camera_ext_dict.values()])
        
        logger.info('Updating camera parameters')
        self.cameras_batch.R = cam_batch_R.to(self.device)
        self.cameras_batch.T = cam_batch_T.to(self.device)
