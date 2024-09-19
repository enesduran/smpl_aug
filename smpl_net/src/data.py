import os
import glob
import torch
import trimesh 
import numpy as np
from tqdm import tqdm



class DFaustDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train_flag, setting_name='kinect', gt_flag=False, aug_flag=False, garment_flag=False):
 
        self.gt_flag = gt_flag
        self.aug_flag = aug_flag
        self.data_path = data_path
        self.setting_name = setting_name
        self.max_point_num = 37500

        self.test_subject = '50027'

        self.train_flag = 'train' if train_flag else 'test'
        self.garment = 'clothed' if garment_flag else 'minimal'

        self.dataloader_path = f'data/{self.train_flag}_{self.garment}.npz'

        if not os.path.exists(self.dataloader_path):
            print(f"No processed data, processing data: {self.train_flag}")
            superset_data = self.process_data()
            self.save_data(superset_data)
        else:
            print(f"Loading already processed data: {self.train_flag}")
            superset_data = self.load_data()
        
        self.discard_data(superset_data)
        
    
    def process_data(self):
        """
        This method takes all the data from the DFaust dataset and processes it into a dictionary. For each 
        sequence point clouds from both the ground truth and the kinect noisy point clouds are loaded. 
        Discarding samples through aug_flag and gt_flag to be performed in load data method.
        """


        pose_data_list = sorted(glob.glob(f'../{self.data_path}_{self.garment}/DFaust_67/*/*/{self.setting_name}/body_data/*.npy'))
        pc_gt_data_list = sorted(glob.glob(f'../{self.data_path}_{self.garment}/DFaust_67/*/*/{self.setting_name}/point_cloud_gt/*.ply'))
        pc_noised_data_list = sorted(glob.glob(f'../{self.data_path}_{self.garment}/DFaust_67/*/*/{self.setting_name}/point_cloud_noised/*.ply'))

        
        if self.train_flag == 'train':
            pose_data_list = [elem for elem in pose_data_list if self.test_subject not in elem]
            pc_gt_data_list = [elem for elem in pc_gt_data_list if self.test_subject not in elem]
            pc_noised_data_list = [elem for elem in pc_noised_data_list if self.test_subject not in elem]
        else:

            # first select the test subject
            pose_data_list = [elem for elem in pose_data_list if self.test_subject in elem]
            pc_gt_data_list = [elem for elem in pc_gt_data_list if self.test_subject in elem]
            pc_noised_data_list = [elem for elem in pc_noised_data_list if self.test_subject in elem]

            # then discard the augmented poses
            pose_data_list = [elem for elem in pose_data_list if 'aug' not in elem]
            pc_gt_data_list = [elem for elem in pc_gt_data_list if 'aug' not in elem]
            pc_noised_data_list = [elem for elem in pc_noised_data_list if 'aug' not in elem]
            

    
        assert len(pose_data_list) == len(pc_gt_data_list), "Number of pose data and point cloud data should be the same"
        assert len(pose_data_list) == len(pc_noised_data_list), "Number of pose data and point cloud data should be the same"

        data_dict = dict()

        print("Processing data")
        
        for _i_ in tqdm(range(len(pose_data_list))):
            
            data_dict_i = dict()

            seq_smpl_params = np.load(pose_data_list[_i_], allow_pickle=True).item()

            ptc_gt = trimesh.load(pc_gt_data_list[_i_])
            ptc_noisy = trimesh.load(pc_noised_data_list[_i_])

            # meta variables 
            data_dict_i['sequence_name'] = seq_smpl_params['seq_name']
            data_dict_i['aug_flag'] = seq_smpl_params['aug_flag']
            data_dict_i['timestep'] = seq_smpl_params['timestep']
            data_dict_i['gender'] = seq_smpl_params['gender']

            # learning-related variables
            data_dict_i['betas'] = seq_smpl_params['betas'][0]
            data_dict_i['trans'] = seq_smpl_params['transl'][0]
            data_dict_i['global_orient'] = seq_smpl_params['global_orient'][0]
            data_dict_i['pose'] = seq_smpl_params['body_pose'][0]

            # make sure the number of points is the same, if less pad, if more sample
            if ptc_gt.vertices.shape[0] > self.max_point_num:
                idx = np.random.choice(ptc_gt.vertices.shape[0], size=self.max_point_num, replace=False)
                ptc_gt = ptc_gt.vertices[idx]
            else:
                ptc_gt = np.concatenate([ptc_gt.vertices, np.zeros((self.max_point_num - ptc_gt.vertices.shape[0], 3))], axis=0)
            
            if ptc_noisy.vertices.shape[0] > self.max_point_num:
                idx = np.random.choice(ptc_noisy.vertices.shape[0], size=self.max_point_num, replace=False)
                ptc_noisy = ptc_noisy[idx]
            else:
                ptc_noisy = np.concatenate([ptc_noisy.vertices, np.zeros((self.max_point_num - ptc_noisy.vertices.shape[0], 3))], axis=0)


            data_dict_i['point_cloud_gt'] = ptc_gt
            data_dict_i['point_cloud_noisy'] = ptc_noisy
            data_dict_i['index'] = _i_
       
            data_dict[str(_i_)] = data_dict_i


        return data_dict

    def discard_data(self, superset_data):
        """
        This method first loads the superset data. Then given 
        two flags, load the corresponding data.
        """
        self.data_dict = dict()

        pc_key = 'point_cloud_gt' if self.gt_flag else 'point_cloud_noisy'
        other_pc_key = 'point_cloud_noisy' if self.gt_flag else 'point_cloud_gt'
 
        for k, v in superset_data.items():

            if (self.aug_flag or (v['aug_flag'] == self.aug_flag)):
                self.data_dict[k] = v
                self.data_dict[k].pop(other_pc_key)
                self.data_dict[k]['point_cloud'] = self.data_dict[k].pop(pc_key)

            
    def load_data(self):
        return np.load(self.dataloader_path, allow_pickle=True)["arr_0"].item()
    
    def save_data(self, superset_data):
        np.savez(self.dataloader_path, superset_data)     


    def __len__(self):
        return len(self.data_dict)


    def __getitem__(self, index):

        # if index in [2390, 2863, 506, 2383, 1731, 3037, 3648, 1237, 1129, 1809, 3211, 3504, 3452,
        #              2465]:
        #     index = 0 

        if self.train_flag == 'train' and index in [1991, 1989, 3725, 1257, 1261, 389, 3741, 3395, 2731, 303, 1263, 3531]:
            index = 0

        # if self.train_flag == 'test' and index in [2390, 2863, 506, 2383, 1731, 3037, 3648, 1237, 1129, 1809, 3211, 3504, 3452,
        #              2465]:
        #     index = 0

        return self.data_dict[str(index)]
    


