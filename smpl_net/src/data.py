import os
import glob
import torch
import trimesh 
import numpy as np
from tqdm import tqdm



class DFaustDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train_flag, setting_name='kinect', gt_flag=False, aug_flag=False):
 
        self.gt_flag = gt_flag
        self.aug_flag = aug_flag
        self.data_path = data_path
        self.setting_name = setting_name
        self.max_point_num = 120000


        self.train_flag = 'train' if train_flag else 'test'
        self.dataloader_path = f'data/{self.train_flag}_GT_{gt_flag}_AUG_{aug_flag}.npz'


        if not os.path.exists(self.dataloader_path):
            print(f"No processed data, processing data: {self.train_flag}")
            self.data_dict = self.process_data()
            self.save_data()
        else:
            print(f"Loading already processed data: {self.train_flag}")
            self.data_dict = self.load_data()
        
        

    def process_data(self):

        pc_folder_name = 'point_cloud_gt' if self.gt_flag else 'point_cloud_noised'

        pose_data_list = sorted(glob.glob(f'../{self.data_path}/*/*/{self.setting_name}/body_data/*.npy'))
        pc_data_list = sorted(glob.glob(f'../{self.data_path}/*/*/{self.setting_name}/{pc_folder_name}/*.ply'))
       

        assert len(pose_data_list) == len(pc_data_list), "Number of pose data and point cloud data should be the same"

        if not self.aug_flag:   
            pose_data_list = [elem for elem in pose_data_list if 'aug' not in elem]
            pc_data_list = [elem for elem in pc_data_list if 'aug' not in elem]
 
        data_dict = dict()

        print("Processing data")
        
        for _i_ in tqdm(range(len(pose_data_list))):
            
            data_dict_i = dict()

            seq_smpl_params = np.load(pose_data_list[_i_], allow_pickle=True).item()
            ptc = trimesh.load(pc_data_list[_i_])

            # meta variables 
            data_dict_i['sequence_name'] = seq_smpl_params['seq_name']
            data_dict_i['aug_flag'] = seq_smpl_params['aug_flag']
            data_dict_i['timestep'] = seq_smpl_params['timestep']
            data_dict_i['gt_flag'] = self.gt_flag
            data_dict_i['gender'] = 'neutral'
   
            # learning-related variables
            data_dict_i['betas'] = seq_smpl_params['betas'][0]
            data_dict_i['trans'] = seq_smpl_params['transl'][0]
            data_dict_i['global_orient'] = seq_smpl_params['global_orient'][0]
            data_dict_i['pose'] = seq_smpl_params['body_pose'][0]
  
            # make sure the number of points is the same, if not, pad with zeros
            data_dict_i['point_cloud'] = np.concatenate([ptc.vertices, np.zeros((self.max_point_num - ptc.vertices.shape[0], 3))], axis=0)
            data_dict[str(_i_)] = data_dict_i
          
        return data_dict
    

    # given two flags, load the corresponding data
    def load_data(self):
        return np.load(self.dataloader_path, allow_pickle=True)["arr_0"].item()

    # save the data to the corresponding files
    def save_data(self):
                
        sub_data_dict = dict()

        for key, value in self.data_dict.items():
            if (value['gt_flag'] == self.gt_flag) and (self.aug_flag or (value['aug_flag'] == self.aug_flag)):
                sub_data_dict[key] = value

        assert len(sub_data_dict) == len(self.data_dict), "Data is not processed correctly"

        np.savez(self.dataloader_path, self.data_dict)     


    def __len__(self):
        return len(self.data_dict)


    def __getitem__(self, index):
        return self.data_dict[str(index)]
    


