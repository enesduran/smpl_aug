import os
import torch
import joblib
import trimesh 
import argparse
import numpy as np
from pathlib import Path
import tqdm.auto as tqdm
import webdataset as wds
from data import DFaustDataset
from torch.utils.data import DataLoader
from pytorch3d.ops.knn import knn_points 

from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train_ours import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2, get_gpu, str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, metavar="N", help="input batch size for training (default: 128)")
    parser.add_argument("--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)")
    parser.add_argument("--epoch", type=int, default=15, metavar="N", help="which model epoch to use (default: 15)")
    parser.add_argument("--rep_type", type=str, default="6d", metavar="N", help="aa, 6d")
    parser.add_argument("--part_num", type=int, default=22, metavar="N", help="part num of the SMPL body")
    parser.add_argument("--garment-flag", type=str2bool)
    parser.add_argument("--test-gt-flag", type=str2bool)
    parser.add_argument("--train-gt-flag", type=str2bool)
    parser.add_argument("--train-aug-flag", type=str2bool)
    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--aug_type", type=str, default="so3", metavar="N", help="so3, zrot, no")
    parser.add_argument("--gt_part_seg", type=str, default="auto", metavar="N", help="")
    parser.add_argument("--EPN_input_radius", type=float, default=0.4, help="train from pretrained model")
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--kinematic_cond", type=str, default="yes", metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--i", type=int, default=None, help="")
    parser.add_argument("--paper_model", action="store_true")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda

    args.device = torch.device("cuda")

    exps_folder = "GARMENT_{}_TESTGT_{}_TRAINGT_{}_TRAINAUG_{}".format(args.garment_flag,
                                                                        args.test_gt_flag,
                                                                        args.train_gt_flag,
                                                                        args.train_aug_flag)

    model_folder = "GARMENT_{}_GT_{}_AUG_{}".format(args.garment_flag,
                                                    args.train_gt_flag,
                                                    args.train_aug_flag)


    output_folder = os.path.sep.join(["./experiments_test", exps_folder])

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)

    if args.paper_model:
        output_folder = "./data/papermodel/"
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"

    nc, _ = get_nc_and_view_channel(args)

    torch.backends.cudnn.benchmark = True

    body_model_male = get_body_model(model_type="smpl", gender="male", batch_size=1, device="cuda")
    body_model_female = get_body_model(model_type="smpl", gender="female", batch_size=1, device="cuda")
    body_model_neutral = get_body_model(model_type="smpl", gender="neutral", batch_size=1, device="cuda")

    bm_dict = {"neutral": body_model_neutral, 
               "female": body_model_female,
               "male": body_model_male}

    parents = body_model_neutral.parents
    # parents = parents[:22]

    base_path = Path(output_folder)
    torch.manual_seed(1)

    test_dataset = DFaustDataset(data_path='outdir/DFaust', 
                                 train_flag=False, 
                                 gt_flag=args.test_gt_flag, 
                                 aug_flag=False, 
                                 garment_flag=args.garment_flag)
  
    test_loader = DataLoader(test_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              drop_last=True)

    model_path = os.path.join('experiments_train', model_folder, f"model_epochs_{args.epoch-1:08d}.pth")
 
    model = PointCloud_network_equiv(option=args, z_dim=args.latent_num, nc=nc, part_num=len(parents)).to(args.device)

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
       
    v2v, joint_err, acc = {}, {}, {}
    

    with torch.inference_mode():

        for i, batch_data in enumerate(tqdm.tqdm(test_loader)):

            assert len(batch_data['gender']) == 1
            bm_list = [bm_dict[g] for g in batch_data["gender"]]
 
            pcl_data = batch_data["point_cloud"][:].cuda().float()
            pcl_data = pcl_data[pcl_data.sum(-1) != 0][None]


            body_pose_gt = batch_data["pose"].cuda()
            body_pose_gt[:, -6:] = 0


            joints_gt, vertices_gt = SMPLX_layer(bm_list,
                                                batch_data["betas"].cuda(),
                                                batch_data["trans"].cuda(),
                                                torch.cat([batch_data["global_orient"].cuda(), body_pose_gt], dim=1),
                                                rep="aa")
            
            pc_data_list_idx = []

            # compute closest points
            for _i_, elem in enumerate(pcl_data):
                clo_pts = knn_points(vertices_gt[_i_][None], pcl_data[_i_][None], K=1, return_nn=True)[1][0, :, 0]
                pc_data_list_idx.append(pcl_data[_i_][clo_pts])
  
            pcl_data = torch.stack(pc_data_list_idx)

            pred_joint, pred_pose, pred_shape, trans_feat = \
                model(pcl_data, None, None, None, is_optimal_trans=False, parents=parents)

            # get_gpu()

            pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)

            
            pred_joints_pos, pred_vertices = SMPLX_layer(bm_list,
                                                        pred_shape,
                                                        translation=torch.zeros((1, 3)).cuda(),
                                                        motion_pose=pred_joint_pose,
                                                        rep="rotmat")
            
             
            pred_joints_pos = pred_joints_pos[0]
            pred_vertices = pred_vertices[0]
 
  
            
            vertices_gt = vertices_gt[0]
            joints_gt = joints_gt[0]
            
            v2v[f"{i}"] = (100 * (vertices_gt - pred_vertices).square().sum(dim=-1).sqrt().mean().item())
            joint_err[f"{i}"] = (100 * (joints_gt - pred_joints_pos).square().sum(dim=-1).sqrt().mean().item())
            # acc[f"{i}"] = ((pred_joint[0].argmax(dim=1).cpu() == batch_data["label_data"][: args.num_point]).mean(dtype=float).item())

            trimesh.Trimesh(vertices=pred_vertices.cpu(), faces=body_model_neutral.faces).export(os.path.join(output_folder, 'results', f'body_pred_{i:04d}.obj'))
            trimesh.Trimesh(vertices=vertices_gt.cpu(), faces=body_model_neutral.faces).export(os.path.join(output_folder, 'results', f'body_gt_{i:04d}.obj'))
            trimesh.PointCloud(pcl_data[0].cpu().numpy()).export(os.path.join(output_folder, 'results', f'pc_{i:04d}.ply'))  

    metric_dict = {"v2v":np.mean(list(v2v.values())),
                    "j2j":np.mean(list(joint_err.values()))}

    print('Saving metrics')    
    joblib.dump(metric_dict, os.path.join(output_folder, "metrics.pkl"))
