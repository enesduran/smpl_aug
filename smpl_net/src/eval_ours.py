import argparse
import os
from pathlib import Path

import numpy as np
import trimesh 
import smplx
import torch
import tqdm.auto as tqdm
import webdataset as wds
from data import DFaustDataset
from torch.utils.data import DataLoader

from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train_ours import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument(
        "--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument("--epoch", type=int, default=15, metavar="N", help="which model epoch to use (default: 15)")
    parser.add_argument("--rep_type", type=str, default="6d", metavar="N", help="aa, 6d")
    parser.add_argument("--part_num", type=int, default=22, metavar="N", help="part num of the SMPL body")
    parser.add_argument("--gt-flag", type=bool)
    parser.add_argument("--aug-flag", type=bool)
    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--aug_type", type=str, default="so3", metavar="N", help="so3, zrot, no")
    parser.add_argument("--gt_part_seg", type=str, default="auto", metavar="N", help="")
    parser.add_argument("--EPN_input_radius", type=float, default=0.4, help="train from pretrained model")
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument(
        "--kinematic_cond", type=str, default="yes", metavar="N", help="point num sampled from mesh surface"
    )
    parser.add_argument("--i", type=int, default=None, help="")
    parser.add_argument("--paper_model", action="store_true")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda

    args.device = torch.device("cuda")

    exps_folder = "gt_flag_{}_aug_flag_{}_num_point_{}".format(args.gt_flag,
                                                                args.aug_flag,
                                                                args.num_point)

     
    output_folder = os.path.sep.join(["./experiments", exps_folder])

    if args.paper_model:
        output_folder = "./data/papermodel/"
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"

    nc, _ = get_nc_and_view_channel(args)

    torch.backends.cudnn.benchmark = True

    body_model = get_body_model(model_type="smpl", gender="neutral", batch_size=1, device="cuda")
    parents = body_model.parents[:22]

    body_model_gt = smplx.SMPL("../body_models/smpl/SMPL_NEUTRAL.pkl", batch_size=1, num_betas=10)

    base_path = Path(output_folder)
    print(base_path)
    torch.manual_seed(1)

    test_dataset = DFaustDataset(data_path='outdir/DFaust/DFaust_67', train_flag=False, gt_flag=args.gt_flag, aug_flag=False)
    test_loader = DataLoader(test_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              drop_last=True)


    model_path = base_path / f"model_epochs_{args.epoch-1:08d}.pth"
    os.makedirs(os.path.join(os.path.dirname(model_path), 'results'), exist_ok=True)

    model = PointCloud_network_equiv(option=args, z_dim=args.latent_num, nc=nc, part_num=args.part_num).to(args.device)

    model.load_state_dict(torch.load(model_path))
       
    v2v, joint_err = {}, {}
    

    with torch.inference_mode():

        for i, batch_data in enumerate(tqdm.tqdm(test_loader)):
 
            pcl_data = batch_data["point_cloud"][:].cuda().float()
            pcl_data = pcl_data[pcl_data.sum(-1) != 0][None]

            rand_idx = np.random.choice(pcl_data.shape[1], 80000, replace=False)
            pcl_data = batch_data["point_cloud"][:, rand_idx].float().cuda()

            pred_joint, pred_pose, pred_shape, trans_feat = \
                model(pcl_data, None, None, None, is_optimal_trans=False, parents=parents)

            pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)

            trans_feat = torch.zeros((1, 3)).cuda()

            pred_joints_pos, pred_vertices = SMPLX_layer(body_model,
                                                        pred_shape,
                                                        trans_feat,
                                                        pred_joint_pose,
                                                        rep="rotmat")

            pred_joints_pos = pred_joints_pos[0].cpu()
            pred_vertices = pred_vertices[0].cpu()

            betas = batch_data["betas"][:]
            transl = torch.zeros_like(batch_data["trans"])

            body_pose = batch_data["pose"]
            body_pose[:, -6:] = 0

            global_orient = batch_data["global_orient"] 


            smpl_output = body_model_gt(betas=betas, transl=transl, body_pose=body_pose, global_orient=global_orient)

            vertices_gt = smpl_output.vertices[0]
            joints_gt = smpl_output.joints[0]

            v2v[f"{i}"] = (100 * (vertices_gt - pred_vertices).square().sum(dim=-1).sqrt().mean().item())
            
            joint_err[f"{i}"] = (100 * (joints_gt - pred_joints_pos).square().sum(dim=-1).sqrt().mean().item())

            trimesh.Trimesh(vertices=pred_vertices, faces=body_model.faces).export(os.path.join(os.path.dirname(model_path), 'results', f'body_pred_{i:04d}.obj'))
            trimesh.PointCloud(pcl_data[0].cpu().numpy()).export(os.path.join(os.path.dirname(model_path), 'results', f'pc_{i:04d}.ply'))

    print(f"v2v={np.mean(list(v2v.values())):.3f} \
            joint_err={np.mean(list(joint_err.values())):.3f} \
            part_acc={100*np.mean(list(acc.values())):.3f}")
