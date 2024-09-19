import argparse
import os

import numpy as np
import torch
import tqdm
import trimesh
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from pytorch3d.ops.knn import knn_points 

from data import DFaustDataset
from backbones import get_part_seg_loss
from geometry import (
    aug_so3_ptc,
    batch_rodrigues,
    get_body_model,
    rotation_matrix_to_angle_axis,
)
from loss_func import NormalVectorLoss, norm_loss
from models_pointcloud import PointCloud_network_equiv

torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=12)


# fmt: off
markers_idx = torch.tensor([3470, 3171, 3327, 857, 1812, 628, 182, 3116, 3040, 239,
                            1666, 1725, 0, 2174, 1568, 1368, 3387, 2112, 1053, 1058,
                            3336, 3346, 1323, 2108, 3122, 3314, 1252, 1082, 1861, 1454,
                            850, 2224, 3233, 1769, 6728, 4343, 5273, 4116, 3694, 6399,
                            6540, 6488, 3749, 5135, 5194, 3512, 5635, 5210, 4360, 4841,
                            6786, 5573, 4538, 4544, 6736, 6747, 4804, 5568, 6544, 6682,
                            5322, 4927, 5686, 4598, 6633, 3506, 3508])
# fmt: on

import nvidia_smi
nvidia_smi.nvmlInit()

def get_gpu():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {:.2f} GB (total), {:.2f} GB (free), {:.2f} GB (used)".format(
            0, nvidia_smi.nvmlDeviceGetName(handle), 
            100 * info.free / info.total, 
            info.total / 1073741824, 
            info.free / 1073741824, 
            info.used / 1073741824)) 
    

def to_categorical(y, num_classes):
    """1-hot encodes a tensor."""
    new_y = torch.eye(num_classes, device=y.device)[y]
    return new_y


def local_to_global_bone_transformation(local_bone_transformation, parents):
    B, K = local_bone_transformation.shape[:2]
    local_bone_transformation = F.normalize(local_bone_transformation, dim=-2).double()
    # local_bone_transformation = F.normalize(local_bone_transformation, dim=-2).float()

    Rs = [local_bone_transformation[:, 0]]
    for i in range(1, len(parents)):
        try:
            Rs.append(torch.bmm(Rs[parents[i]], local_bone_transformation[:, i]))
        except:
            import ipdb; ipdb.set_trace()

    return torch.stack(Rs, dim=1).float()


def kinematic_layer_SO3_v2(global_bone_transformation, parents):
    """Input per-part global transformation Output local bone transformation based on SMPL kinematic tree, rotational
    information (SO3) as SMPL joint params translational info can be further aligned with SMPL meshes?"""

    B, K, D = global_bone_transformation.size()
    assert D != 6
    global_bone_transformation = global_bone_transformation.reshape(B, K, 3, 3)

    root_joint = global_bone_transformation[:, 0:1]
    joint_indices = torch.arange(1, len(parents))
    R_global = global_bone_transformation[:, joint_indices]
    R_parent = global_bone_transformation[:, parents[1:]]
    R_local = torch.bmm(
        torch.transpose(R_parent, 2, 3).reshape(B * (K - 1), 3, 3), R_global.reshape(B * (K - 1), 3, 3)
    ).reshape(B, K - 1, 3, 3)

    return torch.cat([root_joint, R_local], 1).reshape(B, K, 3, 3)


def get_nc_and_view_channel(args):
    if args.rep_type == "aa":
        nc = 3
    elif args.rep_type == "6d":
        nc = 6

    view_channel = 4

    return nc, view_channel


def trimesh_sampling(vertices, faces, count, gender):
    body_mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
    _, sample_face_idx = trimesh.sample.sample_surface_even(body_mesh, count)
    if sample_face_idx.shape[0] != count:
        print("add more face idx to match num_point")
        missing_num = count - sample_face_idx.shape[0]
        add_face_idx = np.random.choice(sample_face_idx, missing_num)
        sample_face_idx = np.hstack((sample_face_idx, add_face_idx))
    r = np.random.rand(count, 2)

    A = vertices[:, faces[sample_face_idx, 0], :]
    B = vertices[:, faces[sample_face_idx, 1], :]
    C = vertices[:, faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    
    lbs_w = gt_lbs_dict[gender].cpu().numpy()
    A_lbs = lbs_w[faces[sample_face_idx, 0], :]
    B_lbs = lbs_w[faces[sample_face_idx, 1], :]
    C_lbs = lbs_w[faces[sample_face_idx, 2], :]
    P_lbs = (
        (1 - np.sqrt(r[:, 0:1])) * A_lbs
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B_lbs
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C_lbs
    )

    return P, P_lbs


def sample_points(vertices, faces, count, gender, fix_sample=False, sample_type="trimesh"):
    assert not fix_sample and sample_type == "trimesh"
    return trimesh_sampling(vertices, faces, count, gender)


def get_pointcloud_(vertices_list, n_points_surface, gender_list):

    points_surface_list, points_surface_lbs_list, points_label_list = [], [], []

    for _j_, vertices in enumerate(vertices_list):

        n_points_surface = vertices.shape[0]
        # n_points_surface = 5000
        if n_points_surface != 6890:
            import ipdb; ipdb.set_trace()
 
        points_surface, points_surface_lbs = sample_points(vertices=vertices.cpu().numpy()[None], 
                                                        faces=body_model_faces, 
                                                            count=n_points_surface,
                                                            gender=gender_list[_j_])

        points_surface = torch.from_numpy(points_surface).to(vertices.device)
        points_surface_lbs = torch.from_numpy(points_surface_lbs).to(vertices.device)
        
        if len(parents) == 22:
            points_label = get_joint_label_merged_arteq(points_surface_lbs)[None, :].repeat(1, 1)
        else:
            points_label = get_joint_label_merged(points_surface_lbs)[None, :].repeat(1, 1)
            

        points_surface_list.append(points_surface)
        points_label_list.append(points_label)
        points_surface_lbs_list.append(points_surface_lbs)

    points_surface = torch.vstack(points_surface_list)
    points_label = torch.vstack(points_label_list)
    points_surface_lbs = torch.vstack(points_surface_lbs_list)

    return points_surface.float(), points_label, points_surface_lbs


def get_pointcloud(pc_list):
   
    points_surface_list = []

    for pc in pc_list:

        rand_idx = np.random.choice(len(pc), args.num_point, replace=False)
        points_surface_list.append(pc[None, rand_idx])

    return torch.vstack(points_surface_list).to(args.device).float()


def get_joint_label_merged_arteq(lbs_weights):
    gt_joint = torch.argmax(lbs_weights, dim=1)
    gt_joint1 = torch.where((gt_joint == 22), 20, gt_joint)
    gt_joint2 = torch.where((gt_joint1 == 23), 21, gt_joint1)
    gt_joint2 = torch.where((gt_joint2 == 10), 7, gt_joint2)
    gt_joint2 = torch.where((gt_joint2 == 11), 8, gt_joint2)

    return gt_joint2


def get_joint_label_merged(lbs_weights): 
    return torch.argmax(lbs_weights, dim=1)
     

def SMPLX_layer(body_model_list, betas, translation, motion_pose, rep="6d"):
    
    bz = 1

    mesh_j_pose_list, mesh_rec_list = [], []

    for _k_, body_model in enumerate(body_model_list):

        if rep == "rotmat":
            motion_pose_aa = rotation_matrix_to_angle_axis(motion_pose[_k_].reshape(-1, 3, 3)).reshape(bz, -1)
        else:
            motion_pose = motion_pose[_k_].squeeze().reshape(bz, -1)
            motion_pose_aa = motion_pose

        zero_center = torch.zeros_like(translation[_k_].reshape(-1, 3).cuda())
        body_param_rec = {}
        body_param_rec["transl"] = zero_center
        body_param_rec["global_orient"] = motion_pose_aa[:, :3].cuda()
        body_param_rec["body_pose"] = torch.cat([motion_pose_aa[:, 3:66].cuda(), torch.zeros(bz, 6).cuda()], dim=1)
        body_param_rec["betas"] = betas[_k_].reshape(bz, -1)[:, :10].cuda()



        body_mesh = body_model(return_verts=True, **body_param_rec)
        
        mesh_rec_list.append(body_mesh.vertices)
        mesh_j_pose_list.append(body_mesh.joints)

    mesh_rec = torch.vstack(mesh_rec_list)
    mesh_j_pose = torch.vstack(mesh_j_pose_list)

    return mesh_j_pose, mesh_rec


def train(args, model, bodymodel_dict, optimizer, train_loader):
    model.train()

    pbar = tqdm.tqdm(train_loader)
  
    for batch_data in pbar:

        motion_pose_aa = batch_data["pose"].to(args.device)        
        motion_trans = batch_data["trans"].to(args.device)
        betas = batch_data["betas"][:, None, :].to(args.device)
        pc_data = batch_data["point_cloud"].to(args.device).float()
      
        global_root = batch_data["global_orient"].to(args.device)

        bm_list = [bm_dict[g] for g in batch_data["gender"]]
  
        B, _ = motion_trans.size()
  
        # concatenate with the global_orient
        motion_pose_aa = torch.cat([global_root, motion_pose_aa], dim=1).reshape(-1, 3)
        motion_pose_rotmat = batch_rodrigues(motion_pose_aa).reshape(B, -1, 3, 3)
        motion_pose_rotmat_global = local_to_global_bone_transformation(motion_pose_rotmat, parents)
        motion_pose_rotmat_global = motion_pose_rotmat_global.to(args.device)
        
        gt_joints_pos, gt_vertices = SMPLX_layer(bm_list, betas, motion_trans, motion_pose_rotmat, rep="rotmat")

        # discard te 0 entries 
        pc_data_list = [elem[elem.sum(-1) != 0] for elem in pc_data]
        pc_data_list_idx = []
        
        # compute closest points
        for _i_, elem in enumerate(pc_data_list):
            
            clo_pts = knn_points(gt_vertices[_i_][None], pc_data_list[_i_][None], K=1, return_nn=True)[1][0, :, 0]
            pc_data_list_idx.append(pc_data_list[_i_][clo_pts])

            # trimesh.Trimesh(vertices=gt_vertices[0].cpu(), faces=bodymodel_dict[batch_data["gender"][_i_]].faces).export(os.path.join(f'rezz/body_gt_{_i_:04d}.obj'))
            # trimesh.PointCloud(pc_data_list[_i_][clo_pts].cpu().numpy()).export(os.path.join(f'rezz/pc_{_i_:04d}.ply'))
            # import ipdb; ipdb.set_trace()
 
        pcl_data, label_data, pcl_lbs = get_pointcloud_(pc_data_list_idx, args.num_point, gender_list=batch_data["gender"])

        losses = {}

        if epoch < 1:
            gt_part_seg = to_categorical(label_data, len(parents)).cuda()
        else:
            gt_part_seg = None
 

        # get_gpu()

        pred_joint, pred_pose, pred_shape, trans_feat = model(
            pcl_data, gt_part_seg, None, pcl_lbs, is_optimal_trans=False, parents=parents)

        # pred_joint, pred_pose, pred_shape, trans_feat = \
        #     model(pcl_data, None, None, None, is_optimal_trans=False, parents=parents)

        # gt_joints_set = [10, 11]
        # pred_pose[:, gt_joints_set] = motion_pose_rotmat_global.reshape(pred_pose.shape[0], pred_pose.shape[1], -1)[:, gt_joints_set]

        pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)

        angle_loss = norm_loss(pred_pose[:, :len(parents)], motion_pose_rotmat_global[:, :len(parents)], loss_type="l2")
 
        pred_joints_pos, pred_vertices = SMPLX_layer(bm_list, pred_shape, motion_trans, pred_joint_pose, rep="rotmat")

        pred_pcl_part_mean = model.soft_aggr_norm(pcl_data.unsqueeze(3), pred_joint).squeeze()
        gt_pcl_part_mean = scatter_mean(pcl_data, label_data.cuda(), dim=1)

      

        losses["angle_recon"] = angle_loss * args.angle_w
        losses["beta"] = F.mse_loss(pred_shape, betas.reshape(B, -1)[:, :10])
        losses["vertices"] = F.mse_loss(gt_vertices, pred_vertices) * args.vertex_w
        losses["normal"] = surface_normal_loss(pred_vertices, gt_vertices).mean() * args.normal_w
        losses["marker"] = F.mse_loss(pred_vertices[:, markers_idx, :], gt_vertices[:, markers_idx, :]) * args.vertex_w * 2

        try: 
            losses["pcl_part_mean"] = F.mse_loss(pred_pcl_part_mean, gt_pcl_part_mean) * args.vertex_w
        except:
            print(f"Issue with pcl part mean {batch_data['index']}")
        
        
        losses["joints_pos"] = (F.mse_loss(gt_joints_pos.reshape(-1, 45, 3), pred_joints_pos.reshape(-1, 45, 3)) * args.jpos_w)
        
        losses["seg_loss"] = (point_cls_loss(pred_joint.contiguous().view(-1, pred_joint.shape[-1]),
                                label_data.reshape(-1), trans_feat) * args.part_w)


        all_loss = 0.0
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        if all_loss.isnan().any():
            print("Loss is NaN issue")
            all_loss = 0.0

        losses["all_loss"] = all_loss

        optimizer.zero_grad()

        all_loss.backward()
        for nm, param in model.named_parameters():
            if param.grad is None:
                pass
            elif param.grad.sum().isnan():
                param.grad = torch.zeros_like(param.grad)

        optimizer.step()

        pred_choice = pred_joint.clone().reshape(-1, part_num).data.max(1)[1]
        correct = (pred_choice.eq(label_data.reshape(-1,).data).cpu().sum())

        batch_correct = correct.item() / (args.batch_size * args.num_point)

        pbar.set_description(f"Batch part acc: {batch_correct:.03f} Training Loss: {all_loss.item():.02f}")
        
    return all_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, metavar="N", help="input batch size for training (default: 128)")
    
    parser.add_argument("--latent_num", type=int, default=128, metavar="N", help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=15, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="N", help="learning rate")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--rep_type", type=str, default="6d", metavar="N", help="aa, 6d")
    parser.add_argument("--part_num", type=int, default=22, metavar="N", help="part num of the SMPL body")
    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--aug_type", type=str, default="so3", metavar="N", help="so3, zrot, no")
    parser.add_argument("--gt_part_seg", type=str, default="auto", metavar="N", help="")
    parser.add_argument("--gt-flag", type=bool)
    parser.add_argument("--garment-flag", type=bool)
    parser.add_argument("--aug-flag", type=bool)
    parser.add_argument("--EPN_input_radius", type=float, default=0.4, help="train from pretrained model")
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--kinematic_cond", type=str, default="yes", metavar="N", help="point num sampled from mesh surface")
    parser.add_argument("--part_w", type=float, default=5, help="")
    parser.add_argument("--angle_w", type=float, default=5, help="")
    parser.add_argument("--jpos_w", type=float, default=1e2, help="")
    parser.add_argument("--vertex_w", type=float, default=1e2, help="")
    parser.add_argument("--normal_w", type=float, default=1e0, help="")
    

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda

    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    exps_folder = "GARMENT_{}_GT_{}_AUG_{}_num_point_{}".format(args.garment_flag,
                                                                args.gt_flag,
                                                                args.aug_flag,
                                                                args.num_point)
   
    output_folder = os.path.sep.join(["./experiments", exps_folder])
 
    body_model_neutral = get_body_model(model_type="smpl", gender="neutral", 
                                        batch_size=1, device="cuda")
    body_model_female = get_body_model(model_type="smpl", gender="female",
                                        batch_size=1, device="cuda")
    body_model_male = get_body_model(model_type="smpl", gender="male", 
                                     batch_size=1, device="cuda")
 

    bm_dict = {"neutral": body_model_neutral, 
               "female": body_model_female,
               "male": body_model_male}
    

    body_model_faces = bm_dict["neutral"].faces.astype(int)
    # parents = bm_dict["neutral"].parents[:22]
    parents = bm_dict["neutral"].parents
    gt_lbs_dict = {'neutral': bm_dict["neutral"].lbs_weights,
                   'male': bm_dict["male"].lbs_weights,
                   'female': bm_dict["female"].lbs_weights} 
    

    part_num = len(parents)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nc, _ = get_nc_and_view_channel(args)

    model = PointCloud_network_equiv(option=args,
                                    z_dim=args.latent_num,
                                    nc=nc,
                                    part_num=part_num).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = DFaustDataset(data_path='outdir/DFaust', 
                                  train_flag=True, 
                                  gt_flag=args.gt_flag, 
                                  aug_flag=args.aug_flag, 
                                  garment_flag=args.garment_flag)
 

    surface_normal_loss = NormalVectorLoss(face=bm_dict["neutral"].faces.astype(int))
    point_cls_loss = get_part_seg_loss()
 
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              drop_last=True)


    for epoch in range(args.epochs):
        average_all_loss = train(args, model, bm_dict, optimizer, train_loader) 
        torch.save(model.state_dict(), os.path.join(output_folder, f"model_epochs_{epoch:08d}.pth"))
