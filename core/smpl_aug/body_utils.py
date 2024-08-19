import torch 
from typing import Optional
from smplx.utils import Tensor


SMPLX_JOINT_MIRROR_DICT = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6, 7: 8, 8: 7, 9: 9, 10: 11, 11: 10, 12: 12, 13: 14, 14: 13, 15: 15,
                            16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 22, 23: 24, 24: 23, 25: 40, 26: 41, 27: 42, 28: 43, 
                            29: 44, 30: 45, 31: 46, 32: 47, 33: 48, 34: 49, 35: 50, 36: 51, 37: 52, 38: 53, 39: 54, 40: 25, 41: 26,
                            42: 27, 43: 28, 44: 29, 45: 30, 46: 31, 47: 32, 48: 33, 49: 34, 50: 35, 51: 36, 52: 37, 53: 38, 54: 39}

SMPL_JOINT_MIRROR_DICT = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6, 7: 8, 8: 7, 9: 9, 10: 11, 11: 10, 12: 12, 13: 14, 14: 13, 15: 15, 
                          16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 23, 23: 22}

# discard root joint and update the rest
SMPL_JOINT_MIRROR_DICT_WO_ROOT = SMPL_JOINT_MIRROR_DICT.copy()
SMPL_JOINT_MIRROR_DICT_WO_ROOT.pop(0)
SMPL_JOINT_MIRROR_DICT_WO_ROOT = {k-1: v-1 for k, v in SMPL_JOINT_MIRROR_DICT_WO_ROOT.items()}

R = torch.tensor([1, -1, -1, -1, 1, 1, -1, 1, 1]).view(3, 3).float()

def flip_pose(pose: Optional[Tensor], 
              rotmat_flag: bool = False) -> Optional[Tensor]:
    """
    pose: tensor of shape (B, 24, 3) or (B, 24, 3, 3) 
    rotmat_flag = bool if True pose is in rotation matrix format
    """
    flipped_pose = pose.clone()

    if rotmat_flag:
        flipped_pose = (R * flipped_pose.view(-1, 3, 3)).view(pose.shape)
        # flipped_pose = torch.matmul(R, flipped_pose.view(-1, 3, 3)).view(pose.shape)
    else:
        flipped_pose[..., 1:] *= -1
        flipped_pose = flipped_pose.reshape(-1, 72)

    return flipped_pose