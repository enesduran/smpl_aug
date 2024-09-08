# adapted from https://github.com/ankurhanda/simkinect
import torch
import numpy as np 
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.implicitron.tools import point_cloud_utils
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer 

from simkinect.camera_utils import filterDisp, filterDisp_batch, add_gaussian_shifts

 
def capture_mesh_depth(meshes, camera, image_size):

    raster_settings = RasterizationSettings(image_size=image_size)

    rasterizer = MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings)

    fragments = rasterizer(meshes)

    # N x image_size x image_size
    depth = fragments.zbuf.cpu().numpy()[..., 0] 

    return depth

def recover_pcd_from_depth(depth_noised_tensor, cameras, single_pc_flag=False):

    bs, h, w = depth_noised_tensor.shape
        
    depth_noised_tensor = depth_noised_tensor.reshape(bs, 1, h, w)
    # dummy rgb values
    rgb = torch.ones((bs, 3, h, w)).to(depth_noised_tensor) * 0.5

    if single_pc_flag:
        projected_pcd_noised_list = point_cloud_utils.get_rgbd_point_cloud(camera=cameras, 
                                                                        image_rgb=rgb, 
                                                                        depth_map=depth_noised_tensor)
    else:
        projected_pcd_noised_list = [point_cloud_utils.get_rgbd_point_cloud(camera=cameras[i], 
                                                                  image_rgb=rgb[i][None], 
                                                                  depth_map=depth_noised_tensor[i][None]) for i in range(bs)]
    
    return projected_pcd_noised_list


def mesh2pcd(vertices, 
             faces, 
             camera_config_dict, 
             cameras_batch, 
             kinect_dot_pattern, 
             upsample_mesh=False, 
             single_pc_flag=False,
             device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):
    """ 
    Takes in the vertices and faces of the mesh and the camera configuration and
    returns the point cloud of the mesh with the noise added to the depth image

    params:
    vertices: np.array (V, 3)
    faces: np.array (F, 3)
    camera_config_dict: dict
    cameras_batch: list of cameras
    kinect_dot_pattern: np.array (H, W)

    """
 
    image_size = (camera_config_dict['height'], camera_config_dict['width'])
    setting_name = camera_config_dict['setting_name']

    num_cameras = len(cameras_batch)

 
    # various variables to handle the noise modelling
    baseline_m    = 0.075   # baseline in m 0.075
    INVALID_DISP_THRESHOLD = 99999999.9
    scale_factor  = camera_config_dict['depth_scale']
    filter_size_window = camera_config_dict['conv_filter_size_window']
    fx = camera_config_dict['focal_length_x']

     
    if upsample_mesh:
        # Apply the subdivision algorithm to the mesh
        print("Upsampling the mesh")
        mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertices), 
                triangles=o3d.utility.Vector3iVector(faces)).subdivide_midpoint(number_of_iterations=1)
        
        # Check the initial number of vertices and triangles
        print(f"Number of vertices: {len(mesh.vertices)}")
        meshes = Meshes(verts=[torch.tensor(mesh.vertices).to(device)], 
                        faces=[torch.tensor(mesh.faces.astype(np.int32)).to(device)])

    else: 
        meshes = Meshes(verts=[torch.tensor(vertices).to(device) for _ in range(num_cameras)], 
                        faces=[torch.tensor(faces.astype(np.int32)).to(device) for _ in range(num_cameras)])
 
    # minmax scaling 
    depth_gt_unscaled = capture_mesh_depth(meshes, cameras_batch, image_size=image_size)    
    depth_gt_unscaled[depth_gt_unscaled>camera_config_dict["depth_max"]] = camera_config_dict["depth_max"]
 
    depth_gt_scaled = (depth_gt_unscaled - depth_gt_unscaled.min()) / (depth_gt_unscaled.max() - depth_gt_unscaled.min() + 1e-10)
 
    depth_f = fx * baseline_m / (add_gaussian_shifts(depth_gt_unscaled)  + 1e-10)
    out_disp = np.array([filterDisp(depth_f[i], kinect_dot_pattern, INVALID_DISP_THRESHOLD, filter_size_window) for i in range(num_cameras)])
    depth = fx * baseline_m / out_disp
        
    # The depth here needs to converted to centimeters so scale factor is introduced 
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
    # noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=depth.shape)*(1.0/6.0) + 0.5))/scale_factor # discretization
    noisy_depth = (35130/np.round(depth*scale_factor)) 
    noisy_depth += np.random.normal(size=depth.shape)*(1.0/6.0)
    noisy_depth = np.round(noisy_depth + 0.5)
    noisy_depth = 35130 / noisy_depth
    noisy_depth /= scale_factor

    depth[out_disp == INVALID_DISP_THRESHOLD] = 0     
    projected_pcd_gt = recover_pcd_from_depth(torch.from_numpy(depth_gt_unscaled).to(device), cameras_batch, single_pc_flag=single_pc_flag)
    projected_pcd_noised = recover_pcd_from_depth(torch.from_numpy(noisy_depth).to(device), cameras_batch, single_pc_flag=single_pc_flag)
 
    return depth_gt_unscaled, depth_gt_scaled, noisy_depth, projected_pcd_gt, projected_pcd_noised
    

    
