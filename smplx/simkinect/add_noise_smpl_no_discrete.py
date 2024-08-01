# adapted from https://github.com/ankurhanda/simkinect
import os 
import cv2 
import json
import torch
import numpy as np 
import open3d as o3d
 
from pytorch3d.io import IO, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.implicitron.tools import point_cloud_utils
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, look_at_view_transform 
 

def add_gaussian_shifts(depth, std=1/2.0):

    if len(depth.shape) == 2:
        rows, cols = depth.shape 
        bs = 1 
    else:
        bs, rows, cols = depth.shape 

    gaussian_shifts = np.random.normal(0, std, size=(bs, rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)
 
    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp = np.repeat(xp[None, ...], bs, axis=0) 
    yp = np.repeat(yp[None, ...], bs, axis=0) 

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[..., 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[..., 1], 0.0), rows)

    depth_interp = np.array([cv2.remap(depth[i], xp_interp[i], yp_interp[i], cv2.INTER_LINEAR) for i in range(bs)])

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_, size_filt_=6):

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    if len(disp.shape) == 2:
        disp_rows, disp_cols = disp.shape 
        dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape
    else:
        bs, disp_rows, disp_cols = disp.shape
        dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    dot_pattern_ = np.repeat(dot_pattern_[None, ...], bs, axis=0)

    lim_rows = disp_rows - size_filt_
    lim_cols = disp_cols - size_filt_

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):
            r_dot, c_dot = r, c
            if (dot_pattern_rows - size_filt_ < r):
                r_dot = r % (dot_pattern_rows - size_filt_)

            if (dot_pattern_cols - size_filt_ < c):
                c_dot = c % (dot_pattern_cols - size_filt_)
            
            # since the dot_pattern_ has the same values for all the channels, observing one channel is enough
            if dot_pattern_[0, r_dot+center, c_dot+center] > 0:
                
                # c and r are the top left corner 
                window  = disp[:, r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[:, r_dot:r_dot+size_filt_, c_dot:c_dot+size_filt_] 

                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        #out_disp[r+center, c + center] = round((accu)*8.0) / 8.0 # discretization

                        import ipdb; ipdb.set_trace()

                        out_disp[r+center, c + center] = accu ########################

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def capture_mesh_depth(meshes, camera, image_size):

    raster_settings = RasterizationSettings(image_size=image_size)
                                            # , faces_per_pixel=1)

    rasterizer = MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings)

    fragments = rasterizer(meshes)

    # N x image_size x image_size
    depth = fragments.zbuf.cpu().numpy()[..., 0] 

    return depth

def recover_pcd_from_depth(depth_noised_tensor, camera):

    if len(depth_noised_tensor.shape) == 2:
        h, w = depth_noised_tensor.shape
        bs = 1
    else:
        bs, h, w = depth_noised_tensor.shape
        
    # dummy rgb values
    rgb = torch.ones((bs, 3, h, w)).to(depth_noised_tensor) * 0.5

    projected_pcd_noised = point_cloud_utils.get_rgbd_point_cloud(camera=camera, 
                                                                  image_rgb=rgb, 
                                                                  depth_map=depth_noised_tensor.reshape(bs, 1, h, w))

    return projected_pcd_noised


def mesh_2_kinectpcd(vertices, faces, camera_config_file="", depth_noise_file=""):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # read config file
    if camera_config_file == '':
        camera_config_file = 'camera_configs/kinect.json'

    with open(camera_config_file, 'r') as f:
        camera_config_dict = json.load(f)

    
    upsample_mesh = False

    image_size = (camera_config_dict['height'], camera_config_dict['width'])
    setting_name = camera_config_dict['setting_name']
    T, verts_num, _ = vertices.shape

    camera_ext_dict = {_cam_['camera_id']: {'R': _cam_['cam_R'], 'T': _cam_['cam_T']} for _cam_ in camera_config_dict["cameras"]}
    num_cameras = len(camera_ext_dict)

    io_object = IO()

    if depth_noise_file == "":
        depth_noise_file = "smplx/simkinect/data/sample_pattern.png"

    # reading the image directly in gray with 0 as input 
    kinect_dot_pattern = cv2.imread(depth_noise_file, cv2.IMREAD_GRAYSCALE)
    

     # various variables to handle the noise modelling
    scale_factor  = 100.0   # converting depth from m to cm 
    baseline_m    = 0.075   # baseline in m 0.075
    INVALID_DISP_THRESHOLD = 99999999.9
    filter_size_window = camera_config_dict['conv_filter_size_window']

    # world units
    fx, fy = camera_config_dict['focal_length_x'], camera_config_dict['focal_length_y']

    # camera_ext_dict_ = {0:(2, -90, 180), 
    #                         1:(2, 90, 0),
    #                         2:(2, 90, 90),
    #                         3:(2, 90, -90)}

    for _t_ in range(T):
    
        if upsample_mesh:
            # Apply the subdivision algorithm to the mesh
            print("Upsampling the mesh")
            mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices[_t_]), 
                    triangles=o3d.utility.Vector3iVector(faces)).subdivide_midpoint(number_of_iterations=1)
            
            # Check the initial number of vertices and triangles
            print(f"Number of vertices: {len(mesh.vertices)}")
            meshes = Meshes(verts=[torch.tensor(mesh.vertices).to(device)], 
                            faces=[torch.tensor(mesh.faces.astype(np.int32)).to(device)])

        else: 
            meshes = Meshes(verts=[torch.tensor(vertices[_t_]).to(device) for _ in range(num_cameras)], 
                            faces=[torch.tensor(faces.astype(np.int32)).to(device) for _ in range(num_cameras)])

    
        cam_batch_R = torch.tensor([elem['R'] for elem in camera_ext_dict.values()])
        cam_batch_T = torch.tensor([elem['T'] for elem in camera_ext_dict.values()])

        focal_length = torch.tensor([fx, fy]*num_cameras).reshape(-1, 2)
        image_size_mult = torch.tensor([image_size]*num_cameras).reshape(-1, 2)
        
        cameras_batch = PerspectiveCameras(focal_length=focal_length, device=device, R=cam_batch_R, T=cam_batch_T, image_size=image_size_mult)

        # minmax scaling 
        depth_gt_unscaled = capture_mesh_depth(meshes, cameras_batch, image_size=image_size)
        depth_gt = (depth_gt_unscaled- depth_gt_unscaled.min()) / (depth_gt_unscaled.max() - depth_gt_unscaled.min() + 1e-10)

        depth_interp = add_gaussian_shifts(depth_gt)
        disp_ = fx * baseline_m / (depth_interp + 1e-10)

        # depth_f = np.round(disp_ * 8.0)/8.0 # discretization
        depth_f = disp_ ######################## 

        out_disp = filterDisp(depth_f, kinect_dot_pattern, INVALID_DISP_THRESHOLD, size_filt_=filter_size_window)

        depth = fx * baseline_m / out_disp
        
        # The depth here needs to converted to centimeters so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=depth.shape)*(1.0/6.0) + 0.5))/scale_factor # discretization

        import ipdb; ipdb.set_trace()

        depth[out_disp == INVALID_DISP_THRESHOLD] = 0 
    

        os.makedirs(f'outdir/{setting_name}', exist_ok=True)
        # save depth for each camera
        for cam_id in range(num_cameras):
            cv2.imwrite(f'outdir/{setting_name}/perfect_depth_{cam_id}_{image_size}.png', depth_gt[cam_id] * 255)
            cv2.imwrite(f'outdir/{setting_name}/processed_depth_{cam_id}_{image_size}.png', depth[cam_id] * 255)
            cv2.imwrite(f'outdir/{setting_name}/noised_depth_{cam_id}_{image_size}.png', noisy_depth[cam_id] * 255)

        # Recover point cloud from noised depth image 
        projected_pcd_noised = recover_pcd_from_depth(torch.from_numpy(noisy_depth).to(device), cameras_batch)


        for cam_id in range(num_cameras):
            io_object.save_pointcloud(projected_pcd_noised[cam_id], f'outdir/{setting_name}_noised_smpl_p3d_no_discret_filt6_{cam_id}_{image_size}.ply')
            io_object.save_pointcloud(projected_pcd_noised, f'outdir/{setting_name}_noised_smpl_p3d_no_discret_filt6_{cam_id}_{image_size}.ply')
    