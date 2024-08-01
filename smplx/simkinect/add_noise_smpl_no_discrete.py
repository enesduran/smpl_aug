# adapted from https://github.com/ankurhanda/simkinect
import os 
import cv2 
import sys 
import torch
import numpy as np 
import open3d as o3d
 
from pytorch3d.io import IO, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.implicitron.tools import point_cloud_utils
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, look_at_view_transform 
 

def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

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

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

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

            if dot_pattern_[r_dot+center, c_dot+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r_dot:r_dot+size_filt_, c_dot:c_dot+size_filt_] 
  
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
                        out_disp[r+center, c + center] = accu ########################

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def capture_mesh_depth(meshes, camera, image_size):
    raster_settings = RasterizationSettings(
        image_size=image_size, 
    )

    rasterizer = MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    )

    fragments = rasterizer(meshes)

    depth = fragments.zbuf.cpu().numpy()[0,:,:,0] # image_size x image_size

    return depth

def recover_pcd_from_depth(depth_noised_tensor, camera):
    h, w = depth_noised_tensor.shape[2:]
    rgb = torch.rand(1, 3, h, w).to(depth_noised_tensor) # dummy rgb values
    # rgb = torch.ones((1, 3, h, w)) # dummy rgb values
    projected_pcd_noised = point_cloud_utils.get_rgbd_point_cloud(camera, rgb, depth_noised_tensor)
    return projected_pcd_noised


def mesh_2_kinectpcd(vertices, faces):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  
    upsample_mesh = False

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices[0]), 
                                     triangles=o3d.utility.Vector3iVector(faces))
    
    if upsample_mesh:
        # Apply the subdivision algorithm to the mesh
        print("Upsampling the mesh")
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)

    # Check the initial number of vertices and triangles
    print(f"Initial number of vertices: {len(mesh.vertices)}")
    print(f"Initial number of triangles: {len(mesh.triangles)}")
 
    meshes = Meshes(verts=[torch.tensor(vertices[0]).to(device)], 
                    faces=[torch.tensor(faces.astype(np.int32)).to(device)])


    image_size = (1500, 1500)

    camera_ext_dict = {0:(2, -90, 180), 
                       1:(2, 90, 0),
                       2:(2, 90, 90),
                       3:(2, 90, -90)}

    io_object = IO()

    # reading the image directly in gray with 0 as input 
    kinect_dot_pattern = cv2.imread("smplx/simkinect/data/sample_pattern.png", cv2.IMREAD_GRAYSCALE)

    # various variables to handle the noise modelling
    scale_factor  = 100.0   # converting depth from m to cm 
    focal_length  = 480.0   # focal length of the camera used 480
    baseline_m    = 0.075   # baseline in m 0.075
    invalid_disp_ = 99999999.9
    filter_size_window = 6


    for i in range(4):

        # create camera (distance, elevation, azimuth)
        R, T = look_at_view_transform(*camera_ext_dict[i])
        cameras = PerspectiveCameras(focal_length=2, device=device, R=R, T=T, image_size=image_size)
        camera = cameras[0]
    
        # capture depth
        depth_gt_unscaled = capture_mesh_depth(meshes, camera, image_size=image_size)
        depth_gt = (depth_gt_unscaled- depth_gt_unscaled.min()) / (depth_gt_unscaled.max() - depth_gt_unscaled.min())

        cv2.imwrite(f'outdir/perfect_depth_{i}_{image_size}.png', depth_gt*255)

 
        # depth_interp = add_gaussian_shifts(depth_gt_unscaled)
        depth_interp = add_gaussian_shifts(depth_gt)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        # depth_f = np.round(disp_ * 8.0)/8.0 # discretization
        depth_f = disp_ ######################## 

        out_disp = filterDisp(depth_f, kinect_dot_pattern, invalid_disp_, size_filt_=filter_size_window)

        depth = focal_length * baseline_m / out_disp
        cv2.imwrite(f'outdir/processed_depth_{i}_{image_size}.png', depth * 255)


        noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=depth.shape)*(1.0/6.0) + 0.5))/scale_factor # discretization
        depth[out_disp == invalid_disp_] = 0 


        # The depth here needs to converted to centimeters so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        # noisy_depth = depth 
        cv2.imwrite(f'outdir/noised_depth_{i}.png', noisy_depth * 255)

        # minmax scaling 
        depth_noised_tensor = torch.from_numpy(noisy_depth)[None, None, :, :].to(device)

        ## Recover noised depth ##
        projected_pcd_noised = recover_pcd_from_depth(depth_noised_tensor, camera)
        io_object.save_pointcloud(projected_pcd_noised, f'outdir/noised_smpl_p3d_no_discret_filt6_{i}_{image_size}.ply')
        
    