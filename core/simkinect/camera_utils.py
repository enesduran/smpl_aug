import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage

def vertex_from_depth(depth,
                      fl,
                      pp,
                      depth_range,
                      png_scale_factor): 
    '''
        This function takes depth map (png image) and 
        converts it into a point cloud 
        
        @params
        fl: focal lenth of the camera

        pp: principal point, center of the camera

        depth_range: depth range from min to max

        pnt_scale_factor: the scale factor by which the 
                          depth maps were scaled to convert
                          to png image 

    '''
    
    fl_x, fl_y = fl 
    pp_x, pp_y = pp 
    min_depth, max_depth = depth_range

    rows, cols = depth.shape
    # bs, rows, cols = depth.shape

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    # convert depth to values in meters 
    depth_m = np.copy(depth) / png_scale_factor 

    # check for range 
    depth_m[depth_m < min_depth] = min_depth
    depth_m[depth_m > max_depth] = max_depth

    # depth_mm = np.max(depth_mm, 0.0)

    vx = (xp - pp_x) * depth_m / fl_x 
    vy = (yp - pp_y) * depth_m / fl_y 
    vz = depth_m 
 
    vertices = np.transpose(np.stack([vx, vy, vz]), (1, 2, 0))

    return vertices # , depth_m 

def add_gaussian_shifts(depth, std=1/2.0):

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

def filterDisp(disp, dot_pattern_, invalid_disp_, size_filt_):

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
    fill_weights[sqr_radius > size_filt_] = -1.0 

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