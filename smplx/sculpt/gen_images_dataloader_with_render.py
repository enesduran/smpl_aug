# Modified from StyleGAN3 codebase

"""Generate images using pretrained network pickle.
Here we have provided the precomputed parameters used to generate the images for 
the main paper and the SUPMAT video in the website. One can easily 
modify these or make different combinations of these."""

import os
import time 
import torch
import dnnlib
import legacy
import numpy as np
from typing import List, Optional 
 

class SCULPT(object):
    def __init__(self, 
                 geo_network_pkl_path: str,
                 texture_network_pkl_path: str,
                 outdir: str) -> None:
        
        time_start = time.time()
    

        super().__init__()

        self.outdir = outdir
        self.device = torch.device('cuda')
  
        with dnnlib.util.open_url(geo_network_pkl_path) as f:
            self.G_geometry = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        # 'blazerlong' is skipped      
        self.clothing_names = {0: 'longlong', 1: 'shirtlong', 2: 'shortshort_poloshort', 3: 'longshort_jerseyshort', 4: 'shirtshort', 5: 'shortlong'}

        # num_samples = 50
        os.makedirs(outdir, exist_ok=True)
        print('Time taken in total:', time.time()-time_start)


    def generate_images(self,
        seeds: List[int],
        body_pose: Optional[torch.Tensor],
        cloth_types: Optional[str]):

        time_start_creation = time.time()

        assert cloth_types.shape[0] == body_pose.shape[0]
        sample_size = body_pose.shape[0]

        # style vector dim is 512 for both goemetry and texture networks      
        z_geo = torch.from_numpy(np.random.RandomState(seeds).randn(sample_size, self.G_geometry.z_dim)).to(self.device) 
        label_ct = torch.from_numpy(cloth_types).to(self.device)
          
        body_pose = body_pose.to(self.device)

        mesh_ply_names = []

        # find corresponding clothing
        for i in range(6):
            mesh_ply_names += [self.clothing_names[i]] * 20 
        mesh_ply_names = np.array(mesh_ply_names)
            
        ## Mapping network
        ws_geo = self.G_geometry.mapping(z_geo, torch.cat((label_ct, body_pose[:,3:66].to(self.device)), 1), truncation_psi=1.0)

        ## Texture Network
        ws_geo = ws_geo.to(torch.float32).unbind(dim=1)
        
        # Execute layers.
        x_geo = self.G_geometry.synthesis.input(ws_geo[0])

        for name, w_geo in zip(self.G_geometry.synthesis.layer_names, ws_geo[1:]):
            x_geo = getattr(self.G_geometry.synthesis, name)(x_geo, w_geo, update_emas=False)

        if self.G_geometry.synthesis.output_scale != 1:
            x_geo = x_geo * self.G_geometry.synthesis.output_scale

        # Ensure correct shape and dtype.
        UV_geo = x_geo.to(torch.float32)

        disp_img_geo = (UV_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
        vert_disps = self.G_geometry.displacement_Layer(disp_img_geo)

        print('Time taken for creating images:', time.time()-time_start_creation)
        return vert_disps

        

        