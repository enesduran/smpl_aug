# SMPL AUGMENTATION LIBRARY 

We provide a framework for batch depth rendering and data augmentation of SMPL body model. We aim to show the effectiveness of our framework in some plausible use cases:

    1) Data augmentation tool for multiview noisy depth images
    2) Simulation accompanying kinect camera noise for a given RGB video 

We also release our code for training and evaluation at this link (todo)

Huge shoutout to [SMPLX](https://github.com/vchoutas/smplx), [SCULPT](https://github.com/soubhiksanyal/SCULPT_release), [SimKinect](https://github.com/ankurhanda/simkinect) implementations. 

Contributors: Enes Duran, Mattia Masiero, Yunhan Wang

### Creating Environment 

Create envrionment by running:

```
conda env create -f env.yml
```

### Setup 

Here are the instructions for setting SCULPT and SMPL models.  

#### Body Models 

Our framework is compatible with [SMPL](https://smpl.is.tue.mpg.de/), [SMPLH](https://mano.is.tue.mpg.de) or [SMPLX](https://smpl-x.is.tue.mpg.de/) explicit body models. and register. Download version 1.1.0 and put them under `body_models` folder. 

```
./body_models
    ├── smpl
    |   ├── SMPL_FEMALE.pkl
    |   ├── SMPL_MALE.pkl
    |   └── SMPL_NEUTRAL.pkl   
    ├── smplh(*) 
    |   ├── SMPLH_FEMALE.npz
    |   ├── SMPLH_MALE.npz
    |   └── SMPLH_NEUTRAL.npz
    └── smplx(*) 
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_MALE.npz
        └── SMPLX_NEUTRAL.npz   
(*) optional
```

#### SCULPT (Optional)

Our framework uses SCULPT as garment generation model. If you want to make use of this optional feature go to [SCULPT webpage](https://sculpt.is.tue.mpg.de/) and register. Download Pre-trained weights for the Geometry Network  and place them under `smplx/sculpt/data`. 

If you plan to use SCUPT to dress SMPL body, please set the flag `--clothing-option clothed`.

#### Config file

You can set the body model path, motion npz file path, and camera config path in `configs/config.yaml`.

### Run Demo 

After setting the environment up, downloading models, and placing them under the corresponding paths, you are good to go! To run the augmentation:

To forward SMPL and get the corresponding Kinect depth and point cloud **with** cloth:

```
bash scripts/create_smpl_garment.sh
```

To forward SMPL and get the corresponding Kinect depth and point cloud **without** cloth:
```
bash scripts/create_smpl_minimal.sh
```

If you would like to use SMPL-X or SMPL-H for the forward pass, please change --body-model-type to smplx or smplh respectively.

When running demo.py, it first creates an SMPL wrapper class with the provided SMPL model. Then, it loads the provided motions and synthesizes the corresponding augmented human point cloud.