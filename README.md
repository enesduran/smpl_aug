# SMPL AUGMENTATION LIBRARY 

We provide a framework for depth rendering and data augmentation of SMPL body model. We aim to show the effectiveness of our framework in some plausible use cases:

    1) Data augmentation tool for multiview noisy depth images
    2) Simulation accompanying kinect camera noise for a given RGB video 


Huge shoutout to [SMPLX](https://github.com/vchoutas/smplx), [SCULPT](https://github.com/soubhiksanyal/SCULPT_release), [SimKinect](https://github.com/ankurhanda/simkinect) implementations. 

# Creating Environment 

Create envrionment by running:

```conda env create -f env.yml```

before running load cuda 12.1
module load cuda/12.1

# Setup 

Here are the instructions for setting SCULPT and SMPL models.  

### Body Models 

Our framework is compatible with [SMPL](https://smpl.is.tue.mpg.de/), [SMPLH](https://mano.is.tue.mpg.de) or [SMPLX](https://smpl-x.is.tue.mpg.de/) explicit body models. and register. Download version 1.1.0 and put them under `smplx/smpl_all_models`. 

```
./body_models
    ├── smpl
    |   ├── SMPL_FEMALE.pkl
    |   ├── SMPL_MALE.pkl
    |   └── SMPL_NEUTRAL.pkl   
    ├── smplh(*) 
    |   ├── female
    |   ├── male
    |   └── neutral
    └── smplx(*) 
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_MALE.npz
        └── SMPLX_NEUTRAL.npz   
(*) optional
```

### SCULPT (Optional)

Our framework uses SCULPT as garment generation model. If you want to make use of this optional feature go to [SCULPT webpage](https://sculpt.is.tue.mpg.de/) and register. Download Pre-trained weights for the Geometry Network  and place them under `smplx/sculpt/data`

# Run Demo 

Having set environment up, downloaded models and placed them under the corresponding paths, you are good to go! To run the augmentation:
 
```python core/demo.py --model-folder body_models --body-model-type smpl --motion-path motion_data/sample_motion_data_smpl.npz --camera-config camera_configs/kinect.json``` 
```python core/demo.py --model-folder body_models --body-model-type smplh --motion-path motion_data/sample_motion_data_smpl.npz --camera-config camera_configs/kinect.json``` 
```python core/demo.py --model-folder body_models --body-model-type smplx --motion-path motion_data/sample_motion_data_smplx.npz --camera-config camera_configs/kinect.json``` 

Notice that our method uses SMPL body model. It can be extended to smplx model (pose data and garment generator should be compatible with SMPLX body model)


TODO (with importance order)

1) Decide on the final architecture to continue and rewrite the codebase
2) Further purge the SCULPT and SimKInect codebase.
3) Extend it to other body models SMPLX, SMPLH without garment support 
4) Providing 3D scene or already rendered depth image of the scene as an optional argument 
5) Formulating tasks to be used in tandem with this framework 
6) Write the report 
7) Presentation 
