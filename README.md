# SMPL AUGMENTATION LIBRARY 

We provide a framework for depth rendering and data augmentation of SMPL body model. We aim to show the effectiveness BLABLABLABLA



Huge shoutout to [SMPLX](https://github.com/vchoutas/smplx), [SCULPT](https://github.com/soubhiksanyal/SCULPT_release), [SimKinect](https://github.com/ankurhanda/simkinect) implementations. 

# Creating Environment 

Create envrionment by running:

```conda env create -f env.yml```

before running load cuda 12.1
module load cuda/12.1

# Setup 

Here are the instructions for setting SCULPT and SMPL models.  

### SMPL 

Our framework uses SMPL as explicit body model. To [SMPL webpage](https://smpl.is.tue.mpg.de/) and register. Download version 1.1.0 and put them under `smplx/smpl_all_models`. 

### SCULPT (Optional)

Our framework uses SCULPT as garment generation model. If you want to make use of this optional feature go to [SCULPT webpage](https://sculpt.is.tue.mpg.de/) and register. Download Pre-trained weights for the Geometry Network  and place them under `smplx/sculpt/data`

# Run Demo 

Having set environment up, downloaded models and placed them under the corresponding paths, you are good to go! To run the augmentation:

```python smplx/demo.py --model-folder smpl_all_models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl --camera-config camera_configs/kinect.json``` 

Notice that our method uses SMPL body model. It can be extended to smplx model (pose data and garment generator should be compatible with SMPLX body model)


TODO (with importance order)

1) Decide on the final architecture to continue and rewrite the codebase
2) Further purge the SCULPT and SimKInect codebase.
3) Read camera intrinsics/extrinsics from a config file 
4) Extend it to other body models SMPLX, SMPLH without garment support 
5) Providing 3D scene or already rendered depth image of the scene as an optional argument 
6) Formulating tasks to be used in tandem with this framework 
7) Multiprocess implementation in depth rendering ==> Needs motion data 
8) Write the report 
9) Presentation 
