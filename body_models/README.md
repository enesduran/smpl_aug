# Body Models

Our framework is compatible with [SMPL](https://smpl.is.tue.mpg.de/), [SMPLH](https://mano.is.tue.mpg.de) or [SMPLX](https://smpl-x.is.tue.mpg.de/) explicit body models. You need to register, download them under `body_models` folder. Finally the folder structure should be:

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

**Note**: You have to perform merging operation of the downloaded SMPLH model with the MANO model. To this end run for each gender model: 

```
python core/tools/merge_smplh_mano.py --smplh-fn body_models/smplh/SMPLH_MALE.npz --mano-right-fn body_models/mano/MANO_RIGHT.pkl --mano-left-fn body_models/mano/MANO_LEFT.pkl --output-folder body_models/smplh_mano
```
