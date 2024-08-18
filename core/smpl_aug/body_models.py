# This code is a modified version of the original code from the SMPL-X project
# https://github.com/vchoutas/smplx/tree/main
# Contact: ps-license@tuebingen.mpg.de

import os
import sys 
import torch
import numpy as np
import os.path as osp
from loguru import logger
from typing import Optional, Dict, Union


import smplx 
from smplx.lbs import lbs
from smplx.utils import Struct, Tensor, Array, SMPLOutput
from smpl_aug.body_utils import SMPL_JOINT_MIRROR_DICT, SMPLX_JOINT_MIRROR_DICT, flip_pose


class SMPL(smplx.SMPL):

    NUM_BODY_JOINTS = 23

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        num_betas: int = 10,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_body_pose: bool = True,
        body_pose: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        dtype=torch.float32,
        batch_size: int = 1,
        joint_mapper=None,
        gender: str = 'neutral',
        age: str = 'adult',
        vertex_ids: Dict[str, int] = None,
        v_template: Optional[Union[Tensor, Array]] = None,
        clothing_option: str = 'minimal',
        **kwargs
    ) -> None:
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            num_betas: int, optional
                Number of shape components to use
                (default = 10).
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        
        super(SMPL, self).__init__(model_path=model_path,
	                        kid_template_path=kid_template_path,
                            data_struct=data_struct,
                            create_betas=create_betas,
                            betas=betas,
                            num_betas=num_betas,
                            create_global_orient=create_global_orient,
                            global_orient=global_orient,
                            create_body_pose=create_body_pose,
                            body_pose=body_pose,
                            create_transl=create_transl,
                            transl=transl,
                            dtype=dtype,
                            batch_size=batch_size,
                            joint_mapper=joint_mapper,
                            gender=gender,
                            age=age,
                            vertex_ids=vertex_ids,
                            v_template=v_template,
                            **kwargs)

        clothing_option = clothing_option.lower()
        self.clothing_option = clothing_option

        if clothing_option != 'minimal':
            logger.info('Using the clothed SMPL model, initializing the SCULPT model. This may take a while.')

            sys.path.append('core/sculpt')
            from sculpt.gen_images_dataloader_with_render import SCULPT

            self.sculpt_instance = SCULPT(geo_network_pkl_path='core/sculpt/data/network-snapshot-013440.pkl',
                                          outdir = './outdir')           

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        augment_pose: bool = False,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)
            augment_pose: bool, optional
                If True, then the pose is augmented through flipping 

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        if augment_pose:
            augment_idx = torch.tensor(list(SMPL_JOINT_MIRROR_DICT.values())) 
 
            if not pose2rot:
                full_pose = full_pose[:, augment_idx]
            else:
                full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)[:, augment_idx]

            full_pose = flip_pose(full_pose, rotmat_flag= not pose2rot)      
        
        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)


        v_template_cano = self.v_template

        # no workaround of lbs function as we also need minimal body
        # vertices to compute joints throught vertex_joint_selector.
        vertices, joints = lbs(betas, full_pose, v_template_cano,
                            self.shapedirs, self.posedirs,
                            self.J_regressor, self.parents,
                            self.lbs_weights, pose2rot=pose2rot)  
        joints = self.vertex_joint_selector(vertices, joints) 


        # depending on the clothing option, use the SCULPT model or SMPL model to generate human mesh 
        if self.clothing_option!='minimal':
            clothing_displacements = self.sculpt_instance.generate_images(seeds=[7],    # plate code of Antalya
                                                body_pose=full_pose,
                                                cloth_types=kwargs.get("cloth_types"),
                                                rotmat_flag= not pose2rot)
            
            v_template_cano = v_template_cano + clothing_displacements.to(v_template_cano)

            # override the vertices if clothing option is on 
            vertices, _ = lbs(betas, full_pose, v_template_cano,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.lbs_weights, pose2rot=pose2rot)     

 
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output


class SMPLLayer(SMPL):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        # Just create a SMPL module without any member variables
        super(SMPLLayer, self).__init__(
            create_body_pose=False,
            create_betas=False,
            create_global_orient=False,
            create_transl=False,
            *args,
            **kwargs,
        )


    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body.  Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format.  (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                Body pose. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''

        return super().forward(betas=betas,
                                body_pose=body_pose,
                                global_orient=global_orient,
                                transl=transl,
                                return_verts=return_verts,
                                return_full_pose=return_full_pose,
                                pose2rot=False,
                                **kwargs)



class SMPLH(smplx.SMPLH):

    def __init__(
        self, model_path,
        kid_template_path: str = '',
        data_struct: Optional[Struct] = None,
        create_left_hand_pose: bool = True,
        left_hand_pose: Optional[Tensor] = None,
        create_right_hand_pose: bool = True,
        right_hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        num_betas=16,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        gender: str = 'neutral',
        age: str = 'adult',
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' SMPLH model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_left_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the left
                hand. (default = True)
            left_hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            create_right_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            right_hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
 
        super(SMPLH, self).__init__(
                        model_path=model_path,
                        kid_template_path=kid_template_path,
                        data_struct=None,
                        create_left_hand_pose=create_left_hand_pose,
                        left_hand_pose=left_hand_pose,
                        create_right_hand_pose=create_right_hand_pose,
                        right_hand_pose=right_hand_pose,
                        use_pca=use_pca,
                        num_pca_comps=num_pca_comps,
                        num_betas=num_betas,
                        flat_hand_mean=flat_hand_mean,
                        batch_size=batch_size,
                        gender=gender,
                        age=age,
                        dtype=dtype,
                        vertex_ids=vertex_ids,
                        use_compressed=use_compressed,
                        ext=ext,
                        **kwargs)


class SMPLHLayer(smplx.SMPLHLayer):

    def __init__(
        self, *args, **kwargs
    ) -> None:
        ''' SMPL+H as a layer model constructor
        '''
        super(SMPLHLayer, self).__init__(*args, 
                                         **kwargs)



class SMPLX(smplx.SMPLX):
    '''
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    '''

    def __init__(
        self, model_path: str,
        kid_template_path: str = '',
        num_expression_coeffs: int = 10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour: bool = False,
        batch_size: int = 1,
        gender: str = 'neutral',
        age: str = 'adult',
        dtype=torch.float32,
        ext: str = 'npz',
        **kwargs
    ) -> None:
        ''' SMPLX model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''
 
        super(SMPLX, self).__init__(
            model_path=model_path,
            kid_template_path=kid_template_path,
            num_expression_coeffs=num_expression_coeffs,
            create_expression=create_expression,
            expression=expression,
            create_jaw_pose=create_jaw_pose,
            jaw_pose=jaw_pose,
            create_leye_pose=create_leye_pose,
            leye_pose=leye_pose,
            create_reye_pose=create_reye_pose,
            reye_pose=reye_pose,
            use_face_contour=use_face_contour,
            batch_size=batch_size,
            gender=gender,
            age=age,
            dtype=dtype,
            ext=ext,
            **kwargs)

class SMPLXLayer(smplx.SMPLXLayer):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
    
       super(SMPLXLayer, self).__init__(*args, 
                                        **kwargs) 

class MANO(smplx.MANO):

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        create_hand_pose: bool = True,
        hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the right
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the right hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        super(MANO, self).__init__(model_path=model_path,
                        is_rhand=is_rhand,
                        data_struct=data_struct,
                        create_hand_pose=create_hand_pose,
                        hand_pose=hand_pose,
                        use_pca=use_pca,
                        num_pca_comps=num_pca_comps,
                        flat_hand_mean=flat_hand_mean,
                        batch_size=batch_size,
                        dtype=dtype,
                        vertex_ids=vertex_ids,
                        use_compressed=use_compressed,
                        ext=ext,
                        **kwargs) 

class MANOLayer(smplx.MANOLayer):
    def __init__(self, *args, **kwargs) -> None:
        ''' MANO as a layer model constructor
        '''
        super(MANOLayer, self).__init__(*args, 
                                        **kwargs) 

class FLAME(smplx.FLAME):

    def __init__(
        self,
        model_path: str,
        data_struct=None,
        num_expression_coeffs=10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_neck_pose: bool = True,
        neck_pose: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour=False,
        batch_size: int = 1,
        gender: str = 'neutral',
        dtype: torch.dtype = torch.float32,
        ext='pkl',
        **kwargs
    ) -> None:
        ''' FLAME model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_neck_pose: bool, optional
                Flag for creating a member variable for the neck pose.
                (default = False)
            neck_pose: torch.tensor, optional, Bx3
                The default value for the neck pose variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''

        super(FLAME, self).__init__(model_path=model_path,
                                data_struct=data_struct,
                                num_expression_coeffs=num_expression_coeffs,
                                create_expression=create_expression,
                                expression=expression,
                                create_neck_pose=create_neck_pose,
                                neck_pose=neck_pose,
                                create_jaw_pose=create_jaw_pose,
                                jaw_pose=jaw_pose,
                                create_leye_pose=create_leye_pose,
                                leye_pose=leye_pose,
                                create_reye_pose=create_reye_pose,
                                reye_pose=reye_pose,
                                use_face_contour=use_face_contour,
                                batch_size=batch_size,
                                gender=gender,
                                dtype=dtype,
                                ext=ext,
                                **kwargs)


class FLAMELayer(smplx.FLAMELayer):
    def __init__(self, *args, **kwargs) -> None:
        ''' FLAME as a layer model constructor '''
        super(FLAMELayer, self).__init__(*args,
                                        **kwargs)


def build_layer(
    model_path: str,
    model_type: str,
    **kwargs
) -> Union[SMPLLayer, SMPLHLayer, SMPLXLayer, MANOLayer, FLAMELayer]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT
            |-- flame
                |-- FLAME_FEMALE
                |-- FLAME_MALE
                |-- FLAME_NEUTRAL

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPLLayer(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLHLayer(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLXLayer(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANOLayer(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAMELayer(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')


def create(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPL, SMPLH, SMPLX, MANO, FLAME]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

     # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANO(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAME(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')