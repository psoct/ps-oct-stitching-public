#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# File       : pipeline_mouse.py
# Time       :2023/6/25 11:25
# Author     :Bot Zhao
# version    :python 3.9
# Description: End-to-end PS-OCT stitching and post-processing pipeline.
"""

import glob
import os
import sys
from pathlib import Path

import tqdm
import SimpleITK as sitk
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (bias_field_correction, file_io, image_utils, img_ds,
                   segment, shading_corr_psoct, xy_stitching)


def downSample_stit(arg, stit_file_lr, intensity_files, save_root, slice_id, dowmsampled_save_suffix,
                    ds_model, shading_intensity_lr, ij, alloc_memory):
    if not os.path.isfile(stit_file_lr):
        for i in tqdm.tqdm(range(len(intensity_files))):
            _, name = os.path.split(intensity_files[i])
            if os.path.isfile(os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix))):
                continue

            #########################
            # resize
            #########################
            arr, img_info = ds_model.run(intensity_files[i])
            print(arr.shape)

            #########################
            # shading correction
            #########################
            arr = shading_intensity_lr.flat_corr(arr)

            #########################
            # Norm
            #########################
            arr[arr < arg.general.norm_lower] = 0
            arr[arr > arg.general.norm_upper] = 100
            norm = ((arr / (arg.general.norm_upper - arg.general.norm_lower)) * 255).astype("uint8")

            #########################
            # SAVE
            #########################
            file_io.save_nii(norm, os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix)),
                             image_info=img_info)

        #############################
        # Stitching by BigStitcher  #
        #############################
        intensity_imgs_lr = glob.glob(
            os.path.join(save_root, 'Slice_{}_*.oct_intensity{}'.format(slice_id, dowmsampled_save_suffix)))

        # # 10x10x10
        root_dir = save_root
        dataset_args = ['', arg.stitching.overlap_x, arg.stitching.overlap_y, arg.stitching.overlap_z,
                        arg.stitching.tiles_x,
                        arg.stitching.tiles_y, arg.stitching.tiles_z, arg.stitching.grid_type, slice_id]  # [Up & Left]
        # 10x10x10
        pairwise_args = arg.stitching.motion_est, arg.stitching.motion_est_ds_x, \
                        arg.stitching.motion_est_ds_y, arg.stitching.motion_est_ds_z
        # # raw
        # pairwise_args = 'Phase Correlation', 4, 4, 2
        opt_args = 'Two-Round using Metadata to align unconnected Tiles', arg.stitching.fixed_tile  # fixed tile used during stitching; adjust as needed
        fusion_args = 'Linear Interpolation', '32-bit floating point', 'stitched_{}.nii'.format(slice_id)
        #
        big_imag_macro1 = xy_stitching.XY_Stitching(ij, root_dir, slice_id, dataset_args, pairwise_args, opt_args,
                                                    fusion_args, alloc_memory)

        # if not os.path.isfile(stit_file_lr):
        xml_file, stit_file_lr = big_imag_macro1.run(files=intensity_imgs_lr)


def downSample_apply_stit(arg, stit_file_lr, hr_image_files, save_root, slice_id,
                          dowmsampled_save_suffix, norm_lower, norm_upper,
                          ds_model, shading_intensity_lr, ij, alloc_memory):
    if not os.path.isfile(stit_file_lr):
        images = []
        for i in tqdm.tqdm(range(len(hr_image_files))):
            _, name = os.path.split(hr_image_files[i])
            if os.path.isfile(os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix))):
                images.append(os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix)))
                continue

            #########################
            # resize
            #########################
            arr, img_info = ds_model.run(hr_image_files[i])
            print(arr.shape)

            #########################
            # shading correction
            #########################
            if shading_intensity_lr is not None:
                arr = shading_intensity_lr.flat_corr(arr)

            #########################
            # Norm
            #########################
            arr[arr < norm_lower] = norm_lower
            arr[arr > norm_upper] = norm_upper
            norm = (((arr - norm_lower) / (norm_upper - norm_lower)) * 255).astype("uint8")

            #########################
            # SAVE
            #########################
            file_io.save_nii(norm, os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix)),
                             image_info=img_info)
            images.append(os.path.join(save_root, name.replace('.nii', dowmsampled_save_suffix)))

        #############################
        # Stitching by BigStitcher  #
        #############################
        # # 10x10x10
        root_dir = save_root
        xml_file = "{}/{}_dataset_{}.xml".format(save_root, slice_id, slice_id)
        dataset_args = ['', arg.stitching.overlap_x, arg.stitching.overlap_y, arg.stitching.overlap_z, arg.stitching.tiles_x,
                        arg.stitching.tiles_y, arg.stitching.tiles_z, arg.stitching.grid_type, slice_id]  # [Up & Left]
        # 10x10x10
        pairwise_args = arg.stitching.motion_est, arg.stitching.motion_est_ds_x, \
                        arg.stitching.motion_est_ds_y, arg.stitching.motion_est_ds_z
        # # raw
        # pairwise_args = 'Phase Correlation', 4, 4, 2
        opt_args = 'Two-Round using Metadata to align unconnected Tiles', arg.stitching.fixed_tile  # fixed tile used during stitching; adjust as needed
        fusion_args = 'Linear Interpolation', '32-bit floating point', 'stitched_{}.nii'.format(slice_id)
        #
        big_imag_macro1 = xy_stitching.XY_Stitching(ij, root_dir, slice_id, dataset_args, pairwise_args, opt_args,
                                                    fusion_args, alloc_memory)

        if not os.path.isfile(stit_file_lr):
            stit_retardance_file = big_imag_macro1.apply_transform(xml_file, save_file=stit_file_lr, files=images)


def apply_mask2img(mask_path, img_path, save_dir_lr, st_crop):
    img_arr, meta = file_io.read_nii(img_path)
    lr_mask_img = sitk.ReadImage(mask_path)
    # lr
    _, name = os.path.split(img_path)
    lr_save_path = os.path.join(save_dir_lr, name)
    lr_mask_arr = sitk.GetArrayFromImage(lr_mask_img)
    img_arr = img_arr[st_crop:st_crop + lr_mask_arr.shape[0], ...]
    file_io.save_nii(img_arr, lr_save_path.replace("attn_corr", "cropped"), meta)
    img_arr[lr_mask_arr == 0] = 0
    file_io.save_nii(img_arr, lr_save_path, meta)
    return lr_save_path


def pipeline_mouse(arg):
    #####################
    # Define Data
    #####################
    test_slice = arg.general.test_slice
    if not test_slice:
        raise ValueError(
            "general.test_slice is empty; add the section identifiers to the configuration file."
        )
    data_dir = arg.general.data_dir
    save_root = arg.general.save_root
    save_dir_lr_mask = "{}/001mm/pred/".format(save_root)
    save_dir_lr = "{}/001mm/attn_corr".format(save_root)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    print(test_slice)
    alloc_memory = arg.general.alloc_memory
    ij = xy_stitching.init_imagej(alloc_memory)

    #####################
    # Define Down-samping
    #####################
    new_spacing = arg.downsample.new_spacing
    inter_med = arg.downsample.inter_med
    dowmsampled_save_suffix = arg.downsample.save_suffix
    ds_model = img_ds.DS_mean(new_spacing, inter_med=inter_med)

    #######################
    # Define crop_slice
    ########################
    # st_crop = 50
    crop_length = arg.crop.crop_length  # micrometers
    contain_length = arg.crop.contain_length  # micrometers
    st_crop = int(crop_length // (new_spacing[-1] * 1000))
    ed_crop = st_crop + int(contain_length // (new_spacing[-1] * 1000))
    print('st_crop:{}, ed_crop:{}'.format(st_crop, ed_crop))

    ###########################
    # Define shading correction
    ###########################
    intensity_flat_raw = arg.shading_corr.intensity_flat_raw
    intensity_flat_file = arg.shading_corr.intensity_flat_save_dir

    retardance_flat_raw = arg.shading_corr.retardance_flat_raw
    retardance_flat_file = arg.shading_corr.retardance_flat_save_dir

    if not os.path.isfile(intensity_flat_file):
        shading_corr_psoct.cal_flat_file(intensity_flat_raw, intensity_flat_file, contain_info=6, crop_z=0)

    if not os.path.isfile(retardance_flat_file):
        shading_corr_psoct.cal_flat_file(retardance_flat_raw, retardance_flat_file, contain_info=6, crop_z=0)

    img = sitk.ReadImage(intensity_flat_file)
    itk_lr = image_utils.sample_2_fixed_spacing(img, new_spacing)
    sitk.WriteImage(itk_lr, intensity_flat_file.replace("intensity_flat.nii", "lr_intensity_flat.nii"))
    shading_intensity_lr = shading_corr_psoct.Flat_correction(intensity_flat_file.replace("intensity_flat.nii",
                                                                                          "lr_intensity_flat.nii"))
    img = sitk.ReadImage(retardance_flat_file)
    itk_lr = image_utils.sample_2_fixed_spacing(img, new_spacing)
    sitk.WriteImage(itk_lr, retardance_flat_file.replace("retardance_flat.nii", "lr_retardance_flat.nii"))
    shading_retardance_lr = shading_corr_psoct.Flat_correction(
    retardance_flat_file.replace("retardance_flat.nii", "lr_retardance_flat.nii"))

    # shading_axisorient = shading_corr_psoct.Flat_correction(flat_file=axisorient_flat, crop_z=crop_slice)

    #################
    # Segment
    #################
    # input_box = [[400, 400, 860, 760]]
    input_box = arg.segment.input_box
    print("input_box", input_box)
    #################################
    # DOWNSAMPLE & SHADING CORRECTION & XY STITCHING
    ################################
    for iid, slice_id in enumerate(test_slice):
        print(slice_id)
        ##############
        # Collect Datac
        ##############
        intensity_files = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_intensity.nii'.format(slice_id)))
        stit_file_lr = os.path.join(save_root, 'stitched_{}.nii'.format(slice_id))
        downSample_stit(arg, stit_file_lr, intensity_files, save_root, slice_id, dowmsampled_save_suffix,
                        ds_model, shading_intensity_lr, ij, alloc_memory)

        stit_file_lr = os.path.join(save_root, 'stitched_oct_retardance_{}.nii'.format(slice_id))
        retardance_files = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_retardance.nii'.format(slice_id)))
        downSample_apply_stit(arg, stit_file_lr, retardance_files, save_root, slice_id,
                              dowmsampled_save_suffix, 0, 2,
                              ds_model, shading_retardance_lr, ij, alloc_memory)

        stit_file_lr = os.path.join(save_root, 'stitched_oct_axisorient_{}.nii'.format(slice_id))
        axisorient_files = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_axisorient.nii'.format(slice_id)))
        downSample_apply_stit(arg, stit_file_lr, axisorient_files, save_root, slice_id,
                              dowmsampled_save_suffix, -np.pi, np.pi,
                              ds_model, shading_intensity_lr=None,
                              ij=ij, alloc_memory=alloc_memory)

    # ##############################
    # # Segment & Crop & ATTN      #
    # ##############################

    iid = len(test_slice) // 2

    while iid >= 0:
        slice_id = test_slice[iid]
        if not os.path.isdir(save_dir_lr_mask):
            os.makedirs(save_dir_lr_mask)
        if not os.path.isdir(save_dir_lr.replace("attn_corr", "cropped")):
            os.makedirs(save_dir_lr.replace("attn_corr", "cropped"))

        stit_file_lr = os.path.join(save_root, 'stitched_{}.nii'.format(slice_id))
        files = [stit_file_lr]

        _, name = os.path.split(stit_file_lr)
        if os.path.isfile(os.path.join(save_dir_lr, name)):
            iid -= 1
            continue

        SAM_model = segment.Sam_seg(save_dir=save_dir_lr_mask, st_crop=st_crop, ed_crop=ed_crop, gpus="0",
                                    model_type="vit_h", device='cuda', evaluate=False)
        mask_path, input_box = SAM_model.run_onecase(files, input_box=input_box, max_bbox=True)
        if not os.path.isdir(save_dir_lr):
            os.makedirs(save_dir_lr)
        _ = apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        iid -= 1

    iid = (len(test_slice) // 2)
    input_box = arg.segment.input_box
    print("input_box", input_box)
    while iid < len(test_slice):
        slice_id = test_slice[iid]
        if not os.path.isdir(save_dir_lr_mask):
            os.makedirs(save_dir_lr_mask)
        stit_file_lr = os.path.join(save_root, 'stitched_{}.nii'.format(slice_id))
        files = [stit_file_lr]
        _, name = os.path.split(stit_file_lr)
        if os.path.isfile(os.path.join(save_dir_lr, name)):
            iid += 1
            continue
        SAM_model = segment.Sam_seg(save_dir=save_dir_lr_mask, st_crop=st_crop, ed_crop=ed_crop, gpus="0",
                                    model_type="vit_h", device='cuda', evaluate=False)
        mask_path, input_box = SAM_model.run_onecase(files, input_box=input_box, max_bbox=True)

        if not os.path.isdir(save_dir_lr):
            os.makedirs(save_dir_lr)

        _ = apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        iid += 1

    #################################################
    # apply mask to oct_retardance & oct_axisorient
    #################################################
    retardance_files = []
    axisorient_files = []
    for iid, slice_id in enumerate(test_slice):
        print(slice_id)
        mask_path = save_dir_lr_mask + '/stitched_{}.nii'.format(slice_id)
        stit_file_lr = os.path.join(save_root, 'stitched_oct_retardance_{}.nii'.format(slice_id))
        save_file = os.path.join(save_dir_lr, 'stitched_oct_retardance_{}.nii'.format(slice_id))
        if not os.path.isfile(save_file):
            _ = apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        retardance_files.append(os.path.join(save_dir_lr, 'stitched_oct_retardance_{}.nii'.format(slice_id)))

        stit_file_lr = os.path.join(save_root, 'stitched_oct_axisorient_{}.nii'.format(slice_id))
        save_file = os.path.join(save_dir_lr, 'stitched_oct_axisorient_{}.nii'.format(slice_id))
        if not os.path.isfile(save_file):
            _ = apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        axisorient_files.append(os.path.join(save_dir_lr, 'stitched_oct_axisorient_{}.nii'.format(slice_id)))

    #################################################
    # Bias-field correction (intensity only)
    #################################################
    save_suffix = arg.bias_corr.save_suffix
    save_bias = False
    intensity_files = []
    for iid, slice_id in tqdm.tqdm(enumerate(test_slice)):
        print("bias_field_correction:===================>")
        lr_save_path = os.path.join(save_dir_lr, 'stitched_{}.nii'.format(slice_id))
        if not os.path.isfile(lr_save_path.replace('.nii', save_suffix)):
            FBC = bias_field_correction.Correct_Bias(lr_save_path, lr_save_path.replace('.nii', save_suffix),
                                               is_save_bias=save_bias)
            FBC.run()
        intensity_files.append(lr_save_path.replace('.nii', save_suffix))
    correct_p = bias_field_correction.correct_n4(
        intensity_files,
        overlap=arg.bias_corr.overlap_z / 100,
        iter=arg.bias_corr.iter,
    )
    intensity_files = bias_field_correction.apply_corr(
        intensity_files, correct_p, suffix=arg.bias_corr.save_corr_suffix
    )

    ###################
    # Stitcher Z
    ###################
    root_dir = save_dir_lr
    files = intensity_files
    dataset_args = ['', 0, 0, arg.z_stit.overlap_z, 1, 1, len(files), '[Up & Right]', "whole_brain"]
    pairwise_args = 'Phase Correlation', 2, 2, 1
    fixed_id = arg.z_stit.fixed_id
    if fixed_id is None:
        fixed_id = len(files) // 2
    opt_args = 'Two-Round using Metadata to align unconnected Tiles', fixed_id
    fusion_args = 'Linear Interpolation', '16-bit floating point', 'stitched_{}.nii.gz'.format("whole_brain_n4_corr")

    big_imag_macro1 = xy_stitching.Z_Stitching(ij, root_dir, "whole_brain", dataset_args, pairwise_args, opt_args,
                                               fusion_args)
    xml_file, stit_file = big_imag_macro1.run(files=files)
    _ = big_imag_macro1.apply_z_transform(xml_file,
                                          save_file=os.path.join(root_dir, 'stitched_whole_retardance_n4_corr.nii.gz'),
                                          files=retardance_files)
    _ = big_imag_macro1.apply_z_transform(xml_file,
                                          save_file=os.path.join(root_dir, 'stitched_whole_axisorient_n4_corr.nii.gz'),
                                          files=axisorient_files)


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description='Run the PS-OCT stitching and post-processing pipeline')
    parser.add_argument('-i', '--config', required=True,
                        help='path to the pipeline configuration file')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    pipeline_mouse(cfg)
