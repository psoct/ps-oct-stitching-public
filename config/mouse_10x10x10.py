# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : 0601_mouse.py
# Time       :2023/8/8 15:34
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
from pathlib import Path

from easydict import EasyDict as edict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
_OUTPUT_DIR = _PROJECT_ROOT / "outputs"

__C = edict()
cfg = __C

__C.general = {}
# Example project-relative paths; adjust them for your dataset when needed.
__C.general.data_dir = str(_DATA_DIR)
__C.general.save_root = str(_OUTPUT_DIR)
# Section identifiers to process, in acquisition order (for example, ['0001', '0002']).
__C.general.test_slice = []
__C.general.norm_upper = 100
__C.general.norm_lower = 0
__C.general.alloc_memory = '256g'
__C.downsample = {}
__C.downsample.new_spacing = (0.01, 0.01, 0.01)
__C.downsample.inter_med = 'Linear'


__C.downsample.save_suffix = 'x10.nii'

#######################
# Define crop_slice
########################
# st_crop = 50
__C.crop = {}
__C.crop.crop_length = 330  # micrometers
__C.crop.contain_length = 500  # micrometers

###########################
# Define shading correction
###########################
__C.shading_corr = {}
__C.shading_corr.intensity_flat_raw = str(_DATA_DIR / "intensity_flat_raw.nii")
__C.shading_corr.intensity_flat_save_dir = str(_OUTPUT_DIR / "intensity_flat.nii")
__C.shading_corr.retardance_flat_raw = str(_DATA_DIR / "retardance_flat_raw.nii")
__C.shading_corr.retardance_flat_save_dir = str(_OUTPUT_DIR / "retardance_flat.nii")

#######################
# Stitching Setting
#######################
__C.stitching = {}
__C.stitching.overlap_x = 20  # %
__C.stitching.overlap_y = 20  # %
__C.stitching.overlap_z = 0  # %
__C.stitching.tiles_x = 5
__C.stitching.tiles_y = 4
__C.stitching.tiles_z = 1  # for xy stitching
__C.stitching.grid_type = '[Up & Right]'
__C.stitching.motion_est = 'Phase Correlation'
__C.stitching.motion_est_ds_x = 2
__C.stitching.motion_est_ds_y = 2
__C.stitching.motion_est_ds_z = 1
__C.stitching.fixed_tile = 12

#######################
# Fusion Setting
#######################
__C.fusion = {}
__C.fusion.padding = 50
__C.fusion.blending_met = "alpha"

#################
# Segment
#################
__C.segment = {}

# SimpleITK coordinates minx, miny, maxx maxy
# 84:458, 48:565
# for 0.02 [[30, 60, 600, 500]]
# for 0.01 [[110, 110, 1190, 900]]
__C.segment.input_box = [[342, 85, 982, 1017]]  # slice30-85410   590


#################
# bias_corr
#################
__C.bias_corr = {}
__C.bias_corr.save_suffix = '_n4.nii'
__C.bias_corr.save_corr_suffix = '_corr_auto.nii'
__C.bias_corr.overlap_z = 60  # (Z_CropLength-sectionThickness)/Z_CropLength
__C.bias_corr.iter = 1
__C.bias_corr.down_sample = 1

#################
# z_stit
#################
__C.z_stit = {}
__C.z_stit.overlap_z = 60
# Use None to select the middle section automatically, or set an explicit zero-based index.
__C.z_stit.fixed_id = None
