# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : img_ds.py
# Time       :2023/4/6 10:57
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import os.path

import numpy as np
import tqdm
from utils import image_utils
import torch
import glob, sys

sys.path.append('../')
from utils import file_io
import SimpleITK as sitk


class Resize_Img(object):
    def __init__(self, new_spacing, inter_med):
        self.new_spacing = new_spacing
        self.inter_med = inter_med

    def run(self, path, save_dir=None):
        img = sitk.ReadImage(path)
        if self.inter_med == 'Linear':
            inter_med = sitk.sitkLinear
        elif self.inter_med == 'NearestNeighbor':
            inter_med = sitk.sitkNearestNeighbor
        else:
            raise ValueError('Unsupported resampling method')
        ds_image = image_utils.sample_2_fixed_spacing(img, self.new_spacing, resamplemethod=inter_med)
        if save_dir is not None:
            sitk.WriteImage(ds_image, save_dir)
        # ds_array = sitk.GetArrayFromImage(ds_image)
        # return ds_array


class DS_mean(object):
    def __init__(self, new_spacing, med='mean', inter_med='Linear'):
        self.factor = None
        self.pooling = None
        # factor = [factor[2], factor[1], factor[0]]
        self.new_spacing = new_spacing
        if inter_med == 'Linear':
            self.inter_med = sitk.sitkLinear
        elif inter_med == 'NearestNeighbor':
            self.inter_med = sitk.sitkNearestNeighbor
        else:
            raise ValueError('Unsupported resampling method')
        self.med = med

    def run(self, path, save_dir=None):
        arr, meta_info = file_io.read_nii(path)
        spacing, orgin, dirction = meta_info
        self.factor = np.array([int(self.new_spacing[2] / spacing[2]), int(self.new_spacing[1] / spacing[1]),
                                int(self.new_spacing[0] / spacing[0])])

        zz, yy, xx = arr.shape
        if self.med == 'mean':
            self.pooling = torch.nn.AvgPool3d(tuple(self.factor), stride=tuple(self.factor))
        elif self.med == 'max':
            self.pooling = torch.nn.MaxPool3d(tuple(self.factor), stride=tuple(self.factor))

        img = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :, :])
        new_img = self.pooling(img)
        new_arr = new_img.squeeze().numpy()
        zzz, yyy, xxx = new_arr.shape

        spacing = (spacing[0] * xx / xxx, spacing[1] * yy / yyy, spacing[2] * zz / zzz)
        img = sitk.GetImageFromArray(new_arr)
        img.SetSpacing(spacing)
        img.SetOrigin(orgin)
        img.SetDirection(dirction)
        ds_image = image_utils.sample_2_fixed_spacing(img, self.new_spacing, resamplemethod=self.inter_med)
        if save_dir is not None:
            sitk.WriteImage(ds_image, save_dir)
            # file_io.save_nii(new_arr, save_dir, [spacing, orgin, dirction])
        return sitk.GetArrayFromImage(ds_image), [self.new_spacing, orgin, dirction]   #NumPy array axes are ordered z, y, x,


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Resize')
    parser.add_argument('-img_path', '--img_path',
                        required=True,
                        help='input file')
    parser.add_argument('-new_spacing', '--new_spacing',
                        default=(0.02, 0.02, 0.02),
                        help='resampling spacing')
    parser.add_argument('-inter_med', '--inter_med',
                        default='Linear',
                        help='resampling method')
    parser.add_argument('-proj_med', '--proj_med',
                        default='Linear',
                        help='resampling method')
    parser.add_argument('-save_path', '--save_path',
                        required=True,
                        help='output filename suffix')

    arguments = parser.parse_args()
    img_path = arguments.img_path
    save_path = arguments.save_path
    inter_med = arguments.inter_med
    proj_med = arguments.proj_med
    new_spacing = arguments.new_spacing

    # model = Resize_Img(new_spacing, inter_med)
    # model.run()

    model = DS_mean(new_spacing, proj_med, "Linear")
    model.run(img_path, save_dir=save_path)
