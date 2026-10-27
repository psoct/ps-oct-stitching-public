# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : image_utils.py
# Time       :2023/3/29 14:02
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import SimpleITK as sitk
import numpy as np


def sample_2_fixed_spacing(img, spacing, resamplemethod=sitk.sitkLinear):
    # Resample with SimpleITK
    resampler = sitk.ResampleImageFilter()
    old_size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())
    new_size = (old_size * old_spacing) / spacing

    # Set output image metadata
    resampler.SetOutputSpacing(spacing)  # output image size
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetSize(new_size.astype("uint16").tolist())

    # Set the output pixel type according to the resampling use case
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # use nearest-neighbor interpolation for masks and save as uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # use linear interpolation for continuous-valued images and save as float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(img)  # execute resampling
    return itk_img_resampled


def sample_2_fixed_size(img, new_size, resamplemethod=sitk.sitkLinear):
    # Resample with SimpleITK
    resampler = sitk.ResampleImageFilter()
    old_size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())
    new_spacing = old_size * old_spacing / np.array(new_size)

    # Set output image metadata
    resampler.SetOutputSpacing(new_spacing.tolist())  # output image size
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetSize(new_size)

    # Set the output pixel type according to the resampling use case
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # use nearest-neighbor interpolation for masks and save as uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # use linear interpolation for continuous-valued images and save as float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(img)  # execute resampling
    return itk_img_resampled


def get_bbox(img_arr, ):
    """
    get the bounding box of a img_array.
    :param img_arr: image_array
    :return: start point;end point;
    """
    index_info = np.nonzero(img_arr)
    if len(img_arr.shape) < 3:
        st = [np.max([np.min(index_info[0]), 0]), np.max([np.min(index_info[1]), 0])]
        et = [np.min([np.max(index_info[0]), img_arr.shape[0]]), np.min([np.max(index_info[1]), img_arr.shape[1]])]
    elif len(img_arr.shape) == 3:
        st = [np.max([np.min(index_info[0]), 0]), np.max([np.min(index_info[1]), 0]),
              np.max([np.min(index_info[2]), 0])]
        et = [np.min([np.max(index_info[0]), img_arr.shape[0]]), np.min([np.max(index_info[1]), img_arr.shape[1]]),
              np.min([np.max(index_info[2]), img_arr.shape[2]])]
    else:
        raise ValueError('ERROR: we just support 2d or 3d image!')
    return st, et
