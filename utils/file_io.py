# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : file_io.py
# Time       :2023/3/29 14:00
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import SimpleITK as sitk
from skimage import io
import os, shutil
import numpy as np
import moviepy


def read_nii(data_dir):
    img = sitk.ReadImage(data_dir)
    img_arr = sitk.GetArrayFromImage(img)
    [spacing, orgin, dirction] = img.GetSpacing(), img.GetOrigin(), img.GetDirection()
    return img_arr, [spacing, orgin, dirction]


def save_nii(arr, save_dir, image_info):
    arr = sitk.GetImageFromArray(arr)
    if image_info[0] is not None:
        arr.SetSpacing(image_info[0])
    if image_info[1] is not None:
        arr.SetOrigin(image_info[1])
    if image_info[2] is not None:
        arr.SetDirection(image_info[2])
    sitk.WriteImage(arr, save_dir)
    print('==>The image has been saved at {}'.format(save_dir))


def saveTif2Nifty(data_dir, save_dir, spacing=(0.02, 0.02, 0.02), orgin=None, img_type=None):
    img = io.imread(data_dir)
    print(img.shape)  # (9, 20)
    print(type(img))  # <class 'numpy.ndarray'>
    print(type(img[0][0]))  # <class 'numpy.uint16'>
    print(img.dtype)  # uint16
    print(np.max(img), np.min(img))
    img = img.transpose(0, 2, 1)
    if img_type is not None:
        img = img.astype(img_type)
    nii_img = sitk.GetImageFromArray(img)
    nii_img.SetSpacing(spacing)
    if orgin:
        nii_img.SetOrigin(orgin)
    sitk.WriteImage(nii_img, save_dir)
    print('{} Done!!'.format(save_dir))
    # return nii_img


def load_module_from_disk(file):
    import importlib.util

    spec = importlib.util.spec_from_file_location('ps_oct_config', file)
    if spec is None or spec.loader is None:
        raise ImportError('Unable to load configuration file: {}'.format(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_temp_dir(dirs, idx, save_datas, x_num, y_num):
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    else:
        shutil.rmtree(dirs)
        os.makedirs(dirs)

    i = 0
    for x in range(x_num):
        for y in range(y_num):
            if x < 10:
                str_x = "0" + str(x)
            else:
                str_x = str(x)
            if y < 10:
                str_y = "0" + str(y)
            else:
                str_y = str(y)
            shutil.copyfile(save_datas[i][0],
                            os.path.join(dirs, 'Slice_' + idx + '_X_' + str_x + '_Y_' + str_y + '_.nii'))
            i += 1
    return True


def parse_path(files):
    slices_dict = {}
    for p in files:
        dirs_n, file_n = os.path.split(p)
        temp = file_n.split('_')
        slices = temp[1]
        tiles = temp[2]
        # coordinate = [round(float(temp[4]), 2), round(float(temp[6]), 2), round(float(temp[8]), 2)]
        coordinate = [float(np.floor(float(temp[i]) * 100) / 100) for i in [4, 6, 8]]    ###Correction: floor coordinates to two decimal places. Values ending in .99 and integer values are edge cases, so avoid relying solely on coordinates for file naming.
        if slices not in slices_dict:
            slices_dict[slices] = [(p, coordinate)]
        else:
            slices_dict[slices].append((p, coordinate))

    # print(slices_dict)
    for i in slices_dict:
        temp2 = slices_dict[i]
        temp2 = sorted(temp2, key=lambda x: (x[1][0], x[1][1]), reverse=False)
        slices_dict[i] = temp2
        print(i)

    macro_input = [] # *.nii
    for i in slices_dict:
        temp3 = slices_dict[i]
        temp3 = sorted(temp3, key=lambda x: (x[1][0], x[1][1]), reverse=False)
        slices_dict[i] = temp3
        print(i)
    return slices_dict



# def convert_avi_2_mp4(src_fpath, dst_fpath):
#     clip = moviepy.VideoFileClip(src_fpath)
#     clip.write_videofile(dst_fpath)


