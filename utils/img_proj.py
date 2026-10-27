# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : img_proj.py
# Time       :2023/4/6 9:11
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import numpy as np
from utils import file_io


def img_proj(arr, length=3, avg_max='mean', direction='z'):
    """
    Conventional projection.
    :param arr:
    :param length:
    :param avg_max:
    :param direction:
    :return:
    """
    if direction == 'x':
        arr = np.transpose(arr, (2, 1, 0))
    elif direction == 'y':
        arr = np.transpose(arr, (0, 2, 1))
    elif direction == 'z':
        pass
    else:
        raise ValueError('Unsupported projection method')

    xx, yy, zz = arr.shape
    if length > zz:
        raise ValueError('Unsupported projection direction')

    res = np.zeros((xx, yy, zz // length))
    i = 0
    while i <= zz - length:
        temp = arr[:, :, i:i + length]
        if avg_max == 'mean':
            res[:, :, i // length] = np.mean(temp, axis=2)
        elif avg_max == 'max':
            res[:, :, i // length] = np.max(temp, axis=2)
        else:
            print('ERROR')
        i += length

    if direction == 'x':
        res = np.transpose(res, (2, 1, 0))
    elif direction == 'y':
        res = np.transpose(res, (0, 2, 1))
    elif direction == 'z':
        pass
    else:
        raise ValueError('Unsupported projection method')
    return res


def img_proj_v2(arr, length=3, avg_max='mean', direction='z'):
    """
    Improve the signal-to-noise ratio by convolution.
    :param arr:
    :param length:
    :param avg_max:
    :param direction:
    :return:
    """
    if length % 2 == 0:
        raise ValueError('The length must be an odd number')

    if direction == 'x':
        arr = np.transpose(arr, (2, 1, 0))
    elif direction == 'y':
        arr = np.transpose(arr, (0, 2, 1))
    elif direction == 'z':
        pass
    else:
        raise ValueError('Unsupported projection direction')
    xx, yy, zz = arr.shape
    res = np.zeros((xx, yy, zz - length // 2))

    i = length // 2
    while i < zz - length // 2:
        temp = arr[:, :, i - length // 2:i + length // 2]
        if avg_max == 'mean':
            res[:, :, i - length // 2] = np.mean(temp, axis=2)
        elif avg_max == 'max':
            res[:, :, i - length // 2] = np.max(temp, axis=2)
        else:
            print('ERROR')
        i += 1

    if direction == 'x':
        res = np.transpose(res, (2, 1, 0))
    elif direction == 'y':
        res = np.transpose(res, (0, 2, 1))
    elif direction == 'z':
        pass
    else:
        raise ValueError('Unsupported projection method')
    return res


if __name__ == '__main__':
    import argparse, time
    from utils import file_io

    parser = argparse.ArgumentParser(description='Denoise')
    parser.add_argument('-img_path', '--img_path', default=r'../demo/data/stitched_0000.nii.gz', help='working directory')
    parser.add_argument('-save_suffix', '--save_suffix', default=r'_median.nii',
                        help='output suffix')
    parser.add_argument('-length', '--length', default=5,
                        help='length')
    parser.add_argument('-method', '--method', default=r'mean',
                        help='method')
    parser.add_argument('-direction', '--direction', default=r'z',
                        help='direction')

    arguments = parser.parse_args()
    img_path = arguments.img_path
    save_suffix = arguments.save_suffix
    method = arguments.method
    length = arguments.length
    direction = arguments.direction

    arr, meta_info = file_io.read_nii(img_path)
    arr = np.transpose(arr, (2, 1, 0))
    res = img_proj(arr, length=length, avg_max=method, direction=direction)
    arr = np.transpose(arr, (2, 1, 0))
    file_io.save_nii(res, img_path.replace('.nii', save_suffix), meta_info)
