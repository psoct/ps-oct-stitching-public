# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : n4_Bias_corr.py
# Time       :2022/11/8 9:34
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tqdm
from scipy.optimize import leastsq

from utils import file_io
from utils import image_utils


def ALine_Corr(in_file, save_dir):
    img_arr, meta_info = file_io.read_nii(in_file)
    axis_res = meta_info[0][-1]
    print(axis_res)
    print(np.log(2))
    # bias = np.zeros_like(img_arr)
    img_arr = img_arr[35:, 50:-50, 50:-50].copy()
    # img_arr = img_arr - np.mean(img_arr)
    img_arr = img_arr / 255
    zz, yy, xx = img_arr.shape

    sum_arr = np.zeros_like(img_arr)
    temp = img_arr[-1, :, :]
    for i in range(img_arr.shape[1]):
        plt.plot(img_arr[:, i, xx // 2])
    # plt.imshow(temp)
    # plt.colorbar()
    plt.show()

    plt.plot(np.mean(np.mean(img_arr, axis=1), axis=1))
    print(np.mean(np.mean(img_arr, axis=1), axis=1).shape)
    plt.show()

    sum_arr[-1, ...] = temp
    plt.imshow(temp)
    plt.colorbar()
    plt.show()

    for i in range(2, zz + 1):
        temp = temp + img_arr[-i, ...]
        sum_arr[-i, ...] = temp
    print(np.max(sum_arr))
    print(np.min(sum_arr))
    for i in range(img_arr.shape[1]):
        plt.plot(sum_arr[:, i, xx // 2])
    # plt.imshow(temp)
    # plt.colorbar()
    plt.show()

    mu_matrix = np.zeros_like(img_arr)
    temp = np.log(1 + (img_arr[0, ...] / sum_arr[0 + 1, ...]))
    mu_matrix[0, ...] = temp
    for i in range(1, zz - 1):
        temp = (np.log(1 + (img_arr[i, ...] / sum_arr[i + 1, ...])))
        mu_matrix[i, ...] = temp

    # mu_matrix = np.zeros_like(img_arr)
    # temp = img_arr[0, ...] / sum_arr[0 + 1, ...]
    # mu_matrix[0, ...] = temp
    # for i in range(1, zz - 1):
    #     temp = img_arr[i, ...] / sum_arr[i + 1, ...]
    #     mu_matrix[i, ...] = temp

    for i in range(img_arr.shape[1]):
        plt.plot(mu_matrix[:, i, xx // 2])
    # plt.imshow(temp)
    # plt.colorbar()
    plt.show()

    # cul
    sum_exp = np.zeros_like(img_arr)
    temp = mu_matrix[0, ...]
    for i in range(1, zz - 1):
        # temp += -np.log(1 + img_arr[i, ...] / sum_arr[i + 1, ...]) / (axis_res * i)
        temp += mu_matrix[i]
        sum_exp[i, ...] = temp

    plt.imshow(temp)
    plt.colorbar()
    plt.show()
    print(np.max(sum_exp))
    print(np.min(sum_exp))
    bias = np.exp(-sum_exp)
    # for i in range(zz):
    #     if i == 0:
    #         temp_bias = np.exp()
    #     else:
    #         for j in range(i):
    #             temp += -np.log(1 + img_arr[j, ...] / sum_arr[j + 1, ...]) / axis_res
    #         temp_bias = np.exp(temp)
    #     bias[i, ...] = temp_bias
    print(np.max(bias))
    print(np.min(bias))
    plt.imshow(bias[50, ...])
    plt.colorbar()
    plt.show()
    plt.plot(np.mean(np.mean(bias, axis=1), axis=1))
    print(np.mean(np.mean(bias, axis=1), axis=1).shape)
    plt.show()
    img_arr_corr = img_arr / (bias * mu_matrix)
    file_io.save_nii(img_arr_corr, save_dir, meta_info)


def ALine_WangHui_Corr(in_file, save_dir):
    img_arr, meta_info = file_io.read_nii(in_file)
    axis_res = meta_info[0][-1]
    bias = np.zeros_like(img_arr)

    print(bias.shape)
    # img_arr = img_arr[5:-50, 50:-50, 50:-50]
    # img_arr = img_arr - np.mean(img_arr)

    avg_pool = torch.nn.AvgPool3d((3, 3, 3), stride=2, padding=1)
    ds_arr = avg_pool(torch.from_numpy(img_arr[np.newaxis, np.newaxis, ...])).numpy().squeeze()
    print(ds_arr.shape, img_arr.shape)
    zz, yy, xx = img_arr.shape
    for i in range(1, zz - 1):
        if i <= zz // 2:
            l_z = np.exp((abs(2 * (i + 1) - zz) / zz) * np.log(ds_arr[i, ...]))
        else:
            pass
        # sum_exp[i, ...] = temp

    # for i in range(zz):
    #     if i == 0:
    #         temp_bias = np.exp()
    #     else:
    #         for j in range(i):
    #             temp += -np.log(1 + img_arr[j, ...] / sum_arr[j + 1, ...]) / axis_res
    #         temp_bias = np.exp(temp)
    #     bias[i, ...] = temp_bias
    print(np.max(bias))
    print(np.min(bias))
    plt.imshow(bias[50, ...])
    plt.colorbar()
    plt.show()
    img_arr_corr = img_arr / bias
    file_io.save_nii(img_arr_corr, save_dir, meta_info)


class Correct_Bias(object):
    """
    Corrects the bias using SimpleITK N4BiasFieldCorrection.
    :param in_file: .nii.gz input path
    :param out_file: .nii.gz path for the corrected output file
    :return: full path of the corrected NIfTI file
    """

    # Correct the MRI bias field with SimpleITK N4BiasFieldCorrection
    def __init__(self, in_file, out_file, mask_file=None, is_save_bias=False):
        self.in_file = in_file
        self.mask_file = mask_file
        self.out_file = out_file
        self.is_save_bias = is_save_bias

    def n4_corr(self):
        if self.mask_file is not None:
            input_image = sitk.ReadImage(self.in_file, sitk.sitkFloat32)
            mask_image = sitk.ReadImage(self.mask_file, sitk.sitkUInt8)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            IMG = corrector.Execute(input_image, mask_image)
            bias = corrector.GetLogBiasFieldAsImage(input_image)
            sitk.WriteImage(IMG, self.out_file)
            if self.is_save_bias:
                sitk.WriteImage(bias, self.out_file.replace('.nii', "_bias.nii"))
        else:
            input_image = sitk.ReadImage(self.in_file, sitk.sitkFloat32)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            IMG = corrector.Execute(input_image)
            bias = corrector.GetLogBiasFieldAsImage(input_image)
            sitk.WriteImage(IMG, self.out_file)
            if self.is_save_bias:
                sitk.WriteImage(bias, self.out_file.replace('.nii', "_bias.nii"))
        # except Exception as e:
        #     print('Error {}'.format(e))

    def run(self):
        self.n4_corr()

    def apply(self, img_dir, save_dir, bias_dir):
        # An input image can be corrected by: input/exp(bias_field).
        bias = sitk.ReadImage(bias_dir)
        img_arr, meta_info = file_io.read_nii(img_dir)
        img_arr = np.transpose(img_arr, (0, 2, 1))
        hr_bias = image_utils.sample_2_fixed_size(bias, [img_arr.shape[2], img_arr.shape[1], img_arr.shape[0]])
        hr_bias_arr = sitk.GetArrayFromImage(hr_bias)
        out = img_arr / np.exp(hr_bias_arr)

        file_io.save_nii(out, save_dir, meta_info)


def show_res(dataset):
    for iid, f in enumerate(dataset):
        img = f
        # print(f, img.shape[0])
        lists = []
        xx = []
        for i in range(len(img)):
            # print(np.mean(img[i, ...]))
            temp = img[i]
            lists.append(temp)
            xx.append(25 * iid + i)
        plt.plot(xx, lists)
    plt.show()


def error(p, x, y):  # fit residual
    return np.sum((Fun(p, x) - y) ** 2)


def Fun(p, x):  # define the fitting function
    # a1,a2,a3 = p
    return x + p


def correct_n4(files, overlap=0.5, iter=10):
    """

    :param files: N4 bias-corrected images, typically intensity images
    :param overlap: overlap ratio along z
    :param iter: number of iterations
    :return: None
    """
    dataset = []
    print(files)
    files = sorted(files)
    for iid, f in enumerate(files):
        xx = []
        img, meta_info = file_io.read_nii(f)
        print(f, img.shape[0])
        for i in range(img.shape[0]):
            temp = img[i, ...]
            xx.append(np.mean(temp[temp != 0]))
        dataset.append(xx)
    old_dataset = dataset.copy()
    upbound = []
    for iii in range(iter):
        est_para = []
        for iid, f in enumerate(dataset[:-1]):
            if iid == 0:
                continue
            else:
                img_pre = dataset[iid - 1]
                img_next = dataset[iid + 1]
                img = f
            xx = []
            yy = []
            overlap_slice = int(len(img) * overlap + 0.5)
            for i in range(overlap_slice):
                xx.append(img[i])
                yy.append(img_pre[i + len(img) - overlap_slice])
            for i in range(len(img) - overlap_slice, len(img)):
                xx.append(img[i])
                yy.append(img_next[i - len(img) + overlap_slice])
            p_value = np.array([0])
            para = leastsq(error, p_value, args=(xx, yy))
            # print(para)
            est_para.append(para[0][0])

        print(max(est_para))
        upbound.append(max(est_para))
        temp = dataset[est_para.index(max(est_para)) + 1]
        dataset[est_para.index(max(est_para)) + 1] = [i + max(est_para) for i in temp]
    correct_p = []
    for corr, raw in zip(dataset, old_dataset):
        correct_p.append([i - j for i, j in zip(corr, raw)])
    # show_res(correct_p)
    return correct_p


def apply_corr(files, correct_p, suffix="_corr.nii"):
    new_files = []
    for iid, f in enumerate(files):
        if not os.path.isfile(f.replace('.nii', suffix)):
            img, meta_info = file_io.read_nii(f)
            print(f, img.shape[0])
            corr_p = correct_p[iid]
            img[img != 0] += corr_p[0]
            file_io.save_nii(img, f.replace('.nii', suffix), meta_info)
        new_files.append(f.replace('.nii', suffix))
    return new_files


def batch_bias_corr(test_slice, save_dir_lr, postfix="stitched", suffix="_corr.nii", save_bias=True):
    save_suffix = suffix
    save_bias = save_bias
    intensity_files = []
    for iid, slice_id in tqdm.tqdm(enumerate(test_slice)):
        lr_save_path = os.path.join(save_dir_lr, '{}_{}.nii'.format(postfix, slice_id))
        if not os.path.isfile(lr_save_path.replace('.nii', '_n4.nii')):
            FBC = Correct_Bias(lr_save_path, lr_save_path.replace('.nii', save_suffix), is_save_bias=save_bias)
            FBC.run()
        intensity_files.append(lr_save_path.replace('.nii', save_suffix))
        # hr_intensity_files.append(hr_save_path.replace('.nii', save_suffix))
    correct_p = correct_n4(intensity_files, overlap=0.6, iter=10)
    # correct_p = None
    _ = apply_corr(intensity_files, correct_p, suffix='_corr_auto.nii')

# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Optical Attenuation Correction')
#     parser.add_argument('-img_path', '--img_path',
#                         default=r'../demo/test_2x3/002_slice_0025_fused_tp_0_ch_0_corr_crop.nii.gz',
#                         help='input file')
#     parser.add_argument('-mask_path', '--mask_path',
#                         default=r'../demo/test_2x3/002_slice_0025_fused_tp_0_ch_0_corr_cropped_mask.nii.gz',
#                         help='mask corresponding to the image')
#     parser.add_argument('-save_suffix', '--save_suffix', default='_n4.nii',
#                         help='output suffix for cropped images')
#     parser.add_argument('-save_bias', '--save_bias', default=False,
#                         help='output suffix for cropped images')
#
#     arguments = parser.parse_args()
#     img_path = arguments.img_path
#     mask_path = arguments.mask_path
#     save_suffix = arguments.save_suffix
#     save_bias = arguments.save_bias
#
#     correct_bias(img_path, mask_path, img_path.replace('.nii', save_suffix), is_save_bias=save_bias)
