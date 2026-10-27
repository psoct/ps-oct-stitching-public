# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : blending.py
# Time       :2023/6/21 14:20
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs

import glob
import os
import sys
import xml.dom.minidom
from scipy import ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
from . import image_utils
from scipy.ndimage import convolve

from utils import file_io


def trans_3d(img, affine):
    affine = torch.Tensor(affine)
    affine_values = affine.view((1, 3, 4))
    #     arr = np.transpose(img, (2, 1, 0))
    arr_torch = torch.from_numpy(img[np.newaxis, np.newaxis, ...]).float()

    out_size = arr_torch.size()
    grid = F.affine_grid(affine_values, out_size, align_corners=True)
    #     print(grid.size())
    transformed_x = F.grid_sample(arr_torch, grid, mode='bilinear', align_corners=True)
    return transformed_x.squeeze().numpy()


def parse_path(files):
    slices_dict = {}
    for p in files:
        dirs_n, file_n = os.path.split(p)
        temp = file_n.split('_')
        slices = temp[1]
        tiles = temp[2]
        coordinate = [float(temp[4]), float(temp[6]), float(temp[8])]
        if slices not in slices_dict:
            slices_dict[slices] = [(p, coordinate)]
        else:
            slices_dict[slices].append((p, coordinate))

    #     print(slices_dict)
    for i in slices_dict:
        temp2 = slices_dict[i]
        temp2 = sorted(temp2, key=lambda x: (x[1][0], x[1][1]), reverse=False)
        slices_dict[i] = temp2

    macro_input = []  # *.nii
    for i in slices_dict:
        temp3 = slices_dict[i]
        temp3 = sorted(temp3, key=lambda x: (x[1][0], x[1][1]), reverse=False)
        slices_dict[i] = temp3

    return slices_dict


def laplacian(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        # i ranges from 5 to 0, giving six Laplacian-pyramid levels
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        temp_lap = cv2.subtract(gaussian_pyramid[i - 1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


def get_bbox(arr):
    points = np.nonzero(arr)
    minx, miny, minz = np.min(points[0]), np.min(points[1]), np.min(points[2])
    maxx, maxy, maxz = np.max(points[0]), np.max(points[1]), np.max(points[2])
    return minx, miny, minz, maxx, maxy, maxz


def crop_arr(arr, minx, miny, minz, maxx, maxy, maxz):
    return arr[minx:maxx + 1, miny:maxy + 1, minz:maxz + 1]


def get_gaussion_pyramid(img, down_times, resamplemethod=sitk.sitkNearestNeighbor):
    img = sitk.GetImageFromArray(img)
    xx, yy, zz = img.GetSize()
    target_origin = img.GetOrigin()
    target_direction = img.GetDirection()
    spacing_x, spacing_y, spacing_z = img.GetSpacing()
    gaussian_pyramid = [img]
    for i in range(down_times):
        # Gaussion
        sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
        #         sitk_gassuian.SetSigma(0.01*(2**(i+1)))
        sitk_gassuian.SetSigma(0.01)
        sitk_gassuian.NormalizeAcrossScaleOff()
        if i == 0:
            sitk_gassuian = sitk_gassuian.Execute(img)
        else:
            sitk_gassuian = sitk_gassuian.Execute(itk_img_resampled)

        # Downsampling
        sitk_gassuian = sitk_gassuian
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_gassuian)  # reference image for resampling
        # Set output image metadata
        resampler.SetSize((xx // (2 ** (i + 1)), yy // (2 ** (i + 1)), zz // (2 ** (i + 1))))  # output image size
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing((spacing_x * (2 ** (i + 1)), spacing_y * (2 ** (i + 1)), spacing_z * (2 ** (i + 1))))

        print((xx // (2 ** (i + 1)), yy // (2 ** (i + 1)), zz // (2 ** (i + 1))))
        print((spacing_x * (2 ** (i + 1)), spacing_y * (2 ** (i + 1)), spacing_z * (2 ** (i + 1))))
        # Set the output pixel type according to the resampling use case
        if resamplemethod == sitk.sitkNearestNeighbor:
            resampler.SetOutputPixelType(sitk.sitkUInt8)  # use nearest-neighbor interpolation for masks and save as uint8
        else:
            resampler.SetOutputPixelType(sitk.sitkFloat32)  # use linear interpolation for continuous-valued images and save as float32
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(sitk_gassuian)  # execute resampling

        gaussian_pyramid.append(itk_img_resampled)

    return gaussian_pyramid


def get_laplacian_pyramid_v2(gaussian_pyramid, up_times=3, resamplemethod=sitk.sitkNearestNeighbor):
    laplacian_pyramid = [sitk.GetArrayFromImage(gaussian_pyramid[-1])]
    #     laplacian_pyramid = [gaussian_pyramid[-1]]
    img = gaussian_pyramid[-1]
    xx, yy, zz = img.GetSize()
    target_origin = img.GetOrigin()
    target_direction = img.GetDirection()
    spacing_x, spacing_y, spacing_z = img.GetSpacing()

    for i in range(up_times, 0, -1):
        # i ranges from 5 to 0, giving six Laplacian-pyramid levels
        # Downsampling
        sitk_gassuian = gaussian_pyramid[i]
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_gassuian)  # reference image for resampling

        # Set output image metadata

        resampler.SetSize(gaussian_pyramid[i - 1].GetSize())  # output image size
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing(gaussian_pyramid[i - 1].GetSpacing())

        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(sitk_gassuian)  # execute resampling

        temp_lap = sitk.GetArrayFromImage(gaussian_pyramid[i - 1]) - sitk.GetArrayFromImage(itk_img_resampled)
        laplacian_pyramid.append(temp_lap)
        temp_lap = sitk.GetImageFromArray(temp_lap)

    return laplacian_pyramid


def get_laplacian_pyramid(gaussian_pyramid, up_times=3, resamplemethod=sitk.sitkNearestNeighbor):
    #     laplacian_pyramid = [sitk.GetArrayFromImage(gaussian_pyramid[-1])]
    laplacian_pyramid = [gaussian_pyramid[-1]]
    img = gaussian_pyramid[-1]
    # img = sitk.GetImageFromArray(img)
    xx, yy, zz = img.GetSize()
    target_origin = img.GetOrigin()
    target_direction = img.GetDirection()
    spacing_x, spacing_y, spacing_z = img.GetSpacing()

    for i in range(up_times, 0, -1):
        # i ranges from 5 to 0, giving six Laplacian-pyramid levels

        sitk_gassuian = gaussian_pyramid[i]

        # UP sampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_gassuian)  # reference image for resampling

        # Set output image metadata
        resampler.SetSize(
            (xx * (2 ** (up_times - i + 1)), yy * (2 ** (up_times - i + 1)), zz * (2 ** (up_times - i + 1))))  # output image size
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing((spacing_x / (2 ** (up_times - i + 1)), spacing_y / (2 ** (up_times - i + 1)),
                                    spacing_z / (2 ** (up_times - i + 1))))

        resampler.SetInterpolator(resamplemethod)
        itk_img_resampled = resampler.Execute(sitk_gassuian)  # execute resampling

        temp_lap = sitk.GetArrayFromImage(gaussian_pyramid[i - 1]) - sitk.GetArrayFromImage(itk_img_resampled)
        temp_lap = sitk.GetImageFromArray(temp_lap)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


def recon_laplacian_pyramid(laplacian_pyramid, resamplemethod=sitk.sitkNearestNeighbor):
    laplacian_pyramid = [sitk.GetImageFromArray(i) for i in laplacian_pyramid]
    img = laplacian_pyramid[0]
    xx, yy, zz = img.GetSize()
    target_origin = img.GetOrigin()
    target_direction = img.GetDirection()
    spacing_x, spacing_y, spacing_z = img.GetSpacing()

    for i, temp in enumerate(laplacian_pyramid[1:]):
        #         sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
        #         sitk_gassuian.SetSigma(0.02*(2**(i+1)))
        #         sitk_gassuian.NormalizeAcrossScaleOff()
        #         sitk_gassuian = sitk_gassuian.Execute(img)

        # Upsampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)  # reference image for resampling

        # Set output image metadata
        resampler.SetSize(temp.GetSize())  # output image size
        resampler.SetOutputOrigin(target_origin)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputSpacing((spacing_x / (2 ** (i + 1)), spacing_y / (2 ** (i + 1)), spacing_z / (2 ** (i + 1))))

        # Set the output pixel type according to the resampling use case
        #         if resamplemethod == sitk.sitkNearestNeighbor:
        #             resampler.SetOutputPixelType(sitk.sitkUInt8)   # use nearest-neighbor interpolation for masks and save as uint8
        #         else:
        #             resampler.SetOutputPixelType(sitk.sitkFloat32)  # use linear interpolation for continuous-valued images and save as float32
        resampler.SetInterpolator(resamplemethod)

        if i == 0:
            itk_img_resampled = resampler.Execute(img)  # execute resampling
        else:
            itk_img_resampled = resampler.Execute(temp_lap)

        print(sitk.GetArrayFromImage(temp).shape)
        print(sitk.GetArrayFromImage(temp).max(), sitk.GetArrayFromImage(itk_img_resampled).max())
        print(sitk.GetArrayFromImage(itk_img_resampled).shape)
        temp_lap = sitk.GetArrayFromImage(itk_img_resampled) + sitk.GetArrayFromImage(temp)
        temp_lap = sitk.GetImageFromArray(temp_lap)
        temp_lap.SetOrigin(target_origin)
        temp_lap.SetDirection(target_direction)
        temp_lap.SetSpacing((spacing_x / (2 ** (i + 1)), spacing_y / (2 ** (i + 1)), spacing_z / (2 ** (i + 1))))

    return temp_lap


# def get_blending_alpha(img1, img2):
#     print("---------------B L E N D I N G------------------")
#     print(img2.shape)
#     print(img1.shape)
#
#     mask = np.zeros(img2.shape)
#     mask[np.logical_and(img2 != 0, img1 != 0)] = 1
#
#     # plt.imshow(mask[50, ...], cmap='gray')
#     # plt.colorbar()
#     # plt.show()
#     overlap_edge = edge_detection_3d(mask)
#
#     # Keep only the outermost edge enclosing the mask
#     overlap_edge[overlap_edge != 0] = 1
#     overlap_edge[mask != 0] = 0
#
#     target_line = overlap_edge.copy()
#     target_line[img2 != 0] = 0
#
#     start_line = overlap_edge.copy()
#     start_line[img2 == 0] = 0
#
#     # calculate dist
#     ### mid case
#     #     print("---------------B L E N D I N G 2------------------")
#     # dist_map = np.zeros_like(overlap_edge[overlap_edge.shape[0]//2, ...])
#     mid_index = mask.shape[0] // 2
#     prox_overlap_line_voxnum = np.sum(overlap_edge[mid_index, ...])
#     mid_mask = mask[mid_index, ...].copy()
#
#     mid_mask_points = np.stack(np.nonzero(mask[mid_index, ...]), axis=0).T
#
#     mid_target_points = np.stack(np.nonzero(target_line[mid_index, ...]), axis=0).T
#     mid_start_points = np.stack(np.nonzero(start_line[mid_index, ...]), axis=0).T
#
#     temp_target = np.zeros_like(mid_mask).astype('float32')
#     temp_target[mid_mask == 1] = 100000000
#
#     for i in range(mid_target_points.shape[0]):
#         xx, yy = mid_target_points[i]
#         overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
#         reshape2raw = np.zeros_like(temp_target)
#         reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
#         temp_target = np.min(np.stack([temp_target, reshape2raw], axis=0), axis=0)
#
#     temp_target[np.logical_and(mask[mid_index, ...] == 0, img2[mid_index, ...] != 0)] = 1
#
#     temp_start = np.zeros_like(mid_mask).astype('float32')
#     temp_start[mid_mask == 1] = 100000000
#
#     for i in range(mid_start_points.shape[0]):
#         xx, yy = mid_start_points[i]
#         overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
#         reshape2raw = np.zeros_like(temp_start)
#         reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
#         temp_start = np.min(np.stack([temp_start, reshape2raw], axis=0), axis=0)
#     temp_start[np.logical_and(mask[mid_index, ...] == 0, img1[mid_index, ...] != 0)] = 1
#
#     mid_alpha_map = temp_target / (temp_start + temp_target)
#     res_alpha = np.zeros_like(mask).astype('float32')
#     for i in range(mask.shape[0]):
#         #     print(np.sum(overlap_edge[i, ...]), prox_overlap_line_voxnum)
#         if np.sum(overlap_edge[i, ...]) == prox_overlap_line_voxnum:
#             res_alpha[i, ...] = mid_alpha_map
#     res_alpha[np.isnan(res_alpha)] = 0
#     # plt.imshow(res_alpha[100, ...], cmap='gray')
#     # plt.colorbar()
#     # plt.show()
#     #
#     # plt.imshow(img2[100, ...], vmin=-2, vmax=2, cmap='gray')
#     # plt.colorbar()
#     # plt.show()
#     return res_alpha
def get_alpha_quick(img1, img2):
    print("---------------B L E N D I N G------------------")
    mask = np.zeros(img2.shape, dtype=np.bool_)
    mask[np.logical_and(img2 != 0, img1 != 0)] = True

    # calculate dist under middle case
    mid_index = mask.shape[0] // 2
    mid_mask = mask[mid_index, ...].copy()
    mid_mask = mid_mask.astype(np.float_)

    # pading
    mid_mask = np.pad(mid_mask, [(1, 1), (1, 1)])
    mid_overlap_edge = edge_detection_2d(mid_mask)
    mid_overlap_edge[mid_overlap_edge != 0] = 1
    mid_overlap_edge[mid_mask != 0] = 0

    target_line = mid_overlap_edge.copy()
    target_line[1:-1, 1:-1][img2[mid_index, ...] != 0] = 0

    start_line = mid_overlap_edge.copy()
    start_line[target_line == 1] = 0

    prox_overlap_line_voxnum = np.sum(mid_overlap_edge)

    mid_mask_points = np.stack(np.nonzero(mid_mask), axis=0).T

    mid_target_points = np.stack(np.nonzero(target_line), axis=0).T
    mid_start_points = np.stack(np.nonzero(start_line), axis=0).T

    temp_target = np.zeros_like(mid_mask).astype('float32')
    temp_target[mid_mask == 1] = 100000000

    for i in range(mid_target_points.shape[0]):
        xx, yy = mid_target_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_target)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_target = np.min(np.stack([temp_target, reshape2raw], axis=0), axis=0)

    temp_target[1:-1, 1:-1][np.logical_and(mask[mid_index, ...] == 0, img2[mid_index, ...] != 0)] = 1

    temp_start = np.zeros_like(mid_mask).astype('float32')
    temp_start[mid_mask == 1] = 100000000

    for i in range(mid_start_points.shape[0]):
        xx, yy = mid_start_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_start)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_start = np.min(np.stack([temp_start, reshape2raw], axis=0), axis=0)
    temp_start[1:-1, 1:-1][np.logical_and(mask[mid_index, ...] == 0, img1[mid_index, ...] != 0)] = 1

    mid_alpha_map = temp_target / (temp_start + temp_target)
    mid_alpha_map = mid_alpha_map[1:-1, 1:-1]
    res_alpha = np.zeros_like(mask).astype('float32')

    try:
        non_zero_z = np.nonzero(mask[:, mid_mask_points[0][0], mid_mask_points[0][1]])
        min_z, max_z = np.min(non_zero_z), np.max(non_zero_z)
    except:
        min_z, max_z = 0, res_alpha.shape[0]
        print('Notice: no overlap region!')

    res_alpha[min_z:max_z, :, :] = np.repeat([mid_alpha_map], max_z - min_z, axis=0)
    # for i in range(mask.shape[0]):
    #     #     print(np.sum(overlap_edge[i, ...]), prox_overlap_line_voxnum)
    #     tmp_overlap_edge = edge_detection_2d(mask[i, ...])
    #     tmp_overlap_edge[tmp_overlap_edge != 0] = 1
    #     tmp_overlap_edge[mask[i, ...] != 0] = 0
    #     if np.sum(tmp_overlap_edge) == prox_overlap_line_voxnum:
    #         res_alpha[i, ...] = mid_alpha_map

    # stit_res = img1 * (1 - res_alpha) + img2 * res_alpha
    # stit_res[np.isnan(stit_res)] = 0
    #     print(stit_res)
    # print(np.max(stit_res))
    # plt.imshow(stit_res[100, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    return res_alpha


def get_edge(tile):
    tile[tile != 0] = 1
    temp = sitk.GetImageFromArray(tile)
    image_float = sitk.Cast(temp, sitk.sitkFloat32)
    temp = sitk.GetArrayFromImage(image_float)
    print(np.unique(temp))
    #     edges = sitk.CannyEdgeDetection(image_float, lowerThreshold=0, upperThreshold=100)
    gra_filter = sitk.SobelEdgeDetectionImageFilter()
    sobel_data = gra_filter.Execute(image_float)
    #     sobel_data = sitk.Cast(sobel_data, sitk.sitkInt8)
    tile_edge = sitk.GetArrayFromImage(sobel_data)
    return tile_edge


# Edge-detection function
def edge_detection_3d(image, threshold=0):
    # Define the 3D Sobel operator
    sobel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                        [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                        [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])

    sobel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                        [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])

    sobel_z = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
    image = image.astype('float32')
    # Apply 3D convolution to compute the gradient magnitude
    gradient_magnitude = np.sqrt(
        convolve(image, sobel_x) ** 2 + convolve(image, sobel_y) ** 2 + convolve(image, sobel_z) ** 2)
    # Convert the gradient magnitude to a binary image
    # Adjust the threshold as needed
    gradient_magnitude[gradient_magnitude > threshold] = 1
    gradient_magnitude[gradient_magnitude != 1] = 0
    return gradient_magnitude.astype(np.bool_)


def edge_detection_2d(image, threshold=0.1):
    # Define the 3D Sobel operator
    dx = ndimage.sobel(image, 0)  # horizontal derivative
    dy = ndimage.sobel(image, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag[mag > threshold] = 1
    mag[mag != 1] = 0
    return mag


def parse_motion_xml(xml_dir):
    data_dir = xml_dir
    dom = xml.dom.minidom.parse(data_dir)
    # Read elements from the XML file
    elementobj = dom.documentElement
    # Get child elements
    subElementObj = elementobj.getElementsByTagName("StitchingResults")
    # Distinguish elements that share the same tag name
    subElementObj1 = elementobj.getElementsByTagName("ViewRegistrations")

    grids = subElementObj1[0].getElementsByTagName("ViewRegistration")

    print(len(grids))
    grids_info = {}
    for loc in grids:
        print(loc.getAttribute('setup'))

        transforms = loc.getElementsByTagName("ViewTransform")
        temp = {}
        for i in transforms:
            if i.getElementsByTagName("Name")[0].childNodes[0].nodeValue == "Stitching Transform":
                aa = i.getElementsByTagName("affine")[0].childNodes[0].nodeValue
                temp["Stitching Transform"] = [float(a) for a in aa.split(' ')]
            #             print("Stitching Transform",i.getElementsByTagName("affine")[0].childNodes[0].nodeValue)
            #         print(i.childNodes[0].data)
            if i.getElementsByTagName("Name")[0].childNodes[0].nodeValue == "Translation to Regular Grid":
                bb = i.getElementsByTagName("affine")[0].childNodes[0].nodeValue
                temp["Translation to Regular Grid"] = [float(b) for b in bb.split(' ')]
        grids_info[loc.getAttribute('setup')] = temp

    return grids_info


def multi_band_blending(laplacian_pyramid1, laplacian_pyramid2, down_times=2, init_alpha_layer=3):
    assert len(laplacian_pyramid2) >= init_alpha_layer
    res_alpha = get_blending_alpha(laplacian_pyramid1[down_times + 1 - init_alpha_layer],
                                   laplacian_pyramid2[down_times + 1 - init_alpha_layer])

    res_alpha_pyramid = []
    for i in range(len(laplacian_pyramid1)):
        if res_alpha.shape != laplacian_pyramid1[i].shape:
            temp_shape = (
                laplacian_pyramid1[i].shape[2], laplacian_pyramid1[i].shape[1], laplacian_pyramid1[i].shape[0])
            arr = image_utils.sample_2_fixed_size(sitk.GetImageFromArray(res_alpha), temp_shape)
            arr = sitk.GetArrayFromImage(arr)
            res_alpha_pyramid.append(arr)
        else:
            res_alpha_pyramid.append(res_alpha)
    # res_alpha_pyramid.reverse()

    # img1 = sitk.GetImageFromArray(img1)
    # gaussian_pyramid = get_gaussion_pyramid(img1, down_times=down_times)
    # laplacian_pyramid1 = get_laplacian_pyramid(gaussian_pyramid, up_times=down_times)
    #
    # img2 = sitk.GetImageFromArray(img2)
    # gaussian_pyramid = get_gaussion_pyramid(img2, down_times=down_times)
    # laplacian_pyramid2 = get_laplacian_pyramid(gaussian_pyramid, up_times=down_times)

    fused_laplacian_pyramid = []
    # for i in laplacian_pyramid1:
    #     # aaa = sitk.GetArrayFromImage(i)
    #     plt.imshow(i[i.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()

    # for i in laplacian_pyramid2:
    #     # aaa = sitk.GetArrayFromImage(i)
    #     plt.imshow(i[i.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()

    fused_laplacian_pyramid = []
    for i in range(len(laplacian_pyramid1)):
        #         print(laplacian_pyramid1[i].GetSize(), res_alpha_pyramid[i].shape)
        temp_fused = laplacian_pyramid1[i] * (
                1 - res_alpha_pyramid[i]) + laplacian_pyramid2[i] * res_alpha_pyramid[i]
        temp_fused[np.isnan(temp_fused)] = 0
        fused_laplacian_pyramid.append(temp_fused)

    # for i in res_alpha_pyramid:
    #     aaa = i
    #     plt.imshow(aaa[aaa.shape[0] // 2, ...], )
    #     plt.colorbar()
    #     plt.show()
    # for i in fused_laplacian_pyramid:
    #     aaa = i
    #     plt.imshow(aaa[aaa.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()
    #     print(len(fused_laplacian_pyramid), len(laplacian_pyramid1[i]))

    return fused_laplacian_pyramid


def alpha_blending_fusion(img1, img2):
    print("---------------B L E N D I N G------------------")
    # print(img2.shape)
    # print(img1.shape)
    # plt.imshow(img1[100, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(img2[100, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    mask = np.zeros(img2.shape, dtype=np.bool_)
    mask[np.logical_and(img2 != 0, img1 != 0)] = 1

    # plt.imshow(mask[100, ...], cmap='gray')
    # plt.colorbar()
    # plt.show()
    # Down sampling
    # mask

    overlap_edge = edge_detection_3d(mask)

    # Keep only the outermost edge enclosing the mask
    overlap_edge[overlap_edge != 0] = 1
    overlap_edge[mask != 0] = 0

    target_line = overlap_edge.copy()
    target_line[img2 != 0] = 0

    start_line = overlap_edge.copy()
    start_line[img2 == 0] = 0

    # calculate dist
    ### mid case
    #     print("---------------B L E N D I N G 2------------------")
    # dist_map = np.zeros_like(overlap_edge[overlap_edge.shape[0]//2, ...])
    mid_index = mask.shape[0] // 2
    # mid_overlap_edge =
    prox_overlap_line_voxnum = np.sum(overlap_edge[mid_index, ...])
    mid_mask = mask[mid_index, ...].copy()

    mid_mask_points = np.stack(np.nonzero(mask[mid_index, ...]), axis=0).T

    mid_target_points = np.stack(np.nonzero(target_line[mid_index, ...]), axis=0).T
    mid_start_points = np.stack(np.nonzero(start_line[mid_index, ...]), axis=0).T

    temp_target = np.zeros_like(mid_mask).astype('float32')
    temp_target[mid_mask == 1] = 100000000

    for i in range(mid_target_points.shape[0]):
        xx, yy = mid_target_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_target)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_target = np.min(np.stack([temp_target, reshape2raw], axis=0), axis=0)

    temp_target[np.logical_and(mask[mid_index, ...] == 0, img2[mid_index, ...] != 0)] = 1

    temp_start = np.zeros_like(mid_mask).astype('float32')
    temp_start[mid_mask == 1] = 100000000

    for i in range(mid_start_points.shape[0]):
        xx, yy = mid_start_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_start)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_start = np.min(np.stack([temp_start, reshape2raw], axis=0), axis=0)
    temp_start[np.logical_and(mask[mid_index, ...] == 0, img1[mid_index, ...] != 0)] = 1

    mid_alpha_map = temp_target / (temp_start + temp_target)
    res_alpha = np.zeros_like(mask).astype('float32')
    for i in range(mask.shape[0]):
        #     print(np.sum(overlap_edge[i, ...]), prox_overlap_line_voxnum)
        if np.sum(overlap_edge[i, ...]) == prox_overlap_line_voxnum:
            res_alpha[i, ...] = mid_alpha_map

    stit_res = img1 * (1 - res_alpha) + img2 * res_alpha
    stit_res[np.isnan(stit_res)] = 0
    #     print(stit_res)
    # print(np.max(stit_res))
    # plt.imshow(stit_res[100, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    return stit_res


def alpha_blending_fusion_quick(img1, img2):
    print("---------------B L E N D I N G------------------")
    mask = np.zeros(img2.shape, dtype=np.bool_)
    mask[np.logical_and(img2 != 0, img1 != 0)] = True

    # calculate dist under middle case
    mid_index = mask.shape[0] // 2
    mid_mask = mask[mid_index, ...].copy()
    mid_mask = mid_mask.astype(np.float_)

    # pading
    mid_mask = np.pad(mid_mask, [(1, 1), (1, 1)])
    mid_overlap_edge = edge_detection_2d(mid_mask)
    mid_overlap_edge[mid_overlap_edge != 0] = 1
    mid_overlap_edge[mid_mask != 0] = 0

    target_line = mid_overlap_edge.copy()
    target_line[1:-1, 1:-1][img2[mid_index, ...] != 0] = 0

    start_line = mid_overlap_edge.copy()
    start_line[target_line == 1] = 0

    prox_overlap_line_voxnum = np.sum(mid_overlap_edge)

    mid_mask_points = np.stack(np.nonzero(mid_mask), axis=0).T

    mid_target_points = np.stack(np.nonzero(target_line), axis=0).T
    mid_start_points = np.stack(np.nonzero(start_line), axis=0).T

    temp_target = np.zeros_like(mid_mask).astype('float32')
    temp_target[mid_mask == 1] = 100000000

    for i in range(mid_target_points.shape[0]):
        xx, yy = mid_target_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_target)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_target = np.min(np.stack([temp_target, reshape2raw], axis=0), axis=0)

    temp_target[1:-1, 1:-1][np.logical_and(mask[mid_index, ...] == 0, img2[mid_index, ...] != 0)] = 1

    temp_start = np.zeros_like(mid_mask).astype('float32')
    temp_start[mid_mask == 1] = 100000000

    for i in range(mid_start_points.shape[0]):
        xx, yy = mid_start_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_start)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_start = np.min(np.stack([temp_start, reshape2raw], axis=0), axis=0)
    temp_start[1:-1, 1:-1][np.logical_and(mask[mid_index, ...] == 0, img1[mid_index, ...] != 0)] = 1

    mid_alpha_map = temp_target / (temp_start + temp_target)
    mid_alpha_map = mid_alpha_map[1:-1, 1:-1]
    res_alpha = np.zeros_like(mask).astype('float16')

    try:
        non_zero_z = np.nonzero(mask[:, mid_mask_points[0][0], mid_mask_points[0][1]])
        min_z, max_z = np.min(non_zero_z), np.max(non_zero_z)
    except:
        min_z, max_z = 0, res_alpha.shape[0]
        print('Notice: no overlap region!')

    res_alpha[min_z:max_z, :, :] = np.repeat([mid_alpha_map], max_z - min_z, axis=0)
    # for i in range(mask.shape[0]):
    #     #     print(np.sum(overlap_edge[i, ...]), prox_overlap_line_voxnum)
    #     tmp_overlap_edge = edge_detection_2d(mask[i, ...])
    #     tmp_overlap_edge[tmp_overlap_edge != 0] = 1
    #     tmp_overlap_edge[mask[i, ...] != 0] = 0
    #     if np.sum(tmp_overlap_edge) == prox_overlap_line_voxnum:
    #         res_alpha[i, ...] = mid_alpha_map

    stit_res = img1 * (1 - res_alpha) + img2 * res_alpha
    stit_res[np.isnan(stit_res)] = 0
    #     print(stit_res)
    # print(np.max(stit_res))
    # plt.imshow(stit_res[100, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    return stit_res


def alpha_blending_fusion_quick_z(img1, img2):
    print("---------------B L E N D I N G------------------")
    mask = np.zeros(img2.shape, dtype=np.bool_)
    mask[np.logical_and(img2 != 0, img1 != 0)] = True

    # calculate dist under middle case
    mid_index = mask.shape[1] // 2
    mid_mask = mask[:, mid_index, ...].copy()
    mid_mask = mid_mask.astype(np.float_)

    # plt.imshow(mid_mask); plt.colorbar(); plt.show();

    # pading
    mid_mask = np.pad(mid_mask, [(1, 1), (1, 1)])
    mid_overlap_edge = edge_detection_2d(mid_mask)
    mid_overlap_edge[mid_overlap_edge != 0] = 1
    mid_overlap_edge[mid_mask != 0] = 0

    target_line = mid_overlap_edge.copy()
    target_line[1:-1, 1:-1][img2[:, mid_index, ...] != 0] = 0

    start_line = mid_overlap_edge.copy()
    start_line[target_line == 1] = 0

    # prox_overlap_line_voxnum = np.sum(mid_overlap_edge)

    mid_mask_points = np.stack(np.nonzero(mid_mask), axis=0).T

    mid_target_points = np.stack(np.nonzero(target_line), axis=0).T
    mid_start_points = np.stack(np.nonzero(start_line), axis=0).T

    temp_target = np.zeros_like(mid_mask).astype('float32')
    temp_target[mid_mask == 1] = 100000000

    for i in range(mid_target_points.shape[0]):
        xx, yy = mid_target_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_target)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_target = np.min(np.stack([temp_target, reshape2raw], axis=0), axis=0)

    temp_target[1:-1, 1:-1][np.logical_and(mask[:, mid_index, ...] == 0, img2[:, mid_index, ...] != 0)] = 1

    temp_start = np.zeros_like(mid_mask).astype('float32')
    temp_start[mid_mask == 1] = 100000000

    for i in range(mid_start_points.shape[0]):
        xx, yy = mid_start_points[i]
        overlap_dist = ((mid_mask_points[:, 0] - xx) ** 2 + (mid_mask_points[:, 1] - yy) ** 2) ** 0.5
        reshape2raw = np.zeros_like(temp_start)
        reshape2raw[mid_mask_points[:, 0], mid_mask_points[:, 1]] = overlap_dist
        temp_start = np.min(np.stack([temp_start, reshape2raw], axis=0), axis=0)
    temp_start[1:-1, 1:-1][np.logical_and(mask[:, mid_index, ...] == 0, img1[:, mid_index, ...] != 0)] = 1

    mid_alpha_map = temp_target / (temp_start + temp_target)
    mid_alpha_map = mid_alpha_map[1:-1, 1:-1]
    res_alpha = np.zeros_like(mask).astype('float16')

    # try:
    #     non_zero_z = np.nonzero(mask[mid_mask_points[len(mid_mask_points)//2][0], :, mid_mask_points[len(mid_mask_points)//2][1]])
    #     min_z, max_z = np.min(non_zero_z), np.max(non_zero_z)
    # except:
    #     min_z, max_z = 0, res_alpha.shape[0]
    #     print('Notice: no overlap region!')

    res_alpha[:, :, :] = np.transpose(np.repeat([mid_alpha_map], res_alpha.shape[1], axis=0), (1, 0, 2))
    # for i in range(mask.shape[0]):
    #     #     print(np.sum(overlap_edge[i, ...]), prox_overlap_line_voxnum)
    #     tmp_overlap_edge = edge_detection_2d(mask[i, ...])
    #     tmp_overlap_edge[tmp_overlap_edge != 0] = 1
    #     tmp_overlap_edge[mask[i, ...] != 0] = 0
    #     if np.sum(tmp_overlap_edge) == prox_overlap_line_voxnum:
    #         res_alpha[i, ...] = mid_alpha_map
    # plt.imshow(res_alpha[69,...]); plt.colorbar(); plt.show();
    # plt.imshow(res_alpha[:, 500, ...])
    # plt.colorbar()
    # plt.show()

    stit_res = img1 * (1 - res_alpha) + img2 * res_alpha
    stit_res[np.isnan(stit_res)] = 0
    #     print(stit_res)
    # print(np.max(stit_res))
    # plt.imshow(stit_res[:, 500, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    return stit_res


def multi_band_blending_quick(img1, img2, down_times=4, ratio=0.1):
    # assert down_times =
    laplacian_pyramid1 = get_laplacian_pyramid_v2(
        get_gaussion_pyramid(img1, down_times=down_times, resamplemethod=sitk.sitkLinear), up_times=down_times)
    laplacian_pyramid2 = get_laplacian_pyramid_v2(
        get_gaussion_pyramid(img2, down_times=down_times, resamplemethod=sitk.sitkLinear), up_times=down_times)

    res_alpha = get_alpha_quick(img1, img2)
    # mask = np.zeros(img2.shape)
    # mask[np.logical_and(img2 != 0, img1 != 0)] = 1
    # pyramid_alpha = get_gaussion_pyramid(mask, down_times=down_times, resamplemethod=sitk.sitkLinear)
    # pyramid_alpha.reverse()
    # pyramid_alpha = [sitk.GetArrayFromImage(i) for i in pyramid_alpha]
    pyramid_alpha = []
    for i in range(len(laplacian_pyramid1)):
        if res_alpha.shape != laplacian_pyramid1[i].shape:
            temp_shape = (
                laplacian_pyramid1[i].shape[2], laplacian_pyramid1[i].shape[1], laplacian_pyramid1[i].shape[0])
            arr = image_utils.sample_2_fixed_size(sitk.GetImageFromArray(res_alpha), temp_shape)
            arr = sitk.GetArrayFromImage(arr)
            pyramid_alpha.append(arr)
        else:
            pyramid_alpha.append(res_alpha)
    # pyramid_alpha.reverse()

    # img1 = sitk.GetImageFromArray(img1)
    # gaussian_pyramid = get_gaussion_pyramid(img1, down_times=down_times)
    # laplacian_pyramid1 = get_laplacian_pyramid(gaussian_pyramid, up_times=down_times)
    #
    # img2 = sitk.GetImageFromArray(img2)
    # gaussian_pyramid = get_gaussion_pyramid(img2, down_times=down_times)
    # laplacian_pyramid2 = get_laplacian_pyramid(gaussian_pyramid, up_times=down_times)
    fused_laplacian_pyramid = []
    # for i in laplacian_pyramid1:
    #     # aaa = sitk.GetArrayFromImage(i)
    #     plt.imshow(i[i.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()

    # for i in laplacian_pyramid2:
    #     # aaa = sitk.GetArrayFromImage(i)
    #     plt.imshow(i[i.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()

    fused_laplacian_pyramid = []
    coef = [1 - i * ratio for i in range(1, down_times + 2)]
    for i in range(len(laplacian_pyramid1)):
        #         print(laplacian_pyramid1[i].GetSize(), res_alpha_pyramid[i].shape)

        coef_matrix = pyramid_alpha[i].copy()
        # plt.imshow(coef_matrix[coef_matrix.shape[0]//2, ...])
        # plt.show()

        coef_matrix[coef_matrix > coef[i]] = coef[i]
        coef_matrix[np.isnan(coef_matrix)] = 0

        coef_matrix = (coef_matrix - np.min(coef_matrix)) / (np.max(coef_matrix) - np.min(coef_matrix))
        # plt.imshow(coef_matrix[coef_matrix.shape[0]//2, ...])
        # plt.show()

        temp_fused = laplacian_pyramid1[i] * (
                1 - coef_matrix) + laplacian_pyramid2[i] * coef_matrix

        # plt.imshow(temp_fused[coef_matrix.shape[0]//2, ...])
        # plt.show()
        # temp_fused[np.isnan(temp_fused)] = 0
        fused_laplacian_pyramid.append(temp_fused)

    # for i in res_alpha_pyramid:
    #     aaa = i
    #     plt.imshow(aaa[aaa.shape[0] // 2, ...], )
    #     plt.colorbar()
    #     plt.show()
    # for i in fused_laplacian_pyramid:
    #     aaa = i
    #     plt.imshow(aaa[aaa.shape[0] // 2, ...])
    #     plt.colorbar()
    #     plt.show()
    #     print(len(fused_laplacian_pyramid), len(laplacian_pyramid1[i]))
    reconed = recon_laplacian_pyramid(fused_laplacian_pyramid)
    reconed = sitk.GetArrayFromImage(reconed)
    reconed[np.logical_and(img2 == 0, img1 == 0)] = 0
    # plt.imshow(reconed[100, ...], vmin=100, vmax=200, cmap='gray'); plt.show()
    return reconed


def mean_blending_quick(img1, img2):
    # assert down_times =
    mask = np.zeros(img2.shape, dtype=np.bool_)
    mask[np.logical_and(img2 != 0, img1 != 0)] = 1

    reconed = np.max(np.stack([img1, img2], axis=0), axis=0)
    reconed[mask == 1] = (img1 + 0.5 + img2 * 0.5)[mask == 1]
    # plt.imshow(reconed[100, ...], vmin=100, vmax=200, cmap='gray'); plt.show()
    return reconed
