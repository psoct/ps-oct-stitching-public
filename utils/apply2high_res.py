# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : apply2high_res.py
# Time       :2023/6/13 10:09
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import math

import numpy as np
import SimpleITK as sitk
import sys, glob, os
# coding=utf-8
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm

from utils import file_io, blending


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


def trans_3d(img, affine):
    affine = torch.Tensor(affine)
    affine_values = affine.view((1, 3, 4))
    #     arr = np.transpose(img, (2, 1, 0))
    img = torch.from_numpy(img[np.newaxis, np.newaxis, ...]).float()

    out_size = img.size()
    grid = F.affine_grid(affine_values, out_size, align_corners=True)
    print(grid.size())
    transformed_x = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    return transformed_x.squeeze().numpy()


def parse_xml(xml_file):
    # Open the XML document
    dom = xml.dom.minidom.parse(xml_file)
    elementobj = dom.documentElement
    # Get child elements
    subElementObj = elementobj.getElementsByTagName("StitchingResults")
    # Distinguish elements that share the same tag name
    subElementObj1 = elementobj.getElementsByTagName("ViewRegistrations")

    grids = subElementObj1[0].getElementsByTagName("ViewRegistration")

    print(len(grids))
    grids_info = {}
    ming_grid = [1000000, -1000000, 10000000]
    for loc in grids:
        # print(loc.getAttribute('setup'))
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
                aa = [float(b) for b in bb.split(' ')]
                temp["Translation to Regular Grid"] = aa
                ming_grid = [min(aa[3], ming_grid[0]), max(aa[7], ming_grid[1]), min(aa[11], ming_grid[2])]
        grids_info[loc.getAttribute('setup')] = temp

    subElementObj1 = elementobj.getElementsByTagName("SequenceDescription")
    imgs_info = subElementObj1[0].getElementsByTagName("ViewSetups")
    temp = imgs_info[0].getElementsByTagName("ViewSetup")
    temp1 = temp[0].getElementsByTagName("voxelSize")
    spacing = temp1[0].getElementsByTagName("size")[0].childNodes[0].nodeValue
    spacing = [float(i) for i in spacing.split(' ')]
    return grids_info, ming_grid, spacing


def apply_to_highres(xml_dir, img_files, dataset_arg, save_dir, slice_id, padding=20, is_norm=True, shading=None,
                     blending_med="max", norm_lower=0, norm_upper=100):
    grids_info, min_grid, spacing = parse_xml(xml_dir)
    files = file_io.parse_path(img_files)[slice_id]
    img_files = [i[0] for i in files]
    # x_num = ['', 20, 20, 4, 5, '[Up & Right]', slice_id]
    x_num = dataset_arg[3]
    y_num = dataset_arg[4]
    z_num = 1
    overlap = dataset_arg[1] / 100
    padding = padding

    arr, meta_info = file_io.read_nii(img_files[0])
    spacing_x, spacing_y, spacing_z = meta_info[0]
    zz, yy, xx = arr.shape

    grid_values = []
    affine_values = []

    out_y = y_num * yy - int((y_num - 1) * yy * overlap) + padding
    out_x = x_num * xx - int((x_num - 1) * xx * overlap) + padding
    out_z = zz + padding
    stit_arr = np.zeros((out_z, out_y, out_x), dtype='uint8')
    # print(stit_arr.shape)

    for idx, file in tqdm.tqdm(enumerate(img_files)):
        grid_values.append((grids_info[str(idx)]['Translation to Regular Grid'][3],
                            grids_info[str(idx)]['Translation to Regular Grid'][7],
                            grids_info[str(idx)]['Translation to Regular Grid'][11]))
        affine_values.append((grids_info[str(idx)]['Stitching Transform'][3],
                              grids_info[str(idx)]['Stitching Transform'][7],
                              grids_info[str(idx)]['Stitching Transform'][11]))

        arr, meta_info = file_io.read_nii(file)
        if shading is not None:
            arr = shading.flat_corr(arr)

        ####################
        # transform 3D
        ####################
        temp_affine = grids_info[str(idx)]['Stitching Transform'].copy()
        temp_affine[3], temp_affine[7] = -(temp_affine[3] * (0.01 / spacing_x)) / (xx / 2), -(
                temp_affine[7] * (0.01 / spacing_y)) / (yy / 2)
        temp_affine[11] = -(temp_affine[11] * (0.01 / spacing_z)) / (zz / 2)
        # print(temp_affine)
        arr = trans_3d(arr, temp_affine)

        if is_norm:
            arr[arr < norm_lower] = norm_lower
            arr[arr > norm_upper] = norm_upper
            arr = (((arr - norm_lower) / (norm_upper - norm_lower)) * 255).astype("uint8")

        # temp = np.zeros((out_z, out_y, out_x), dtype='uint8')
        temp_x, temp_y, temp_z = grid_values[-1]
        temp_x = int((temp_x - min_grid[0]) * (spacing[0] / spacing_x))
        temp_y = int((-temp_y + min_grid[1]) * (spacing[1] / spacing_y))
        temp_z = int((temp_z - min_grid[2]) * (spacing[2] / spacing_z))

        temp_x, temp_y = temp_y, temp_x

        st_x, ed_x = padding // 2 + temp_z, padding // 2 + temp_z + zz
        st_y, ed_y = out_y - temp_x - padding // 2 - arr.shape[1], out_y - temp_x - padding // 2
        st_z, ed_z = padding // 2 + temp_y, padding // 2 + (temp_y + arr.shape[2])
        # print(temp_x, temp_y, temp_z, arr.shape)
        # stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        if blending_med == "max":
            # stit_arr = np.max(np.stack([temp, stit_arr], axis=0), axis=0)
            stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(
                np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)
        elif blending_med == "alpha":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.alpha_blending_fusion_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "mean":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.mean_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "multi_band":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.multi_band_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)

        #
        # moved_temp = temp
        #     print(moved_temp.shape, temp.shape)
    #     stit_arr[stit_arr==0]=moved_temp[stit_arr==0]
    #     break
    #     plt.imshow(stit_arr[150, ...], vmin=100, vmax=200, cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    # stit_arr
    file_io.save_nii(stit_arr, save_dir, meta_info)


def apply_to_highres_multiband(xml_dir, img_files, dataset_arg, save_dir, slice_id,
                               padding=20, shading=None, down_times=3):
    grids_info, min_grid, spacing = parse_xml(xml_dir)
    files = parse_path(img_files)[slice_id]
    img_files = [i[0] for i in files]
    # x_num = ['', 20, 20, 4, 5, '[Up & Right]', slice_id]
    x_num = dataset_arg[3]
    y_num = dataset_arg[4]
    z_num = 1
    overlap = dataset_arg[1] / 100
    padding = padding

    arr, meta_info = file_io.read_nii(img_files[0])
    spacing_x, spacing_y, spacing_z = meta_info[0]
    zz, yy, xx = arr.shape

    grid_values = []
    affine_values = []
    padding_x, padding_y, padding_z = 0, 0, 0
    out_y = y_num * yy - int((y_num - 1) * yy * overlap) + padding_y
    out_x = x_num * xx - int((x_num - 1) * xx * overlap) + padding_x
    out_z = zz + padding_z
    stit_arr = np.zeros((out_z, out_y, out_x), dtype='float32')
    laplacian_pyramid_arr = [np.zeros((out_z // (2 ** i), out_y // (2 ** i), out_x // (2 ** i)), dtype='float32') for i
                             in range(down_times)]
    laplacian_pyramid_arr.reverse()
    # print(stit_arr.shape)

    for idx, file in enumerate(img_files):
        grid_values.append((grids_info[str(idx)]['Translation to Regular Grid'][3],
                            grids_info[str(idx)]['Translation to Regular Grid'][7],
                            grids_info[str(idx)]['Translation to Regular Grid'][11]))
        affine_values.append((grids_info[str(idx)]['Stitching Transform'][3],
                              grids_info[str(idx)]['Stitching Transform'][7],
                              grids_info[str(idx)]['Stitching Transform'][11]))

        arr, meta_info = file_io.read_nii(file)
        if shading is not None:
            arr = shading.flat_corr(arr)

        ####################
        # transform 3D
        ####################
        temp_affine = grids_info[str(idx)]['Stitching Transform'].copy()
        temp_affine[3], temp_affine[7] = -(temp_affine[3] * (0.01 / spacing_x)) / (xx / 2), -(
                temp_affine[7] * (0.01 / spacing_y)) / (yy / 2)
        temp_affine[11] = -(temp_affine[11] * (0.01 / spacing_z)) / (zz / 2)
        print(temp_affine)
        arr = trans_3d(arr, temp_affine)

        # arr_img = sitk.GetImageFromArray(arr)
        arr_laplacian_pyramid = blending.get_laplacian_pyramid_v2(
            blending.get_gaussion_pyramid(arr, down_times=down_times),
            up_times=down_times)
        temp_pyramid_arr = [np.zeros((out_z // (2 ** i), out_y // (2 ** i), out_x // (2 ** i)), dtype='float32')
                            for i in range(down_times + 1)]
        temp_pyramid_arr.reverse()
        #
        # arr[arr < 0] = 0
        # arr[arr > 100] = 100
        # arr = ((arr / 100) * 255).astype("uint8")
        #
        # temp = np.zeros((out_z, out_y, out_x), dtype='uint8')
        temp_x, temp_y, temp_z = grid_values[-1]
        temp_x = int((temp_x - min_grid[0]) * (spacing[0] / spacing_x))
        temp_y = int((-temp_y + min_grid[1]) * (spacing[1] / spacing_y))
        temp_z = int((temp_z - min_grid[2]) * (spacing[2] / spacing_z))

        temp_x, temp_y = temp_y, temp_x
        tile_minx, tile_maxx = padding_z // 2 + temp_z, padding_z // 2 + temp_z + zz
        tile_miny, tile_maxy = out_y - temp_x - padding_y // 2 - arr.shape[1], out_y - temp_x - padding_y // 2,
        tile_minz, tile_maxz = padding_x // 2 + temp_y, padding_x // 2 + (temp_y + arr.shape[2])

        for i in range(down_times + 1):
            print(i)
            temp_pyramid_arr[i][tile_minx // (2 ** (down_times - i)):tile_maxx // (2 ** (down_times - i)),
            tile_miny // (2 ** (down_times - i)):tile_maxy // (2 ** (down_times - i)),
            tile_minz // (2 ** (down_times - i)):tile_maxz // (2 ** (down_times - i))] = sitk.GetArrayFromImage(
                arr_laplacian_pyramid[i])

        if idx == 0:
            laplacian_pyramid_arr = temp_pyramid_arr
            continue

        laplacian_pyramid_arr = blending.multi_band_blending(laplacian_pyramid_arr, temp_pyramid_arr)
        reconed = blending.recon_laplacian_pyramid(laplacian_pyramid_arr)
        reconed = sitk.GetArrayFromImage(reconed)
        plt.imshow(reconed[50, ...], vmin=100, vmax=200, cmap='gray')
        plt.colorbar()
        plt.show()
        #
        # moved_temp = temp
        #     print(moved_temp.shape, temp.shape)
    #     stit_arr[stit_arr==0]=moved_temp[stit_arr==0]
    #     break
    # plt.imshow(stit_arr[150, ...], vmin=100, vmax=200, cmap='gray')
    # plt.colorbar()
    # plt.show()
    # stit_arr
    file_io.save_nii(stit_arr.astype(np.uint8), save_dir, meta_info)


def apply_to_highres_z(xml_dir, img_files, dataset_arg, save_dir, slice_id, padding=20, is_norm=False, shading=None,
                       blending_med="max", norm_lower=0, norm_upper=100):
    grids_info, min_grid, spacing = parse_xml(xml_dir)
    img_files = sorted(img_files)
    # x_num = ['', 20, 20, 4, 5, '[Up & Right]', slice_id]
    x_num = dataset_arg[4]
    y_num = dataset_arg[5]
    z_num = dataset_arg[6]
    overlap_x = dataset_arg[1] / 100
    overlap_y = dataset_arg[2] / 100
    overlap_z = dataset_arg[3] / 100
    padding = padding

    arr, meta_info = file_io.read_nii(img_files[0])
    spacing_x, spacing_y, spacing_z = meta_info[0]
    zz, yy, xx = arr.shape

    grid_values = []
    affine_values = []

    out_y = y_num * yy - int((y_num - 1) * yy * overlap_y) + padding
    out_x = x_num * xx - int((x_num - 1) * xx * overlap_x) + padding
    out_z = z_num * zz - int((z_num - 1) * zz * overlap_z) + padding
    stit_arr = np.zeros((out_z, out_y, out_x), dtype='uint8')
    # print(stit_arr.shape)

    for idx, file in tqdm.tqdm(enumerate(img_files)):
        grid_values.append((grids_info[str(idx)]['Translation to Regular Grid'][3],
                            grids_info[str(idx)]['Translation to Regular Grid'][7],
                            grids_info[str(idx)]['Translation to Regular Grid'][11]))
        affine_values.append((grids_info[str(idx)]['Stitching Transform'][3],
                              grids_info[str(idx)]['Stitching Transform'][7],
                              grids_info[str(idx)]['Stitching Transform'][11]))

        arr, meta_info = file_io.read_nii(file)
        if shading is not None:
            arr = shading.flat_corr(arr)
        ####################
        # transform 3D
        ####################

        temp_affine = grids_info[str(idx)]['Stitching Transform'].copy()
        temp_affine[3], temp_affine[7] = -(temp_affine[3] * (spacing[0] / spacing_x)) / (xx / 2), -(
                temp_affine[7] * (spacing[1] / spacing_y)) / (yy / 2)
        temp_affine[11] = -(temp_affine[11] * (spacing[2] / spacing_z)) / (zz / 2)
        # print(temp_affine)

        # temp = np.zeros((out_z, out_y, out_x), dtype='uint8')
        temp_x, temp_y, temp_z = grid_values[-1]
        temp_x = int((temp_x - min_grid[0]) * (spacing[0] / spacing_x))
        temp_y = int((-temp_y + min_grid[1]) * (spacing[1] / spacing_y))
        temp_z = int((temp_z - min_grid[2]) * (spacing[2] / spacing_z))

        temp_x, temp_y = temp_y, temp_x

        st_x, ed_x = padding // 2 + temp_z, padding // 2 + temp_z + zz
        st_y, ed_y = out_y - temp_x - padding // 2 - arr.shape[1], out_y - temp_x - padding // 2
        st_z, ed_z = padding // 2 + temp_y, padding // 2 + (temp_y + arr.shape[2])

        arr = trans_3d(arr, temp_affine)
        # arr = arr
        if is_norm:
            arr[arr < norm_lower] = norm_lower
            arr[arr > norm_upper] = norm_upper
            norm = (((arr - norm_lower) / (norm_upper - norm_lower)) * 255).astype("uint8")

        # print(temp_x, temp_y, temp_z, arr.shape)
        # stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        if blending_med == "max":
            # stit_arr = np.max(np.stack([temp, stit_arr], axis=0), axis=0)
            stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(
                np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)
        elif blending_med == "alpha":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                print(arr.shape, stit_arr.shape, ed_x)
                # print(stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z].shape, (st_x-ed_x))
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.alpha_blending_fusion_quick_z(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "mean":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.mean_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "multi_band":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.multi_band_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)

        #
        # moved_temp = temp
        #     print(moved_temp.shape, temp.shape)
    #     stit_arr[stit_arr==0]=moved_temp[stit_arr==0]
    #     break
    #     plt.imshow(stit_arr[150, ...], vmin=100, vmax=200, cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    # stit_arr
    file_io.save_nii(stit_arr, save_dir, meta_info)


def apply_to_highres_z_adaptor_temp(xml_dir, img_files, dataset_arg, save_dir, padding=20, is_norm=False, shading=None,
                                    blending_med="max", norm_lower=0, norm_upper=100):
    grids_info, min_grid, spacing = parse_xml(xml_dir)
    img_files = sorted(img_files)
    # x_num = ['', 20, 20, 4, 5, '[Up & Right]', slice_id]
    x_num = dataset_arg[4]
    y_num = dataset_arg[5]
    z_num = dataset_arg[6]
    overlap_x = dataset_arg[1] / 100
    overlap_y = dataset_arg[2] / 100
    overlap_z = dataset_arg[3] / 100
    padding = padding

    arr, meta_info = file_io.read_nii(img_files[0])
    spacing_x, spacing_y, spacing_z = meta_info[0]
    zz, yy, xx = arr.shape

    grid_values = []
    affine_values = []

    out_y = y_num * yy - int((y_num - 1) * yy * overlap_y) + padding
    out_x = x_num * xx - int((x_num - 1) * xx * overlap_x) + padding
    out_z = z_num * zz - int((z_num - 1) * zz * overlap_z) + padding
    stit_arr = np.zeros((out_z, out_y, out_x), dtype='uint8')
    # print(stit_arr.shape)

    print(stit_arr.shape)
    for idx, file in tqdm.tqdm(enumerate(img_files)):
        grid_values.append((grids_info[str(idx)]['Translation to Regular Grid'][3],
                            grids_info[str(idx)]['Translation to Regular Grid'][7],
                            grids_info[str(idx)]['Translation to Regular Grid'][11]))
        affine_values.append((grids_info[str(idx)]['Stitching Transform'][3],
                              grids_info[str(idx)]['Stitching Transform'][7],
                              grids_info[str(idx)]['Stitching Transform'][11]))

        arr, meta_info = file_io.read_nii(file)
        if shading is not None:
            arr = shading.flat_corr(arr)
        ####################
        # transform 3D
        ####################
        temp_affine = grids_info[str(idx)]['Stitching Transform'].copy()

        adapt_z_padding = math.ceil(abs(temp_affine[11]))
        adapt_y_padding = math.ceil(abs(temp_affine[7]))
        adapt_x_padding = math.ceil(abs(temp_affine[3]))

        xxx, yyy, zzz = arr.shape[2] + adapt_x_padding, arr.shape[1] + adapt_y_padding, arr.shape[0] + adapt_z_padding
        temp_affine[3] = -(temp_affine[3] * (spacing[0] / spacing_x)) / (xxx / 2)
        temp_affine[7] = -(temp_affine[7] * (spacing[1] / spacing_y)) / (yyy / 2)
        temp_affine[11] = -(temp_affine[11] * (spacing[2] / spacing_z)) / (zzz / 2)
        # print(temp_affine)

        # temp = np.zeros((out_z, out_y, out_x), dtype='uint8')
        temp_x, temp_y, temp_z = grid_values[-1]
        temp_x = int((temp_x - min_grid[0]) * (spacing[0] / spacing_x))
        temp_y = int((-temp_y + min_grid[1]) * (spacing[1] / spacing_y))
        temp_z = int((temp_z - min_grid[2]) * (spacing[2] / spacing_z))

        temp_x, temp_y = temp_y, temp_x

        temp_arr = np.zeros((zzz, yyy, xxx), dtype='uint8')
        if grids_info[str(idx)]['Stitching Transform'][11] > 0:
            st_x, ed_x = padding // 2 + temp_z, padding // 2 + temp_z + zzz
            temp_arr[:zz, ...] = arr
        else:
            st_x, ed_x = padding // 2 + temp_z - adapt_z_padding, padding // 2 + temp_z + zz
            temp_arr[adapt_z_padding:, ...] = arr

        st_y, ed_y = out_y - temp_x - padding // 2 - arr.shape[1], out_y - temp_x - padding // 2
        st_z, ed_z = padding // 2 + temp_y, padding // 2 + (temp_y + arr.shape[2])
        arr = trans_3d(temp_arr, temp_affine)
        # arr = arr
        if is_norm:
            arr[arr < norm_lower] = norm_lower
            arr[arr > norm_upper] = norm_upper
            arr = (((arr - norm_lower) / (norm_upper - norm_lower)) * 255).astype("uint8")
        # print(temp_x, temp_y, temp_z, arr.shape)
        # stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        if blending_med == "max":
            # stit_arr = np.max(np.stack([temp, stit_arr], axis=0), axis=0)
            print(st_x, ed_x, st_y, ed_y, st_z, ed_z)
            stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(
                np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        elif blending_med == "alpha":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                print(arr.shape, stit_arr.shape, ed_x)
                # print(stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z].shape, (st_x-ed_x))
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.alpha_blending_fusion_quick_z(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "mean":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.mean_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "multi_band":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.multi_band_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)

        #
        # moved_temp = temp
        #     print(moved_temp.shape, temp.shape)
    #     stit_arr[stit_arr==0]=moved_temp[stit_arr==0]
    #     break
    #     plt.imshow(stit_arr[150, ...], vmin=100, vmax=200, cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    # stit_arr
    file_io.save_nii(stit_arr, save_dir, meta_info)


def apply_to_highres_adaptor(xml_dir, img_files, dataset_arg, save_dir, slice_id=None, padding=20, is_norm=False,
                             shading=None,
                             blending_med="max", norm_lower=0, norm_upper=100):
    grids_info, min_grid, spacing = parse_xml(xml_dir)

    if dataset_arg[6] == 1:
        files = file_io.parse_path(img_files)[slice_id]
        img_files = [i[0] for i in files]
    else:
        img_files = sorted(img_files)[:70]
    # x_num = ['', 20, 20, 4, 5, '[Up & Right]', slice_id]
    x_num = dataset_arg[4]
    y_num = dataset_arg[5]
    z_num = dataset_arg[6]
    overlap_x = dataset_arg[1] / 100
    overlap_y = dataset_arg[2] / 100
    overlap_z = dataset_arg[3] / 100
    padding = padding

    arr, meta_info = file_io.read_nii(img_files[0])
    spacing_x, spacing_y, spacing_z = meta_info[0]
    zz, yy, xx = arr.shape

    grid_values = []
    affine_values = []

    out_y = y_num * yy - int((y_num - 1) * yy * overlap_y) + padding
    out_x = x_num * xx - int((x_num - 1) * xx * overlap_x) + padding
    out_z = z_num * zz - int((z_num - 1) * zz * overlap_z) + padding
    stit_arr = np.zeros((out_z, out_y, out_x), dtype='uint8')
    # print(stit_arr.shape)

    print(stit_arr.shape)
    for idx, file in tqdm.tqdm(enumerate(img_files)):
        grid_values.append((grids_info[str(idx)]['Translation to Regular Grid'][3],
                            grids_info[str(idx)]['Translation to Regular Grid'][7],
                            grids_info[str(idx)]['Translation to Regular Grid'][11]))
        affine_values.append((grids_info[str(idx)]['Stitching Transform'][3],
                              grids_info[str(idx)]['Stitching Transform'][7],
                              grids_info[str(idx)]['Stitching Transform'][11]))

        arr, meta_info = file_io.read_nii(file)
        if shading is not None:
            arr = shading.flat_corr(arr)
        ####################
        # transform 3D
        ####################
        temp_affine = grids_info[str(idx)]['Stitching Transform'].copy()

        adapt_z_padding = math.ceil(abs(temp_affine[11]))
        adapt_y_padding = math.ceil(abs(temp_affine[7]))
        adapt_x_padding = math.ceil(abs(temp_affine[3]))

        xxx, yyy, zzz = arr.shape[2] + adapt_x_padding, arr.shape[1] + adapt_y_padding, arr.shape[0] + adapt_z_padding
        temp_affine[3] = -(temp_affine[3] * (spacing[0] / spacing_x)) / (xxx / 2)
        temp_affine[7] = -(temp_affine[7] * (spacing[1] / spacing_y)) / (yyy / 2)
        temp_affine[11] = -(temp_affine[11] * (spacing[2] / spacing_z)) / (zzz / 2)
        # print(temp_affine)

        # temp = np.zeros((out_z, out_y, out_x), dtype='uint8')
        temp_x, temp_y, temp_z = grid_values[-1]
        temp_x = int((temp_x - min_grid[0]) * (spacing[0] / spacing_x))
        temp_y = int((-temp_y + min_grid[1]) * (spacing[1] / spacing_y))
        temp_z = int((temp_z - min_grid[2]) * (spacing[2] / spacing_z))

        temp_x, temp_y = temp_y, temp_x
        if grids_info[str(idx)]['Stitching Transform'][11] > 0:
            st_x, ed_x = padding // 2 + temp_z, padding // 2 + temp_z + arr.shape[0] + adapt_z_padding
            arr = np.pad(arr, ((0, adapt_z_padding), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            st_x, ed_x = padding // 2 + temp_z - adapt_z_padding, padding // 2 + temp_z + zz
            arr = np.pad(arr, ((adapt_z_padding, 0), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))

        if grids_info[str(idx)]['Stitching Transform'][7] > 0:
            st_y, ed_y = out_y - temp_x - padding // 2 - arr.shape[1], out_y - temp_x - padding // 2 + adapt_y_padding
            arr = np.pad(arr, ((0, 0), (0, adapt_y_padding), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            st_y, ed_y = out_y - temp_x - padding // 2 - arr.shape[1] - adapt_y_padding, out_y - temp_x - padding // 2
            arr = np.pad(arr, ((0, 0), (adapt_y_padding, 0), (0, 0)), 'constant', constant_values=(0, 0))

        if grids_info[str(idx)]['Stitching Transform'][3] > 0:
            st_z, ed_z = padding // 2 + temp_y, padding // 2 + (temp_y + arr.shape[2]) + adapt_x_padding
            arr = np.pad(arr, ((0, 0), (0, 0), (0, adapt_x_padding)), 'constant', constant_values=(0, 0))
        else:
            st_z, ed_z = padding // 2 + temp_y - adapt_x_padding, padding // 2 + (temp_y + arr.shape[2])
            arr = np.pad(arr, ((0, 0), (0, 0), (adapt_x_padding, 0)), 'constant', constant_values=(0, 0))

        # arr = np.transpose(arr, (0, 2, 1))
        arr = trans_3d(arr, temp_affine)
        # arr = np.transpose(arr, (0, 2, 1))
        # arr = arr
        if is_norm:
            arr[arr < norm_lower] = norm_lower
            arr[arr > norm_upper] = norm_upper
            arr = (((arr - norm_lower) / (norm_upper - norm_lower)) * 255).astype("uint8")
        # print(temp_x, temp_y, temp_z, arr.shape)
        # stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        if blending_med == "max":
            # stit_arr = np.max(np.stack([temp, stit_arr], axis=0), axis=0)
            print(st_x, ed_x, st_y, ed_y, st_z, ed_z)
            stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.max(
                np.stack([stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr], axis=0), axis=0)

        elif blending_med == "alpha":
            print(st_x, ed_x, st_y, ed_y, st_z, ed_z)
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                print(arr.shape, stit_arr.shape, ed_x)
                # print(stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z].shape, (st_x-ed_x))
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.alpha_blending_fusion_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "mean":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.mean_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)
        elif blending_med == "multi_band":
            if idx == 0:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = arr
            else:
                stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = blending.multi_band_blending_quick(
                    stit_arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], arr)

        #
        # moved_temp = temp
        #     print(moved_temp.shape, temp.shape)
    #     stit_arr[stit_arr==0]=moved_temp[stit_arr==0]
    #     break
    #     plt.imshow(stit_arr[150, ...], vmin=100, vmax=200, cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    # stit_arr
    stit_arr = np.transpose(stit_arr, (0, 2, 1))
    [spacing, orgin, dirction] = meta_info
    print(spacing)
    print(orgin)
    print(dirction)
    file_io.save_nii(stit_arr, save_dir, meta_info)


#
# if __name__ == '__main__':
#     import argparse, time
#
#     parser = argparse.ArgumentParser(description='Image Stitching by BigStitcher')
#     parser.add_argument('-img_path', '--img_path', default=r'../demo/data', help='working directory')
#     parser.add_argument('-suffix', '--suffix', default=r'_shading_corr_v2.nii',
#                         help='input file suffix')
#     parser.add_argument('-slice_id', '--slice_id', default=r'0009',
#                         help='section to stitch')
#     parser.add_argument('-overlap', '--overlap', default=[20, 20, 0],
#                         help='overlap ratios in x, y, and z')
#     parser.add_argument('-tiles_num', '--tiles_num', default=[19, 24, 1],
#                         help='numbers of tiles in x, y, and z')
#     parser.add_argument('-grid_type', '--grid_type', default='[Up & Right]',
#                         help='grid layout')
#     arg = parser.parse_args()
#
#     data_dir = "/path/to/processed"
#     slice_id = '0009'
#
#     xml_file = '/path/to/0009_dataset_0009.xml'
#     save_root = "/path/to/output"
#
#     dataset_args = ['', arg.overlap[0], arg.overlap[1], arg.tiles_num[0],
#                     arg.tiles_num[1], arg.grid_type, slice_id]  # [Up & Left]
#
#     # apply_to_highres(xml_file, hr_intensity_imgs, dataset_args,
#     #                  save_root + "/stitched_raw_alpha_blend_{}.nii".format(slice_id),
#     #                  slice_id, padding=20, shading=None,
#     #                  blending_med="alpha")
#
#     hr_retardance_imgs = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_retardance.nii'.format(slice_id)))
#     apply_to_highres(xml_file, hr_retardance_imgs, dataset_args,
#                      save_root + "/stitched_raw_alpha_blend_{}_oct_retardance.nii".format(slice_id),
#                      slice_id, padding=20, shading=None,
#                      blending_med="alpha")
#
#     hr_axisorient_imgs = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_axisorient.nii'.format(slice_id)))
#     apply_to_highres(xml_file, hr_axisorient_imgs, dataset_args,
#                      save_root + "/stitched_raw_alpha_blend_{}_oct_retardance.nii".format(slice_id),
#                      slice_id, padding=20, shading=None,
#                      blending_med="alpha")
#
#
#     pass


if __name__ == '__main__':
    import argparse, time

    parser = argparse.ArgumentParser(description='Image Stitching by BigStitcher')
    parser.add_argument('-img_path', '--img_path',
                        required=True,
                        help='working directory')
    parser.add_argument('-xml_file', '--xml_file', required=True,
                        help='BigStitcher XML file')
    parser.add_argument('-save_path', '--save_path', required=True,
                        help='stitching output path')
    parser.add_argument('-suffix', '--suffix', default=r'_n4.nii',
                        help='input file suffix')
    parser.add_argument('-slice_id', '--slice_id', default=r'whole_test',
                        help='section to stitch')
    parser.add_argument('-overlap', '--overlap', default=[0, 0, 50],
                        help='overlap ratios in x, y, and z')
    parser.add_argument('-tiles_num', '--tiles_num', default=[1, 1, 72],
                        help='numbers of tiles in x, y, and z')
    parser.add_argument('-grid_type', '--grid_type', default='[Up & Right]',
                        help='grid layout')
    arg = parser.parse_args()

    data_dir = arg.img_path
    slice_id = arg.slice_id
    xml_file = arg.xml_file

    dataset_args = ['', arg.overlap[0], arg.overlap[1], arg.overlap[2], arg.tiles_num[0],
                    arg.tiles_num[1], arg.tiles_num[2], arg.grid_type, slice_id]  # [Up & Left]

    print(dataset_args)

    hr_intensity_imgs = glob.glob(os.path.join(data_dir, '*{}'.format(arg.suffix)))
    apply_to_highres_adaptor(xml_file, hr_intensity_imgs, dataset_args, arg.save_path,
                             padding=120, is_norm=False,
                             shading=None,
                             blending_med="max", norm_lower=0, norm_upper=100)
    # apply_to_highres(xml_file, hr_intensity_imgs, dataset_args,
    #                  save_root + "/stitched_raw_alpha_blend_{}.nii".format(slice_id),
    #                  slice_id, padding=20, shading=None,
    #                  blending_med="alpha")
    #
    # hr_retardance_imgs = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_retardance.nii'.format(slice_id)))
    # apply_to_highres(xml_file, hr_retardance_imgs, dataset_args,
    #                  save_root + "/stitched_raw_alpha_blend_{}_oct_retardance.nii".format(slice_id),
    #                  slice_id, padding=20, shading=None,

    #                  blending_med="alpha")
    #
    # hr_axisorient_imgs = glob.glob(os.path.join(data_dir, 'Slice_{}_*.oct_axisorient.nii'.format(slice_id)))
    # apply_to_highres(xml_file, hr_axisorient_imgs, dataset_args,
    #                  save_root + "/stitched_raw_alpha_blend_{}_oct_retardance.nii".format(slice_id),
    #                  slice_id, padding=20, shading=None,
    #                  blending_med="alpha")

    pass
