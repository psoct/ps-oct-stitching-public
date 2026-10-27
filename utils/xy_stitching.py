# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : xy_stitching.py
# Time       :2023/3/29 14:02
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

import glob, sys
import math
import os
import SimpleITK as sitk
import numpy as np

import shutil
from xml.dom import minidom

from utils import file_io
import imagej
import scyjava


def init_imagej(alloc_memory):
    scyjava.config.add_option('-Xmx{}'.format(alloc_memory))
    ij = imagej.init(['net.imagej:imagej:2.9.0', 'net.preibisch:BigStitcher:0.8.0'])
    print(f"ImageJ version: {ij.getVersion()}")
    return ij


class XY_Stitching(object):
    def __init__(self, ij, root_dir, slice_id, dataset_args, pairwise_args, opt_args, fusion_args, alloc_memory='24g'):
        self.xml_path = None
        self.root_dir = root_dir
        self.slice_id = slice_id
        self.dataset_args = dataset_args
        self.pairwise_args = pairwise_args
        self.opt_args = opt_args
        self.fusion_args = fusion_args
        self.ij = ij

    def define_dataset_by_macro(self, xml_name, xml_save_dir, input_path, grid_type, min_memory_kb,
                                x_tiles, y_tiles, z_tiles, x_overlap, y_overlap, z_overlap):
        """
        Define the dataset.
        :param xml_name: output XML filename
        :param xml_save_dir: directory for the output XML file
        :param input_path: input path; a temporary directory is created automatically for stitching
        :param grid_type: grid type (normally unchanged)
        :param min_memory_kb: exclude images smaller than this size
        :param x_tiles: number of tiles along x
      s :param y_tiles: number of tiles along y
        :param z_tiles: number of tiles along z
        :param x_overlap: overlap ratio along x
        :param y_overlap: overlap ratio along y
        :param z_overlap: overlap ratio along z
        :return: ImageJ macro command
        """
        dataset_args = ["define_dataset=[Automatic Loader (Bioformats based)]",
                        "project_filename={}".format(xml_name),
                        "path={}".format(input_path),
                        "exclude={}".format(min_memory_kb),
                        "pattern_0=Tiles pattern_1=Tiles",
                        "move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)]",
                        "grid_type={}".format(grid_type),  # [Down & Right             ]
                        "tiles_x={} tiles_y={} tiles_z={} overlap_x_(%)={} overlap_y_(%)={} overlap_z_(%)={}".format(
                            x_tiles, y_tiles, z_tiles, x_overlap, y_overlap, z_overlap),
                        "keep_metadata_rotation how_to_load_images=[Load raw data]",
                        "dataset_save_path={} check_stack_sizes".format(xml_save_dir)
                        ]
        #     print("_".join(dataset_args))
        #     run("Define dataset ...", "_".join(dataset_args))
        msg = 'run("Define dataset ...", "{}");'.format(" ".join(dataset_args))
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Finished Define dataset ...')
        return msg

    def cal_pairwise_shifts_by_macro(self, xml_path, method="Phase Correlation", ds_x=2, ds_y=2, ds_z=1):
        """
        Estimate pairwise tile displacements and store them in the XML file.
        :param xml_path: dataset XML path
        :param method: estimation method; defaults to phase correlation
        :param ds_x: x-axis downsampling factor used to accelerate computation
        :param ds_y:y-axis downsampling factor used to accelerate computation
        :param ds_z:z-axis downsampling factor used to accelerate computation
        :return: ImageJ macro command
        """
        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[{}] " \
                       "show_expert_algorithm_parameters downsample_in_x={} downsample_in_y={} downsample_in_z={} " \
                       "number_of_peaks_to_check=5 minimal_overlap=0 subpixel_accuracy".format(xml_path, method, ds_x,
                                                                                               ds_y, ds_z)
        msg = 'run("Calculate pairwise shifts ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Calculate pairwise shifts ...Done')
        return msg

    def glo_opt_by_macro(self, xml_path, method, fix_id):
        """
        Estimate the optimal displacement for each tile and store the result in the XML file.
        :param xml_path: dataset XML path
        :param method:estimation method; defaults to phase correlation
        :param fix_id:fixed tile for multi-tile stitching; preferably choose a central tile
        :return: ImageJ macro command
        """
        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 " \
                       "absolute=3.500 global_optimization_strategy=[{}] " \
                       "show_expert_grouping_options how_to_treat_timepoints=[treat individually] " \
                       "how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat " \
                       "individually] how_to_treat_tiles=compare fix_group_0-{}".format(xml_path, method, fix_id)
        msg = 'run("Optimize globally and apply shifts ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Optimize globally and apply shifts ...Done')
        return msg

    def fuse_save_by_macro(self, xml_path, out_dir, fusion_interp, pixel_savetype, file_name, blend=' blend'):
        """
        Fuse multiple tiles into a single output file.
        :param xml_path:  dataset XML path
        :param out_dir: output directory
        :param fusion_interp: interpolation method used during fusion
        :param pixel_savetype: output data type: 16-bit or 32-bit floating point
        :param file_name: output filename
        :param blend: use the default value
        :return: ImageJ macro command
        """
        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[" \
                       "Currently Selected Views] downsampling=1 pixel_type=[{}] interpolation=[{}]" \
                       " image=[Precompute Image] interest_points_for_non_rigid=[-= Disable Non-Rigid =-]" \
                       "{} produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] " \
                       "output_file_directory={}/. filename_addition={}".format(xml_path, pixel_savetype, fusion_interp,
                                                                                blend,
                                                                                out_dir, file_name)
        msg = 'run("Fuse dataset ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print(msg)
        print('==>Fuse dataset ...Done')
        return msg

    def stiching_main(self, root_dir, slice_num, dataset_args, pairwise_args, opt_args, fusion_args):
        xml_save_dir = root_dir + "/temp_{}".format(slice_num)
        input_path = xml_save_dir
        [temp_file_postfix, overlap_x, overlap_y, overlap_z, tile_num_x, tile_num_y, tile_num_z, grid_type,
         xml_save_path] = dataset_args
        xml_name = xml_save_path + '_dataset_{}.xml'.format(slice_num)
        print(xml_name)

        define_msg = self.define_dataset_by_macro(xml_name, xml_save_dir, input_path, grid_type, 10,
                                                  tile_num_x, tile_num_y, tile_num_z, overlap_x, overlap_y, overlap_z)

        xml_path = '{}/{}'.format(xml_save_dir, xml_name)
        xy_pairwise_method, xy_pairwise_dx, xy_pairwise_dy, xy_pairwise_dz = pairwise_args
        self.cal_pairwise_shifts_by_macro(xml_path, method=xy_pairwise_method,
                                          ds_x=xy_pairwise_dx, ds_y=xy_pairwise_dy, ds_z=xy_pairwise_dz)

        opt_method, opt_fixed_id = opt_args
        self.glo_opt_by_macro(xml_path, opt_method, fix_id=opt_fixed_id)

        fusion_interp, pixel_savetype, save_xystit_tif = fusion_args
        print("xml_path:", xml_path)
        self.fuse_save_by_macro(xml_path, input_path, fusion_interp, pixel_savetype,
                                'temp_slice_{}'.format(slice_num))
        self.xml_path = xml_path
        print(define_msg)

        return glob.glob(input_path + '/*.tif')[0], xml_name

    def stiching_fuse(self, root_dir, slice_num, dataset_args, fusion_args):
        xml_save_dir = root_dir + "/temp_{}".format(slice_num)
        input_path = xml_save_dir
        [temp_file_postfix, overlap_x, overlap_y, tile_num_x, tile_num_y, grid_type, xml_save_path] = dataset_args
        xml_name = xml_save_path + '_dataset_{}.xml'.format(slice_num)
        print(xml_name)

        define_msg = self.define_dataset_by_macro(xml_name, xml_save_dir, input_path, grid_type, 10,
                                                  tile_num_x, tile_num_y, 1, overlap_x, overlap_y, 10)

        xml_path = '{}/{}'.format(xml_save_dir, xml_name)
        fusion_interp, pixel_savetype, save_xystit_tif = fusion_args
        print("xml_path:", xml_path)
        self.fuse_save_by_macro(xml_path, input_path, fusion_interp, pixel_savetype,
                                'temp_slice_{}'.format(slice_num))
        self.xml_path = xml_path
        print(define_msg)

        return glob.glob(input_path + '/*.tif')[0], xml_name

    # def hierarchical_run(self, img_suffix=None, files=None, stitch_cell_x=4, stitch_cell_y=5):
    #     if files is not None:
    #         files = files
    #     elif img_suffix is not None:
    #         files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
    #     else:
    #
    #     slices_dict = {}
    #     for p in files:
    #         dirs_n, file_n = os.path.split(p)
    #         temp = file_n.split('_')
    #         slices = temp[1]
    #         tiles = temp[2]
    #         coordinate = [float(temp[4]), float(temp[6]), float(temp[8])]
    #         if slices not in slices_dict:
    #             slices_dict[slices] = [(p, coordinate)]
    #         else:
    #             slices_dict[slices].append((p, coordinate))
    #
    #     print(slices_dict)
    #     for i in slices_dict:
    #         temp2 = slices_dict[i]
    #         temp2 = sorted(temp2, key=lambda x: (x[1][0], x[1][1]), reverse=False)
    #         slices_dict[i] = temp2
    #
    #     macro_input = []  # *.nii
    #     for i in slices_dict:
    #         temp3 = slices_dict[i]
    #         temp3 = sorted(temp3, key=lambda x: (x[1][0], x[1][1]), reverse=False)
    #         slices_dict[i] = temp3
    #
    #     idx, save_datas, x_num, y_num = self.slice_id, slices_dict[self.slice_id], \
    #                                     self.dataset_args[3], self.dataset_args[4]
    #     temp_dirs = os.path.join(self.root_dir, 'temp_' + self.slice_id)
    #     global_x_num = math.ceil(x_num / stitch_cell_x)
    #     global_y_num = math.ceil(y_num / stitch_cell_y)
    #
    #     for x in range(x_num):
    #
    #         for y in range(x_num):
    #             # dir_n, file_n = os.path.split(save_datas[i][0])
    #             if x < 10:
    #                 str_x = "0" + str(x)
    #             else:
    #                 str_x = str(x)
    #             if y < 10:
    #                 str_y = "0" + str(y)
    #             else:
    #                 str_y = str(y)
    #
    #             shutil.copyfile(save_datas[i][0],
    #                             os.path.join(temp_dirs, 'Slice_' + idx + '_X_' + str_x + '_Y_' + str_y + '_.nii'))
    #
    #             if x + 1 % stitch_cell_x and y+1 % global_y_num:
    #
    #
    #
    #
    #             out_file, xml_name = self.stiching_main(self.root_dir, self.slice_id, self.dataset_args,
    #                                                     self.pairwise_args, self.opt_args, self.fusion_args)
    #             dir_name, file_name = os.path.split(out_file)
    #             file_io.saveTif2Nifty(out_file, os.path.join(dir_name, self.fusion_args[-1]),
    #                                   spacing=meta_info[0], orgin=meta_info[1])
    #             shutil.copyfile(os.path.join(dir_name, self.fusion_args[-1]),
    #                             os.path.join(self.root_dir, self.fusion_args[-1]))
    #             shutil.copyfile(os.path.join(dir_name, xml_name), os.path.join(self.root_dir, xml_name))
    #             shutil.rmtree(dir_name)
    #
    #             i += 1
    #
    #     slices_dict = file_io.parse_path(files)
    #     _, meta_info = file_io.read_nii(files[0])
    #     is_prep_data = file_io.create_temp_dir(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
    #                                           slices_dict[self.slice_id], self.dataset_args[3], self.dataset_args[4])
    #
    #     # dirs, idx, save_datas, x_num, y_num = os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id, \
    #     #                                       slices_dict[self.slice_id], self.dataset_args[3], self.dataset_args[4]
    #     # if not os.path.isdir(dirs):
    #     #     os.makedirs(dirs)
    #     # else:
    #     #     shutil.rmtree(dirs)
    #     #     os.makedirs(dirs)
    #     #
    #     # i = 0
    #     # for x in range(x_num):
    #     #     for y in range(y_num):
    #     #         # dir_n, file_n = os.path.split(save_datas[i][0])
    #     #         if x < 10:
    #     #             str_x = "0" + str(x)
    #     #         else:
    #     #             str_x = str(x)
    #     #         if y < 10:
    #     #             str_y = "0" + str(y)
    #     #         else:
    #     #             str_y = str(y)
    #     #         shutil.copyfile(save_datas[i][0],
    #     #                         os.path.join(dirs, 'Slice_' + idx + '_X_' + str_x + '_Y_' + str_y + '_.nii'))
    #     #         i += 1
    #
    #     if is_prep_data:
    #         pass

    def just_fuse(self, img_suffix=None, files=None):
        if files is not None:
            files = files
        elif img_suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
        else:
            raise ValueError('Please select files or a file suffix')

        slices_dict = file_io.parse_path(files)
        _, meta_info = file_io.read_nii(files[0])
        is_prep_data = file_io.create_temp_dir(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
                                              slices_dict[self.slice_id], self.dataset_args[3],
                                              self.dataset_args[4])
        if is_prep_data:
            out_file, xml_name = self.stiching_fuse(self.root_dir, self.slice_id, self.dataset_args,
                                                    self.fusion_args)
            dir_name, file_name = os.path.split(out_file)
            file_io.saveTif2Nifty(out_file, os.path.join(dir_name, self.fusion_args[-1]),
                                  spacing=meta_info[0], orgin=meta_info[1])
            shutil.copyfile(os.path.join(dir_name, self.fusion_args[-1]),
                            os.path.join(self.root_dir, self.fusion_args[-1]))
            shutil.copyfile(os.path.join(dir_name, xml_name), os.path.join(self.root_dir, xml_name))
            shutil.rmtree(dir_name)
            return os.path.join(self.root_dir, xml_name), os.path.join(self.root_dir, self.fusion_args[-1])
        else:
            raise ValueError('Please check the input data')

    def run(self, img_suffix=None, files=None):
        if files is not None:
            files = files
        elif img_suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
        else:
            raise ValueError('Please select files or a file suffix')
        # Sort files using the coordinates encoded in their original names
        ####sun####################
        for k in files:
            print(k)
        ####sun####################
        slices_dict = file_io.parse_path(files)

        _, meta_info = file_io.read_nii(files[0])
        ###sun
        for ss in slices_dict[self.slice_id]:
            print(ss)
        ###sun
        ##print(slices_dict[self.slice_id]) ###sun
        # Create a temporary workspace and copy the files to be stitched into it; return whether setup succeeded
        is_prep_data = file_io.create_temp_dir(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
                                              slices_dict[self.slice_id], self.dataset_args[4], self.dataset_args[5])
        if is_prep_data:
            # Run the stitching task
            out_file, xml_name = self.stiching_main(self.root_dir, self.slice_id, self.dataset_args,
                                                    self.pairwise_args, self.opt_args, self.fusion_args)
            dir_name, file_name = os.path.split(out_file)
            # Convert the generated .tif file to NIfTI format
            file_io.saveTif2Nifty(out_file, os.path.join(dir_name, self.fusion_args[-1]),
                                  spacing=meta_info[0], orgin=meta_info[1])
            shutil.copyfile(os.path.join(dir_name, self.fusion_args[-1]),
                            os.path.join(self.root_dir, self.fusion_args[-1]))
            shutil.copyfile(os.path.join(dir_name, xml_name), os.path.join(self.root_dir, xml_name))
            shutil.rmtree(dir_name)
            return os.path.join(self.root_dir, xml_name), os.path.join(self.root_dir, self.fusion_args[-1])
        else:
            raise ValueError('Please check the input data')

    def just_run(self, data_dir, save_postfix, dataset_args, pairwise_args, opt_args, fusion_args, img_suffix=None,
                 files=None):
        _, meta_info = file_io.read_nii(files[0])
        out_file, xml_name = self.stiching_main(data_dir, save_postfix, dataset_args,
                                                pairwise_args, opt_args, fusion_args)
        dir_name, file_name = os.path.split(out_file)
        file_io.saveTif2Nifty(out_file, os.path.join(dir_name, self.fusion_args[-1]),
                              spacing=meta_info[0], orgin=meta_info[1])
        # shutil.copyfile(os.path.join(dir_name, self.fusion_args[-1]),
        #                 os.path.join(self.root_dir, self.fusion_args[-1]))
        # shutil.copyfile(os.path.join(dir_name, xml_name), os.path.join(self.root_dir, xml_name))
        # shutil.rmtree(dir_name)
        return os.path.join(self.root_dir, xml_name), os.path.join(self.root_dir, self.fusion_args[-1])

    def run_quick(self, img_suffix=None, files=None, edge=False):
        if files is not None:
            files = files
        elif img_suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
        else:
            raise ValueError('Please select files or a file suffix')

        slices_dict = file_io.parse_path(files)
        _, meta_info = file_io.read_nii(files[0])
        if edge:
            is_prep_data = self.get_edge(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
                                         slices_dict[self.slice_id], self.dataset_args[4], self.dataset_args[5])
        else:
            is_prep_data = file_io.create_temp_dir(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
                                                  slices_dict[self.slice_id], self.dataset_args[4],
                                                  self.dataset_args[5])
        print(files, is_prep_data)
        if is_prep_data:
            xml_save_dir = self.root_dir + "/temp_{}".format(self.slice_id)
            input_path = xml_save_dir
            [temp_file_postfix, overlap_x, overlap_y, overlap_z, tile_num_x, tile_num_y, tile_num_z, grid_type,
             xml_save_path] = self.dataset_args
            xml_name = xml_save_path + '_dataset_{}.xml'.format(self.slice_id)
            print(xml_name)

            define_msg = self.define_dataset_by_macro(xml_name, xml_save_dir, input_path, grid_type, 10,
                                                      tile_num_x, tile_num_y, tile_num_z, overlap_x, overlap_y,
                                                      overlap_z)
            print(define_msg)
            xml_path = '{}/{}'.format(xml_save_dir, xml_name)
            xy_pairwise_method, xy_pairwise_dx, xy_pairwise_dy, xy_pairwise_dz = self.pairwise_args
            self.cal_pairwise_shifts_by_macro(xml_path, method=xy_pairwise_method,
                                              ds_x=xy_pairwise_dx, ds_y=xy_pairwise_dy, ds_z=xy_pairwise_dz)
            opt_method, opt_fixed_id = self.opt_args
            self.glo_opt_by_macro(xml_path, opt_method, fix_id=opt_fixed_id)
            shutil.copyfile(os.path.join(os.path.join(self.root_dir, 'temp_' + self.slice_id), xml_name),
                            os.path.join(self.root_dir, xml_name))
            shutil.rmtree(os.path.join(self.root_dir, 'temp_' + self.slice_id))
            return os.path.join(self.root_dir, xml_name), os.path.join(self.root_dir, 'temp_' + self.slice_id)
        else:
            raise ValueError('Please check the input data')

    def get_edge(self, dirs, idx, save_datas, x_num, y_num):
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
        else:
            shutil.rmtree(dirs)
            os.makedirs(dirs)

        i = 0
        for x in range(x_num):
            for y in range(y_num):
                # dir_n, file_n = os.path.split(save_datas[i][0])
                if x < 10:
                    str_x = "0" + str(x)
                else:
                    str_x = str(x)
                if y < 10:
                    str_y = "0" + str(y)
                else:
                    str_y = str(y)

                img_dir = save_datas[i][0]
                img = sitk.ReadImage(img_dir)
                spacing = img.GetSpacing()
                gaussion = sitk.SmoothingRecursiveGaussianImageFilter()
                executor = sitk.SobelEdgeDetectionImageFilter()
                # executor.SetVariance(0.01)
                gaussion.SetSigma(2 * spacing[0])
                img = gaussion.Execute(img)
                img = executor.Execute(img)
                sitk.WriteImage(img, os.path.join(dirs, 'Slice_' + idx + '_X_' + str_x + '_Y_' + str_y + '_.nii'))
                i += 1
        return True

    def apply_transform(self, xml_path, save_file, suffix=None, files=None):
        if files is not None:
            files = files
        elif suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(suffix))
        else:
            raise ValueError('Please select files or a file suffix')

        _, meta_info = file_io.read_nii(files[0])
        slices_dict = file_io.parse_path(files)
        _ = file_io.create_temp_dir(os.path.join(self.root_dir, 'temp_' + self.slice_id), self.slice_id,
                                   slices_dict[self.slice_id], self.dataset_args[4], self.dataset_args[5])
        fusion_interp, pixel_savetype, save_xystit_tif = self.fusion_args

        _, name = os.path.split(xml_path)
        new_dir = self.root_dir + '/temp_' + self.slice_id + '/' + name
        shutil.copyfile(xml_path, new_dir)
        self.fuse_save_by_macro(new_dir, self.root_dir + '/temp_' + self.slice_id, fusion_interp, pixel_savetype,
                                save_xystit_tif + '_slice_{}'.format(self.slice_id))
        out_file = glob.glob(self.root_dir + '/temp_' + self.slice_id + '/*.tif')[0]
        file_io.saveTif2Nifty(out_file, save_file, spacing=meta_info[0], orgin=meta_info[1])
        shutil.rmtree(self.root_dir + '/temp_' + self.slice_id)
        return out_file.replace('.tif', '.nii.gz')


class Z_Stitching(object):
    def __init__(self, ij, root_dir, slice_id, dataset_args, pairwise_args, opt_args, fusion_args):
        self.xml_path = None
        self.root_dir = root_dir
        self.slice_id = slice_id
        self.dataset_args = dataset_args
        self.pairwise_args = pairwise_args
        self.opt_args = opt_args
        self.fusion_args = fusion_args
        self.ij = ij

    def define_dataset_by_macro(self, xml_name, xml_save_dir, input_path, grid_type, min_memory_kb,
                                x_tiles, y_tiles, z_tiles, x_overlap, y_overlap, z_overlap):
        dataset_args = ["define_dataset=[Automatic Loader (Bioformats based)]",
                        "project_filename={}".format(xml_name),
                        "path={}".format(input_path),
                        "exclude={}".format(min_memory_kb),
                        "pattern_0=Tiles pattern_1=Tiles",
                        "move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)]",
                        "grid_type={}".format(grid_type),  # [Down & Right             ]
                        "tiles_x={} tiles_y={} tiles_z={} overlap_x_(%)={} overlap_y_(%)={} overlap_z_(%)={}".format(
                            x_tiles, y_tiles, z_tiles, x_overlap, y_overlap, z_overlap),
                        "keep_metadata_rotation how_to_load_images=[Load raw data]",
                        "dataset_save_path={} check_stack_sizes".format(xml_save_dir)
                        ]
        #     print("_".join(dataset_args))
        #     run("Define dataset ...", "_".join(dataset_args))
        msg = 'run("Define dataset ...", "{}");'.format(" ".join(dataset_args))
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Finished Define dataset ...')
        return msg

    def cal_pairwise_shifts_by_macro(self, xml_path, method="Phase Correlation", ds_x=2, ds_y=2, ds_z=1):
        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[{}] " \
                       "show_expert_algorithm_parameters downsample_in_x={} downsample_in_y={} downsample_in_z={} " \
                       "number_of_peaks_to_check=5 minimal_overlap=0 subpixel_accuracy".format(xml_path, method, ds_x,
                                                                                               ds_y, ds_z)
        msg = 'run("Calculate pairwise shifts ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Calculate pairwise shifts ...Done')
        return msg

    def glo_opt_by_macro(self, xml_path, method, fix_id):
        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 " \
                       "absolute=3.500 global_optimization_strategy=[{}] " \
                       "show_expert_grouping_options how_to_treat_timepoints=[treat individually] " \
                       "how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat " \
                       "individually] how_to_treat_tiles=compare fix_group_0-{}".format(xml_path, method, fix_id)
        msg = 'run("Optimize globally and apply shifts ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print('==>Optimize globally and apply shifts ...Done')
        return msg

    def fuse_save_by_macro(self, xml_path, out_dir, fusion_interp, pixel_savetype, file_name, blend=' blend'):

        dataset_args = "select={} process_angle=[All angles] process_channel=[All channels] process_illumination=[All " \
                       "illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[" \
                       "Currently Selected Views] downsampling=1 pixel_type=[{}] interpolation=[{}]" \
                       " image=[Precompute Image] interest_points_for_non_rigid=[-= Disable Non-Rigid =-]" \
                       "{} produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] " \
                       "output_file_directory={}/. filename_addition={}".format(xml_path, pixel_savetype, fusion_interp,
                                                                                blend,
                                                                                out_dir, file_name)
        msg = 'run("Fuse dataset ...", "{}");'.format(dataset_args)
        self.ij.py.run_macro("""{}""".format(msg))
        print(msg)
        print('==>Fuse dataset ...Done')
        return msg

    def stiching_main(self, root_dir, slice_num, dataset_args, pairwise_args, opt_args, fusion_args):
        xml_save_dir = root_dir + "/temp_{}".format(slice_num)
        input_path = xml_save_dir
        [temp_file_postfix, overlap_x, overlap_y, overlap_z, tile_num_x, tile_num_y, tile_num_z, grid_type,
         xml_save_path] = dataset_args
        xml_name = xml_save_path + '_dataset_{}.xml'.format(slice_num)
        print(xml_name)

        define_msg = self.define_dataset_by_macro(xml_name, xml_save_dir, input_path, grid_type, 10,
                                                  tile_num_x, tile_num_y, tile_num_z, overlap_x, overlap_y, overlap_z)

        xml_path = '{}/{}'.format(xml_save_dir, xml_name)
        xy_pairwise_method, xy_pairwise_dx, xy_pairwise_dy, xy_pairwise_dz = pairwise_args
        self.cal_pairwise_shifts_by_macro(xml_path, method=xy_pairwise_method,
                                          ds_x=xy_pairwise_dx, ds_y=xy_pairwise_dy, ds_z=xy_pairwise_dz)

        opt_method, opt_fixed_id = opt_args
        self.glo_opt_by_macro(xml_path, opt_method, fix_id=opt_fixed_id)

        fusion_interp, pixel_savetype, save_xystit_tif = fusion_args
        print("xml_path:", xml_path)
        self.fuse_save_by_macro(xml_path, input_path, fusion_interp, pixel_savetype,
                                'temp_slice_{}'.format(slice_num))
        self.xml_path = xml_path
        print(define_msg)

        return glob.glob(input_path + '/*.tif')[0], xml_name

    def hierarchical_run(self, img_suffix=None, files=None, stitch_cell_x=4, stitch_cell_y=4):
        if files is not None:
            files = files
        elif img_suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
        else:
            raise ValueError('Please select files or a file suffix')

        dirs = os.path.join(self.root_dir, 'temp_' + self.slice_id)
        if not os.path.isdir(dirs):
            os.makedirs(dirs)

        for f in files:
            _, f_name = os.path.split(f)
            if f_name[-len(".gz"):] == ".gz":
                temp, meta = file_io.read_nii(f)
                file_io.save_nii(temp, os.path.join(dirs, f_name.replace(".gz", "")), meta)
            else:
                shutil.copyfile(f, os.path.join(dirs, f_name))

    def run(self, img_suffix=None, files=None):
        if files is not None:
            files = files
        elif img_suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(img_suffix))
        else:
            raise ValueError('Please select files or a file suffix')

        dirs = os.path.join(self.root_dir, 'temp_' + self.slice_id)
        if not os.path.isdir(dirs):
            os.makedirs(dirs)

        for f in files:
            _, f_name = os.path.split(f)
            if f_name[-len(".gz"):] == ".gz":
                temp, meta = file_io.read_nii(f)
                file_io.save_nii(temp, os.path.join(dirs, f_name.replace(".gz", "")), meta)
            else:
                shutil.copyfile(f, os.path.join(dirs, f_name))

        # slices_dict = file_io.parse_path(files)
        _, meta_info = file_io.read_nii(files[0])
        out_file, xml_name = self.stiching_main(self.root_dir, self.slice_id, self.dataset_args,
                                                self.pairwise_args, self.opt_args, self.fusion_args)
        dir_name, file_name = os.path.split(out_file)
        file_io.saveTif2Nifty(out_file, os.path.join(dir_name, self.fusion_args[-1]),
                              spacing=meta_info[0], orgin=meta_info[1])
        shutil.copyfile(os.path.join(dir_name, self.fusion_args[-1]),
                        os.path.join(self.root_dir, self.fusion_args[-1]))
        shutil.copyfile(os.path.join(dir_name, xml_name), os.path.join(self.root_dir, xml_name))
        shutil.rmtree(dir_name)
        return os.path.join(self.root_dir, xml_name), os.path.join(self.root_dir, self.fusion_args[-1])

    def apply_z_transform(self, xml_path, save_file, suffix=None, files=None, modality=None):
        if files is not None:
            files = files
        elif suffix is not None:
            files = glob.glob(self.root_dir + '/*{}'.format(suffix))
        else:
            raise ValueError('Please select files or a file suffix')
        print(files)

        _, meta_info = file_io.read_nii(files[0])
        if not os.path.isdir(os.path.join(self.root_dir, 'temp_' + self.slice_id)):
            os.makedirs(os.path.join(self.root_dir, 'temp_' + self.slice_id))
        for f in files:
            _, name = os.path.split(f)
            new_name = "stitched_{}_n4_corr_auto.nii".format(name.split("_")[-1][:-4])
            ####################################################################
            if modality == "axisorient":
                arr, meta = file_io.read_nii(f)
                print(np.min(arr), np.max(arr))
                arr += (np.pi / 2)
                file_io.save_nii(arr, os.path.join(self.root_dir, 'temp_' + self.slice_id, new_name), meta)
            elif modality == "retardance":
                arr, meta = file_io.read_nii(f)
                print(np.min(arr), np.max(arr))
                norm = (((arr - 0) / (np.pi / 2 - 0)) * 255).astype("uint8")
                file_io.save_nii(norm, os.path.join(self.root_dir, 'temp_' + self.slice_id, new_name), meta)
            else:
                shutil.copyfile(f, os.path.join(self.root_dir, 'temp_' + self.slice_id, new_name))
            ############################################################
            # shutil.copyfile(f, os.path.join(self.root_dir, 'temp_' + self.slice_id, new_name))
        fusion_interp, pixel_savetype, save_xystit_tif = self.fusion_args
        _, name = os.path.split(xml_path)
        new_dir = self.root_dir + '/temp_' + self.slice_id + '/' + name
        shutil.copyfile(xml_path, new_dir)
        self.fuse_save_by_macro(new_dir, self.root_dir + '/temp_' + self.slice_id, fusion_interp, pixel_savetype,
                                save_xystit_tif + '_slice_{}'.format(self.slice_id))
        out_file = glob.glob(self.root_dir + '/temp_' + self.slice_id + '/*.tif')[0]
        file_io.saveTif2Nifty(out_file, save_file, spacing=meta_info[0], orgin=meta_info[1])
        if modality == "axisorient":
            arr, meta = file_io.read_nii(save_file)
            arr -= (np.pi / 2)
            file_io.save_nii(arr, save_file, meta)
        elif modality == "retardance":
            arr, meta = file_io.read_nii(save_file)
            arr = (arr / 255) * (np.pi / 2 - 0) + 0
            file_io.save_nii(arr, save_file, meta)
        shutil.rmtree(os.path.join(self.root_dir, 'temp_' + self.slice_id))
        return out_file.replace('.tif', '.nii.gz')


# if __name__ == '__main__':
#     import glob, os, shutil, math
#     import numpy as np
#     import SimpleITK as sitk
#     import sys, glob, os
#
#
#     from utils import file_io
#     alloc_memory = '64g'
#     ij = init_imagej(alloc_memory)
#
#     save_root = r'/path/to/processing'
#     slice_id = "0007"
#     dowmsampled_save_suffix = 'x20.nii'
#
#     files = glob.glob(os.path.join(save_root, 'Slice_{}_*.oct_intensity{}'.format(slice_id, dowmsampled_save_suffix)))
#
#     slices_dict = {}
#     for p in files:
#         dirs_n, file_n = os.path.split(p)
#         temp = file_n.split('_')
#         slices = temp[1]
#         tiles = temp[2]
#         coordinate = [float(temp[4]), float(temp[6]), float(temp[8])]
#         if slices not in slices_dict:
#             slices_dict[slices] = [(p, coordinate)]
#         else:
#             slices_dict[slices].append((p, coordinate))
#
#     print(slices_dict)
#     for i in slices_dict:
#         temp2 = slices_dict[i]
#         temp2 = sorted(temp2, key=lambda x: (x[1][0], x[1][1]), reverse=False)
#         slices_dict[i] = temp2
#
#     macro_input = []  # *.nii
#     for i in slices_dict:
#         temp3 = slices_dict[i]
#         temp3 = sorted(temp3, key=lambda x: (x[1][0], x[1][1]), reverse=False)
#         slices_dict[i] = temp3
#
#     idx, save_datas, x_num, y_num = slice_id, slices_dict[slice_id], 19, 24
#     stitch_cell_x = 4
#     stitch_cell_y = 5
#     global_x_num = math.ceil(x_num / stitch_cell_x)
#     global_y_num = math.ceil(y_num / stitch_cell_y)
#
#     i = 0
#     stitch_cell_x = 4
#     stitch_cell_y = 5
#     global_x_num = math.ceil(x_num / stitch_cell_x)
#     global_y_num = math.ceil(y_num / stitch_cell_y)
#
#     global_x = 0
#     global_y = 0
#
#     y_enough = False
#     x_enough = False
#
#     for x in range(x_num):
#         for y in range(y_num):
#             # dir_n, file_n = os.path.split(save_datas[i][0])
#             if x < 10:
#                 str_x = "0" + str(x)
#             else:
#                 str_x = str(x)
#             if y < 10:
#                 str_y = "0" + str(y)
#             else:
#                 str_y = str(y)
#
#             temp_dirs = os.path.join(save_root, 'temp_{}_{}_'.format(str(x//stitch_cell_x), str(y//stitch_cell_y)) + slice_id)
#             if not os.path.isdir(temp_dirs):
#                 os.makedirs(temp_dirs)
#             shutil.copyfile(save_datas[i][0],
#                             os.path.join(temp_dirs, 'Slice_' + idx + '_X_' + str_x + '_Y_' + str_y + '_.nii'))
#
#             i += 1
#
#     final_stitch_dir = os.path.join(save_root, "whole_temp_{}".format(slice_id))
#     if not os.path.isdir(final_stitch_dir):
#         os.makedirs(final_stitch_dir)
#     for x in range(global_x_num):
#         for y in range(global_y_num):
#             temp_dirs = os.path.join(save_root, 'temp_{}_{}_'.format(str(x), str(y)) + slice_id)
#             print(temp_dirs)
#
#             # root_dir = temp_dirs
#             if x == (global_x_num-1):
#
#                 dataset_args = ['', 20, 20, 4, 5, '[Up & Right]',
#                                 'temp_{}_{}_{}'.format(str(x), str(y), slice_id)]  # [Up & Left]
#             # 10x10x10
#             pairwise_args = 'Phase Correlation', 2, 2, 2
#             # raw
#             opt_args = 'Two-Round using Metadata to align unconnected Tiles', 12  # fixed tile used during stitching; adjust as needed
#             fusion_args = 'Linear Interpolation', '32-bit floating point', 'stitched_{}.nii'.format(
#                 'temp_{}_{}_{}'.format(str(x), str(y), slice_id))
#
#             big_imag_macro1 = XY_Stitching(ij, save_root, slice_id, dataset_args, pairwise_args, opt_args,
#                                                         fusion_args, alloc_memory)
#
#             # if not os.path.isfile(stit_file_lr):
#             xml_file, stit_file_lr = big_imag_macro1.just_run(save_root, '{}_{}_{}'.format(str(x), str(y), slice_id),
#                                                               dataset_args, pairwise_args, opt_args, fusion_args,
#                                                               img_suffix=None, files=files)
#             shutil.copyfile(os.path.join(temp_dirs, 'stitched_{}.nii'.format('temp_{}_{}_{}'.format(str(x), str(y), slice_id))),
#                             os.path.join(final_stitch_dir, 'stitched_{}.nii'.format('Slice_{}_X_{}_Y_{}_.nii'.format(slice_id, str(x), str(y)))))
#             shutil.rmtree(temp_dirs)
#             break
#             pass
#
#     # root_dir = temp_dirs
#     dataset_args = ['', 20, 20, 5, 5, '[Up & Right]',
#                     'whole_{}'.format(slice_id)]  # [Up & Left]
#     # 10x10x10
#     pairwise_args = 'Phase Correlation', 2, 2, 1
#     # raw
#     opt_args = 'Two-Round using Metadata to align unconnected Tiles', 12  # fixed tile used during stitching; adjust as needed
#     fusion_args = 'Linear Interpolation', '32-bit floating point', 'stitched_whole_{}.nii'.format(
#         'final_temp_{}'.format(slice_id))
#
#     big_imag_macro1 = XY_Stitching(ij, save_root, slice_id, dataset_args, pairwise_args, opt_args, fusion_args, alloc_memory)
#
#     # if not os.path.isfile(stit_file_lr):
#     xml_file, stit_file_lr = big_imag_macro1.just_run(save_root, 'whole_{}'.format(slice_id),
#                                                       dataset_args, pairwise_args, opt_args, fusion_args,
#                                                       img_suffix=None, files=files)


# class hierarchical_Stit(object):


if __name__ == '__main__':
    import argparse, time

    parser = argparse.ArgumentParser(description='Image Stitching by BigStitcher')
    parser.add_argument('-img_path', '--img_path', default=r'../demo/data', help='working directory')
    parser.add_argument('-suffix', '--suffix', default=r'.oct_intensityx20.nii',
                        help='input file suffix')
    parser.add_argument('-slice_id', '--slice_id', default=r'0009',
                        help='section to stitch')
    parser.add_argument('-overlap', '--overlap', default=[20, 20, 0],
                        help='overlap ratios in x, y, and z')
    parser.add_argument('-tiles_num', '--tiles_num', default=[4, 5, 1],  # [19, 24, 1],
                        help='numbers of tiles in x, y, and z')
    parser.add_argument('-grid_type', '--grid_type', default='[Up & Right]',
                        help='grid layout')
    parser.add_argument('-stit_med', '--stit_med', default='Phase Correlation',
                        help='alignment method for adjacent tiles')
    parser.add_argument('-edge_img', '--edge_img', default=False,
                        help='use a Sobel gradient image to estimate the optimal displacement for each tile')
    parser.add_argument('-ds_ratio', '--ds_ratio', default=[2, 2, 1],
                        help='downsampling ratios')
    parser.add_argument('-opt_med', '--opt_med', default='Two-Round using Metadata to align unconnected Tiles',
                        help='global optimization method')
    parser.add_argument('-fix_id', '--fix_id', default=12,  # 228,
                        help='ID of the fixed tile')
    parser.add_argument('-save_name', '--save_name', default='stitched_0009_shading.nii.gz',
                        help='final output filename')
    parser.add_argument('-alloc_memory', '--alloc_memory', default='64g',
                        help='maximum memory allocation')
    arguments = parser.parse_args()

    root_dir1 = arguments.img_path
    suffix = arguments.suffix
    slice_id1 = arguments.slice_id
    alloc_memory = arguments.alloc_memory
    overlap = arguments.overlap
    tiles_num = arguments.tiles_num
    grid_type = arguments.grid_type
    stit_med = arguments.stit_med
    ds_ratio = arguments.ds_ratio
    opt_med = arguments.opt_med
    fix_id = arguments.fix_id
    save_name = arguments.save_name
    is_edge = arguments.edge_img

    st = time.time()
    dataset_args1 = ['', overlap[0], overlap[1], tiles_num[0], tiles_num[1], grid_type, slice_id1]  # Up & Left
    pairwise_args1 = stit_med, ds_ratio[0], ds_ratio[1], ds_ratio[2]
    opt_args1 = opt_med, fix_id
    fusion_args1 = 'Linear Interpolation', '32-bit floating point', save_name

    ij = init_imagej(alloc_memory)

    big_imag_macro1 = XY_Stitching(ij, root_dir1, slice_id1, dataset_args1, pairwise_args1, opt_args1, fusion_args1)
    if is_edge:
        xml_file, file_dir = big_imag_macro1.run_quick(img_suffix=suffix, edge=True)
        files = glob.glob(file_dir + '/*{}'.format(suffix))
        apply2high_res.apply_to_highres(xml_file, files, dataset_args1,
                                        file_dir + "/{}".format(save_name),
                                        slice_id1, padding=20, shading=None, is_norm=False,
                                        blending_med="alpha")
    # big_imag_macro1.run(suffix)
    print(time.time() - st)
