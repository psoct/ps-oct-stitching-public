# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : img_process.py
# Time       :2022/8/26 13:56
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

import glob
import os
import pickle
import sys
from pathlib import Path

# import libs
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.restoration as sci_re
import tqdm
from skimage import measure as ms

sys.path.append('../')
# from utils.file_io import *
from segment_anything import SamPredictor, sam_model_registry
from utils import file_io, image_utils


def check_info_by_z(new_arr):
    avg_intensity = []
    max_intensity = []
    var_intensity = []
    for i in range(new_arr.shape[0]):
        temp = new_arr[i, ...]
        avg_intensity.append(np.mean(temp))
        max_intensity.append(np.max(temp))
        var_intensity.append(np.std(temp))
    plt.plot(range(0, new_arr.shape[0]), avg_intensity)
    plt.plot(range(0, new_arr.shape[0]), max_intensity)
    plt.plot(range(0, new_arr.shape[0]), var_intensity)
    plt.show()


def delete_zero_plane(arr):
    max_intensity = []
    for i in range(arr.shape[0]):
        temp = arr[i, ...]
        max_intensity.append(np.max(temp))
    index = np.nonzero(max_intensity)
    st, ed = np.min(index), np.max(index)
    return arr[st:ed + 1, ...]


def replace_empty_by_noise(arr1, mean=49.636, std=1.970):
    """
    replace_empty_by_noise
    :param arr1:
    :param mean: mean value of Gaussian noise
    :param std:
    :return:
    """
    nums = len(arr1[arr1 == 0])
    print('===> Number of Empty Position:', nums)
    gaussian_noise = np.random.normal(loc=mean, scale=std, size=nums)
    arr1[arr1 == 0] = gaussian_noise
    return arr1


def norm_2_interval(arr, minv=0, maxv=255):
    """
    IMAGE NORMALIZATION; bad implementation
    :param arr:
    :param minv:
    :param maxv:
    :return:
    """
    old_min, old_max = np.min(arr), np.max(arr)
    # arr[arr<0] = 0
    # arr[arr>255] = 255
    arr = (old_min - np.min(arr)) + ((arr - old_min) / (old_max - old_min)) * (maxv - minv)
    return arr, [old_min, old_max]


def otsu_binarization_3d(img, threshold=-1):
    """
    otsu FOR 3D image
    :param img:
    :param threshold:
    :return: BINARY IMAGE
    """
    out = img.copy()
    max_sigma = 0
    max_t = 0
    x, y, z = img.shape
    voxels = (x * y * z)
    for temp_t in tqdm.tqdm(range(1, 255)):
        v0 = out[np.where(out < temp_t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / voxels
        v1 = out[np.where(out >= temp_t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / voxels
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = temp_t
        threshold = max_t
    out[out < threshold] = 0
    out[out >= threshold] = 1
    print("Threshold >>", max_t)
    return out


def smooth_by_gaussian(img, sigma):
    img1 = sitk.GetImageFromArray(img)
    smoother = sitk.SmoothingRecursiveGaussianImageFilter()
    smoother.SetSigma(sigma)
    out = smoother.Execute(img1)
    out = sitk.GetArrayFromImage(out)
    return out


def otsu_binarization_sitk(img):
    """
    otsu FOR 3D image
    :param img:
    :return: BINARY IMAGE
    """
    img1 = sitk.GetImageFromArray(img)
    otsu_method = sitk.OtsuThresholdImageFilter()
    otsu_method.SetNumberOfHistogramBins(500)
    otsu_method.SetNumberOfThreads(8)
    _ = otsu_method.Execute(img1)
    print("===> Threshold: ", otsu_method.GetThreshold())
    binary_out = np.zeros_like(img)
    binary_out[img >= otsu_method.GetThreshold()] = 1
    return binary_out


def delete_tiny_noise(img, open_num=5, close_num=5):
    """
    open and close to remove little noise
    :param img:  Binary image
    :param open_num:
    :param close_num:
    :return:
    """
    img = sitk.GetImageFromArray(img)
    binary_arr_img = sitk.Cast(img, sitk.sitkUInt16)
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius(5)
    dilater = sitk.BinaryDilateImageFilter()
    dilater.SetKernelRadius(5)
    out = binary_arr_img

    for i in range(open_num):
        out = eroder.Execute(out)
        out = dilater.Execute(out)
    out = sitk.GetArrayFromImage(out)
    return out


def find_max_component(data):
    """
    contain the largest connected component .
    :param data: array 3d;
    :return:
    """
    # c = np.ones((3, 3, 3))
    #     dilation
    # data = morphology.erosion(data, c)
    #     find cc
    [heart_res, num] = ms.label(data, connectivity=1, background=0, return_num=True)
    areas = []
    region = ms.regionprops(heart_res)
    for i in range(num):
        areas.append(region[i].area)
    label_num = np.argsort(-np.array(areas))
    res = np.zeros_like(data)
    res[heart_res == (label_num[0] + 1)] = 1
    return res


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


def crop_img(img_arr, st, et):
    """
    crop img by voxel index;
    :param img_arr: img array;
    :param st: start point;
    :param et: end point;
    :return: new image
    """
    temp = img_arr.copy()
    return temp[st[0]:et[0], st[1]:et[1], st[2]:et[2]]


def draw_ratios(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(data)
    for i in data:
        plt.plot(range(len(i)), i)
    plt.title('ratios')
    plt.show()


def draw_sts(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(data)
    temp = []
    for i in data:
        temp.append(i[0])
    plt.plot(range(len(temp)), temp)
    plt.title('Start Slices')
    plt.xticks(range(len(temp)), range(5, 5 + len(temp)))
    plt.show()


class Segment_Crop(object):
    def __init__(self, data_dir, stop_z=None, crop_img_save_suffix=None,
                 seg_img_save_suffix=None, cropped_mask_save_suffix=None):
        self.binary_arr_wo_tinynoise = None
        self.sts = None
        self.binary_arr = None
        self.denoised_img = None
        self.res = None
        self.data_dir = data_dir
        self.stop_z = stop_z
        self.crop_img_save_suffix = crop_img_save_suffix
        self.seg_img_save_suffix = seg_img_save_suffix
        self.cropped_mask_save_suffix = cropped_mask_save_suffix

    def cal_pos_ratio(self, threshold=None):
        temp = []
        st, et = get_bbox(self.res)
        sts = []
        bbaray = crop_img(self.res, st, et)
        z, y, x = bbaray.shape
        for i in range(z):
            ratio = np.sum(bbaray[i, ...]) / (y * x)
            sts.append(i + st[0])
            temp.append(ratio)
        if threshold is not None:
            sts_index = np.where(np.array(temp) >= threshold)[0][0] + 1 + st[0]
        else:
            first_derivative = np.gradient(np.array(temp))
            sts_index = np.where(first_derivative == np.max(first_derivative))[0][0] + 3 + st[0]
        # plt.plot(range(z), temp)
        # plt.title('ratios')
        # plt.show()
        return [sts_index], temp

    def run(self):
        arr, meta_info = file_io.read_nii(self.data_dir)

        # prepare
        est_sigma = sci_re.estimate_sigma(arr)
        est_mean = np.mean(arr.flatten()).item()
        print('===> Estimated mean&sigma: {},{}'.format(est_mean, est_sigma))
        add0_img = replace_empty_by_noise(arr, mean=est_mean, std=est_sigma)

        # denoise
        self.denoised_img = smooth_by_gaussian(add0_img, est_sigma)

        # norm
        norm_arr, _ = norm_2_interval(self.denoised_img, minv=0, maxv=255)

        # binary
        self.binary_arr = otsu_binarization_sitk(norm_arr)

        # tiny_noise
        self.binary_arr_wo_tinynoise = delete_tiny_noise(self.binary_arr)

        # max_component
        self.res = find_max_component(self.binary_arr_wo_tinynoise)

        # Get start point by ratio
        sts, ratios = self.cal_pos_ratio()
        self.sts = sts
        print('===> Estimated START SLICE: {}'.format(sts[0]))

        if self.stop_z is None:
            self.stop_z = self.binary_arr.shape[0] - sts[0]

        if self.cropped_mask_save_suffix is not None:
            img_cropped = self.denoised_img[sts[0]:sts[0] + self.stop_z, ...]
            file_io.save_nii(img_cropped, self.data_dir.replace('.nii', self.crop_img_save_suffix), meta_info)
        if self.seg_img_save_suffix is not None:
            file_io.save_nii(self.binary_arr_wo_tinynoise, self.data_dir.replace('.nii', self.seg_img_save_suffix),
                             meta_info)
        if self.cropped_mask_save_suffix is not None:
            binary_arr = self.binary_arr_wo_tinynoise[sts[0]:sts[0] + self.stop_z, ...]
            file_io.save_nii(binary_arr, self.data_dir.replace('.nii', self.cropped_mask_save_suffix), meta_info)

    # # QC show img
    def show(self):
        sts = self.sts
        plt.figure(figsize=(15, 15))
        plt.subplot(331), plt.imshow(self.binary_arr[sts[0], ...])
        plt.subplot(332), plt.imshow(self.binary_arr_wo_tinynoise[sts[0], ...])
        plt.subplot(333), plt.imshow(self.denoised_img[sts[0], ...])

        plt.subplot(334), plt.imshow(self.binary_arr[sts[len(sts) // 2], ...])
        plt.subplot(335), plt.imshow(self.binary_arr_wo_tinynoise[sts[len(sts) // 2], ...])
        plt.subplot(336), plt.imshow(self.denoised_img[sts[len(sts) // 2], ...])

        plt.subplot(337), plt.imshow(self.binary_arr[sts[-1], ...])
        plt.subplot(338), plt.imshow(self.binary_arr_wo_tinynoise[sts[-1], ...])
        plt.subplot(339), plt.imshow(self.denoised_img[sts[-1], ...])
        try:
            plt.savefig(self.data_dir.replace('.nii', '_QC_v2.png'))
        except Exception as e:
            plt.savefig(self.data_dir.replace('.nii.gz', '_QC_v2.png'))
        print('==> FINISHED {}'.format(self.data_dir))


class Sam_seg(object):
    def __init__(self, save_dir, st_crop=60, ed_crop=160, model_type="vit_h", gpus="0", device='cuda', evaluate=False,
                 sam_checkpoint=None, gt_dir=None):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        """
        Perform foreground segmentation with SAM.
        :param save_dir: segmentation output directory
        :param st_crop:  front crop position along z before stitching each section, in micrometers
        :param ed_crop: end crop position along z before stitching each section, in micrometers
        :param model_type: SAM model variant: vit_h, vit_l, or vit_b; vit_h has the largest parameter count
        :param gpus: GPU index
        :param device: compute device (GPU or CPU)
        :param evaluate: whether to evaluate segmentation results
        :param sam_checkpoint: SAM checkpoint path; defaults to checkpoints/sam under the project root
        :param gt_dir: ground-truth directory used for evaluation
        """
        self.device = device
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.save_dir = save_dir
        self.gt_dir = gt_dir

        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        if model_type not in checkpoint_names:
            raise ValueError("model_type must be one of: vit_h, vit_l, vit_b")

        if sam_checkpoint is None:
            project_root = Path(__file__).resolve().parents[1]
            sam_checkpoint = project_root / "checkpoints" / "sam" / checkpoint_names[model_type]
        else:
            sam_checkpoint = Path(sam_checkpoint).expanduser().resolve()

        if not sam_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_checkpoint}. "
                "Download it from the official links in README.md."
            )

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        self.predictor = SamPredictor(self.sam)
        self.eval = evaluate
        self.st_crop = st_crop
        self.ed_crop = ed_crop
        # input_point = np.array([[500, 375], [1125, 625]])
        input_label = np.array([1, 1])

    def run(self, files, input_box=None):
        files = sorted(files)
        # input_box = np.array([[430, 420, 852, 730]])
        if input_box == None:
            input_box = [[154, 189, 1148, 784]]
        input_label = np.array([1, ])

        res_pathes = []
        for path in tqdm.tqdm(files):
            img, meta_info = file_io.read_nii(path)
            img = img[self.st_crop:, ...]
            median = img.shape[0] // 2
            input_point = np.array([[img.shape[1] // 2, img.shape[2] // 2], ])

            res = []
            i = median
            while i >= 0:
                if i == median:
                    temp_arr = img[i, ...].astype('uint8')
                    temp_img = cv2.cvtColor(temp_arr, cv2.COLOR_GRAY2RGB)
                    # print(temp_img.shape)
                    self.predictor.set_image(temp_img)
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=np.array(input_box)[None, :],
                        multimask_output=False,
                    )
                    middle_logits = logits
                    middle_masks = masks
                else:
                    # plt.imshow(masks[0,...]); plt.show()
                    temp_arr2 = img[i, ...].astype('uint8')
                    temp_img2 = cv2.cvtColor(temp_arr2, cv2.COLOR_GRAY2RGB)
                    self.predictor.set_image(temp_img2)
                    mask_input = logits[0, :, :]
                    st, ed = image_utils.get_bbox(masks[0, ...])
                    input_box = [[min(st[0], input_box[0][0]), min(st[1] - 10, input_box[0][1]),
                                  max(ed[1] + 10, input_box[0][2]), max(ed[0] + 10, input_box[0][3])]]
                    print('slice {}:'.format(i), input_box)
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=np.array(input_box)[None, :],
                        mask_input=mask_input[None, :],
                        multimask_output=False,
                    )
                res.append(masks[0, ...])
                i -= 1
            res.reverse()

            logits = middle_logits
            masks = middle_masks
            i = median + 1
            while i < img.shape[0]:
                # plt.imshow(masks[0,...]); plt.show()
                temp_arr2 = img[i, ...].astype('uint8')
                temp_img2 = cv2.cvtColor(temp_arr2, cv2.COLOR_GRAY2RGB)
                self.predictor.set_image(temp_img2)
                mask_input = logits[0, :, :]
                st, ed = image_utils.get_bbox(masks[0, ...])
                input_box = [
                    [min(st[0], input_box[0][0]), min(st[1] - 10, input_box[0][1]), max(ed[1] + 10, input_box[0][2]),
                     max(ed[0] + 10, input_box[0][3])]]
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=np.array(input_box)[None, :],
                    mask_input=mask_input[None, :],
                    multimask_output=False,
                )
                res.append(masks[0, ...])
                i += 1
                print('slice {}:'.format(i), input_box)
            # res.reverse()
            res = np.stack(res, axis=0)
            # print(res.shape)
            st, ed = image_utils.get_bbox(middle_masks[0, ...])
            input_box = [
                [min(st[0], input_box[0][0]), min(st[1] - 10, input_box[0][1]), max(ed[1] + 10, input_box[0][2]),
                 max(ed[0] + 10, input_box[0][3])]]
            # file_io.save_nii(img[30:80, ...], path.replace('.nii.gz', '_v1.nii.gz'), meta_info)
            _, file_name = os.path.split(path)
            pred_path = os.path.join(self.save_dir, file_name.replace('.nii.gz', '_mask.nii.gz'))
            file_io.save_nii(res.astype('uint8'), pred_path, meta_info)
            res_pathes.append(pred_path)
            # img[res == 0] = 0
            # file_io.save_nii(img, os.path.join(self.save_dir, file_name.replace('.nii.gz', '_masked.nii.gz')), meta_info)

            if self.eval:
                collect_dice([pred_path], [pred_path.replace("/pred/", "/GT/").replace('.nii.gz', '_v2_masked.nii')])
        return res_pathes

    def run_onecase(self, files, input_box=None, max_bbox=False):
        files = sorted(files)
        # input_box = np.array([[430, 420, 852, 730]])
        if input_box == None:
            input_box = [[154, 189, 1148, 784]]  # max bbox

        input_label = np.array([1, ])

        path = files[0]
        img, meta_info = file_io.read_nii(path)
        img = img[self.st_crop:self.ed_crop, ...]
        median = img.shape[0] // 2
        input_point = np.array([[img.shape[1] // 2, img.shape[2] // 2], ])

        res = []
        i = median
        while i >= 0:
            if i == median:
                temp_arr = img[i, ...].astype('uint8')
                temp_img = cv2.cvtColor(temp_arr, cv2.COLOR_GRAY2RGB)
                # print(temp_img.shape)
                self.predictor.set_image(temp_img)
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=np.array(input_box)[None, :],
                    multimask_output=False,
                )
                middle_logits = logits
                middle_masks = masks
                st, ed = image_utils.get_bbox(masks[0, ...])
                input_box = [[st[1] - 10, st[0] - 10, ed[1] + 10, ed[0] + 10]]

            else:
                # plt.imshow(masks[0,...]); plt.show()
                temp_arr2 = img[i, ...].astype('uint8')
                temp_img2 = cv2.cvtColor(temp_arr2, cv2.COLOR_GRAY2RGB)
                self.predictor.set_image(temp_img2)
                mask_input = logits[0, :, :]
                st, ed = image_utils.get_bbox(masks[0, ...])
                if max_bbox:
                    input_box = [
                        [min(st[1] - 10, input_box[0][0]), min(st[0] - 10, input_box[0][1]),
                         max(ed[1] + 10, input_box[0][2]),
                         max(ed[0] + 10, input_box[0][3])]]

                else:
                    input_box = [[st[1] - 10, st[0] - 10, ed[1] + 10, ed[0] + 10]]

                print('slice {}:'.format(i), input_box)
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=np.array(input_box)[None, :],
                    mask_input=mask_input[None, :],
                    multimask_output=False,
                )
            res.append(masks[0, ...])
            i -= 1
        res.reverse()

        logits = middle_logits
        masks = middle_masks
        i = median + 1
        while i < img.shape[0]:
            # plt.imshow(masks[0,...]); plt.show()
            temp_arr2 = img[i, ...].astype('uint8')
            temp_img2 = cv2.cvtColor(temp_arr2, cv2.COLOR_GRAY2RGB)
            self.predictor.set_image(temp_img2)
            mask_input = logits[0, :, :]
            st, ed = image_utils.get_bbox(masks[0, ...])

            if max_bbox:
                input_box = [
                    [min(st[1] - 10, input_box[0][0]), min(st[0] - 10, input_box[0][1]),
                     max(ed[1] + 10, input_box[0][2]),
                     max(ed[0] + 10, input_box[0][3])]]

            else:
                input_box = [[st[1] - 10, st[0] - 10, ed[1] + 10, ed[0] + 10]]

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=np.array(input_box)[None, :],
                mask_input=mask_input[None, :],
                multimask_output=False,
            )
            res.append(masks[0, ...])
            i += 1
            print('slice {}:'.format(i), input_box)
        # res.reverse()
        res = np.stack(res, axis=0)
        # print(res.shape)
        st, ed = image_utils.get_bbox(middle_masks[0, ...])
        if max_bbox:
            input_box = [
                [min(st[1] - 10, input_box[0][0]), min(st[0] - 10, input_box[0][1]), max(ed[1] + 10, input_box[0][2]),
                 max(ed[0] + 10, input_box[0][3])]]

        else:
            input_box = [[st[1] - 10, st[0] - 10, ed[1] + 10, ed[0] + 10]]

        # file_io.save_nii(img[30:80, ...], path.replace('.nii.gz', '_v1.nii.gz'), meta_info)
        _, file_name = os.path.split(path)
        pred_path = os.path.join(self.save_dir, file_name.replace('.nii.gz', '_mask.nii.gz'))
        file_io.save_nii(res.astype('uint8'), pred_path, meta_info)
        # img[res == 0] = 0
        # file_io.save_nii(img, os.path.join(self.save_dir, file_name.replace('.nii.gz', '_masked.nii.gz')), meta_info)

        if self.eval:
            if self.gt_dir is None:
                raise ValueError("gt_dir is required when evaluate=True")
            gt_path = os.path.join(self.gt_dir, file_name.replace('.nii.gz', '_v2_masked.nii'))
            collect_dice([pred_path], [gt_path])
        return pred_path, input_box

    def run_new(self, files, input_box=None):
        pass


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


def apply_mask2other(mask_path, img_path, save_dir, st_crop):
    img_arr, meta = file_io.read_nii(img_path)
    lr_mask_img = sitk.ReadImage(mask_path)
    _, name = os.path.split(img_path)
    if meta[0] != lr_mask_img.GetSpacing():
        img = sitk.ReadImage(img_path)

        # up-sampling mask
        meta_hr = img.GetSpacing(), img.GetOrigin(), img.GetDirection()
        new_size = (img.GetSize()[1], img.GetSize()[0],
                    int(lr_mask_img.GetSize()[2] * lr_mask_img.GetSpacing()[2] / img.GetSpacing()[2]))
        hr_mask_img = image_utils.sample_2_fixed_size(lr_mask_img, new_size)

        # apply to img
        img_arr = sitk.GetArrayFromImage(img)
        hr_mask_arr = np.transpose(sitk.GetArrayFromImage(hr_mask_img), (0, 2, 1))
        st_slice_hr = int(st_crop * lr_mask_img.GetSpacing()[2] / meta_hr[0][2])
        img_arr = img_arr[st_slice_hr:st_slice_hr + hr_mask_arr.shape[0], ...]
        img_arr[hr_mask_arr == 0] = 0
        file_io.save_nii(img_arr, save_dir, meta_hr)
    else:
        lr_mask_arr = sitk.GetArrayFromImage(lr_mask_img)
        img_arr = img_arr[st_crop:st_crop + lr_mask_arr.shape[0], ...]
        file_io.save_nii(img_arr, save_dir.replace("attn_corr", "cropped"), meta)
        img_arr[lr_mask_arr == 0] = 0
        file_io.save_nii(img_arr, save_dir, meta)

    # hr


def batch_segment(test_slice, data_root, postfix, save_dir_lr_mask, save_dir_lr, st_crop, ed_crop,
                  input_box):
    """
    Segment all sections in one dataset.
    :param test_slice: section IDs as a list
    :param data_root: data root directory
    :param save_dir_lr_mask: output directory for segmentation masks
    :param save_dir_lr: output directory for masked images
    :param st_crop: front crop position along z before stitching each section, in micrometers
    :param ed_crop: end crop position along z before stitching each section, in micrometers
    :param input_box: initial bounding box, for example[[110, 110, 1190, 900]]
    :return: None
    """
    iid = len(test_slice) // 2
    old_bbox = input_box
    while iid >= 0:
        slice_id = test_slice[iid]
        if not os.path.isdir(save_dir_lr_mask):
            os.makedirs(save_dir_lr_mask)
        if not os.path.isdir(save_dir_lr.replace("attn_corr", "cropped")):
            os.makedirs(save_dir_lr.replace("attn_corr", "cropped"))

        stit_file_lr = os.path.join(data_root, '{}_{}.nii'.format(postfix, slice_id))
        files = [stit_file_lr]

        _, name = os.path.split(stit_file_lr)
        SAM_model = Sam_seg(save_dir=save_dir_lr_mask, st_crop=st_crop, ed_crop=ed_crop, gpus="0",
                            model_type="vit_h", device='cuda', evaluate=False)
        mask_path, input_box = SAM_model.run_onecase(files, input_box=input_box, max_bbox=True)

        if not os.path.isdir(save_dir_lr):
            os.makedirs(save_dir_lr)
        apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        iid -= 1

    iid = (len(test_slice) // 2)
    input_box = old_bbox
    print("input_box", input_box)
    while iid < len(test_slice):
        slice_id = test_slice[iid]
        if not os.path.isdir(save_dir_lr_mask):
            os.makedirs(save_dir_lr_mask)
        stit_file_lr = os.path.join(data_root, 'stitched_{}.nii'.format(slice_id))
        files = [stit_file_lr]
        _, name = os.path.split(stit_file_lr)
        SAM_model = Sam_seg(save_dir=save_dir_lr_mask, st_crop=st_crop, ed_crop=ed_crop, gpus="0",
                            model_type="vit_h", device='cuda', evaluate=False)
        mask_path, input_box = SAM_model.run_onecase(files, input_box=input_box, max_bbox=True)
        if not os.path.isdir(save_dir_lr):
            os.makedirs(save_dir_lr)
        apply_mask2img(mask_path, stit_file_lr, save_dir_lr, st_crop)
        iid += 1


def cal_dice(arr1, arr2):
    arr1[arr1 != 0] = 1
    arr2[arr2 != 0] = 1
    arr1[arr1 != 1] = 0
    arr2[arr2 != 1] = 0
    inter = np.sum(arr1 * arr2) * 2
    union = np.sum(arr1) + np.sum(arr2)

    return inter / union
    # pass


def collect_dice(files_pred, files_gt):
    files_pred = sorted(files_pred)
    files_gt = sorted(files_gt)
    print(len(files_gt))
    dices = []
    for pred, gt in zip(files_pred, files_gt):
        print(pred, gt)
        pred_arr, _ = file_io.read_nii(pred)
        gt_arr, _ = file_io.read_nii(gt)
        dice_v = cal_dice(pred_arr[:100, ...], gt_arr)
        dices.append(dice_v)
        print(dice_v)
        pass
    return dices
    pass


