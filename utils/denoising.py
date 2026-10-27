# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : denoising.py
# Time       :2023/3/29 14:01
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import SimpleITK as sitk
import bm3d
import skimage.restoration as sci_re
import numpy as np
from utils import file_io


def smooth_by_gaussian(img, sigma=1.97):
    img1 = sitk.GetImageFromArray(img)
    smoother = sitk.SmoothingRecursiveGaussianImageFilter()
    smoother.SetSigma(sigma)
    out = smoother.Execute(img1)
    out = sitk.GetArrayFromImage(out)
    return out


def smooth_by_mean(img, radius=5):
    sitk_mean = sitk.MeanImageFilter()
    sitk_mean.SetRadius(radius)
    sitk_mean = sitk_mean.Execute(img)
    out = sitk.GetArrayFromImage(sitk_mean)
    return out


def smooth_by_median(img, radius=5):
    sitk_mean = sitk.MedianImageFilter()
    sitk_mean.SetRadius(radius)
    sitk_mean = sitk_mean.Execute(img)
    out = sitk.GetArrayFromImage(sitk_mean)
    return out


def bm3d_speckle(img, sigma_psd=None):
    if sigma_psd is None:
        sigma_psd = sci_re.estimate_sigma(img)
        print("Using estimated sigma {}".format(sigma_psd))
    res = bm3d.bm3d(img, sigma_psd)
    return res


def HistogramEqual(image):
    sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
    sitk_hisequal.SetAlpha(0.9)
    sitk_hisequal.SetBeta(0.9)
    sitk_hisequal.SetRadius(3)
    sitk_hisequal = sitk_hisequal.Execute(image)
    return sitk_hisequal
    # sitk.WriteImage(sitk_hisequal, "sitk_hisequal.mha")


def nlm_speckle(img, patch_size=7, patch_distance=11, h=None):
    # h is the filtering cutoff in intensity units. Larger values accept more dissimilar patches and produce smoother but potentially blurrier images. For Gaussian noise with standard deviation sigma, a common choice is h slightly below sigma.
    if h is None:
        h = sci_re.estimate_sigma(img) * 0.5
        print("Using estimated h {}".format(h))
    denoised_img = sci_re.denoise_nl_means(img, patch_size, patch_distance=patch_distance, h=h, fast_mode=True)
    return denoised_img


def IndexLowPass(h, w, z, d0, n=2):
    mid_x = int(w / 2)
    mid_y = int(h / 2)
    mid_z = int(z / 2)
    grid_x = np.arange(w)
    grid_y = np.arange(h)
    grid_z = np.arange(z)
    [X, Y, Z] = np.meshgrid(grid_x, grid_y, grid_z)
    grid = np.stack([X, Y, Z])
    d = np.sqrt((grid[0, ...] - mid_x) ** 2 + (grid[1, ...] - mid_y) ** 2 + (grid[2, ...] - mid_z) ** 2)
    H = np.e ** (-(d / d0) ** n)
    return H


def LowPass(h, w, z, d0, n=2):
    H = np.zeros(shape=[h, w, z], dtype=float)
    mid_x = int(w / 2)
    mid_y = int(h / 2)
    mid_z = int(z / 2)
    grid_x = np.arange(w)
    grid_y = np.arange(h)
    grid_z = np.arange(z)
    [X, Y, Z] = np.meshgrid(grid_x, grid_y, grid_z)
    grid = np.stack([X, Y, Z])
    d = np.sqrt((grid[0, ...] - mid_x) ** 2 + (grid[1, ...] - mid_y) ** 2 + (grid[2, ...] - mid_z) ** 2)
    H[d <= d0] = 1
    return H


def frequence_pass(img, d0, save_fft=False):
    # print(np.min(img))
    # print(np.max(img))
    res = np.fft.fftn(img)
    xx, yy, zz = img.shape
    filter = IndexLowPass(xx, yy, zz, d0, n=2)

    unit_fftshift = np.fft.fftshift(res)
    img_fft_abs = None
    if save_fft:
        img_fft_abs = 10 * np.log(np.abs(unit_fftshift))

    unit_fftshift = unit_fftshift * filter  # Apply filtering

    unit_fft2_2 = np.fft.ifftshift(unit_fftshift)
    unit_handle = np.fft.ifftn(unit_fft2_2)
    unit_img = np.array(np.real(unit_handle))
    # unit_img[unit_img > 255] = 255
    # unit_img[unit_img < 0] = 0
    return unit_img, img_fft_abs


class denoise_oct(object):
    def __init__(self, file_name, save_suffix, method='median'):
        self.file_name = file_name
        self.save_suffix = save_suffix
        self.med = method

    def run(self):
        img_arr, img_info = file_io.read_nii(self.file_name)
        print(img_arr.shape)
        if self.med == 'bm3d':
            denoised_img = bm3d_speckle(img_arr)
        elif self.med == 'nlm':
            denoised_img = nlm_speckle(img_arr)
        elif self.med == 'median':
            img = sitk.GetImageFromArray(img_arr)
            denoised_img = smooth_by_median(img, radius=2)
        elif self.med == 'frequence_pass':
            img = sitk.GetImageFromArray(img_arr)
            denoised_img = smooth_by_median(img, radius=2)
        else:
            raise ValueError('Undefined denoising method')
        file_io.save_nii(denoised_img, self.file_name.replace('.nii', self.save_suffix), img_info)


if __name__ == '__main__':
    import argparse, time

    parser = argparse.ArgumentParser(description='Denoise')
    parser.add_argument('-img_path', '--img_path', default=r'../demo/data/stitched_0000.nii.gz', help='working directory')
    parser.add_argument('-save_suffix', '--save_suffix', default=r'_median.nii',
                        help='input file suffix')
    parser.add_argument('-method', '--method', default=r'median',
                        help='method')

    arguments = parser.parse_args()
    img_path = arguments.img_path
    save_suffix = arguments.save_suffix
    method = arguments.method
    st = time.time()
    model = denoise_oct(img_path, save_suffix, method)
    model.run()
    print(time.time() - st)
