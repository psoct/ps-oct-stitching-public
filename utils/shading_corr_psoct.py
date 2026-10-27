#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flat-field correction utilities for PS-OCT images."""

import numpy as np

from utils import denoising, file_io


def cal_flat_file(flat_file, save_file, contain_info=6, crop_z=0):
    """Create a normalized, low-pass-filtered flat-field image."""
    flat_arr, meta_info = file_io.read_nii(flat_file)
    flat_arr = flat_arr[crop_z:, ...]
    flat_arr = flat_arr / np.mean(flat_arr)
    flat_arr, _ = denoising.frequence_pass(flat_arr, contain_info)
    file_io.save_nii(flat_arr, save_file, meta_info)
    return save_file


class Flat_correction(object):
    """Apply a precomputed flat-field image to an image array."""

    def __init__(self, flat_file):
        flat_arr, meta_info = file_io.read_nii(flat_file)
        self.flat_arr = flat_arr
        self.meta_info = meta_info

    def flat_corr(self, img, save_dir=None):
        corrected = img / self.flat_arr
        if save_dir is not None:
            file_io.save_nii(corrected, save_dir, self.meta_info)
        return corrected
