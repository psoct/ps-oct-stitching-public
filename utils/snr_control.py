# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : snr_control.py
# Time       :2023/3/29 16:41
# Author     :Bot Zhao
# version    :python 3.9
# Description:
"""

# import libs
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np


def get_SNR(img_arr, mask_arr, show_dist=False):
    signal = img_arr[mask_arr == 1]
    noise = img_arr[mask_arr == 2]

    if show_dist:
        plt.plot(range(noise.shape[0]), noise)
        plt.show()

        plt.hist(noise, bins=100,
                 density=True,
                 histtype='stepfilled',
                 color='red',
                 alpha=0.3,
                 label='Histogram',
                 )

        model = KernelDensity(bandwidth=1.0, kernel='gaussian')
        model.fit(noise[:, np.newaxis])
        x_range = np.linspace(noise.min() - 1, noise.max() + 1, len(noise))
        x_log_prob = model.score_samples(x_range[:, np.newaxis])  # Returns the log probability density
        x_prob = np.exp(x_log_prob)

        plt.plot(x_range, x_prob, color='gray')
        plt.show()
    snr = (np.mean(signal) - np.mean(noise)) / np.std(signal)
    return snr, [np.mean(signal), np.std(signal), np.mean(noise), np.std(noise)]
