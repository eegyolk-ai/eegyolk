# -*- coding: utf-8 -*-

# historical functions from Bjorn Bruns et al.  Refined by wvxvw.
# Notes:
#
# I (wvxvw) found out that two functions were essentially duplicates:
# read_cnt_file() was repeated twice with very small variation and
# select_bad_epochs_list() is just a more general version of
# select_bad_epochs() (works on a list rather than on a single
# stimulus.)  Therefore the duplicates have been removed.  Next, I
# rewrote read_cnt_file() as CntReader class with read() method in
# place of the old function.  My goal in rewriting the function was to
# separate constant, one-time and repeated code.  The constant
# information was rewritten as class properties, one-time code was
# rewritten as class initialization code and the rest made the read()
# method.
#
# My understanding so far is that the main goal of read_cnt_file()
# function was to find bad data (channels or epochs) in a particular
# EEG, and later replace it with fictitious data based on nearby
# channels.  The function also tried to record the information about
# what was cleared, but this information doesn't seem to be very
# useful, as it is not connected to the "event ids", which is my
# biggest source of confusion in this rewrite.  My guess is that the
# original author might have himself not clearly understood what
# exactly was this function collecting.  Also, it would be useful to
# find out whether that information about data cleaning was ever used
# for anything.  Perhaps, it was only displayed to the user, without
# affecting any further calcuations, in which case, it's probably safe
# do remove it altogether (as it stands, it is more confusing than
# helping).

"""
Helper functions to work with ePODIUM EEG data

"""
import h5py
import logging
import os
import re
import warnings

from glob import glob

import mne
import numpy as np
import pandas as pd


def select_bad_channels(data_raw, time=100, threshold=5, include_for_mean=0.8):
    '''Historical function to find suspect channels --> still might
    need manual inspection!  This function was written and used by
    Bjorn Bruns, Florian HUber and other collaboartorson the earlier
    work to select bad channels.  It is slightly rewritten.

    Args:
    --------
    data_raw: mne object

    time: int
    Time window to look for ouliers (time in seconds). Default = 100.
    threshold: float/int
    Relative threshold. Anything channel with variance > threshold*mean OR
    < threshold*mean will be considered suspect. Default = 5.
    include_for_mean: float
    Fraction of variances to calculate mean. This is to ignore the highest and
    lowest ones, which coul dbe far outliers.
    '''
    sfreq = data_raw.info['sfreq']
    no_channels = len(data_raw.ch_names) - 1  # Subtract stimuli channel
    data, times = data_raw[
        :no_channels,
        int(sfreq * 10):int(sfreq * (time + 10))
    ]
    variances = []

    for i in range(data.shape[0]):
        variances.append(data[i, :].var())

    var_arr = np.array(variances)
    exclude = int((1 - include_for_mean) * no_channels / 2)
    mean_low = np.mean(np.sort(var_arr)[exclude:(no_channels - exclude)])

    suspects = np.where(
        (var_arr > threshold * mean_low) &
        (var_arr < threshold / mean_low)
    )[0]
    suspects_names = [data_raw.ch_names[x] for x in list(suspects)]
    selected_suspects = [
        data_raw.ch_names.index(x)
        for x in suspects_names if x not in ['HEOG', 'VEOG']
    ]
    selected_suspects_names = [
        x for x in suspects_names if x not in ['HEOG', 'VEOG']
    ]
    print('Suspicious channel(s): ', selected_suspects_names)

    return selected_suspects, selected_suspects_names


# Functions to import and process EEG data from cnt files.


def standardize_EEG(
        data_array,
        std_aim=1,
        centering='per_channel',
        scaling='global',
):
    '''Center data around 0 and adjust standard deviation.

    Args:
    --------
    data_array: np.ndarray
        Input data.
    std_aim: float/int
        Target standard deviation for rescaling/normalization.
    centering: str
        Specify if centering should be done 'per_channel', or 'global'.
    scaling: str
        Specify if scaling should be done 'per_channel', or 'global'.

    '''
    if centering == 'global':
        data_mean = data_array.mean()

        # Center around 0
        data_array = data_array - data_mean

    elif centering == 'per_channel':
        for i in range(data_array.shape[1]):
            data_mean = data_array[:, i, :].mean()

            # Center around 0
            data_array[:, i, :] = data_array[:, i, :] - data_mean

    else:
        print('Centering method not known.')
        return None

    if scaling == 'global':
        data_std = data_array.std()

        # Adjust std to std_aim
        data_array = data_array * std_aim/data_std

    elif scaling == 'per_channel':
        for i in range(data_array.shape[1]):
            data_std = data_array[:, i, :].std()

            # Adjust std to std_aim
            data_array[:, i, :] = data_array[:, i, :] * std_aim / data_std
    else:
        print('Given method is not known.')
        return None

    return data_array
