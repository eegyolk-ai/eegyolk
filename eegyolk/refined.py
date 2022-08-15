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

import glob
import os
import pandas as pd
import numpy as np
import hashlib
import h5py
import warnings
import re
import mne
from collections import Counter


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


def select_bad_epochs_list(
        epochs,
        stimuli,
        threshold=5,
        max_bad_fraction=0.2,
):
    ''' Function to find suspect epochs and channels --> still might
    need manual inspection!  Credit to Bjorn Bruns and Florian Huber

    Args:
    --------
    epochs: epochs object (mne)

    stimuli: list of int/str
    Stimuli to pick epochs for.
    threshold: float/int
    Relative threshold. Anything channel with variance > threshold*mean OR
    < threshold*mean will be considered suspect. Default = 5.
    max_bad_fraction: float
    Maximum fraction of bad epochs.
    If number is higher for one channel, call it a 'bad' channel
    '''
    bad_epochs = set()
    bad_channels = []

    for stimulus in stimuli:
        signals = epochs[str(stimulus)].get_data()
        max_bad_epochs = max_bad_fraction * signals.shape[0]

        # Find outliers in episode STD and max-min difference:
        signals_std = np.std(signals, axis=2)
        signals_minmax = np.amax(signals, axis=2) - np.amin(signals, axis=2)

        outliers_high = np.where(
            (signals_std > threshold * np.mean(signals_std)) |
            (signals_minmax > threshold * np.mean(signals_minmax))
        )
        outliers_low = np.where(
            (signals_std < 1 / threshold * np.mean(signals_std)) |
            (signals_minmax < 1 / threshold * np.mean(signals_minmax))
        )
        outliers = (
            np.concatenate((outliers_high[0], outliers_low[0])),
            np.concatenate((outliers_high[1], outliers_low[1])),
        )

        if len(outliers[0]) > 0:
            print(
                'Found',
                len(set(outliers[0])),
                'bad epochs in a total of',
                len(set(outliers[1])),
                ' channels.',
            )
            occurences = [
                (Counter(outliers[1])[x], x)
                for x in list(Counter(outliers[1]))
            ]
            for occur, channel in occurences:
                if occur > max_bad_epochs:
                    print(
                        'Found bad channel (more than',
                        max_bad_epochs,
                        ' bad epochs): Channel no: ',
                        channel,
                    )
                    bad_channels.append(channel)
                else:
                    # only add bad epochs for non-bad channels
                    bad_epochs = (
                        bad_epochs | set(outliers[0][outliers[1] == channel])
                    )

            print(
                'Marked',
                len(bad_epochs),
                'bad epochs in a total of',
                signals.shape[0],
                ' epochs.',
            )

            # Remove bad data:
            # signals = np.delete(signals, bad_channels, 1)
            # signals = np.delete(signals, list(bad_epochs), 0)

        else:
            print('No outliers found with given threshold.')

    return [epochs.ch_names[x] for x in bad_channels], list(bad_epochs)


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


class CntReader:

    known_channel_sets = {
        '30': (
            'O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8',
            'C4', 'TP8', 'T8', 'P7', 'P3', 'CP3', 'CPZ', 'CZ', 'FC4',
            'FT8', 'TP7', 'C3', 'FCZ', 'FZ', 'F4', 'F8', 'T7', 'FT7',
            'FC3', 'F3', 'FP2', 'F7', 'FP1',
        ),
        '62': (
            'O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7',
            'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ',
            'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1', 'AFZ',
            'PO3', 'P1', 'POZ', 'P2', 'PO4', 'CP2', 'P6', 'M1', 'CP6', 'C6',
            'PO8', 'PO7', 'P5', 'CP5', 'CP1', 'C1', 'C2', 'FC2', 'FC6', 'C5',
            'FC1', 'F2', 'F6', 'FC5', 'F1', 'AF4', 'AF8', 'F5', 'AF7', 'AF3',
            'FPZ',
        ),
    }

    event_idx = 2, 3, 4, 5, 12, 13, 14, 15

    tmin = -0.2

    tmax = 0.8

    lpass = 0.5

    hpass = 40

    threshold = 5

    max_bad_fraction = 0.2

    max_bad_channels = 2

    baseline = None, 0  # means from the first instant to t = 0

    def __init__(self):
        self.channel_set = None
        self.signals = None
        self.labels = []
        self.channel_names = []
        self.channels_exclude = None
        self.annotations = None
        self.events = None
        # MARK: Setting the montage is not verified yet
        # (choice of standard montage)
        self.montage = mne.channels.make_standard_montage(
            'standard_1020',
        )
        self.montage.ch_names = [
            ch_name.upper() for ch_name in montage.ch_names
        ]

    def read(self, cnt, label_group, channel_set='30'):
        '''
        Historical function from previous work to read cnt files.
        Function to read cnt file. Run bandpass filter.
        Then detect and correct/remove bad channels and bad epochs.
        Store resulting epochs as arrays.

        Args:
        --------
        cnt: str
        Name of file to import.
        label_group: int
        Unique ID of specific group (must be > 0).
        channel_set: str
        Select among pre-defined channel sets. Here: '30' or '62'
        '''
        if channel_set not in self.known_channel_sets:
            raise ValueError('Predefined channel set must be either 30 or 62')

        self.channel_set = self.known_channel_sets[channel_set]

        self.signal_collection = np.zeros((0, len(channel_set), 501))

        data_raw = mne.io.read_raw_cnt(cnt, eog='auto', preload=True)

        # Band-pass filter (btwn 0.5 and 40 Hz.;0.5 to 30Hz in Stober 2016)
        data_raw.filter(0.5, 40, fir_design='firwin')

        # This used to be set in the loop, but it looks like a
        # constant expression
        data_raw.set_montage(self.montage)

        # Get events from annotations in the data
        self.annotations, self.events = mne.events_from_annotations(data_raw)

        # Select channels to exclude (if any)
        mask = set(self.channel_set).union(set(('HEOG', 'VEOG')))
        self.channels_exclude = tuple(
            x for x in data_raw.ch_names if x not in mask
        )

        valid_events = tuple(
            eid for eid in self.event_idx if str(eid) in self.events
        )

        for event_id in valid_events:
            picks, epochs = self._picks_epochs(data_raw)
            # Detect potential bad channels and epochs
            #
            # TODO(wvxvw): Find out if select_bad_epochs_list() can be
            # used on valid_events instead of checking each event
            # individually.  The return format of this function is
            # bogus.  The premis of this loop seems to be: modify
            # `raw_data.info['bads']' each iteration and try to
            # interpolate the values from other channel into the bad
            # channels, if bad channels are found.  It's not clear to
            # me, why do this per event id: the `raw_data' doesn't
            # seem to change meaningfully between iterations outside
            # of its info property, which is also not indexed by event
            # id.  select_bad_epochs_list(), while it does generate
            # bad channels and bad epochs depending on event id seems
            # to be unconnected to how later bad channels are treated.
            bad_channels, bad_epochs = select_bad_epochs_list(
                epochs,
                [event_id],
                threshold=self.threshold,
                max_bad_fraction=self.max_bad_fraction,
            )
            if len(bad_channels) > self.max_bad_channels:
                raise ValueError(
                    'Too many bad channels: {}'.format(len(bad_channels)),
                )

            if len(bad_channels) > 0:
                # Mark bad channels:
                data_raw.info['bads'] = bad_channels
                # Interpolate bad channels using functionality of 'mne'
                epochs.interpolate_bads()

                # Get signals as array and add to total collection
                self.channel_names.append(epochs.ch_names)
                signals_cleaned = epochs[str(eid)].drop(bad_epochs).get_data()
                self.signals = np.concatenate(
                    (self.signals, signals_cleaned),
                    axis=0,
                )
                self.labels += (
                    [event_id + label_group] * signals_cleaned.shape[0]
                )

        return self.signals, self.labels, self.channel_names

    def _picks_epochs(self, data_raw):
        # Pick EEG channels
        picks = mne.pick_types(
            data_raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            include=self.channel_set,
            exclude=self.channels_exclude,
        )
        epochs = mne.Epochs(
            data_raw,
            events=self.annotations,
            event_id=self.events,
            tmin=self.tmin,
            tmax=self.tmax,
            proj=True,
            picks=picks,
            baseline=self.baseline,
            preload=True,
            event_repeated='merge',
            verbose=False,
        )
        return picks, epochs
