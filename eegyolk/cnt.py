# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions originally designed
to reproduce work from 2021 to run on a Linux machine.
The specific work was pre-processing of EEG files.
"""

import logging
import os
import re

from collections import Counter

import mne
import mne_features.feature_extraction as fe
import numpy as np
import pandas as pd

from .rawf import RawData


def select_bad_epochs_list(
        epochs,
        stimuli,
        threshold=5,
        max_bad_fraction=0.2,
):
    ''' Function to find suspect epochs and channels --> still might
    need manual inspection!  Credit to Bjorn Bruns and Florian Huber.
    Arguments are as below
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


class CntReader:
    """
    This is class that allows reading in of cnt files in a specific way.
    """

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

    def __init__(self, raw_data):
        self.raw_data = raw_data
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
            ch_name.upper() for ch_name in self.montage.ch_names
        ]

    def read(self, cnt, label_group, channel_set='30'):
        '''
        Historical function from previous work to read cnt files.
        Function to read cnt file. Run bandpass filter.
        Then detect and correct/remove bad channels and bad epochs.
        Store resulting epochs as arrays.
        Arguments include
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

        self.signals = np.zeros((0, len(self.channel_set), 0))
        self.labels = []
        self.channel_names = []

        raw_cnt = self.raw_data.read_cnt(cnt, preload=True)

        # Band-pass filter
        # (between 0.5 and 40 Hz. was 0.5 to 30Hz in Stober 2016)
        raw_cnt.filter(0.5, 40, fir_design='firwin')

        # This used to be set in the loop, but it looks like a
        # constant expression
        raw_cnt.set_montage(self.montage)

        # Get events from annotations in the data
        self.annotations, self.events = mne.events_from_annotations(raw_cnt)

        # Select channels to exclude (if any)
        mask = set(self.channel_set).union(set(('HEOG', 'VEOG')))
        self.channels_exclude = tuple(
            x for x in raw_cnt.ch_names if x not in mask
        )

        valid_events = tuple(
            eid for eid in self.event_idx if str(eid) in self.events
        )

        for eid in valid_events:
            picks, epochs = self._picks_epochs(raw_cnt)
            # Detect potential bad channels and epochs
            #
            # TODO(wvxvw): Find out if select_bad_epochs_list() can be
            # used on valid_events instead of checking each event
            # individually.  The return format of this function is
            # bogus.  The premis of this loop seems to be: modify
            # `raw_cnt.info['bads']' each iteration and try to
            # interpolate the values from other channel into the bad
            # channels, if bad channels are found.  It's not clear to
            # me, why do this per event id: the `raw_cnt' doesn't
            # seem to change meaningfully between iterations outside
            # of its info property, which is also not indexed by event
            # id.  select_bad_epochs_list(), while it does generate
            # bad channels and bad epochs depending on event id seems
            # to be unconnected to how later bad channels are treated.
            bad_channels, bad_epochs = select_bad_epochs_list(
                epochs,
                [eid],
                threshold=self.threshold,
                max_bad_fraction=self.max_bad_fraction,
            )
            if len(bad_channels) > self.max_bad_channels:
                raise ValueError(
                    f'Too many bad channels: {len(bad_channels)}',
                )

            if len(bad_channels) > 0:
                # Mark bad channels:
                raw_cnt.info['bads'] = bad_channels
                # Interpolate bad channels using functionality of
                # 'mne' Also, sometimes the bad channel calculation
                # seems to be wrong...
                epochs.interpolate_bads()

                # Get signals as array and add to total collection
                #
                # TODO(wvxvw): Above is the original comment.  It's
                # not clear to me why the channels are appended like
                # this... For a single CNT file, the channles are
                # going to be the same every time this condition is
                # true.  It should make sense to simply store the
                # first or the last, but I don't know if there's any
                # code that relies on the length of this collection to
                # deduce some information about what was fixed.
                self.channel_names.append(epochs.ch_names)
                signals_cleaned = epochs[str(eid)].drop(
                    bad_epochs,
                ).get_data()
                # TODO(wvxvw): The original code had hardcoded value
                # 501 for the third axis.  I don't know why it's 501
                # and don't want to take my chances with it.  It seems
                # to be correct, though.  Even worse, I don't think
                # the third dimension plays any role, but maybe I'm
                # wrong about it.
                diff = signals_cleaned.shape[2] - self.signals.shape[2]
                if diff > 0:
                    self.signals = np.pad(
                        self.signals,
                        ((0, 0), (0, 0), (0, diff)),
                    )
                self.signals = np.concatenate(
                    (self.signals, signals_cleaned),
                    axis=0,
                )
                self.labels += (
                    [eid + label_group] * signals_cleaned.shape[0]
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

    # TODO(wvxvw): From my reading of the notebook, this method is
    # useless.  The notebook suggests that variance and rms are,
    # basically the same feature (who would've thought?)  I have no
    # idea what was the point of adding this...
    def rms(self, data):
        """Root-mean squared value of the data (per channel).
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        Returns
        -------
        output : ndarray, shape (n_channels,)
        Notes
        -----
        Alias of the feature function: *rms*
        """
        return np.sqrt(np.mean(np.power(data, 2), axis=-1))

    def selected_features(self):
        return {
            'mean',
            ('root_mean_squared', self.rms),
            'hjorth_mobility',
            'hjorth_complexity',
            'variance',
            'kurtosis',
            'skewness',
            'app_entropy',
            'zero_crossings',
            'energy_freq_bands',
            'spect_edge_freq',
        }

    def col_names_from_feautres(self, ch_names, extracted):
        col_names = []

        for i, cname in extracted.columns:
            s, e = re.search(r'\d+', cname).span()
            chname = str(ch_names[0][int(cname[s:e])])
            if '_' in cname:
                mo = cname.split('_')[1]
                parts = i, mo, chname
            else:
                parts = i, chname

            clname = '_'.join(parts)
            col_names.append(clname)

        return col_names

    def save_preprocessed_row(self, row):
        try:
            signals, labels, ch_names = self.read(
                row['cnt_path'],
                row['age_months'],
            )
        except ValueError as e:
            logging.warn(
                f'Cannot read {row["cnt_file"]}: {e}.',
            )
            return

        selected = self.selected_features()

        try:
            extracted = fe.extract_features(
                signals,
                500.0,
                selected,
                return_as_df=1,
            )
        except Exception as e:
            logging.warn(
                f'Cannot extract features from {row["cnt_file"]}: {e}.',
            )
            return

        extracted.columns = self.col_names_from_feautres(ch_names, extracted)

        extracted.to_hdf(row['h5'], key='df', mode='w')

        transposed = pd.DataFrame(row).transpose()
        transposed.to_csv(row['csv'], sep=',', index=False, header=True)

    def csv_h5_paths(self, processed_dir):

        def func(row):
            cnt = row['cnt_file']
            csv = os.path.join(
                processed_dir,
                f'processed_data_{cnt}.csv',
            )
            csv_exists = os.path.isfile(csv)
            h5 = os.path.join(
                processed_dir,
                f'extracted_features_{cnt}.h5',
            )
            h5_exists = os.path.isfile(h5)
            return csv, csv_exists, h5, h5_exists

        return func

    def save_preprocessed(self, processed_dir, limit=None, force=False):
        if limit is None:
            section = self.raw_data.raw_good
        else:
            section = self.raw_data.raw_good.head(limit)

        try:
            os.makedirs(processed_dir)
        except FileExistsError:
            pass

        preprocessed = section.copy()
        columns = ['csv', 'csv_exists', 'h5', 'h5_exists']
        preprocessed[columns] = preprocessed.apply(
            self.csv_h5_paths(processed_dir),
            axis=1,
            result_type='expand',
        )

        if not force:
            preprocessed = preprocessed[
                ~preprocessed['csv_exists'] & ~preprocessed['h5_exists']
            ]

        preprocessed.apply(self.save_preprocessed_row, axis=1)


def preprocess(raw, meta, processed, limit=None, force=False):
    acquired = RawData(raw, meta)
    acquired.fill_unlabeled()
    acquired.filter_broken()
    reader = CntReader(acquired)
    reader.save_preprocessed(processed, limit=limit, force=force)
