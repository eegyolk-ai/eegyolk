# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions related to reproduction
of 2021 work. Specifically, it helps reading in raw data from
known types of files.
"""

import logging
import os
import re
import warnings

from glob import glob

import mne
import numpy as np
import pandas as pd


class MneViewer:

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __getitem__(self, records):
        data = self.raw_data.raw['cnt_path']
        if isinstance(records, int):
            f = data[records]
            return self.raw_data.read_cnt(f)

        return (self.raw_data.read_cnt(f) for f in data[records])


class RawData:

    age_groups = {
        11: '11mnd mmn',
        17: '17mnd mmn',
        23: '23mnd mmn',
        29: '29mnd mmn',
        35: '35mnd mmn',
        41: '41mnd mmn',
        47: '47mnd mmn',
    }

    age_files = {
        11: 'ages_11mnths.txt',
        17: 'ages_17mnths.txt',
        23: 'ages_23mnths.txt',
        29: 'ages_29mnths.txt',
        35: 'ages_35mnths.txt',
        41: 'ages_41mnths.txt',
        47: 'ages_47mnths.txt',
    }

    cnt_read_args = {
        'eog': 'auto',
        # NOTE: This is a way of preventing another waring hen
        # parsing CNT files.  The default value is `auto', but MNE
        # cannot identify how many bytes per event are used, and gives
        # a warning about that.  There seem to be only two possible
        # options: `int16' and `int32'.  `int32' was chosen by the old
        # code by default, so the new code keeps the same behavior
        # while making it more explicit (and avoiding the warning).
        'data_format': 'int16',
        # NOTE: Note that the default is the US format, and all
        # dates are botched because of that.  Also note that
        # "conveniently", the dates are in the worst imaginable
        # format:
        # https://commons.apache.org/proper/commons-lang/apidocs/org/\
        # apache/commons/lang3/time/FastDateFormat.html That is
        # 3-digit year, that Python doesn't support...  The format
        # should really be `dd/mm/yyy', but that is not supported by
        # the MNE library.
        'date_format': 'dd/mm/yy',
        'verbose': False,
    }

    def __init__(self, raw_data_dir, meta_dir):
        self.cnt_files = pd.concat(self.read_all_age_groups(raw_data_dir))
        self.ages = pd.concat(self.read_ages_meta(meta_dir))
        self.raw = pd.merge(
            self.cnt_files,
            self.ages,
            how='left',
            on=('age_group', 'code'),
        )

    def read_cnt(self, fname, preload=True):
        return mne.io.read_raw_cnt(
            fname,
            preload=preload,
            **self.cnt_read_args
        )

    def read_good_cnt(self, fname, preload=True):
        return mne.io.read_raw_cnt(
            fname,
            preload=preload,
            **self.cnt_read_args
        )

    def read_age_group(self, raw_data_dir, age_group, directory):
        pattern = os.path.join(raw_data_dir, directory, '*.cnt')
        logging.info('Acquiring: %s', pattern)
        cnts = glob(pattern)
        names = tuple(
            os.path.splitext(os.path.basename(f))[0]
            for f in cnts
        )
        codes = tuple(
            int(re.search(r'\d+', x).group())
            for x in names
        )
        return pd.DataFrame(
            list(zip(codes, cnts, names)),
            columns=('code', 'cnt_path', 'cnt_file'),
        )

    def read_all_age_groups(self, raw_data_dir):
        for age_group, directory in self.age_groups.items():
            ag_df = self.read_age_group(raw_data_dir, age_group, directory)
            ag_df['age_group'] = age_group
            yield ag_df

    def read_ages_meta(self, meta_dir):
        for age_group, age_file in self.age_files.items():
            df = pd.read_csv(
                os.path.join(meta_dir, 'ages', age_file),
                sep="\t"
                )
            df['age_group'] = age_group
            yield df

    def breakdown_by_age(self):
        return [
            self.raw.loc[self.raw['age_group'] == ag]
            for ag in self.age_groups.keys()
        ]

    def unlabeled(self):
        return self.raw.loc[self.raw['age_days'].isnull()]

    def fill_unlabeled(self):
        '''Fill in the missing age data based on the age group the
        subject is in

        We know the age group (i.e. 11, 17, 23, ... months etc) of all
        the subjects, based on the folder the files are in and based
        on the file name. We have got the exact ages (in days) of most
        subjects seperately, which we have added to the DataFrame
        above. For some of the subjects, we don't have the exact age
        and therefore we set this equal to the age group.
        '''
        self.raw['age_months'].fillna(self.raw['age_group'], inplace=True)
        self.raw['age_days'].fillna(self.raw['age_group'] * 30, inplace=True)
        self.raw['age_years'].fillna(self.raw['age_group'] / 12, inplace=True)

    @property
    def as_mne(self):
        return MneViewer(self)

    def filter_broken(self):
        '''This did not exist in the original code, but it makes it
        easier to deal with MNE file reading errors: we just try
        reading all the files once, and sort them into two groups:
        raw_good and raw_bad.
        '''
        can_read = self.raw.index.to_series()
        for i, row in self.raw.iterrows():
            try:
                # Usually, when MNE fails to read a file, there will
                # also be a warning, as we will throw those away, we
                # don't care about those warnings.
                with warnings.catch_warnings():
                    with mne.utils.use_log_level('error'):
                        self.read_cnt(row['cnt_path'], preload=False)
                can_read[i] = 1
            except Exception as e:
                # This is a problem with MNE library: it simply fails
                # to read some files, but instead of giving a
                # meaningful error, it breaks with all kinds of
                # generic errors.  Luckily, there seem to be
                # relatively few of these files.
                can_read[i] = 0

        can_read = np.array(can_read)
        self.raw_good = self.raw[can_read == 1]
        self.raw_bad = self.raw[can_read == 0]

    def count_events(self):
        tmin = -0.2
        tmax = 0.8
        # means from the first instant to t = 0
        baseline = None, 0
        counts = pd.Series(dtype=np.int32)
        mappings = {}

        for i, path in enumerate(self.raw_good['cnt_path']):
            raw = self.read_cnt(path)

            # events is going to have event ids in the last column
            events, mapping = mne.events_from_annotations(raw, verbose=False)
            uniques = np.unique(events[:, 2], return_counts=True)
            single_counts = dict(zip(*uniques))
            counts = counts.add(pd.Series(single_counts), fill_value=0)
            # The original code used this mapping, but this mapping
            # gives different keys for the same values for each
            # mne.events_from_annotations() result... so, it seems
            # like the original code was simply confused about what it
            # was showing.
            #
            # mapping = {v: int(k) for k, v in mapping.items()}
            # mappings.update(mapping)


class SwitchedRawData:

    age_groups = {
        11: '11mnd mmn',
        17: '17mnd mmn',
        23: '23mnd mmn',
        29: '29mnd mmn',
        35: '35mnd mmn',
        41: '41mnd mmn',
        47: '47mnd mmn',
    }

    age_files = {
        11: 'ages_11mnths.txt',
        17: 'ages_17mnths.txt',
        23: 'ages_23mnths.txt',
        29: 'ages_29mnths.txt',
        35: 'ages_35mnths.txt',
        41: 'ages_41mnths.txt',
        47: 'ages_47mnths.txt',
    }

    cnt_read_args = {
        'eog': 'auto',
        # NOTE: This is a way of preventing another waring hen
        # parsing CNT files.  The default value is `auto', but MNE
        # cannot identify how many bytes per event are used, and gives
        # a warning about that.  There seem to be only two possible
        # options: `int16' and `int32'.  `int16' was chosen by the
        # oldest code by default, so this new code keeps the same behavior
        # while making it more explicit (and avoiding the warning).
        'data_format': 'int32',
        # NOTE: Note that the default is the US format, and all
        # dates are botched because of that.  Also note that
        # "conveniently", the dates are in the worst imaginable
        # format:
        # https://commons.apache.org/proper/commons-lang/apidocs/org/\
        # apache/commons/lang3/time/FastDateFormat.html That is
        # 3-digit year, that Python doesn't support...  The format
        # should really be `dd/mm/yyy', but that is not supported by
        # the MNE library.
        'date_format': 'dd/mm/yy',
        'verbose': False,
    }

    def __init__(self, raw_data_dir, meta_dir):
        self.cnt_files = pd.concat(self.read_all_age_groups(raw_data_dir))
        self.ages = pd.concat(self.read_ages_meta(meta_dir))
        self.raw = pd.merge(
            self.cnt_files,
            self.ages,
            how='left',
            on=('age_group', 'code'),
        )

    def read_cnt(self, fname, preload=True):
        return mne.io.read_raw_cnt(
            fname,
            preload=preload,
            **self.cnt_read_args
        )

    def read_good_cnt(self, fname, preload=True):
        return mne.io.read_raw_cnt(
            fname,
            preload=preload,
            **self.cnt_read_args
        )

    def read_age_group(self, raw_data_dir, age_group, directory):
        pattern = os.path.join(raw_data_dir, directory, '*.cnt')
        logging.info('Acquiring: %s', pattern)
        cnts = glob(pattern)
        names = tuple(
            os.path.splitext(os.path.basename(f))[0]
            for f in cnts
        )
        codes = tuple(
            int(re.search(r'\d+', x).group())
            for x in names
        )
        return pd.DataFrame(
            list(zip(codes, cnts, names)),
            columns=('code', 'cnt_path', 'cnt_file'),
        )

    def read_all_age_groups(self, raw_data_dir):
        for age_group, directory in self.age_groups.items():
            ag_df = self.read_age_group(raw_data_dir, age_group, directory)
            ag_df['age_group'] = age_group
            yield ag_df

    def read_ages_meta(self, meta_dir):
        for age_group, age_file in self.age_files.items():
            df = pd.read_csv(
                os.path.join(meta_dir, 'ages', age_file),
                sep="\t"
                )
            df['age_group'] = age_group
            yield df

    def breakdown_by_age(self):
        return [
            self.raw.loc[self.raw['age_group'] == ag]
            for ag in self.age_groups.keys()
        ]

    def unlabeled(self):
        return self.raw.loc[self.raw['age_days'].isnull()]

    def fill_unlabeled(self):
        '''Fill in the missing age data based on the age group the
        subject is in

        We know the age group (i.e. 11, 17, 23, ... months etc) of all
        the subjects, based on the folder the files are in and based
        on the file name. We have got the exact ages (in days) of most
        subjects seperately, which we have added to the DataFrame
        above. For some of the subjects, we don't have the exact age
        and therefore we set this equal to the age group.
        '''
        self.raw['age_months'].fillna(self.raw['age_group'], inplace=True)
        self.raw['age_days'].fillna(self.raw['age_group'] * 30, inplace=True)
        self.raw['age_years'].fillna(self.raw['age_group'] / 12, inplace=True)

    @property
    def as_mne(self):
        return MneViewer(self)

    def filter_broken(self):
        '''This did not exist in the original code, but it makes it
        easier to deal with MNE file reading errors: we just try
        reading all the files once, and sort them into two groups:
        raw_good and raw_bad.
        '''
        can_read = self.raw.index.to_series()
        for i, row in self.raw.iterrows():
            try:
                # Usually, when MNE fails to read a file, there will
                # also be a warning, as we will throw those away, we
                # don't care about those warnings.
                with warnings.catch_warnings():
                    with mne.utils.use_log_level('error'):
                        self.read_cnt(row['cnt_path'], preload=False)
                can_read[i] = 1
            except Exception as e:
                # This is a problem with MNE library: it simply fails
                # to read some files, but instead of giving a
                # meaningful error, it breaks with all kinds of
                # generic errors.  Luckily, there seem to be
                # relatively few of these files.
                can_read[i] = 0

        can_read = np.array(can_read)
        self.raw_good = self.raw[can_read == 1]
        self.raw_bad = self.raw[can_read == 0]

    def count_events(self):
        tmin = -0.2
        tmax = 0.8
        # means from the first instant to t = 0
        baseline = None, 0
        counts = pd.Series(dtype=np.int32)
        mappings = {}

        for i, path in enumerate(self.raw_good['cnt_path']):
            raw = self.read_cnt(path)

            # events is going to have event ids in the last column
            events, mapping = mne.events_from_annotations(raw, verbose=False)
            uniques = np.unique(events[:, 2], return_counts=True)
            single_counts = dict(zip(*uniques))
            counts = counts.add(pd.Series(single_counts), fill_value=0)
            # The original code used this mapping, but this mapping
            # gives different keys for the same values for each
            # mne.events_from_annotations() result... so, it seems
            # like the original code was simply confused about what it
            # was showing.
            #
            # mapping = {v: int(k) for k, v in mapping.items()}
            # mappings.update(mapping)


class RawDataBdf:
    """
    Doc
    """

    def __init__(self, raw_data_dir, meta_dir):
        raw = pd.read_csv(
            os.path.join(meta_dir, 'children.txt'),
            sep='\t',
        )
        raw['Age_days_a'] = pd.to_numeric(raw['Age_days_a'], errors='coerce')
        raw['Age_days_b'] = pd.to_numeric(raw['Age_days_b'], errors='coerce')
        pre_drop = len(raw)
        raw.dropna(inplace=True)
        post_drop = len(raw)
        logging.warn(
            '%s records were dropped because of bad age values',
            pre_drop - post_drop,
        )
        prefix = (
            raw_data_dir +
            os.path.sep +
            raw['ParticipantID'].astype(str)
        )
        raw['path_a'] = prefix + 'a.bdf'
        raw['path_b'] = prefix + 'b.bdf'
        raw['age_group_a'] = (raw['Age_days_a'] / 365).round().astype(int)
        raw['age_group_b'] = (raw['Age_days_b'] / 365).round().astype(int)
        self.raw = raw

    def read_bdf(self, fname, preload=True):
        return mne.io.read_raw_bdf(fname, preload=preload)

    def breakdown_by_age(self, sample='a'):
        age_col = 'age_group_' + sample
        return [
            self.raw.loc[self.raw[age_col] == ag]
            for ag in sorted(self.raw[age_col].unique())
        ]

    def filter_broken(self):
        '''This did not exist in the original code, but it makes it
        easier to deal with MNE file reading errors: we just try
        reading all the files once, and sort them into two groups:
        raw_good and raw_bad.
        '''
        can_read = self.raw.index.to_series()
        for i, row in self.raw.iterrows():
            try:
                # Usually, when MNE fails to read a file, there will
                # also be a warning, as we will throw those away, we
                # don't care about those warnings.
                with warnings.catch_warnings():
                    with mne.utils.use_log_level('error'):
                        self.read_bdf(row['path_a'], preload=False)
                        self.read_bdf(row['path_b'], preload=False)
                can_read[i] = 1
            except Exception as e:
                # This is a problem with MNE library: it simply fails
                # to read some files, but instead of giving a
                # meaningful error, it breaks with all kinds of
                # generic errors.  Luckily, there seem to be
                # relatively few of these files.
                can_read[i] = 0

        can_read = np.array(can_read)
        self.raw_good = self.raw[can_read == 1]
        self.raw_bad = self.raw[can_read == 0]
