"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains tools for analysing the DDP dataset.
This file is not actively updated and contains functions from
the https://github.com/epodium repository.
"""

import mne
import pandas as pd
import numpy as np
import os
import glob

from IPython.display import clear_output


class DDP:

    """
    ########### --- GENERAL DATASET TOOLS --- #################
    """

    file_extension = ".cnt"
    frequency = 500  # Hz

    metadata_filenames = [
        "children.txt",
        "cdi.txt",
        "parents.txt",
        "CODES_overview.txt",
    ]

    channels_ddp_30 = [
        'O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7',
        'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ',
        'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1',
    ]

    channels_ddp_62 = [
        'O2', 'O1', 'OZ', 'PZ',
        'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7',
        'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ',
        'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1', 'AFZ', 'PO3',
        'P1', 'POZ', 'P2', 'PO4', 'CP2', 'P6', 'M1', 'CP6', 'C6', 'PO8', 'PO7',
        'P5', 'CP5', 'CP1', 'C1', 'C2', 'FC2', 'FC6', 'C5', 'FC1', 'F2', 'F6',
        'FC5', 'F1', 'AF4', 'AF8', 'F5', 'AF7', 'AF3', 'FPZ',
    ]

    channel_names = channels_ddp_30

    montage = 'standard_1020'
    mne_montage = mne.channels.make_standard_montage(montage)
    mne_info = mne.create_info(channel_names, frequency, ch_types='eeg')

    event_dictionary = {
        '12', '13', '14', '15', '2', '3', '4', '5', '55', '66', '77', '88',
    }

    incomplete_experiments = []

    def read_raw(preload=True, verbose=False):

        """
        Concatenate raw files with same id
        Show number of files versus number of experiments
        raws[0] is modified in-place to achieve the concatenation.
        """

        return mne.io.read_raw_bdf(raw_path, preload=read_raw, verbose=verbose)

    def get_events_from_raw(self, raw):
        """
        TODO: add explanation here
        """
        events, event_dict = mne.events_from_annotations(raw)
        events_12 = self.group_events_standard(events)

        return events, [1]

    def group_events_2(events):
        """
        Only returns the standard and deviant events from all events.
        """

        events_3 = copy.deepcopy(events)
        for i in range(len(events)):
            for newValue, minOld, maxOld in event_conversion_12:
                condition = np.logical_and(
                    minOld <= events_12[i], events_12[i] <= maxOld)
                events_12[i] = np.where(condition, newValue, events_12[i])
        return events_12

    def create_labels_raw(
        self,
        path_label_csv,
        dataset_directory,
        ages_directory
    ):
        """
        This function creates a .csv file with the labels:
        filename / code / age_group / age_days
        The labels are saved in 'path_label_csv'.
        """
        age_groups = [5, 11, 17, 23, 29, 35, 41, 47]

        # Store cnt file info per age_group
        list_age_groups = []
        for age_group in age_groups:
            filename_list = []
            code_list = []

            folder = os.path.join(
                dataset_directory,
                str(age_group) + "mnd mmn",
            )
            pattern = os.path.join(folder, '*' + self.file_extension)
            raw_paths = sorted(glob.glob(pattern, recursive=True))
            filenames = [os.path.split(path)[1] for path in raw_paths]

            for filename in filenames:
                filename_list.append(filename)
                code_list.append(int(filename[0:3]))
                # First 3 numbers of filename is participant code

            df_age_group = pd.DataFrame(
                {"filename": filename_list, "code": code_list},
            )
            df_age_group['age_group'] = age_group
            list_age_groups.append(df_age_group)

        cnt_df = pd.concat(list_age_groups)

        # Set correct age labels for each file
        df_list = []
        for age_group in age_groups:
            age_path = os.path.join(
                ages_directory,
                "ages_" + str(age_group) + "mnths.txt",
            )
            df = pd.read_csv(age_path, sep="\t")
            df['age_group'] = age_group
            df_list.append(df)

        age_df = pd.concat(df_list)
        age_df = age_df.drop(columns=['age_months', 'age_years'])
        # age_days may be sufficient
        merged_df = pd.merge(
            cnt_df, age_df,
            how='left',
            on=['age_group', 'code'],
        )
        merged_df['age_days'].fillna(
            merged_df['age_group'] * 30,
            inplace=True,
        )

        if os.path.exists(path_label_csv):
            os.remove(path_label_csv)
        merged_df.to_csv(path_label_csv)

        return merged_df
