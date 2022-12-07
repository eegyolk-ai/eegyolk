"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
Authors: Floris Pauwels <florispauwels@live.nl>

Epodium class for working with the ePodium dataset
as in the thesis of  Floris Pauwels.
"""

import mne
import numpy as np
import pandas as pd
import copy
import os

from sklearn.model_selection import train_test_split


# This class is useful for passing the dataset object as input to a function.
class Epodium:

    def __init__(self):
        pass

    # ########### --- GENERAL DATASET TOOLS --- #################

    file_extension = ".bdf"
    metadata_filenames = ["children.txt", "cdi.txt", "parents.txt"]

    # Sample frequency in Hz:
    frequency = 2048

    channels_epod = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
                     'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
                     'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
                     'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

    # 26 channels that are in both ePod and DDP:
    channels_epod_ddp = ['O2', 'O1', 'P4', 'P8', 'C4', 'T8', 'P7', 'P3',
                         'C3', 'F4', 'F8', 'T7', 'F3', 'F7', 'PO3', 'PO4',
                         'CP2', 'CP6', 'CP5', 'CP1', 'FC2', 'FC6', 'FC1',
                         'FC5', 'AF4', 'AF3']

    channels_mastoid = ['EXG1', 'EXG2']
    channels_drop = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']

    montage = 'standard_1020'
    mne_montage = mne.channels.make_standard_montage(montage)
    mne_info = mne.create_info(channels_epod, frequency, ch_types='eeg')

    conditions = ['GiepM', "GiepS", "GopM", "GopS"]
    conditions_events = ['GiepM_S', 'GiepM_D', 'GiepS_S', 'GiepS_D',
                         'GopM_S', 'GopM_D', 'GopS_S', 'GopS_D']

    event_dictionary = {'GiepM_FS': 1, 'GiepM_S': 2, 'GiepM_D': 3,
                        'GiepS_FS': 4, 'GiepS_S': 5, 'GiepS_D': 6,
                        'GopM_FS': 7,  'GopM_S': 8,   'GopM_D': 9,
                        'GopS_FS': 10, 'GopS_S': 11,  'GopS_D': 12}

    # Order into type (FS/S/D)
    firststandard_id = [1, 4, 7, 10]
    standard_id = [2, 5, 8, 11]
    deviant_id = [3, 6, 9, 12]

    # To be ignored during processing:
    incomplete_experiments = ["113a", "107b (deel 1+2)", "132a", "121b(2)",
                              "113b", "107b (deel 3+4)", "147a", "121a",
                              "134a", "143b", "121b(1)", "145b", "152a",
                              "184a", "165a", "151a", "163a", "207a",
                              "215b"]

    def events_from_raw(self, raw):
        events = mne.find_events(raw, verbose=False,
                                 min_duration=2/self.frequency)
        events_12 = self.group_events_12(events)
        return events_12, self.event_dictionary

    @staticmethod
    def group_events_12(events):
        """
        Puts similar event conditions with different pronounciations together
        Reduces the number of distinctive events from 78 to 12 events,
        by combining different pronounciations into the same event.
        """
        event_conversion_12 = [
            [1, 1, 12], [2, 13, 24], [3, 25, 36],
            [4, 101, 101], [5, 102, 102], [6, 103, 103],
            [7, 37, 48], [8, 49, 60], [9, 61, 72],
            [10, 104, 104], [11, 105, 105], [12, 106, 106]]

        events_12 = copy.deepcopy(events)
        for i in range(len(events)):
            for newValue, minOld, maxOld in event_conversion_12:
                mask = np.logical_and(minOld <= events_12[i],
                                      events_12[i] <= maxOld)
                events_12[i] = np.where(mask, newValue, events_12[i])
        return events_12

    @staticmethod
    def create_labels(metadata_directory, path_save_csv=""):
        """
        This function creates a .csv file with the labels:
        Participant / Age_days_a / Age_days_b / Risk_of_dyslexia
        """
        path_children = os.path.join(metadata_directory, "children.txt")
        path_cdi = os.path.join(metadata_directory, "cdi.txt")
        path_parents = os.path.join(metadata_directory, "parents.txt")

        epod_children = pd.read_table(path_children)

        merged_df = epod_children[['ParticipantID', 'Age_days_a',
                                   'Age_days_b', 'Group_AccToParents']]
        new_names = {'ParticipantID': 'Participant',
                     'Group_AccToParents': 'Risk_of_dyslexia'}
        merged_df = merged_df.rename(columns=new_names)

        # Creates dyslexia continuum:
        parents_dyslexia_scores = pd.read_table(path_parents)
        parents_dyslexia_tests = [
            "emt_mother",
            "klepel_mother",
            "vc_mother",
            "emt_father",
            "klepel_father",
            "vc_father",
        ]
        min_max_dict = {"emt": [50, 116], "klepel": [32, 116], "vc": [10, 26]}

        n_participants = len(epod_children)
        dyslexia_score = np.zeros(n_participants)
        for i in range(n_participants):
            cumulative_score = 0
            n_scores = 0
            for test_parent in parents_dyslexia_tests:
                test = test_parent.split("_")[0]
                score_temp = parents_dyslexia_scores[test_parent][i]
                if score_temp.isdigit():
                    # Add normalised score:
                    cumulative_score += (
                        (int(score_temp) - min_max_dict[test][0]) /
                        (min_max_dict[test][1] - min_max_dict[test][0])
                    )
                    n_scores += 1
            dyslexia_score[i] = cumulative_score / n_scores

        merged_df["Dyslexia_score"] = dyslexia_score

        if path_save_csv:
            if os.path.exists(path_save_csv):
                os.remove(path_save_csv)
            merged_df.to_csv(path_save_csv)

        return merged_df

    def is_valid_experiment(self, events, min_standards,
                            min_deviants, min_firststandards):
        """
        Checks from the events file if there are enough epochs in
        the .fif file to be valid for analysis by counting how many
        events are left in standard, deviant, and FS in the 4 conditions.
        """
        for i in range(4):
            s_invalid = np.count_nonzero(events == self.standard_id[i])\
                < min_standards
            d_invalid = np.count_nonzero(events == self.deviant_id[i])\
                < min_deviants
            fs_invalid = np.count_nonzero(events == self.firststandard_id[i])\
                < min_firststandards
            if s_invalid or d_invalid or fs_invalid:
                return False
        return True

    # ########### --- MACHINE/DEEP LEARNING TOOLS --- #################

    # Normalize age for regressive age prediction:
    min_age = 487
    max_age = 756
    # range_age = 269 (9 months)
    range_age = max_age - min_age

    def normalize_age(self, age_days):
        """
        Normalizes age between -1 and 1.
        """
        return (age_days-self.min_age) / (0.5*self.range_age) - 1

    def denormalize_age(self, value):
        return (value+1)*0.5*self.range_age + self.min_age

    @staticmethod
    def split_dataset(experiment_list, proportion=0.8):
        """
        Each participant that exceeds the minimum epochs is put into
        a test or train set. Both the train and test sets have the same
        ratio of participants that did either a, b, or both experiments
        to keep the distributions independent.
        'Proportion' is set between 0 and 1 and represents the
        percentage of experiments put into the first return value.
        """

        # Split same proportion of participants that did a, b, or both.
        a_set = set(exp[0:3] for exp in experiment_list if 'a' in exp)
        b_set = set(exp[0:3] for exp in experiment_list if 'b' in exp)
        experiments_a_and_b = list(a_set.intersection(b_set))
        experiments_a_only = list(a_set.difference(b_set))
        experiments_b_only = list(b_set.difference(a_set))

        # Split participants into train and test dataset
        train_ab, test_ab =\
            train_test_split(experiments_a_and_b, train_size=proportion)
        train_a, test_a =\
            train_test_split(experiments_a_only, train_size=proportion)
        train_b, test_b =\
            train_test_split(experiments_b_only, train_size=proportion)

        train = [x + 'a' for x in train_ab] + [x + 'b' for x in train_ab] +\
                [x + 'a' for x in train_a] + [x + 'b' for x in train_b]
        test = [x + 'a' for x in test_ab] + [x + 'b' for x in test_ab] +\
               [x + 'a' for x in test_a] + [x + 'b' for x in test_b]

        return train, test
