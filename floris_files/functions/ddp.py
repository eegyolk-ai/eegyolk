"Tools for analysing the DDP dataset. This file is not actively updated and contains functions from the https://github.com/epodium repository" 

import mne
import pandas as pd
import numpy as np
import os
import glob
import copy

from IPython.display import clear_output


class DDP:
    
    ########### --- GENERAL DATASET TOOLS --- #################
    
    file_extension = ".cnt"
    frequency = 500 # Hz    
    metadata_filenames = ["children.txt", "cdi.txt", "parents.txt", "CODES_overview.txt"]

    # Channels
    channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5',
               'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
               'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
               'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
               'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
               'Fz', 'Cz']
    
    channels_ddp_30 = ['O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7', 
                   'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ', 
                   'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1']

    channels_ddp_62 = ['O2', 'O1', 'OZ', 'PZ', 'P4', 'CP4', 'P8', 'C4', 'TP8', 'T8', 'P7', 
                   'P3', 'CP3', 'CPZ', 'CZ', 'FC4', 'FT8', 'TP7', 'C3', 'FCZ', 'FZ', 
                   'F4', 'F8', 'T7', 'FT7', 'FC3', 'F3', 'FP2', 'F7', 'FP1', 'AFZ', 'PO3', 
                   'P1', 'POZ', 'P2', 'PO4', 'CP2', 'P6', 'M1', 'CP6', 'C6', 'PO8', 'PO7', 
                   'P5', 'CP5', 'CP1', 'C1', 'C2', 'FC2', 'FC6', 'C5', 'FC1', 'F2', 'F6', 
                   'FC5', 'F1', 'AF4', 'AF8', 'F5', 'AF7', 'AF3', 'FPZ']
    
    channel_names = channel_names    

    montage = 'standard_1020'
    mne_montage = mne.channels.make_standard_montage(montage)
    mne_info = mne.create_info(channel_names, frequency, ch_types='eeg')
    
    # Events
    # Warning, dictionary differs for each participant.
    event_dictionary_full =  {'12', '13', '14', '15', '2', '3', '4', '5', '55', '66', '77', '88'}
    event_dictionary_3 = {'standard': 1, 'deviant': 2, 'first_standard': 3}
    event_dictionary_2 = {'standard': 1, 'deviant': 2}
    standard_id = [1]
    deviant_id = [2]
    firststandard_id = [3]

    standard_list = ['2', '3', '4', '5']
    deviant_list = ['55', '66', '77', '88']
    first_standard_list = ['12', '13', '14', '15']
    
    incomplete_experiments = ["8_11", "108_11", "156_11", "164_11", "619_11", "636_11", "162_17", "702_11", "7_17", "162_17", "311_17", "735_23", "737_23", "101_29", "741_29", "756_29", "737_23"]
    
    @staticmethod
    def read_raw(raw_paths, preload = True, verbose=False):
        """
        Return the raw experiment from multiple .cnt files
        """
        raws = []
        for raw_path in raw_paths:
            raws.append(mne.io.read_raw_cnt(raw_path, preload=preload, verbose=verbose))
        return mne.concatenate_raws(raws)
    
    def events_from_raw(self, raw):
        """
        Return the events from raw. The original labels are changed to:
        1 for standards, 2 for deviants, 3 for first standards.
        """
        standard_list = ['2', '3', '4', '5']
        deviant_list = ['55', '66', '77', '88']
        first_standard_list = ['12', '13', '14', '15']

        events, event_dict = mne.events_from_annotations(raw)
        events_3 = copy.deepcopy(events)        

        contains_first_standard = False

        for key in event_dict:
            if key in standard_list:
                events_3[events==event_dict[key]] = 1        
            if key in deviant_list:
                    events_3[events==event_dict[key]] = 2
            if key in first_standard_list:
                    events_3[events==event_dict[key]] = 3
                    contains_first_standard = True
        if contains_first_standard:
            return events_3, self.event_dictionary_3        
        else:
            return events_3, self.event_dictionary_2
        return events_3

    def create_labels_raw(self, dataset_directory, ages_directory, path_save_csv=""):
        """
        This function creates a .csv file with the labels: filename / code / age_group / age_days
        The labels are saved in 'path_save_csv'.
        """
        age_groups = [5, 11, 17, 23, 29, 35, 41, 47]

        # Store cnt file info per age_group
        list_age_groups = []
        for age_group in age_groups:
            filename_list = []
            code_list = []

            folder = os.path.join(dataset_directory, str(age_group) + "mnd mmn")
            pattern = os.path.join(folder, '*' + self.file_extension)
            raw_paths = sorted(glob.glob(pattern, recursive=True))
            filenames = [os.path.split(path)[1] for path in raw_paths]

            for filename in filenames:
                filename_list.append(filename)
                code_list.append(int(filename[0:3])) # First 3 numbers of filename is participant code

            df_age_group = pd.DataFrame({"filename": filename_list, "code": code_list})
            df_age_group['age_group'] = age_group
            list_age_groups.append(df_age_group)

        cnt_df = pd.concat(list_age_groups)

        # Set correct age labels for each file
        df_list = []
        for age_group in age_groups:
            age_path = os.path.join(ages_directory, "ages_" + str(age_group) + "mnths.txt") 
            df = pd.read_csv(age_path, sep = "\t")
            df['age_group'] = age_group
            df_list.append(df)

        age_df = pd.concat(df_list)
        age_df = age_df.drop(columns=['age_months', 'age_years']) # age_days is sufficient
        merged_df = pd.merge(cnt_df, age_df, how = 'left', on = ['age_group', 'code'])
        merged_df['age_days'].fillna(merged_df['age_group'] * 30, inplace = True)
    
        if path_save_csv:
            if os.path.exists(path_label_csv):
                os.remove(path_label_csv)
            merged_df.to_csv(path_label_csv)
        
        return merged_df

    
    def is_valid_experiment(self, events, min_standards, min_deviants, min_firststandards):
        "Checks from the events file if there are enough epochs in the .fif file to be valid for analysis."
        # Counts how many events are left in standard, deviant, and FS in the 4 conditions.
        if np.count_nonzero(events == 1) < min_standards\
        or np.count_nonzero(events == 2) < min_deviants\
        or np.count_nonzero(events == 3) < min_firststandards:
                return False
        return True
        

