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
    
    
    @staticmethod
    def read_raw(raw_paths, preload=True, verbose=False):
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
        
    incomplete_experiments = ["8_11", "108_11", "156_11", "164_11", "619_11", "636_11", "162_17", "702_11", "7_17", "162_17", "311_17", "420_23", "733_29", "735_23", "737_23", "101_29", "434_29", "612_29", "702_29", "741_29", "756_29", "737_23", "14_41", "15_41", "17_41", "23_41", "24_41", "25_41", "149_41", "151_41", "152_41", "153_41", "172_41", "173_41", "174_41", "176_41", "177_41", "178_41", "179_41", "180_41", "181_41", "182_41", "304_41", "306_41", "312_41", "313_41", "314_41", "317_41", "320_41", "436_41", "438_41", "441_41", "472_41", "473_41", "474_41", "475_41", "478_41", "479_41", "480_41", "481_41", "482_41", "484_41", "485_41", "486_41", "487_41", "488_41", "491_41", "493_41", "494_41", "496_41", "497_41", "733_41", "734_41", "119_47", "121_47", '18_41', '19_41', '21_41', '27_41', '28_41', '29_41', '30_41', '34_41', '35_41', '39_41', '137_41', '139_41', '140_41', '141_41', '142_41', '146_41', '148_41', '154_41', '155_41', '156_41', '157_41', '158_41', '159_41', '162_41', '163_41', '164_41', '165_41', '166_41', '167_41', '168_41', '169_41', '170_41', '171_41', '321_41', '323_41', '324_41', '325_41', '326_41', '329_41', '330_41', '332_41', '334_41', '335_41', '340_41', '343_41', '344_41', '345_41', '346_41', '348_41', '422_41', '443_41', '445_41', '448_41', '449_41', '450_41', '451_41', '453_41', '454_41', '455_41', '456_41', '457_41', '459_41', '465_41', '466_41', '468_41', '469_41', '471_41', '611_41', '615_41', '616_41', '619_41', '620_41', '621_41', '622_41', '624_41', '625_41', '626_41', '627_41', '628_41', '629_41', '632_41', '633_41', '635_41', '735_41', '738_41', '739_41', '740_41', '741_41', '742_41', '743_41', '745_41', '746_41', '747_41', '748_41', '751_41', '752_41', '753_41', '755_41', '756_41', '10_47', '40_47', '124_47', '345_29', '453_29', '115_35', '115_41', '333_41', '711_47']
