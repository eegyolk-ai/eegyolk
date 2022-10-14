"""Tools for working with the ePodium dataset."""

import mne
import numpy as np
import copy

# This class is useful for passing the dataset object as input to a function.
class Epodium:
    
    ############ --- GENERAL DATASET TOOLS --- #################
    
    file_extension = ".bdf"
    frequency = 2048 # Hz

    metadata_filenames = ["children.txt", "cdi.txt", "parents.txt", "CODES_overview.txt"]

    channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5',
                   'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
                   'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
                   'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
                   'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
                   'Fz', 'Cz']

    channels_mastoid = ['EXG1', 'EXG2']
    channels_drop = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
    
    montage = 'standard_1020'
    mne_montage = mne.channels.make_standard_montage(montage) 
    mne_info = mne.create_info(channel_names, frequency, ch_types='eeg')

    conditions = ['GiepM', "GiepS", "GopM", "GopS"]
    conditions_events = ['GiepM_S', 'GiepM_D', 'GiepS_S', 'GiepS_D', 'GopM_S', 'GopM_D', 'GopS_S', 'GopS_D']

    event_dictionary = {'GiepM_FS': 1, 'GiepM_S': 2, 'GiepM_D': 3, 
                        'GiepS_FS': 4, 'GiepS_S': 5, 'GiepS_D': 6,
                        'GopM_FS': 7,  'GopM_S': 8,   'GopM_D': 9,
                        'GopS_FS': 10, 'GopS_S': 11,  'GopS_D': 12}
    
    # Order into type (FS/S/D)
    firststandard_id = [1, 4, 7, 10]
    standard_id = [2, 5, 8, 11]
    deviant_id = [3, 6, 9, 12]
    
    # To be ignored during processing: 
    incomplete_experiments = ["113a", "107b (deel 1+2)", "132a", "121b(2)", "113b", "107b (deel 3+4)", "147a",
                              "121a", "134a", "143b", "121b(1)", "145b", "152a", "184a", "165a", "151a", "163a",
                              "207a", "215b"]    

    @staticmethod
    def read_raw(preload=True, verbose=False):
         return mne.io.read_raw_bdf(raw_path, preload=read_raw, verbose=verbose)
    
    def events_from_raw(self, raw):
        events = mne.find_events(raw, verbose=False, min_duration=2/self.frequency)
        events_12 = self.group_events_12(events)
        return events_12, event_dictionary
    
    @staticmethod
    def group_events_12(events):
        """
        Puts similar event conditions with different pronounciations together
        Reduces the number of distinctive events from 78 to 12 events.
        This is done by combining different pronounciations into the same event.
        """
        event_conversion_12 = [
            [1, 1, 12], [2, 13, 24], [3, 25, 36],
            [4, 101, 101], [5, 102, 102], [6, 103, 103],
            [7, 37, 48], [8, 49, 60], [9, 61, 72],
            [10, 104, 104], [11, 105, 105], [12, 106, 106]]

        events_12 = copy.deepcopy(events)
        for i in range(len(events)):
            for newValue, minOld, maxOld in event_conversion_12:
                mask = np.logical_and(minOld <= events_12[i], events_12[i] <= maxOld)
                events_12[i] = np.where(mask, newValue, events_12[i])
        return events_12

    def is_valid_experiment(self, events, min_standards, min_deviants, min_firststandards):
        "Checks from the events file if there are enough epochs in the .fif file to be valid for analysis."
        # Counts how many events are left in standard, deviant, and FS in the 4 conditions.
        for i in range(4):
            if np.count_nonzero(events == self.standard_id[i]) < min_standards\
            or np.count_nonzero(events == self.deviant_id[i]) < min_deviants\
            or np.count_nonzero(events == self.firststandard_id[i]) < min_firststandards:
                return False
        return True
    

    ############ --- MACHINE/DEEP LEARNING TOOLS --- #################
        
    # Normalize age for regressive age prediction:
    min_age = 487
    max_age = 756 # 756
    range_age = max_age - min_age # 269 (9 months)

    def normalize_age(age_days): # Normalizes age between -1 and 1
        return (age_days-min_age) / (0.5*range_age) - 1

    def denormalize_age(value):
        return (value+1)*0.5*range_age + min_age


    def split_train_test_datasets(experiment_list, test_size = 0.25):
        """
        Each participant that exceeds the minimum epochs is put into a test or train set.
        Both the train and test sets have the same proportion of participants that did either a, b, or both experiments
        """    

        # Initialise same proportion of participants that did either a, b, or both experiments 
        a_set = set(exp[0:3] for exp in experiment_list if 'a' in exp)
        b_set = set(exp[0:3] for exp in experiment_list if 'b' in exp)

        experiments_a_and_b = list(a_set.intersection(b_set))
        experiments_a_only = list(a_set.difference(b_set)) # participants with only a
        experiments_b_only = list(b_set.difference(a_set)) # participants with only b

        # Split participants into train and test dataset
        train_ab, test_ab = train_test_split(experiments_a_and_b, test_size=test_size)  
        train_a, test_a = train_test_split(experiments_a_only, test_size=test_size) 
        train_b, test_b = train_test_split(experiments_b_only, test_size=test_size) 

        train = [x + 'a' for x in train_ab] + [x + 'b' for x in train_ab] + \
                [x + 'a' for x in train_a] + [x + 'b' for x in train_b]
        test = [x + 'a' for x in test_ab] + [x + 'b' for x in test_ab] + \
           [x + 'a' for x in test_a] + [x + 'b' for x in test_b]

        print(f"The dataset is split up into {len(train)} train and {len(test)} test experiments")    
        return train, test



