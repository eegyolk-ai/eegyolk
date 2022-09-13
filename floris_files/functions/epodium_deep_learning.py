"""Tools for applying deep learning to the ePodium dataset."""

import numpy as np
import pandas as pd
import os
import glob
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

from functions import epodium
import local_paths


def split_train_test_datasets(processing_method, test_size = 0.25, min_standards = 180, min_deviants = 80, min_firststandards = 80):
    """
    This function checks the number of clean epochs in the event files after processing.
    Each participant that exceeds the minimum epochs is put into a test or train set.
    Both the train and test sets have the same proportion of participants that did either a, b, or both experiments
    
    In an ideal experiment, there are 360 standards, 120 deviants and 130 first standards in each of the 4 conditions.
    """
    
    # ePodium setup of the 12 events in 4 conditions.
    firststandard_index = [1, 4, 7, 10]
    standard_index = [2, 5, 8, 11]
    deviant_index = [3, 6, 9, 12]

    # Experiments with enough epochs are added to clean_list
    clean_list = []
    
    metadata_path = os.path.join(local_paths.ePod_metadata, "children.txt")
    metadata = pd.read_table(metadata_path)

    path_events = glob.glob(os.path.join(local_paths.processed, "ePod_" + processing_method, "events", '*.txt'))
    for path_event in path_events:
        file_event = os.path.basename(path_event)
        event = np.loadtxt(path_event, dtype=int)
        
        # Counts how many events are left in standard, deviant, and FS in the 4 conditions.
        for i in range(4):
            if np.count_nonzero(event[:, 2] == standard_index[i]) < min_standards\
            or np.count_nonzero(event[:, 2] == deviant_index[i]) < min_deviants\
            or np.count_nonzero(event[:, 2] == firststandard_index[i]) < min_firststandards:
                break
        else: # No bads found at end of for loop
            clean_list.append(os.path.splitext(file_event)[0])

    clean_list = sorted(clean_list)
    print(f"Analyzed: {len(path_events)}, bad: {len(path_events) - len(clean_list)}")
    print(f"{len(clean_list)} files have enough epochs for analysis.")
    
    # Initialise same proportion of participants that did either a, b, or both experiments 
    a_set = set(exp[0:3] for exp in clean_list if 'a' in exp)
    b_set = set(exp[0:3] for exp in clean_list if 'b' in exp)

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


# Normalize age for regressive age prediction:
min_age = 487
max_age = 756 # 756
range_age = max_age - min_age # 269 (9 months)

def normalize_age(age_days): # Normalizes age between -1 and 1
    return (age_days-min_age) / (0.5*range_age) - 1

def denormalize_age(value):
    return (value+1)*0.5*range_age + min_age


analyse_events_8 = {
    'GiepM_S': 2,
    'GiepM_D': 3,
    'GiepS_S': 5,
    'GiepS_D': 6,
    'GopM_S': 8,
    'GopM_D': 9,
    'GopS_S': 11,
    'GopS_D': 12,
}

analyse_events_4 = {
    'GiepM': 2,
    'GiepS': 5,
    'GopM': 8,
    'GopS': 11,
}

class EvokedDataIterator(Sequence):
    """
        An Iterator Sequence class as input to feed the model.
        The next value is given from the __getitem__ function
    """    
    
    def __init__(self, experiments, split_folder, n_experiments_batch = 8, n_trials_averaged = 60, gaussian_noise = 0):
        self.experiments = experiments
        self.n_experiments_batch = n_experiments_batch
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise
        
        metadata_path = os.path.join(local_paths.ePod_metadata, "children.txt")
        self.metadata = pd.read_table(metadata_path)
        
        self.split_path = os.path.join(local_paths.split, split_folder)
            
    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.n_experiments_batch))
    
    def __getitem__(self, index):        
        x_batch = []
        y_batch = []
        
        for i in range(self.n_experiments_batch):
            
            # Set participant
            participant_index = (index * self.n_experiments_batch + i) % len(self.experiments)
            participant = self.experiments[participant_index]
            participant_id = participant[:3]
            participant_metadata = self.metadata.loc[self.metadata['ParticipantID'] == float(participant_id)]
            
            for key in analyse_events_4:
                
                # Get Standard and Deviant file
                npy_name_S = f"{self.experiments[participant_index]}_{key}_S.npy"
                npy_name_D = f"{self.experiments[participant_index]}_{key}_D.npy"
                npy_path_S = os.path.join(self.split_path, npy_name_S)
                npy_path_D = os.path.join(self.split_path, npy_name_D)
                npy_S = np.load(npy_path_S)
                npy_D = np.load(npy_path_D)

                # Create ERP from averaging 'n_trials_averaged' trials.
                trial_indexes_S = np.random.choice(npy_S.shape[0], self.n_trials_averaged, replace=False)
                evoked_S = np.mean(npy_S[trial_indexes_S,:,:], axis=0)
                trial_indexes_D = np.random.choice(npy_D.shape[0], self.n_trials_averaged, replace=False)
                evoked_D = np.mean(npy_D[trial_indexes_D,:,:], axis=0)
                
                # Merge Standard and Deviant evoked along the channel dimensions.
                evoked = np.concatenate((evoked_S, evoked_D))
                evoked += np.random.normal(0, self.gaussian_noise, evoked.shape)
                x_batch.append(evoked)
                
                # Binary labels:
                # y = np.zeros(2)
                # if participant_metadata["Sex"].item() == "M" :
                #     y[0] = 1
                # if participant_metadata["Group_AccToParents"].item() == "At risk":
                #     y[1] = 1
                
                if str(participant[-1]) == "a":
                    y = normalize_age(int(participant_metadata[f"Age_days_a"].item()))
                if str(participant[-1]) == "b":
                    try: y = normalize_age(int(participant_metadata[f"Age_days_b"].item())) # Not all ages in metadata
                    except:  y = normalize_age(int(participant_metadata[f"Age_days_a"].item()) + 120)
                
                y_batch.append(y)
        
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)
        return np.array(x_batch), np.array(y_batch)

