"""Tools specifically for the ePodium dataset."""

import mne
import numpy as np
import pandas as pd
import copy
import os
import glob
from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

import local_paths

# INFORMATION

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5',
               'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
               'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
               'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
               'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
               'Fz', 'Cz']

channels_mastoid = ['EXG1', 'EXG2']
channels_drop = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
frequency = 2048
mne_info = mne.create_info(channel_names, frequency, ch_types='eeg')


# LOADING

def load_events(folder_events, eeg_filenames):
    if not(os.path.exists(folder_events)):
        print("There is no folder at: ", folder_events,
              "\n first save the events in this folder.")
        return None

    events = []
    for filename in eeg_filenames:
        filepath = os.path.join(folder_events, filename + ".txt")
        events.append(np.loadtxt(filepath, dtype=int))
    print(len(events), "Event Marker files loaded")
    return events


def load_dataset(folder_dataset, file_extension='.bdf', preload=False):
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = sorted(glob.glob(pattern, recursive=True))
    eeg_dataset = []
    eeg_filenames = []

    for i, path in enumerate(eeg_filepaths):
        raw = mne.io.read_raw_bdf(path, preload=preload)
        eeg_dataset.append(raw)
        filename = os.path.split(path)[1].replace(file_extension, '')
        eeg_filenames.append(filename)
        print(i + 1, "EEG files loaded")

        clear_output(wait=True)
    print(i + 1, "EEG files loaded")

    return eeg_dataset, eeg_filenames


def load_metadata(folder, filenames):
    metadata = []
    for filename in filenames:
        path = os.path.join(folder, filename)
        metadata.append(pd.read_table(path))
    return metadata


def load_processed_file(path_npy, path_events):
    npy = np.load(path_npy)
    events_12 = np.loadtxt(path_events, dtype=int)

    epochs = mne.EpochsArray(npy, mne_info, events=events_12, tmin=-0.2,
                             event_id=event_dictionary, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.info.set_montage(montage, on_missing = 'ignore')

    print(f"{npy.shape[0]} different trials loaded \nEach trial contains {npy.shape[2]} timesteps in {npy.shape[1]} channels.")    
    return epochs


# EVENTS

event_dictionary = {
    'GiepM_FS': 1,
    'GiepM_S': 2,
    'GiepM_D': 3,
    'GiepS_FS': 4,
    'GiepS_S': 5,
    'GiepS_D': 6,
    'GopM_FS': 7,
    'GopM_S': 8,
    'GopM_D': 9,
    'GopS_FS': 10,
    'GopS_S': 11,
    'GopS_D': 12,
}

conditions = ['GiepM', "GiepS", "GopM", "GopS"]

event_conversion_12 = [
    [1, 1, 12],
    [2, 13, 24],
    [3, 25, 36],
    [4, 101, 101],
    [5, 102, 102],
    [6, 103, 103],
    [7, 37, 48],
    [8, 49, 60],
    [9, 61, 72],
    [10, 104, 104],
    [11, 105, 105],
    [12, 106, 106]
]

def group_events_12(events):
    """
    Specific to the ePODIUM dataset.
    Reduces the number of distinctive events from 78 to 12 events.
    This is done by combining different pronounciations into the same event.
    """
    events_12 = copy.deepcopy(events)
    for i in range(len(events)):
        for newValue, minOld, maxOld in event_conversion_12:
            condition = np.logical_and(
                minOld <= events_12[i], events_12[i] <= maxOld)
            events_12[i] = np.where(condition, newValue, events_12[i])
    return events_12


def save_events(folder_events, eeg_dataset, eeg_filenames):
    if not os.path.exists(folder_events):
        os.mkdir(folder_events)

    for i in range(len(eeg_dataset)):
        path_events = os.path.join(folder_events, eeg_filenames[i] + ".txt")
        if(os.path.exists(path_events)):
            print(f"Event .txt file for {eeg_filenames[i]} already exists")
        else:
            np.savetxt(path_events, mne.find_events(eeg_dataset[i], min_duration = 2/frequency), fmt='%i')
            print("\n", i + 1, " out of ", len(eeg_dataset), " saved.")                
        clear_output(wait=True)


# PROCESSING



# DEEP LEARNING

def train_test_datasets(method = "autoreject", min_standards = 180, min_deviants = 80, min_firststandards = 80):
    """
    This function checks the number of clean epochs in each file after processing.
    Each participant that exceeds the minimum epochs is put into a test or train set.
    Both the train and test sets have the same proportion of participants that did either a, b, or both experiments
    
    In an ideal experiment, there are 360 standards, 120 deviants and 130 first standards in each of the 4 conditions.
    """
    if(method == "autoreject"):
        path_method = local_paths.ePod_processed_autoreject
    if(method == "ransac"):
        path_method = local_paths.ePod_processed_ransac    

    firststandard_index = [1, 4, 7, 10]
    standard_index = [2, 5, 8, 11]
    deviant_index = [3, 6, 9, 12]

    # Experiments with enough epochs are added to clean_list
    clean_list = []

    path_events = glob.glob(os.path.join(path_method, "events", '*.txt'))
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
    train_ab, test_ab = train_test_split(experiments_a_and_b, test_size=0.25)  
    train_a, test_a = train_test_split(experiments_a_only, test_size=0.25) 
    train_b, test_b = train_test_split(experiments_b_only, test_size=0.25) 

    train = [x + 'a' for x in train_ab] + [x + 'b' for x in train_ab] + \
            [x + 'a' for x in train_a] + [x + 'b' for x in train_b]
    test = [x + 'a' for x in test_ab] + [x + 'b' for x in test_ab] + \
       [x + 'a' for x in test_a] + [x + 'b' for x in test_b]
    
    print(f"The dataset is split up into {len(train)} train and {len(test)} test experiments")    
    return train, test


class EvokedDataIterator(Sequence):
    
    def __init__(self, experiments, n_experiments = 8, n_trials_averaged = 60):
        self.experiments = experiments
        self.n_experiments = n_experiments
        self.n_trials_averaged = n_trials_averaged
        
        metadata_path = os.path.join(local_paths.ePod_metadata, "children.txt")
        self.metadata = pd.read_table(metadata_path)
        
        event_types = 12 # (FS/S/D in 4 conditions)
        self.n_files =  len(self.experiments) * event_types
        self.batch_size = self.n_experiments * event_types
    
    def __len__(self):
        # The number of batches in the Sequence.
        return int(np.ceil(len(self.experiments) / self.n_experiments))
    
    def __getitem__(self, index):
        
        x_batch = []
        y_batch = []
        
        for i in range(self.n_experiments):
            participant_index = (index * self.n_experiments + i) % len(self.experiments)
            participant_id = self.experiments[participant_index][:3]
            participant_metadata = self.metadata.loc[self.metadata['ParticipantID'] == float(participant_id)]
            
            for key in event_dictionary:
            
                # Get file
                npy_name = f"{self.experiments[participant_index]}_{key}.npy"
                npy_path = os.path.join(local_paths.ePod_processed_autoreject, "epochs_split", npy_name)
                npy = np.load(npy_path)
                
                # Create ERP from averaging 'n_trials_averaged' trials.
                trial_indexes = np.random.choice(npy.shape[0], self.n_trials_averaged, replace=False)
                evoked = np.mean(npy[trial_indexes,:,:], axis=0)
                x_batch.append(evoked)
                
                # Create labels
                y = np.zeros(5)
                if(participant_metadata["Sex"].item() == "F"):
                    y[0] = 1
                if(participant_metadata["Group_AccToParents"].item() == "At risk"):
                    y[1] = 1
                if(key.endswith("_FS")):
                    y[2] = 1
                if(key.endswith("_S")):
                    y[3] = 1
                if(key.endswith("_D")):
                    y[4] = 1
                y_batch.append(y)   
        
        return np.array(x_batch), np.array(y_batch)


# 
    
    