"""Tools for working with the ePodium dataset."""

import mne
import numpy as np
import pandas as pd
import copy
import os
import glob
from IPython.display import clear_output

import local_paths


# INFORMATION

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1',
                 'FC5', 'T7', 'C3', 'CP1', 'CP5',
                 'P7', 'P3', 'Pz', 'PO3', 'O1', 
                 'Oz', 'O2', 'PO4', 'P4', 'P8',
                 'CP6', 'CP2', 'C4', 'T8', 'FC6',
                 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
                 'Fz', 'Cz']

channel_dict = {'Fp1':'eeg', 'AF3':'eeg', 'F7':'eeg', 'F3':'eeg', 'FC1':'eeg',
                 'FC5':'eeg', 'T7':'eeg', 'C3':'eeg', 'CP1':'eeg', 'CP5':'eeg',
                 'P7':'eeg', 'P3':'eeg', 'Pz':'eeg', 'PO3':'eeg', 'O1':'eeg', 
                 'Oz':'eeg', 'O2':'eeg', 'PO4':'eeg', 'P4':'eeg', 'P8':'eeg',
                 'CP6':'eeg', 'CP2':'eeg', 'C4':'eeg', 'T8':'eeg', 'FC6':'eeg',
                 'FC2':'eeg', 'F4':'eeg', 'F8':'eeg', 'AF4':'eeg', 'Fp2':'eeg',
                 'Fz':'eeg', 'Cz':'eeg'}

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


def load_cleaned_file(path_npy, path_events):
    npy = np.load(path_npy)
    events_12 = np.loadtxt(path_events, dtype=int)

    epochs = mne.EpochsArray(npy, mne_info, events=events_12, tmin=-0.2,
                             event_id=event_dictionary, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.info.set_montage(montage, on_missing = 'ignore')

    print(f"{npy.shape[0]} different trials loaded \nEach trial contains {npy.shape[2]} timesteps in {npy.shape[1]} channels.")    
    return epochs


# PLOTTING

def plot_ERP(epochs, condition, event_type):
    standard = condition + "_S"
    deviant = condition + "_D"

    if(event_type == "standard"):
        evoked = epochs[standard].average()    
    elif(event_type == "deviant"):
        evoked = epochs[deviant].average()    
    elif(event_type == "MMN"):
        evoked = mne.combine_evoked([epochs[deviant].average(), epochs[standard].average()], weights = [1, -1])

    fig = evoked.plot(spatial_colors = True)

def plot_array_as_evoked(array, frequency=512, baseline_start=-0.2, n_trials = 60, ylim = None):
    info = mne.create_info(channel_names, frequency, ch_types='eeg')
    evoked = mne.EvokedArray(array, info, tmin=baseline_start, nave=n_trials)
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.info.set_montage(montage, on_missing = 'ignore')
    if ylim != None:
        ylim_temp = dict(eeg=ylim)
    fig = evoked.plot(spatial_colors=True, ylim=ylim_temp)


# EVENTS

conditions = ['GiepM', "GiepS", "GopM", "GopS"]

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

# List of events without 'first standards'
analyse_events = ['GiepM_S', 'GiepM_D', 'GiepS_S', 'GiepS_D', 'GopM_S', 'GopM_D', 'GopS_S', 'GopS_D']

# Conversion matrix to put same condition with different pronounciations together
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
    """
    This function loads the events from the raw file and saves them in an external folder.
    Loading from a .txt file is many times faster than loading from raw.
    """
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

    
    