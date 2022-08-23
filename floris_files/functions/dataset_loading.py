""" Functions for loading raw data, metadata, and events."""

import mne
import pandas as pd
import numpy as np
import os
import glob
from IPython.display import clear_output

def load_dataset(folder_dataset, file_extension='.bdf', preload=False):
    """
    This function is for datasets under 5 files. Otherwise
    use generator_load_dataset.
    Reads and returns the files that store the EEG data,
    along with a list of the filenames and paths of these bdf files.
    Takes as input the top folder location of the dataset.
    """
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = glob.glob(pattern, recursive=True)
    eeg_dataset = []
    eeg_filenames = []
    eeg_filenames_failed_to_load = []

    files_loaded = 0
    files_failed_to_load = 0
    for path in eeg_filepaths:
        filename = os.path.split(path)[1].replace(file_extension, '')

        if file_extension == '.bdf':
            raw = mne.io.read_raw_bdf(path, preload=preload)

        if file_extension == '.cnt':  # .cnt files do not always load.
            try:
                raw = mne.io.read_raw_cnt(path, preload=preload)
                # TODO: What kinds of exceptions are expected here?
            except Exception:
                eeg_filenames_failed_to_load.append(filename)
                files_failed_to_load += 1
                print(f"File {filename} could not be loaded.")
                continue

        eeg_dataset.append(raw)
        eeg_filenames.append(filename)
        files_loaded += 1
        print(files_loaded, "EEG files loaded")
        # if preload and files_loaded >= max_files_preloaded : break

        clear_output(wait=True)
    print(len(eeg_dataset), "EEG files loaded")
    if files_failed_to_load > 0:
        print(files_failed_to_load, "EEG files failed to load")

    return eeg_dataset, eeg_filenames


def load_metadata(folder, filenames):
    """
    Reads and returns the four metadata text files.
    Takes as input the folder location of the metadata files.
    """
    metadata = []
    for filename in filenames:
        path = os.path.join(folder, filename)
        metadata.append(pd.read_table(path))
    return metadata


def save_events(folder_events, eeg_dataset, eeg_filenames):
    """
    Events are loaded from raw EEG files and saved in .txt file.
    Loading from .txt file is much faster than from EEG file.
    """
    if not os.path.exists(folder_events):
        os.mkdir(folder_events)

    for i in range(len(eeg_dataset)):
        path_events = os.path.join(folder_events, eeg_filenames[i] + ".txt")
        if(os.path.exists(folder_events)):
            print(f"Event .txt file for {eeg_filenames[i]} already existed")
        else:
            np.savetxt(path_events, mne.find_events(eeg_dataset[i]), fmt='%i')
            print("\n", i + 1, " out of ", len(eeg_dataset), " saved.")                
        clear_output(wait=True)


def load_events(folder_events, eeg_filenames):
    """
    Events are saved and loaded externally from .txt file,
    since loading events from raw EEG file takes much longer.
    NB: eeg_filenames should not include extension or root directory.
    """
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