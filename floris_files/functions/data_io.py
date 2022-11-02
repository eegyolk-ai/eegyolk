"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
Authors: Floris Pauwels <florispauwels@live.nl>

Tools to save and load data to and from storage.
"""

import mne
import numpy as np
import pandas as pd
import os
import glob
from IPython.display import clear_output

# ########### --- LOADING --- ############# #


def load_raw_dataset(dataset_directory,
                     file_extension='.bdf',
                     preload=False,
                     max_files=9999):

    pattern = os.path.join(dataset_directory, '**/*' + file_extension)
    raw_paths = sorted(glob.glob(pattern, recursive=True))
    experiments_raw = []
    experiments_id = []

    for path in raw_paths:
        # Support for multiple file extensions
        if file_extension == '.bdf':
            raw = mne.io.read_raw_bdf(path, preload=preload)
        elif file_extension == '.cnt':
            # .cnt files do not always load.
            try:
                raw = mne.io.read_raw_cnt(path, preload=preload)
            except Exception:
                print(f"File {filename} could not be loaded.")
                continue

        experiments_raw.append(raw)
        filename = os.path.split(path)[1].replace(file_extension, '')
        experiments_id.append(filename)

        print(len(experiments_id), "EEG files loaded")
        if len(experiments_id) >= max_files:
            break

        clear_output(wait=True)

    return experiments_raw, experiments_id


def load_events(events_directory, experiments):
    if not os.path.exists(events_directory):
        print("There is no directory at: ", events_directory,
              "\n first save the events in this directory.")
        return None

    events = []
    for experiment in experiments:
        event_path = os.path.join(events_directory, experiment + ".txt")
        events.append(np.loadtxt(event_path, dtype=int))
    print(len(events), "Event Marker files loaded")
    return events


def load_metadata(metadata_directory, metadata_filenames):
    metadata = []
    for filename in metadata_filenames:
        path = os.path.join(metadata_directory, filename)
        metadata.append(pd.read_table(path))
    return metadata


# ########### --- SAVING --- ############# #

def save_events(events_directory, experiments_raw, experiments_id, freq):
    """
    This function loads the events from the raw file and saves them in
    an external directory. Loading from a .txt file is many times faster
    than loading from raw.
    """
    if not os.path.exists(events_directory):
        os.mkdir(events_directory)

    for i in range(len(experiments_raw)):
        path_events = os.path.join(events_directory, experiments_id[i]+".txt")
        if os.path.exists(path_events):
            print(f"Event .txt file for {experiments_id[i]} already exists")
        else:
            events = mne.find_events(experiments_raw[i], min_duration=2/freq)
            np.savetxt(path_events, events, fmt='%i')
            print("\n", i + 1, " out of ", len(experiments_raw), " saved.")
        clear_output(wait=True)

def save_experiment_names(experiments_set, path_txt):
    """
    This function saves the names of the experiments in experiments_set 
    in path_txt as a .txt file.   
    This is used for saving the train, test and validation sets of a model.    
    """
    with open(path_txt, 'w') as f:
        for experiment in experiments_set:
            f.write(experiment + '\n')
    return
    