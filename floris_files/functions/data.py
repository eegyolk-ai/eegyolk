import mne
import numpy as np
import pandas as pd
import os
import glob
from IPython.display import clear_output

############ --- LOADING --- #################

def load_raw_dataset(dataset_directory, file_extension='.bdf', preload=False):
    pattern = os.path.join(dataset_directory, '**/*' + file_extension)
    raw_paths = sorted(glob.glob(pattern, recursive=True))
    experiments_raw = []
    experiments_id = []

    for i, path in enumerate(raw_paths):
        raw = mne.io.read_raw_bdf(path, preload=preload)
        experiments_raw.append(raw)
        filename = os.path.split(path)[1].replace(file_extension, '')
        experiments_id.append(filename)
        print(i+1, "EEG files loaded")

        clear_output(wait=True)
    print(i+1, "EEG files loaded")

    return experiments_raw, experiments_id

def load_events(events_directory, experiments):
    if not(os.path.exists(events_directory)):
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


############ --- SAVING --- #################

def save_events(events_directory, experiments_raw, experiments_id):
    """
    This function loads the events from the raw file and saves them in an external directory.
    Loading from a .txt file is many times faster than loading from raw.
    """
    if not os.path.exists(events_directory):
        os.mkdir(events_directory)

    for i in range(len(experiments_raw)):
        path_events = os.path.join(events_directory, experiments_id[i] + ".txt")
        if(os.path.exists(path_events)):
            print(f"Event .txt file for {experiments_id[i]} already exists")
        else:
            np.savetxt(path_events, mne.find_events(experiments_raw[i], min_duration = 2/frequency), fmt='%i')
            print("\n", i + 1, " out of ", len(experiments_raw), " saved.")                
        clear_output(wait=True)