"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for loading raw data, metadata, and events."""

import mne
import pandas as pd
import numpy as np
import os
import glob
# from IPython.display import clear_output
from eegyolk.display_helper import make_ordinal  # added eegyolk.


def generator_load_dataset(
    folder_dataset,
    file_extension='.bdf',
    preload=True
):
    """
    Documentation
    """
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = glob.glob(pattern, recursive=True)
    for path in eeg_filepaths:
        bdf_file = mne.io.read_raw_bdf(path, preload=preload)
        print('file is read')
        eeg_filename = os.path.split(path)[1].replace(file_extension, '')
        yield bdf_file, eeg_filename
        # clear_output(wait=True)
    print(len(eeg_filepaths), "EEG files loaded")


def load_dataset(
    folder_dataset,
    file_extension='.bdf',
    preload=True,
    output=False,
    verbose=False
):
    """
    This function is for datasets under five files. Otherwise
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
            raw = mne.io.read_raw_bdf(path, preload=preload, verbose=verbose)

        if file_extension == '.cnt':  # .cnt files do not always load.
            try:
                raw = mne.io.read_raw_cnt(
                    path,
                    preload=preload,
                    verbose=verbose
                )

            except Exception:
                eeg_filenames_failed_to_load.append(filename)
                files_failed_to_load += 1
                print(f"File {filename} could not be loaded.")
                continue

        eeg_dataset.append(raw)
        eeg_filenames.append(filename)
        files_loaded += 1
        if output:
            print(files_loaded, "EEG files loaded")
        # if preload and files_loaded >= max_files_preloaded : break

        # clear_output(wait=True)
    print(len(eeg_dataset), "EEG files loaded")
    if files_failed_to_load > 0:
        print(files_failed_to_load, "EEG files failed to load")

    return eeg_dataset, eeg_filenames


def i_load_metadata(folder, filenames):
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
        np.savetxt(path_events, mne.find_events(eeg_dataset[i]), fmt='%i')
        print("\n", i + 1, " out of ", len(eeg_dataset), " saved.")
        # clear_output(wait=True)


def read_filtered_data(metadata, to_array=False, verbose=False):
    epochs = []
    for index, file in metadata.iterrows():
        path = os.path.join(file['path_epoch'], file['epoch_file'])
        epoch = mne.read_epochs(path, preload=False, verbose=verbose)
        if to_array is True:
            epoch = epoch.get_data()
        epochs.append(epoch)
    return epochs


def caller_save_events(folder_events, generator_argument):
    """
    Events are loaded from raw EEG files and saved in .txt file.
    Loading from .txt file is much faster than from EEG file.
    """
    if not os.path.isdir(folder_events):
        os.mkdir(folder_events)

    for i, (file, filename) in enumerate(generator_argument):
        path_events = os.path.join(folder_events, filename + ".txt")
        np.savetxt(path_events, mne.find_events(file), fmt='%i')
        print("\n", i, " saved.")
        # clear_output(wait=True)


def load_events(folder_events, eeg_filenames):
    """
    Events are saved and loaded externally from .txt file,
    since loading events from raw EEG file takes much longer.
    NB: eeg_filenames should not include extension or root directory.
    """
    if not (os.path.exists(folder_events)):
        print("There is no folder at: ", folder_events,
              "\n first save the events in this folder.")
        return None

    events = []
    for filename in eeg_filenames:
        filepath = os.path.join(folder_events, filename + ".txt")
        events.append(np.loadtxt(filepath, dtype=int))
    print(len(events), "Event Marker files loaded")
    return events


def print_event_info(
    events,
    participant_index=5,
    event_index=500,
    sample_frequency=2048
):
    """
    Prints information on a specified event marker.
    """
    event_time = events[participant_index][event_index][0]
    event_ID = events[participant_index][event_index][2]
    print((
        "Participant {} heard event ID: {} after {:.1f} seconds "
        "as the {} event"
    ).format(
        participant_index,
        event_ID,
        event_time / sample_frequency,
        make_ordinal(event_index),
    ))
