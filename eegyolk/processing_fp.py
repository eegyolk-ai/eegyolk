"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
Authors: Floris Pauwels <florispauwels@live.nl>

Tools for (multi)processing raw EEG data and seperating into
epochs, with the help of the autoreject library.
"""

import mne
import numpy as np
import autoreject
import os
import glob
import multiprocessing
import functools


def process_raw_multiprocess(experiments_name, experiments_paths,
                             dataset, processed_directory, num_processes=8,
                             verbose=False):
    """
    This function sets up multiprocessing for the 'process_raw' function.
    """
    indexes = range(len(experiments_name))
    with multiprocessing.Pool(num_processes) as pool:
        # Set fixed input variables
        partial = functools.partial(process_raw,
                                    experiments_name=experiments_name,
                                    experiments_paths=experiments_paths,
                                    dataset=dataset,
                                    processed_directory=processed_directory,
                                    verbose=verbose)
        # Multiprocess with respect to raw paths
        pool.map(partial, indexes)

    print("All files processed")


def process_raw(index, experiments_name, experiments_paths,
                dataset, processed_directory, verbose=False):
    """
    This function processes a raw EEG file from the MNE class.
    Processing steps: 1. high-pass filter, 2. create epochs,
        3. low-pass filter, 4. AutoReject.
    The processed .fif file is stored in processed_directory.
    This directory should contain an 'events' folder to store
    the events as .txt file.
    The function is tested on both the ePod and the DDP dataset.

    Args:
    raw_path: Path to the raw EEG-file
    dataset: Class containing information on the dataset, e.g.
    processed_directory: Directory for storing the files.
    """
    experiment_name = experiments_name[index]
    experiment_paths = experiments_paths[index]

    # Paths for cleaned data
    path_epoch = os.path.join(processed_directory, experiment_name+"_epo.fif")
    path_events = os.path.join(processed_directory, "events",
                               experiment_name+"_events.txt")

    # Check if file needs to be processed:
    if os.path.exists(path_epoch) or os.path.exists(path_events):
        if verbose:
            print(f"Experiment {experiment_name} already cleaned \n", end='')
        # If the event .txt file is missing:
        if not os.path.exists(path_events):
            print(f"Creating the event file {experiment_name}.txt \n", end='')
            epochs_clean = mne.read_epochs(path_epoch, verbose=0)
            np.savetxt(path_events, epochs_clean.events, fmt='%i')
        return
    if experiment_name in dataset.incomplete_experiments:
        if verbose:
            print(f"Experiment {experiment_name} ignored \n", end='')
        return
    print(f"Cleaning experiment: {experiment_name}  \n", end='')

    # Read-in raw file
    try:
        raw = dataset.read_raw(experiment_paths)
        events, event_dict = dataset.events_from_raw(raw)
    except Exception:
        print(f"Unable to read-in from path {experiment_paths} \n", end='')
        return

    # Set electrodes
    raw.pick_channels(dataset.channel_names)
    raw.info.set_montage(dataset.mne_montage, on_missing='ignore')

    # High-pass filter for detrending
    raw.filter(0.1, None, verbose=False)

    # Create epochs from raw. Epoch creation sometimes returns an error.
    try:
        epochs = mne.Epochs(raw, events, event_dict, -0.2, 0.8,
                            preload=True, verbose=False)
    except Exception:
        print("Not all events of the event_dictionary in file "
              + {experiment_name} + "\n", end='')
        return

    # Low pass filter for high-frequency artifacts
    epochs.filter(None, 40, verbose=False)

    print(epochs.ch_names)
    # epochs.plot( n_epochs=5, n_channels=50)

    # Reject bad trials and repair bad sensors in EEG
    # autoreject.Ransac() is quicker but less accurate than AutoReject.
    ar = autoreject.AutoReject()
    epochs_clean = ar.fit_transform(epochs)

    # # Save data and events
    epochs_clean.save(path_epoch)
    # np.save(path_cleaned_file, epochs_clean.get_data())
    np.savetxt(path_events, epochs_clean.events, fmt='%i')


def valid_experiments(dataset, event_directory, min_standards=180,
                      min_deviants=80, min_firststandards=0):
    """
    This function checks the number of remaining epochs in the event files
    after processing. In an ideal epodium experiment, there are 360 standards,
    120 deviants and 130 first standards in each of the 4 conditions.
    """

    # Experiments with enough epochs are added to valid_experiments
    valid_experiments = []

    paths_events = glob.glob(os.path.join(event_directory, '*.txt'))
    for path_events in paths_events:
        events = np.loadtxt(path_events, dtype=int)

        if dataset.is_valid_experiment(events[:, 2], min_standards,
                                       min_deviants, min_firststandards):

            filename_events = os.path.basename(path_events)
            experiment = filename_events.split(("_events.txt"))[0]
            valid_experiments.append(experiment)

    valid_experiments = sorted(valid_experiments)
    print(f"Analyzed: {len(paths_events)} bad: "
          f"{len(paths_events) - len(valid_experiments)}")
    print(f"{len(valid_experiments)} experiments have "
          "enough epochs for analysis.")

    return valid_experiments
