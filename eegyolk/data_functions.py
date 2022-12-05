"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions designed to  create simulated "dummy" data.
"""

import random
import numpy as np
import mne
import pandas as pd
import os
import glob
from IPython.display import clear_output


def generate_sine_wave(frequency, time_points):
    """
    Function to generate a sine wave.
    """
    return np.sin((2 * np.pi) * (time_points * frequency + random.random()))


def generate_frequency_distribution(
        distribution="planck",
        max_freq=256,
        freq_sample_rate=10,
):
    """
    This function returns the occurence of frequencies up to
    'max_freq' from a 'distribution'.  Returns arrays of frequencies
    and their occurence as x and y value.  The form of :math:`1 / (x^2 *
    (exp(1/x)-1))` is inspired by Planck's law.

    Args:
      distribution: string
        The shape of the distribution. Choose from "planck", "constant",
        "linear" or create a new one.

      max_freq: float
        The largest frequency that the function considers.

    freq_sample_rate: float
    """
    frequencies = np.linspace(
        0,
        max_freq,
        max_freq * freq_sample_rate,
        endpoint=False,
    )
    # divide by zero error if f == 0
    f_temp = np.where(frequencies == 0, 1e-2, frequencies)

    if distribution == "planck":
        return 1 / (f_temp**2 * (np.exp(1 / f_temp) - 1))
    if distribution == "constant":
        return np.ones(max_freq * freq_sample_rate)
    if distribution == "linear":
        return np.linspace(1, 0, max_freq * freq_sample_rate, endpoint=False)
    print("Correct distribution not found")
    return 1 / (f_temp**2 * (np.exp(1 / f_temp) - 1))


def random_frequency_from_density_distribution(max_freq, freq_distribution):
    """
    Returns a single random frequency from a cumulative distribution:
        1. Sum the array cumulatively and scale from 0 to 1.
        2. Pick a random number between 0 and 1.
        3. Loop through the array until the number is >= than a random value.

    Args:
      max_freq: float
        The maximum frequency that the function can return.

      freq_distribution: 1D numpy array
        The density probability distribution
    """
    cumulative = np.cumsum(freq_distribution)
    cumulative /= cumulative[-1]
    random_value = random.random()
    frequencies = np.linspace(
        0,
        max_freq,
        len(freq_distribution),
        endpoint=False,
    )
    for i, cum_value in enumerate(cumulative):
        if cum_value >= random_value:
            return frequencies[i]
    return frequencies[i]


def generate_epoch(
        freq_distribution,
        N_combined_freq=100,
        max_freq=256,
        duration=2,
        sample_rate=512,
):
    """
    Returns a single epoch of EEG dummy data.

    Args:
      freq_distribution: 1D numpy array
        The density probability distribution

      N_combined_freq: float
        Number of frequencies in epoch.
    """
    N_time_points = sample_rate * duration

    # Create epoch by summing up sines of different frequencies
    epoch = np.zeros(N_time_points)
    time_points = np.linspace(0, duration, N_time_points, endpoint=False)
    for i in range(N_combined_freq):
        freq = random_frequency_from_density_distribution(
            max_freq,
            freq_distribution,
        )
        epoch += generate_sine_wave(freq, time_points)

    return epoch


def create_labeled_dataset(size, distributions=["planck", "constant"]):
    """
    Uses the functions from this scripts to create dataset with
    various frequency distributions.

    Args:
      size: float

      distributions: list of strings
        The names of the distributions that are generated and labeled
    """
    X = []
    Y = np.zeros(size)

    dist = []
    for distribution in distributions:
        dist.append(generate_frequency_distribution(distribution))

    for i in range(size):
        randDist = random.randint(0, len(distributions) - 1)
        X.append(generate_epoch(dist[randDist]))
        Y[i] = randDist
    return np.array(X), Y


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


def save_longer_events(
        events_directory,
        experiments_raw,
        experiments_id,
        freq,
):
    """
    This function loads the events from the raw file and saves them in
    an external directory. Loading from a .txt file is many times faster
    than loading from raw. This was written by Floris Pauwels for this thesis,
    and he originally placed it in the data_io module as save_events.
    This function only saves event of a certain legnth based on
    a comparison to frequency.
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


def simple_load_metadata(metadata_directory, metadata_filenames):
    """
    Metadata should usually be loaded with helper_function.load_metadata,
    however this code is inlcuded as it was used in the code of
    Floris Pauwels for this thesis under the function
    name data_io.load_metadata
    """
    metadata = []
    for filename in metadata_filenames:
        path = os.path.join(metadata_directory, filename)
        metadata.append(pd.read_table(path))
    return metadata


def load_raw_dataset(dataset_directory,
                     file_extension='.bdf',
                     preload=False,
                     max_files=9999):
    """
    A function to load existing ePodium datasets. Contact Utrecht for data.
    """
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
