"""
Helper functions to work with ePODIUM EEG data

"""
import glob
import os
import pandas as pd
import numpy as np
import hashlib
import h5py
import warnings
import re
import mne
from collections import Counter


def hash_it_up_right_all(folder, extension):
    raw = {'hash': [], 'file': []}
    files = os.path.join(folder, '*' + extension)

    BUF_SIZE = 65536
    for file in glob.glob(files):
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        raw['hash'].append(result)
        raw['file'].append(file)

    return pd.DataFrame(raw)


def band_pass_filter(data_raw, lo_freq, hi_freq):
    # Band-pass filter
    # note between 1 and 40 Hz. was 0.5 to 30Hz in Stober 2016,
    # choice needs more research
    filtered = data_raw.filter(lo_freq, hi_freq, fir_design='firwin')
    return filtered


def load_metadata(
    filename,
    path_metadata,
    path_output,
    make_excel_files=True,
    make_csv_files=True
):

    """
    This function loads the metadata stored in the metadata folder,
    and makes an excel or csv from the txt.
    Inputs: filename, path_metadata(file where metadata is), )
    make_excel_files, make_csv_files (True makes this type of file),
    path_output(where we put the file)
    Outputs: csv and/or excel file
    """
    original_path = os.path.join(path_metadata, filename + '.txt')
    original_path = os.path.normpath(original_path)
    # TODO: why is it OK to ignore non-existent files here?
    if os.path.exists(original_path):
        metadata = pd.read_table(original_path)
        if make_csv_files:
            csv_path = os.path.join(path_output, filename + '.csv')
            metadata.to_csv(csv_path)
        if make_excel_files:
            excel_path = os.path.join(path_output, filename + '.xlsx')
            metadata.to_excel(excel_path)
        return metadata
    else:
        print("PATH NOT FOUND:", original_path)
        return None


def filter_eeg_raw(eeg, lowpass, highpass, freqs, mastoid_channels, drop_ch):
    eeg = band_pass_filter(eeg, lowpass, highpass)   # bandpass filter
    eeg = eeg.notch_filter(freqs=freqs)  # notch filter
    eeg = eeg.set_eeg_reference(ref_channels=mastoid_channels)  # ref substract
    eeg = eeg.drop_channels(drop_ch)   # remove selected channels
    montage = mne.channels.make_standard_montage('standard_1020')  # set mont
    eeg.info.set_montage(montage, on_missing='ignore')
    if len(eeg.info['bads']) != 0:   # remove bad channels
        eeg = mne.pick_types(eeg.info, meg=False, eeg=True, exclude='bads')
    return eeg


def create_epochs(
    eeg,
    event_markers_simplified,
    time_before_event,
    time_after_event,
):
    """
    This function turns eeg data into epochs.
    inputs: eeg data files, event parkers, time before event, time after event
    output: eeg data divided in epochs
    """
    epochs = []
    for i in range(len(eeg)):
        single_epoch = mne.Epochs(
            eeg[i],
            event_markers_simplified[i],
            tmin=time_before_event,
            tmax=time_after_event
        )
        epochs.append(single_epoch)
    return epochs


def evoked_responses(epochs, avg_variable):
    """
    This function creates an average evoked response for each event.
    input: epoched data, variable where needs to be averaged on e.g.
    average per participant per event
    output: evoked responses
    """
    evoked = []
    for i in range(len(epochs)):
        avg_epoch = []
        for j in range(len(avg_variable)):
            avg_epoch.append(epochs[i][j].average())
        evoked.append(avg_epoch)
    return evoked
