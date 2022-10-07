
"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions originally designed to work with ePODIUM EEG
data, that can be applied to other EEG data as well.

"""
import glob
import os
import pandas as pd
import hashlib
import mne


def hash_it_up_right_all(folder, extension):
    """This function creates a hash signature for every file in a folder,
        with a specified extension (e.g. cnt or fif)
        and then puts the hases into a table.

        :param folder: the folder with files to hasg
        :type folder: string

        :returns: dataframe
        :rtype: pandas.core.frame.DataFrame
        """
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
    """Band pass filter code to filter out certain frequencies.

        :param data_raw: raw EEG data, result of mne.io.read_raw functions
        :type data_raw: mne.io.Raw
        :param lo_freq: Hertz below which to disinclude
        :type lo_freq: int
        :param high_freq: Hertz above which to disinclude
        :type high_freq: int

        :returns: filtered
        :rtype: array
        """
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
    Loads the metadata stored in the metadata folder,
    and makes an excel or csv from the txt.
    Inputs are filename, path_metadata(file where metadata is), )
    make_excel_files, make_csv_files (True makes this type of file),
    path_output(where we put the file)
    Outputs are a csv and/or excel file

    """
    original_path = os.path.join(path_metadata, filename + '.txt')
    original_path = os.path.normpath(original_path)
    # why is it OK to ignore non-existent files here?
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
    """
    This is a filtering function for eeg data.
    """
    eeg = band_pass_filter(eeg, lowpass, highpass)   # bandpass filter
    eeg = eeg.notch_filter(freqs=freqs)  # notch filter
    eeg = eeg.set_eeg_reference(ref_channels=mastoid_channels)  # ref substract
    eeg = eeg.drop_channels(drop_ch)   # remove selected channels
    montage = mne.channels.make_standard_montage('standard_1020')  # set mont
    eeg.info.set_montage(montage, on_missing='ignore')
    if len(eeg.info['bads']) != 0:   # remove bad channels
        eeg = mne.pick_types(eeg.info, meg=False, eeg=True, exclude='bads')
    return eeg


def create_epoch(
    eeg,
    event_markers_simplified,
    time_before_event,
    time_after_event,
    autoreject=False
):
    """
    This function turns eeg data into epochs.
    Inputs are eeg data files, event parkers, time before event,
    and time after event
    Outputs are eeg data divided in epochs

    """
    single_epoch = mne.Epochs(
        eeg,
        event_markers_simplified,
        tmin=time_before_event,
        tmax=time_after_event
    )

    if autoreject:
        ar = autoreject.AutoReject()
        single_epoch = ar.fit_transform(single_epoch)

    return single_epoch


def evoked_responses(epochs, avg_variable):
    """
    This function creates an average evoked response for each event.
    The input is epoched data, variable where needs to be averaged on e.g.
    average per participant per event
    The output is evoked responses
    """
    evoked = []
    for event in avg_variable:
        epoch = epochs[event].average()
        evoked.append(epoch)
    return evoked


def input_mmr_prep(metadata, epochs, standard_events):
    """
    This function creates calculations about mis-match response over a set
    of participant data.
    """
    # create dataframe with expected columns
    df = pd.DataFrame(columns=["eeg_file", "paradigm", "channel", "mean"])

    # loop over all participants
    for i in range(len(metadata['eeg_file'])):

        # loop over every paradigm per participant
        for j in standard_events:
            paradigm = j
            # select the standard and deviant for a specific sequence
            #  and calculate the evoked response
            std_evoked = epochs[i][j].average()
            dev_evoked = epochs[i][j+1].average()

            # calculate the mismatch response between standard and
            #  deviant evoked
            evoked_diff = mne.combine_evoked(
                [std_evoked, dev_evoked],
                weights=[1, -1],
            )

            # get a list of all channels
            chnames_list = evoked_diff.info['ch_names']

            # compute for every channel the distance mean of the mismatch line
            for channel in chnames_list:
                chnames = mne.pick_channels(
                    evoked_diff.info['ch_names'],
                    include=[channel],
                )
                roi_dict = dict(left_ROI=chnames)  # unfortunately
                # combine_channels only takes a dictionary as input
                roi_evoked = mne.channels.combine_channels(
                    evoked_diff,
                    roi_dict,
                    method='mean',
                )
                mmr = roi_evoked.to_data_frame()
                mmr_avg = mmr['left_ROI'].mean()
                mmr_std = mmr['left_ROI'].std()
                mmr_skew = mmr['left_ROI'].skew()

                df = df.append(
                    {
                        'eeg_file': metadata['eeg_file'][i],
                        'paradigm': paradigm,
                        'channel': channel,
                        'mean':  mmr_avg,
                        'std': mmr_std,
                        'skew': mmr_skew,
                    },
                    ignore_index=True,
                )
    return df
