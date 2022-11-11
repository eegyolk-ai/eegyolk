"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
Authors: Floris Pauwels <florispauwels@live.nl>

Tools to help with displaying information.
"""

import matplotlib.pyplot as plt
import numpy as np
import collections.abc
import mne


def show_plot(x=None, y=None, title="", xlabel="", ylabel="",
              xlim=None, ylim=None, legend="", show=True, scatter=False,
              scatter_color=None):
    """
    Show plot with title and lables.
    """
    
    # Initialise
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add optional axis limits.
    if(xlim):
        plt.xlim(xlim[0], xlim[1])
    if(ylim):
        plt.ylim(ylim[0], ylim[1])

    # Multiple plot types
    if isinstance(y, collections.abc.Sequence) and legend != "":
        for i in range(len(y)):
            plt.plot(x, y[i], label=legend[i])
        plt.legend()  
    elif scatter:
        if type(scatter_color) != type(None):
            scatter_color = plt.cm.seismic(scatter_color)
        plt.scatter(x, y, c=scatter_color)
    else:
        plt.plot(x, y)

    if show:
        plt.show()


def plot_raw_fragment(raw, channel_index, duration=1, start=0, average=False):
    """
    Shows a fragment of the raw EEG data from specified raw file
    and channel_index.  `start_time` and `duration` are in seconds.
    """
    data, times = raw[:]
    sfreq = int(raw.info["sfreq"])
    fragment = data[channel_index][start * sfreq: (start + duration) * sfreq]
    if average:
        # Set average to 0
        fragment -= np.average(fragment)
    # From volt to micro volt
    fragment *= 10**6
    time = times[start * sfreq: (start + duration) * sfreq]
    show_plot(
        time,
        fragment,
        "EEG data fragment",
        "time (s)",
        "Channel voltage (\u03BCV)")


def plot_ERP(epochs, condition, event_type, save_path=""):
    standard = condition + "_S"
    deviant = condition + "_D"

    if event_type == "standard":
        evoked = epochs[standard].average()
    elif event_type == "deviant":
        evoked = epochs[deviant].average()
    elif event_type == "MMN":
        evoked = mne.combine_evoked([epochs[deviant].average(),
                                    epochs[standard].average()],
                                    weights=[1, -1])

    fig = evoked.plot(spatial_colors=True)

    if save_path:
        fig.savefig(save_path)


def plot_array_as_evoked(array, channel_names, montage='standard_1020',
                         frequency=512, baseline_start=-0.2,
                         n_trials=60, ylim=None):
    """
    Plot an array as an evoked.
    Information like the sensor montage and the frequency are needed as input.
    """
    info = mne.create_info(channel_names, frequency, ch_types='eeg')
    evoked = mne.EvokedArray(array, info, tmin=baseline_start, nave=n_trials)
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.info.set_montage(montage, on_missing='ignore')
    # if ylim is not None:
    #     # e.g. ylim = dict(eeg=[-20, 20])
    #     ylim_temp = dict(eeg=ylim)
    fig = evoked.plot(spatial_colors=True)


color_dictionary = {
    1: "#8b0000",
    2: "#008000",
    3: "#000080",
    4: "#ff0000",
    5: "#ff1493",
    6: "#911eb4",
    7: "#87cefa",
    8: "#ffd700",
    9: "#696969",
    10: "#000000",
    11: "#1e90ff",
    12: "#7fff00",
}
