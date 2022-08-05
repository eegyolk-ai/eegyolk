""" Tools to help with displaying information."""

import matplotlib.pyplot as plt
import numpy as np
import collections.abc


def show_plot(x = None, y = None, title = "", xlabel = "", ylabel = "", legend = ""):
    """Show plot with title and lables in 1 line.

    Args:
     x: 1D numpy array

     y: 1D numpy array
    """
    
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if isinstance(y, collections.abc.Sequence):
        for i in range(len(y)):
            plt.plot(x, y[i], label = legend[i])
        plt.legend()

    else:
        plt.plot(x, y)
    plt.show()


def show_raw_fragment(raw, channel_index, duration = 1, start = 0, average=False):
    """
    Shows a fragment of the raw EEG data from specified raw file
    and channel_index.  `start_time` and `duration` are in seconds.
    """
    data, times = raw[:]
    sfreq = int(raw.info["sfreq"])
    fragment = data[channel_index][start * sfreq: (start + duration) * sfreq]
    if(average):
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
        "Channel voltage (\u03BCV)",
    )


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
