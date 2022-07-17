""" Tools to help with displaying information."""

import matplotlib.pyplot as plt
import numpy as np


def show_plot(
    x=None,
    y=None,
    title="",
    xlabel="",
    ylabel="",
    legend="",
    show=True
):
    '''
     Show plot with title and lables in 1 line.

    Args:
     x: 1D numpy array

     y: 1D numpy array

    '''
    plt.clf()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)
    if (x is not None) and (y is not None):
        plt.plot(x, y)
    if show:
        plt.show()


def show_raw_fragment(raw, channel_index, duration=1, start=0, average=False):
    """Shows a fragment of the raw EEG data from specified raw file
    and channel_index.  `start_time` and `duration` are in seconds."""
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


def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


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
