""" Functions for creating dummy data. """

import numpy as np
import random


def generate_sine_wave(frequency, time_points):
    return np.sin((2 * np.pi) * (time_points * frequency + random.random()))


def generate_frequency_distribution(
        distribution="planck",
        max_freq=256,
        freq_sample_rate=10,
):
    '''This function returns the occurence of frequencies up to
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
    '''
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
    '''
    Returns a single random frequency from a cumulative distribution:
        1. Sum the array cumulatively and scale from 0 to 1.
        2. Pick a random number between 0 and 1.
        3. Loop through the array until the number is >= than a random value.

    Args:
      max_freq: float
        The maximum frequency that the function can return.

      freq_distribution: 1D numpy array
        The density probability distribution
    '''
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
    '''
    Returns a single epoch of EEG dummy data.

    Args:
      freq_distribution: 1D numpy array
        The density probability distribution

      N_combined_freq: float
        Number of frequencies in epoch.
    '''
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
    '''Uses the functions from this scripts to create dataset with
    various frequency distributions.

    Args:
      size: float

      distributions: list of strings
        The names of the distributions that are generated and labeled
    '''
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
