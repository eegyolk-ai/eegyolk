"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
Authors: Floris Pauwels <florispauwels@live.nl>

Sequence class for each dataset to iterate over the data for
training the deep learning model.
"""

from tensorflow.keras.utils import Sequence
import random
import os
import numpy as np
import mne


class EpodiumSequence(Sequence):
    """
    An Iterator Sequence class as input to feed the model.
    The next value is given from the __getitem__ function.
    For more information on Sequences, go to:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    self.labels contains:
    ['Participant', 'Age_days_a', 'Age_days_b', 'Risk_of_dyslexia']

    *sample_rate*: The number of data points in a second of each channel.
    *n_trials_averaged*: The number of trials averaged to form the ERP.
    *gaussian_noise*: The standard deviation of the noise added
    to each datapoint to reduce overfitting.
    """

    def __init__(
        self,
        experiments,
        target_labels,
        epochs_directory,
        channel_names=None,
        sample_rate=None,
        batch_size=8,
        n_trials_averaged=30,
        gaussian_noise=0,
        standardise=False,
        input_type="standard",
        label='age'
    ):
        self.experiments = experiments
        self.labels = target_labels
        self.epochs_directory = epochs_directory
        self.channel_names = channel_names
        self.label = label
        self.batch_size = batch_size
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise
        self.standardise = standardise
        self.sample_rate = sample_rate
        self.input_type = input_type

    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.batch_size))

    def __getitem__(self, index, verbose=False):
        x_batch = []
        y_batch = []

        for i in range(self.batch_size):
            # Set participant
            index_marker = index * self.batch_size + i
            experiment_index = index_marker % len(self.experiments)
            experiment = self.experiments[experiment_index]
            participant = experiment[:3]
            participant_labels = self.labels.loc[
                self.labels['Participant'] == float(participant)]
            # Load .fif file
            if (verbose):
                print(f"Loading experiment {experiment}")
            path_epochs = os.path.join(
                self.epochs_directory, experiment + "_epo.fif")
            epochs = mne.read_epochs(path_epochs, verbose=0)

            # Modify epochs
            if self.channel_names:
                epochs.pick_channels(self.channel_names)
            if self.sample_rate:
                epochs.resample(self.sample_rate)

            # A data instance is created for each condition
            for condition in ['GiepM', "GiepS", "GopM", "GopS"]:

                # Create ERP from averaging 'n_trials_averaged' trials.
                standard_data = epochs[condition + '_S'].get_data()
                trial_indexes_standards = np.random.choice(
                    standard_data.shape[0],
                    self.n_trials_averaged,
                    replace=False,
                )
                evoked_standard = np.mean(
                    standard_data[trial_indexes_standards, :, :], axis=0)

                # Set data to standard, standard + deviant,
                # or mismatch negativity
                if self.input_type == "standard":
                    data = evoked_standard
                elif self.input_type == "standard_deviant":
                    deviant_data = epochs[condition + '_D'].get_data()
                    trial_indexes_deviants = np.random.choice(
                        deviant_data.shape[0],
                        self.n_trials_averaged,
                        replace=False,
                    )
                    evoked_deviant = np.mean(
                        deviant_data[trial_indexes_deviants, :, :], axis=0)
                    data = np.concatenate((evoked_standard, evoked_deviant))
                elif self.input_type == "MMR":
                    deviant_data = epochs[condition + '_D'].get_data()
                    trial_indexes_deviants = np.random.choice(
                        deviant_data.shape[0],
                        self.n_trials_averaged,
                        replace=False,
                    )
                    evoked_deviant = np.mean(
                        deviant_data[trial_indexes_deviants, :, :], axis=0)
                    data = evoked_deviant - evoked_standard
                else:
                    print(f"Input type: {self.input_type} unknown")

                # Create noise
                if self.gaussian_noise != 0:
                    data += np.random.normal(
                        0, self.gaussian_noise, data.shape)

                if self.standardise:
                    data = data/data.std()

                x_batch.append(data)

                # Labels
                if self.label == 'age':
                    if str(experiment[-1]) == "a":
                        y = int(participant_labels["Age_days_a"].item())
                    elif str(experiment[-1]) == "b":
                        try:
                            y = int(participant_labels["Age_days_b"].item())
                        except KeyError:
                            # If age of 'b' experiment not in metadata
                            y = int(
                                participant_labels["Age_days_a"].item()) + 120
                elif self.label == 'dyslexia':
                    y = participant_labels[f"Dyslexia_score"].item()
                else:
                    print("Label not found")

                if (verbose):
                    print(f"Target y: {y}")

                y_batch.append(y)

        # Shuffle batch
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)

        return np.array(x_batch), np.array(y_batch)


class DDPSequence(Sequence):
    """
    An Iterator Sequence class as input to feed the model.
    The next value is given from the __getitem__ function.
    For more information on Sequences, go to:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    self.labels contains:  ['filename', 'participant', 'age_group', 'age_days']
    """

    def __init__(
        self, experiments, target_labels, epochs_directory, channel_names=None,
        sample_rate=None, batch_size=8, n_instances_per_experiment=4,
        n_trials_averaged=30, gaussian_noise=0, mismatch_negativity=False,
        standardise=False
    ):

        self.experiments = experiments
        self.labels = target_labels
        self.epochs_directory = epochs_directory
        self.channel_names = channel_names

        self.instances = n_instances_per_experiment
        self.batch_size = batch_size
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise
        self.sample_rate = sample_rate
        self.standardise = standardise
        self.mismatch_negativity = mismatch_negativity

    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.batch_size))

    def __getitem__(self, index, verbose=False):
        x_batch = []
        y_batch = []

        for i in range(self.batch_size):

            # Set participant
            index_maker = index * self.batch_size + i
            experiment_index = index_maker % len(self.experiments)
            experiment = self.experiments[experiment_index]

            participant = experiment.split("_")[0]
            age_group = experiment.split("_")[1]

            experiment_labels = self.labels.loc[
                (self.labels['participant'] == float(participant))
                & (self.labels['age_group'] == int(age_group))]

            # Load .fif file
            if (verbose):
                print(f"Loading experiment {experiment}")
            path_epochs = os.path.join(
                self.epochs_directory, experiment + "_epo.fif")
            epochs = mne.read_epochs(path_epochs, verbose=0)

            # Modify epochs
            if self.sample_rate:
                epochs.resample(self.sample_rate)

            # Multiple instances are loaded from
            # the same experiment for loading efficiency.
            for j in range(self.instances):
                # Create ERP from averaging 'n_trials_averaged' trials.
                standard_data = epochs["standard"].get_data()
                trial_indexes_standards = np.random.choice(
                    standard_data.shape[0],
                    self.n_trials_averaged,
                    replace=False,
                )
                evoked_standard = np.mean(
                    standard_data[trial_indexes_standards, :, :], axis=0)

                # Set data to mismatch negativity or standard
                if self.mismatch_negativity:
                    deviant_data = epochs["deviant"].get_data()
                    trial_indexes_standards = np.random.choice(
                        deviant_data.shape[0],
                        self.n_trials_averaged,
                        replace=False,
                    )
                    evoked_deviant = np.mean(
                        deviant_data[trial_indexes_standards, :, :], axis=0)
                    data = evoked_deviant - evoked_standard
                else:
                    data = evoked_standard

                # Create noise
                data += np.random.normal(0, self.gaussian_noise, data.shape)

                if self.standardise:
                    data = data/data.std()

                x_batch.append(data)
                y_batch.append(experiment_labels["age_days"].iloc[0])

        # Shuffle batch
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)

        return np.array(x_batch), np.array(y_batch)
