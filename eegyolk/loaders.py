"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions  designed
to reproduce previous ML work from 2021.
"""

import os

from glob import glob

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class RegressionsLoader:

    def __init__(self, preprocessed, models, samples=None):
        self.preprocessed = preprocessed
        self.models = models
        self.samples = samples

        self.loaded = None
        self.feature_names = None

        # TODO(wvxvw): The self.codes_xxx doesn't seem to be ever
        # used.
        self.x_train = None
        self.y_train = None
        self.codes_train = None

        self.x_val = None
        self.y_val = None
        self.codes_val = None

        self.x_test = None
        self.y_test = None
        self.codes_test = None

        self.x_train_val = None
        self.y_train_val = None
        self.codes_train_val = None

        try:
            os.makedirs(self.models)
        except FileExistsError:
            pass

    def load_all(self, h5s, csvs):
        for h5, csv in zip(h5s, csvs):
            df_features = pd.read_hdf(h5)
            df_metadata = pd.read_csv(csv)

            df_features['label'] = df_metadata['age_months'][0]
            df_features['code'] = df_metadata['code'][0]
            yield df_features

    def load(self):
        h5s = sorted(glob(os.path.join(self.preprocessed, '*.h5')))
        csvs = sorted(glob(os.path.join(self.preprocessed, '*.csv')))

        try:
            self.loaded = pd.concat(self.load_all(h5s, csvs))
        except ValueError as e:
            if not os.path.isdir(self.preprocessed) or (not h5s) or (not csvs):
                raise FileNotFoundError(
                    f'{self.preprocessed} should exist and be non-empty'
                ) from e
            raise
        self.feature_names = self.loaded.columns.values

    def refine_ds(self, ds):
        x = ds.drop(['label', 'code'], axis=1).reset_index(drop=True)
        y = ds['label'].reset_index(drop=True)
        codes = ds['code'].reset_index(drop=True)
        # MARK: reducing from 64 bit float to 32 bit float, to reduce
        # memory usage
        scaler = StandardScaler()
        return (
            pd.DataFrame(scaler.fit_transform(x)).astype('float32'),
            y,
            codes,
        )

    def downscale(self):
        self.x_train_val = self.x_train_val.head(self.samples)
        self.y_train_val = self.y_train_val.head(self.samples)

    def split(self):
        code = self.loaded['code']
        # List all the unique subject IDs
        subject_ids = code.unique()

        ids_train, ids_temp = train_test_split(
            subject_ids,
            test_size=0.3,
            random_state=42,
        )
        ids_test, ids_val = train_test_split(
            ids_temp,
            test_size=0.5,
            random_state=42,
        )

        # Split the DataFrames into train, validation and test
        df_train = self.loaded[code.isin(ids_train)]
        df_val = self.loaded[code.isin(ids_val)]
        df_test = self.loaded[code.isin(ids_test)]

        self.x_train, self.y_train, self.codes_train = self.refine_ds(df_train)
        self.x_val, self.y_val, self.codes_val = self.refine_ds(df_val)
        self.x_test, self.y_test, self.codes_test = self.refine_ds(df_test)

        # For the ML models, the data set has been split into train
        # and test.  Only for the simple feedforward neural network,
        # we've also used a validation set (taken from the train
        # set).  The test set remains the same.
        self.x_train_val = pd.concat((self.x_train, self.x_val))
        self.y_train_val = pd.concat((self.y_train, self.y_val))
        self.codes_train_val = pd.concat((self.codes_train, self.codes_val))

        # Shuffle data before using
        self.x_train_val, y_train_val, codes_train_val = shuffle(
            self.x_train_val,
            self.y_train_val,
            self.codes_train_val,
            random_state=42,
        )

        if self.samples is not None:
            self.downscale()
