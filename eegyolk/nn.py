# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions designed
to reproduce a neural net that worked with 2021 data.
"""

import os

import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    BatchNormalization,
    Flatten,
    Input,
)
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)


class NnOptimizer:

    # TODO(cmhm + wvxvw): Why/What is this?
    input_shape = (450,)

    optimization_params = (
        {},
        {'dropout': 0.5},
        {'dense_max': 300, 'dense_min': 200},
        {'need_dropuot': False},
        {'nlayers': 3},
    )

    def __init__(self, loader, epochs=1500):
        self.loader = loader
        self.epochs = epochs
        self.optimizer = Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
        )

    def fully_connected_model(
            self,
            dropout=0.3,
            dense_max=512,
            dense_min=128,
            need_dropuot=True,
            nlayers=2,
    ):
        model = keras.Sequential()

        model.add(Dense(
            dense_max,
            activation='tanh',
            input_shape=self.input_shape,
        ))
        model.add(BatchNormalization())
        if need_dropuot:
            model.add(Dropout(dropout))

        if nlayers > 2:
            model.add(Dense(
                256,
                activation='tanh',
                input_shape=self.input_shape,
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(dense_min, activation='tanh'))
        model.add(BatchNormalization())

        model.add(Dense(1))

        return model

    def model_name(self, n):
        return os.path.join(self.loader.models, f'fc_regressor_{n}.hdf5')

    def fit(self, model, n):
        model.build(self.input_shape)
        model.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer,
            metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
        )
        model.summary()

        checkpointer = ModelCheckpoint(
            filepath=self.model_name(n),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
        )
        earlystopper = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=1,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=50,
            min_lr=0.0001,
            verbose=1,
        )

        return model.fit(
            x=self.loader.x_train,
            y=self.loader.y_train,
            validation_data=(self.loader.x_val, self.loader.y_val),
            epochs=self.epochs,
            callbacks=[checkpointer, earlystopper, reduce_lr],
        )

    def fit_model(self, n):
        model = self.fully_connected_model(**self.optimization_params[n])
        return self.fit(model, n)

    def optimize(self):
        for n, p in enumerate(self.optimization_params):
            model = self.fully_connected_model(**p)
            yield self.fit(model, n)

    def predict(self, n):
        model = load_model(self.model_name(n))

        predictions = model.predict(self.loader.x_test)
        rmse = mean_squared_error(
            self.loader.y_test,
            predictions,
            squared=False)
        mae = mean_absolute_error(self.loader.y_test, predictions)

        return predictions, rmse, mae
