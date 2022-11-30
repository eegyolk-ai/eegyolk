
"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions originally designed to work with ePODIUM EEG
data, that can be applied to other EEG data as well.

"""
import glob
import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout
from tensorflow.keras.layers import AveragePooling1D, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow import keras
from tensorflow.keras import layers


def SVM(X, y, X_train, y_train, X_test, y_test):
    # gridsearch for hyper parameters
    svm = SVC()
    parameters = {
        'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'C': [0.0001, 10000],
        'gamma': ('auto', 'scale'),
    }
    clf = GridSearchCV(svm, parameters)
    clf.fit(X_train, y_train)
    print("Hyperparameters: ", clf.best_params_)

    # model definition
    svm = SVC(
        C=10000,
        kernel=clf.best_params_.get('kernel'),
        gamma=clf.best_params_.get('gamma'),
    )  # C=0.0001, kernel='linear', gamma='auto', random_state=True
    svm.fit(X_train, y_train)

    # with kfold cross validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    model = svm
    acc_score = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold = fold + 1
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        print('Predictions of fold {}:'.format(fold), pred_values)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score)/k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    return


def LR(X, y, X_train, y_train, X_test, y_test):
    # grid search
    lr = LogisticRegression()
    parameters = {
        'C': [500, 1000],
        'max_iter': [4000, 5000],
        'solver': ('liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'),
    }  # 'penalty':('l1', 'l2', 'elasticnet')

    clf = GridSearchCV(lr, parameters)
    clf.fit(X_train, y_train)
    print("Hyperparameters: ", clf.best_params_)

    # model definition
    lr = LogisticRegression(C=1000, solver=clf.best_params_.get('solver'))
    lr.fit(X_train, y_train)

    # with kfold cross validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    model = lr
    acc_score = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold = fold + 1
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        print('Predictions of fold {}:'.format(fold), pred_values)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score)/k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    return


def DT(X, y, X_train, y_train, X_test, y_test):
    # grid search
    dt = tree.DecisionTreeClassifier()
    parameters = {
        'criterion': ('gini', 'entropy', 'log_loss'),
        'splitter': ('random', 'best'),
        'max_depth': [2, 20],
    }
    clf = GridSearchCV(dt, parameters)
    clf.fit(X_train, y_train)
    print("Hyperparameters: ", clf.best_params_)

    # model definition
    dt = tree.DecisionTreeClassifier(
        criterion=clf.best_params_.get('criterion'),
        splitter=clf.best_params_.get('splitter'),
        max_depth=clf.best_params_.get('max_depth'),
    )
    dt.fit(X_train, y_train)
    tree.plot_tree(dt)

    # with kfold cross validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    model = tree.DecisionTreeClassifier(max_depth=5)
    acc_score = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold = fold + 1
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)
        print('Predictions of fold {}:'.format(fold), pred_values)

    avg_acc_score = sum(acc_score)/k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    return


def MLP(X, y, X_train, y_train, X_test, y_test):
    # grid search
    NN = MLPClassifier()
    parameters = {
        'activation': ('identity', 'relu', 'logistic', 'tanh'),
        'alpha': [0.00001, 10],
        'solver': ('lbfgs', 'sgd', 'adam'),
        'learning_rate': ('invscaling', 'adaptive'),
        'max_iter': [4000, 5000],
    }
    clf = GridSearchCV(NN, parameters)
    clf.fit(X_train, y_train)
    print("Hyperparameters: ", clf.best_params_)

    # model definition
    NN = MLPClassifier(
        activation=clf.best_params_.get('activation'),
        alpha=clf.best_params_.get('alpha'),
        solver=clf.best_params_.get('solver'),
        learning_rate=clf.best_params_.get('learning_rate'),
        max_iter=clf.best_params_.get('max_iter'),
    )
    NN.fit(X_train, y_train)

    # with kfold cross validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    model = NN
    acc_score = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold = fold + 1
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        NN.fit(X_train, y_train)
        pred_values = NN.predict(X_test)
        print('Predictions of fold {}:'.format(fold), pred_values)
        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score)/k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    return


def CNN(df, epochs, batch_size):
    # create tensor
    target = df.pop('Group_AccToParents')
    tf.convert_to_tensor(df)

    # normalize data
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(df)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target,
        test_size=0.2,
        random_state=42,
    )

    # train model
    model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
          ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # fit model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    return
