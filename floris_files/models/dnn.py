"""
Author: Bruce Shuyue Jia
Source: https://github.com/SuperBruceJia/EEG-DL
"""
import tensorflow as tf
import keras

def DNN(Input, keep_prob, weights_1, biases_1, weights_2, biases_2):
    """
    Args:
        Input: The input EEG signals
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer
    Returns:
        FC_2: Final prediction of DNN Model
        FC_1: Extracted features from the first fully connected layer
    """

    # First fully-connected layer
    FC_1 = tf.matmul(Input, weights_1) + biases_1
    FC_1 = tf.layers.batch_normalization(FC_1, training=True)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)

    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)

    return FC_2, FC_1


def NN():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1024,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)])
    return model


# From https://github.com/epodium/EEG_age_prediction:


def fully_connected_model():
    """ Returns the fully connected model from Ismail Fawaz et al. (2019). """
    n_timesteps = 512
    n_features = 32
    n_outputs = 5

    input_shape = (n_timesteps, n_features)

    input_layer = keras.layers.Input(input_shape)

    input_layer_flattened = keras.layers.Flatten()(input_layer)

    layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
    layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

    layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

    layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

    output_layer = keras.layers.Dropout(0.3)(layer_3)
    output_layer = keras.layers.Dense(n_outputs, activation='sigmoid')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


