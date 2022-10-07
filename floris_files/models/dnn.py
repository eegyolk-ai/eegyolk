# From https://github.com/epodium/EEG_age_prediction:


def fully_connected_model():
    """ Returns the fully connected model from Ismail Fawaz et al. (2019). """
    n_timesteps = 512
    n_features = 32
    n_outputs = 3

    input_shape = (n_features, n_timesteps)

    input_layer = keras.layers.Input(input_shape)

    input_layer_flattened = keras.layers.Flatten()(input_layer)

    layer_1 = keras.layers.Dropout(0.5)(input_layer_flattened)
    layer_1 = keras.layers.Dense(100, activation='relu')(layer_1)

    layer_2 = keras.layers.Dropout(0.4)(layer_1)
    layer_2 = keras.layers.Dense(50, activation='relu')(layer_2)

    layer_3 = keras.layers.Dropout(0.3)(layer_2)
    layer_3 = keras.layers.Dense(30, activation='relu')(layer_3)

    output_layer = keras.layers.Dropout(0.2)(layer_3)
    output_layer = keras.layers.Dense(n_outputs, activation='sigmoid')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


