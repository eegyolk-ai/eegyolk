"""
Author: Bruce Shuyue Jia
Source: https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-Transformer.py
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers



# TRANSFORMER: (TransformerBlock, TokenAndPositionEmbedding, TransformerModel)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, embed_dim, maxlen):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'pos_emb': self.pos_emb,
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim
        })
        return config

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, self.maxlen, self.embed_dim])
        out = x + positions
        return out

def transformer_model(input_shape, n_classes):
    embed_dim = input_shape[0]  # Features of each time point
    maxlen = input_shape[1]     # Consider 16 input time points
    num_heads = 8   # Number of attention heads
    ff_dim = 64     # Hidden layer size in feed forward network inside transformer
    
    # Input Time-series
    inputs = layers.Input(shape=(embed_dim, maxlen))
    embedding_layer = TokenAndPositionEmbedding(embed_dim, maxlen)
    x = embedding_layer(tf.reshape(inputs, [-1]))

    # Encoder Architecture
    transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block_1(x)
    x = transformer_block_2(x)

    # Output
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes)(x)

    return keras.Model(inputs=inputs, outputs=outputs)
