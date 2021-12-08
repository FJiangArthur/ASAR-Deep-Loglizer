import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square

NOISE_DIM = 96

def sample_noise(batch_size: int, dim: int) -> tf.Tensor:
    return tf.random.uniform(shape=(batch_size, dim), minval=-1, maxval=1)


class Generator(tf.keras.Model):
    """Model class for the generator"""

    def __init__(self, input_size, hidden_size, num_layers, num_keys, emb_dimension):
        super(Generator, self).__init__()

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = emb_dimension
        self.emb = tf.keras.layers.Embedding(input_dim=self.input_size,
                                             output_dim=self.emb_dimension,
                                             embeddings_initializer='uniform',
                                             mask_zero=True)
        # self.lstm = tf.keras.layers.LSTM(self.hidden_size, return_state=True)

        # self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_size))

        rnn_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm, return_sequences=True,return_state=True)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation=self.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=self.input_size, activation=self.leaky_relu)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        inputs = self.emb(x)
        whole_seq_output, _, state = self.lstm(inputs)
        whole_seq_output = self.dense2(self.dense1(whole_seq_output))
        return whole_seq_output

    @tf.function
    def loss_function(self, predictions, labels):
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictions), logits=predictions))
        return G_loss


class Discriminator(tf.keras.Model):
    """Model class for the discriminator"""

    def __init__(self, input_size, hidden_size, num_layers, num_keys,emb_dimension, num_labels):
        super(Discriminator, self).__init__()

        self.input_size = input_size,
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = emb_dimension
        self.num_labels = num_labels

        self.emb = tf.keras.layers.Embedding(self.input_size[0],
                                             self.emb_dimension,
                                             embeddings_initializer='uniform',
                                             mask_zero=True)

        rnn_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm,return_sequences=True,return_state=True)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='sigmoid')
        # TODO: Tune 100, use leaky relu
        self.dense2 = tf.keras.layers.Dense(units=100, activation='relu')
        # TODO: Change to multi-clas
        self.dense3 = tf.keras.layers.Dense(units=self.num_labels, activation='sigmoid')


    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # inputs = self.emb(x)
        # whole_seq_output, final_memory_state, final_carry_state = self.lstm(inputs)
        whole_seq_output = self.dense1(x)

        # whole_seq_output = tf.concat((whole_seq_output, tf.cast(labels, tf.float32)), axis=1)

        whole_seq_output = self.dense2(whole_seq_output)
        whole_seq_output = self.dense3(whole_seq_output)
        return whole_seq_output

    @tf.function
    def loss_function(self, predictions, labels):
        D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions), logits=predictions))
        D_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(labels), logits=labels))
        return D_loss

