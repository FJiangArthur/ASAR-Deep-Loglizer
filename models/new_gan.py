import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square

from models.tf_basemodel import tf_BasedModel

NOISE_DIM = 96


def sample_noise(batch_size: int, dim: int) -> tf.Tensor:
    return tf.random.uniform(shape=(batch_size, dim), minval=-1, maxval=1)


class Generator(tf_BasedModel):
    """Model class for the generator"""

    def __init__(
            self,
            meta_data,
            batch_sz,
            input_size,
            num_keys,
            hidden_size=100,
            num_directions=2,
            num_layers=3,
            window_size=None,
            use_attention=False,
            embedding_dim=16,
            model_save_path="./cnn_models",
            feature_type="sequentials",
            label_type="next_log",
            eval_type="session",
            topk=5,
            use_tfidf=False,
            freeze=False,
            gpu=-1,
            **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            batch_sz=batch_sz,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        self.num_labels = meta_data["num_labels"]
        self.num_layers = num_layers
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = embedding_dim
        # self.lstm = tf.keras.layers.LSTM(self.hidden_size, return_state=True)

        # self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_size))

        rnn_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation=self.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=self.input_size, activation=self.leaky_relu)
        self.bce = tf.keras.losses.CategoricalCrossentropy()


    @tf.function
    def call(self, x):
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        whole_seq_output = self.lstm(x)
        whole_seq_output = self.dense2(self.dense1(whole_seq_output))
        whole_seq_output = tf.nn.softmax(whole_seq_output)
        return tf.cast(whole_seq_output, dtype=tf.int32)

    @tf.function
    def loss_function(self, D_predictions, D_labels, G_predictions, G_labels):
        G_loss = tf.keras.losses.CategoricalCrossentropy()(D_labels, D_predictions)
        G_loss += tf.keras.metrics.mean_squared_error(G_labels, G_predictions)
        return G_loss


class Discriminator(tf_BasedModel):
    """Model class for the discriminator"""

    def __init__(
            self,
            meta_data,
            batch_sz,
            input_size,
            num_keys,
            hidden_size=100,
            num_directions=2,
            num_layers=3,
            window_size=None,
            use_attention=False,
            embedding_dim=16,
            model_save_path="./cnn_models",
            feature_type="sequentials",
            label_type="next_log",
            eval_type="session",
            topk=5,
            use_tfidf=False,
            freeze=False,
            gpu=-1,
            **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            batch_sz=batch_sz,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        self.num_labels = meta_data["num_labels"]
        self.num_layers = num_layers
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = embedding_dim


        rnn_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation=self.leaky_relu)
        # TODO: Tune 100, use leaky relu
        self.dense2 = tf.keras.layers.Dense(units=self.input_size, activation=self.leaky_relu)
        # TODO: Change to multi-class
        self.dense3 = tf.keras.layers.Dense(units=self.num_labels, activation='sigmoid')

        self.bce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def call(self, x):
        # x = tf.convert_to_tensor(input_dict["features"])

        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        x = self.lstm(x)
        whole_seq_output = self.dense1(x)

        # whole_seq_output = tf.concat((whole_seq_output, tf.cast(labels, tf.float32)), axis=1)

        whole_seq_output = self.dense2(whole_seq_output)
        whole_seq_output = self.dense3(whole_seq_output)
        whole_seq_output = tf.nn.softmax(whole_seq_output,axis=-1)
        whole_seq_output = tf.argmax(whole_seq_output,axis=1)
        return tf.cast(whole_seq_output, dtype=tf.int32)

    @tf.function
    def loss_function(self, D_real, D_real_labels, D_fake, D_fake_labels):
        G_loss = self.bce(D_real, D_real_labels) + \
                 self.bce(D_fake, D_fake_labels)
        return G_loss
