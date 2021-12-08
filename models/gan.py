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
    def __init__(
            self,
            input_size,
            num_keys,
            emb_dimension,
            meta_data,
            batch_sz,
            hidden_size=100,
            num_directions=2,
            num_layers=1,
            window_size=None,
            use_attention=False,
            embedding_dim=16,
            model_save_path="./lstm_models",
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
        self.lstm = tf.keras.layers.RNN(stacked_lstm, return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.hidden_size, activation='sigmoid')

    @tf.function
    def call(self, input_dict):
        x = tf.convert_to_tensor(input_dict["features"])
        # x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        inputs = self.emb(x)
        whole_seq_output, _ = self.lstm(inputs)
        whole_seq_output = self.dense2(self.dense1(whole_seq_output))

        # y_pred = tf.nn.softmax(whole_seq_output)

        # y = tf.one_hot(y, self.num_labels)
        # y = tf.squeeze(y)
        # loss = self.loss_function(whole_seq_output, y)

        # return_dict = {"loss": loss, "y_pred": y_pred}
        return whole_seq_output


    @tf.function
    def loss_function(self, predictions, labels):
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictions), logits=predictions))
        return G_loss


class Discriminator(tf_BasedModel):
    def __init__(
            self,
            input_size,
            num_keys,
            emb_dimension,
            meta_data,
            batch_sz,
            hidden_size=100,
            num_directions=2,
            num_layers=1,
            window_size=None,
            use_attention=False,
            embedding_dim=16,
            model_save_path="./lstm_models",
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

        self.input_size = input_size,
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.emb_dimension = emb_dimension
        self.num_labels = meta_data["num_labels"]

        self.emb = tf.keras.layers.Embedding(input_dim=self.hidden_size,
                                             output_dim=self.emb_dimension,
                                             embeddings_initializer='uniform',
                                             mask_zero=True)

        rnn_cells = [tf.keras.layers.LSTMCell(self.emb_dimension) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm,return_sequences=True,return_state=True)

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='sigmoid')
        # TODO: Tune 100, use leaky relu
        self.dense2 = tf.keras.layers.Dense(units=100, activation='relu')
        # TODO: Change to multi-clas
        self.dense3 = tf.keras.layers.Dense(units=self.num_labels, activation='sigmoid')


    @tf.function
    def call(self, input_dict):

        x = tf.convert_to_tensor(input_dict["features"])
        # x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        inputs = self.emb(x)
        whole_seq_output, _ = self.lstm(inputs)
        whole_seq_output = self.dense3(self.dense2(self.dense1(whole_seq_output)))

        # y_pred = tf.nn.softmax(whole_seq_output)
        #
        # y = tf.one_hot(y, self.num_labels)
        # y = tf.squeeze(y)
        # loss = self.loss_function(whole_seq_output, y)
        #
        # return_dict = {"loss": loss, "y_pred": y_pred}
        return whole_seq_output

    @tf.function
    def loss_function(self, predictions, labels):
        D_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions), logits=predictions))
        D_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(labels), logits=labels))
        return D_loss
