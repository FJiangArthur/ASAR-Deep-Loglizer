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

        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(units=self.input_size, activation='sigmoid')

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


def evaluate_next_log(test_loader, discriminator):
    y_pred = []
    store_dict = defaultdict(list)
    infer_start = time.time()

    for batch_idx, batch_list in enumerate(test_loader):
        y = tf.convert_to_tensor([sub['window_anomalies'] for sub in batch_list])
        x = tf.convert_to_tensor([sub['features'] for sub in batch_list])

        y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk
        logits_real = discriminator(x)
        store_dict["session_idx"].extend(
            tensor2flatten_arr(batch_input["session_idx"])
        )
        store_dict["window_anomalies"].extend(
            tensor2flatten_arr(batch_input["window_anomalies"])
        )
        store_dict["window_labels"].extend(
            tensor2flatten_arr(batch_input["window_labels"])
        )
        store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
        store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
        store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
    infer_end = time.time()
    logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
    self.time_tracker["test"] = infer_end - infer_start
    store_df = pd.DataFrame(store_dict)
    best_result = None
    best_f1 = -float("inf")

    count_start = time.time()

    topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
    logging.info("Calculating acc sum.")
    hit_df = pd.DataFrame()
    for col in sorted(topkdf.columns):
        topk = col + 1
        hit = (topkdf[col] == store_df["window_labels"]).astype(int)
        hit_df[topk] = hit
        if col == 0:
            acc_sum = 2 ** topk * hit
        else:
            acc_sum += 2 ** topk * hit
    acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
    hit_df["acc_num"] = acc_sum

    for col in sorted(topkdf.columns):
        topk = col + 1
        check_num = 2 ** topk
        store_df["window_pred_anomaly_{}".format(topk)] = (
            ~(hit_df["acc_num"] <= check_num)
        ).astype(int)
    # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)

    logging.info("Finish generating store_df.")

    if self.eval_type == "session":
        use_cols = ["session_idx", "window_anomalies"] + [
            f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
        ]
        session_df = (
            store_df[use_cols].groupby("session_idx", as_index=False).sum()
        )
    else:
        session_df = store_df
    # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

    for topk in range(1, self.topk + 1):
        pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
        y = (session_df["window_anomalies"] > 0).astype(int)
        window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)
        eval_results = {
            "f1": f1_score(y, pred),
            "rc": recall_score(y, pred),
            "pc": precision_score(y, pred),
            "top{}-acc".format(topk): window_topk_acc,
        }
        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        if eval_results["f1"] >= best_f1:
            best_result = eval_results
            best_f1 = eval_results["f1"]
    count_end = time.time()
    logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
    return best_result
