import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square
from tensorflow import keras
import os, logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import defaultdict
from functools import partial
import math

LEARN_RATE = 1e-3
BETA_1 = 0.1


class tfEmbedder(tf.keras.Model):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            pretrain_matrix=None,
            freeze=False,
            use_tfidf=False,
    ):
        super(tfEmbedder, self).__init__()
        # For 2470: TF-IDF is the importance of the word based on the whole input log
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            num_tokens = pretrain_matrix.shape[0],
            assert (vocab_size == num_tokens)
            assert (embedding_dim == pretrain_matrix.shape[1])

            self.embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True,
                embeddings_initializer=keras.initializers.Constant(pretrain_matrix),
                trainable=True,
            )
        else:
            self.embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True,
                embeddings_initializer=keras.initializers.Constant(pretrain_matrix),
                trainable=True,
            )

    def call(self, x):
        if self.use_tfidf:
            return tf.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x)


class tf_BasedModel(tf.keras.Model):
    def __init__(
            self,
            meta_data,
            batch_sz,
            model_save_path,
            feature_type,
            label_type,
            eval_type,
            topk,
            use_tfidf,
            embedding_dim,
            cp_callback=None,
            freeze=False,
            gpu=-1,
            anomaly_ratio=None,
            patience=3,
            **kwargs,
    ):
        super(tf_BasedModel, self).__init__()
        self.batch_sz = batch_sz
        self.topk = topk
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio  # only used for auto encoder
        self.patience = patience
        self.time_tracker = {}
        self.learning_rate = LEARN_RATE
        self.beta_1 = BETA_1
        self.optimizer = None
        self.embedding_dim = embedding_dim
        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = model_save_path

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_save_file,
            verbose=1,
            save_weights_only=True,
            save_freq=5)

        assert (feature_type in ["sequentials", "semantics"])
        self.embedding_matrix = tf.Variable(
            tf.random.truncated_normal([meta_data["vocab_size"], self.embedding_dim], dtype=tf.float32, mean=0,
                                       stddev=1 / math.sqrt(self.embedding_dim)))


    def evaluate(self, test_loader, dtype="test"):
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "none":
            raise RuntimeError("Not implemented")

    def __evaluate_anomaly(self, test_loader, dtype="test"):
        y_pred = []
        store_dict = defaultdict(list)
        infer_start = time.time()

        for batch_idx, batch_list in enumerate(test_loader):
            item_dict = defaultdict(list)
            for items in batch_list:
                item_dict['session_idx'].append(items['session_idx'])
                item_dict['features'].append(items['features'])
                item_dict['window_labels'].append(items['window_labels'])
                item_dict['window_anomalies'].append(items['window_anomalies'])

            item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((self.batch_sz, -1))
            item_dict['features'] = np.array(item_dict['features']).reshape((self.batch_sz, -1))
            item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((self.batch_sz, -1))
            item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((self.batch_sz, -1))

            return_dict = self.call(batch_list)
            y_prob, _y_pred = return_dict["y_pred"].max(dim=1)
            y_pred.append(_y_pred)
            store_dict["session_idx"].extend(
                item_dict["session_idx"]
            )
            store_dict["window_anomalies"].extend(
                item_dict["window_anomalies"]
            )
            store_dict["window_preds"].extend(y_pred)

        infer_end = time.time()

        logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
        self.time_tracker["test"] = infer_end - infer_start

        store_df = pd.DataFrame(store_dict)
        use_cols = ["session_idx", "window_anomalies", "window_preds"]
        session_df = store_df[use_cols].groupby("session_idx", as_index=False).sum()
        pred = (session_df[f"window_preds"] > 0).astype(int)
        y = (session_df["window_anomalies"] > 0).astype(int)

        eval_results = {
            "f1": f1_score(y, pred),
            "rc": recall_score(y, pred),
            "pc": precision_score(y, pred),
            "acc": accuracy_score(y, pred),
        }
        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        return eval_results

    def __evaluate_next_log(self, test_loader, dtype="test"):
        y_pred = []
        store_dict = defaultdict(list)
        infer_start = time.time()
        for batch_idx, batch_list in enumerate(test_loader):
            item_dict = defaultdict(list)
            for items in batch_list:
                item_dict['session_idx'].append(items['session_idx'])
                item_dict['features'].append(items['features'])
                item_dict['window_labels'].append(items['window_labels'])
                item_dict['window_anomalies'].append(items['window_anomalies'])

            item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((self.batch_sz, )).tolist()
            item_dict['features'] = np.array(item_dict['features']).reshape((self.batch_sz, -1))
            item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((self.batch_sz,)).tolist()
            item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((self.batch_sz,)).tolist()

            return_dict = self.call(item_dict)
            y_pred = return_dict["y_pred"]
            y_prob_topk, y_pred_topk = tf.math.top_k(input=y_pred, k=self.topk)

            store_dict["session_idx"].extend(
                item_dict["session_idx"]
            )
            store_dict["window_anomalies"].extend(
                item_dict["window_anomalies"]
            )
            store_dict["window_labels"].extend(item_dict["window_labels"])
            store_dict["x"].extend(item_dict["features"])
            store_dict["y_pred_topk"].extend(y_pred_topk)
            store_dict["y_prob_topk"].extend(y_prob_topk)

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

    def save_model(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            tf.keras.models.save_model(
                self,
                self.model_save_file[0],
                save_format="tf",
                overwrite=True
            )
        except:
            raise ValueError(" s")

    def fit(self, train_loader, epoches=100, callbacks=None, test_loader=None, verbose=0):
        if not callbacks:
            callbacks = [self.cp_callback]

        if not self.optimizer:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        logging.info(
            "Start training on {} batches.".format(
                len(train_loader)
            )
        )

        best_f1 = -float("inf")
        best_results = None
        worse_count = 0

        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            train_loss = 0
            num_classes = 10
            loss = None
            batch_cnt = 0
            epoch_loss = 0

            for batch_idx, batch_list in enumerate(train_loader):
                item_dict = defaultdict(list)

                for items in batch_list:
                    item_dict['session_idx'].append(items['session_idx'])
                    item_dict['features'].append(items['features'])
                    item_dict['window_labels'].append(items['window_labels'])
                    item_dict['window_anomalies'].append(items['window_anomalies'])

                item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((self.batch_sz, -1))
                item_dict['features'] = np.array(item_dict['features']).reshape((self.batch_sz, -1))
                item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((self.batch_sz, -1))
                item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((self.batch_sz, -1))

                with tf.GradientTape(persistent=True) as tape:
                    returned_dict = self.call(item_dict)
                    loss = returned_dict['loss']
                    epoch_loss += tf.reduce_mean(loss).numpy()
                    batch_cnt += 1

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            epoch_loss = epoch_loss / batch_cnt
            print(f"Train Epoch: {epoch} \tLoss: {epoch_loss / len(train_loader):.6f}")
            epoch_time_elapsed = time.time() - epoch_time_start

            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] = epoch_time_elapsed

            if test_loader is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(test_loader)
                if eval_results["f1"] > best_f1:
                    best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    # self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= self.patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break

        # tf.keras.models.load_model(self.model_save_file)
        return best_results
