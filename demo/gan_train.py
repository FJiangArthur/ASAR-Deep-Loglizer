#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

sys.path.append("../")
import argparse
import tensorflow as tf
import random
from models.data_generator import tf_data_generator


from models.gan import Discriminator, Generator, sample_noise
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.utils import seed_everything, dump_params, dump_final_results


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="Autoencoder", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_1.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--anomaly_ratio", default=0.1, type=float)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--cache", default=False, type=bool)

params = vars(parser.parse_args())

model_save_path = dump_params(params)
BATCH_SIZE = 1024
PATIENCE = 10
learning_rate = 1e-3
beta_1 = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)


def optimize(tape: tf.GradientTape, model: tf.keras.Model, loss: tf.Tensor) -> None:
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate_next_log(discriminator, test_loader):
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

        item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((BATCH_SIZE, )).tolist()
        item_dict['features'] = np.array(item_dict['features']).reshape((BATCH_SIZE, -1))
        item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((BATCH_SIZE,)).tolist()
        item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((BATCH_SIZE,)).tolist()

        return_dict = discriminator.call(item_dict['features'])

        y_prob_topk, y_pred_topk = tf.math.top_k(input=return_dict, k=5)

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


    session_df = store_df

    for topk in range(1, 6):
        pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
        y = (session_df["window_anomalies"] > 0).astype(int)
        window_topk_acc = 1 - (store_df["window_anomalies"].sum() / len(store_df))
        eval_results = {
            "f1": f1_score(y, pred, average='weighted', labels=np.unique(y)),
            "rc": recall_score(y, pred, average='weighted', labels=np.unique(y)),
            "pc": precision_score(y, pred, labels=np.unique(y)),
            "top{}-acc".format(topk): window_topk_acc,
        }
        logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
        if eval_results["f1"] >= best_f1:
            best_result = eval_results
            best_f1 = eval_results["f1"]
    count_end = time.time()
    logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
    return best_result


def gan_train_and_test(discriminator, generator, epoches, train_loader, test_loader, noise_z):
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

            item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((BATCH_SIZE, -1))
            item_dict['features'] = np.array(item_dict['features']).reshape((BATCH_SIZE, -1))
            item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((BATCH_SIZE, -1))
            item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((BATCH_SIZE, -1))

            y = tf.convert_to_tensor(item_dict['window_anomalies'])
            x = tf.convert_to_tensor(item_dict['features'])
            
            with tf.GradientTape(persistent=True) as tape:
                G_sample = generator(noise_z)
                labels_real = y
                logits_real = discriminator(x)
                # re-use discriminator weights on new inputs
                logits_fake = discriminator(G_sample)

                g_loss = generator.loss_function(logits_fake, logits_real)
                d_loss = discriminator.loss_function(logits_fake, logits_real)
                epoch_loss += (g_loss + d_loss)
            optimize(tape, generator, g_loss)
            optimize(tape, discriminator, d_loss)

        epoch_loss = epoch_loss / (batch_idx + 1)
        epoch_time_elapsed = time.time() - epoch_time_start
        logging.info(
            "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
        )
        if test_loader is not None and (epoch % 1 == 0):
            eval_results = evaluate_next_log(discriminator, test_loader)
            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                best_results = eval_results
                best_results["converge"] = int(epoch)
                # discriminator.save_model()
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= PATIENCE:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break

if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")
    dataset_train = tf_data_generator(session_train, feature_type=params["feature_type"], batch_size=params["batch_size"], shuffle=True)
    dataset_test = tf_data_generator(session_test, feature_type=params["feature_type"], batch_size=params["batch_size"], shuffle=True)

    batch_size = 1024
    # our noise dimension


    # create generator and discriminator
    # ext.meta_data = {'num_labels': 14, 'vocab_size': 14}
    max_input_senquence_len = dataset_train.max_input_size
    feature_len = dataset_train.feature_len
    hidden_size = 256
    num_layers = 2
    num_keys = ext.meta_data['vocab_size']
    emb_dimension = 128

    num_labels = ext.meta_data['num_labels']


    generator_model = Generator(feature_len, hidden_size, num_layers, num_keys, emb_dimension)
    generator_model.build((None, feature_len))

    discriminator_model = Discriminator(feature_len, hidden_size, num_layers, num_keys, emb_dimension, num_labels)
    discriminator_model.build((None, feature_len))

    noise_z = sample_noise(batch_size, feature_len)

    gan_train_and_test(discriminator_model, generator_model, 10, dataset_train, dataset_test, noise_z)
