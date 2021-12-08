#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys
sys.path.append("../")
import argparse

import tensorflow as tf
from models.vae_new import AutoEncoder

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

class tf_data_generator(tf.keras.utils.Sequence):
        def __init__(self, session_dict, batch_size, feature_type="semantics", shuffle=True):
            self.shuffle = shuffle,
            self.max_input_size = 0
            self.batch_size = batch_size,
            flatten_data_list = []
            self.feature_len = 0
            # flatten all sessions
            for session_idx, data_dict in enumerate(session_dict.values()):
                features = data_dict["features"][feature_type]
                window_labels = data_dict["window_labels"]
                window_anomalies = data_dict["window_anomalies"]

                # This is for making the input work
                self.max_input_size = max(self.max_input_size, len(window_labels))
                for window_idx in range(len(window_labels)):
                    sample = {
                        "session_idx": session_idx,  # not session id
                        "features": features[window_idx],
                        "window_labels": window_labels[window_idx],
                        "window_anomalies": window_anomalies[window_idx],
                    }
                    self.feature_len = len(features[window_idx])
                    flatten_data_list.append(sample)
            self.flatten_data_list = flatten_data_list

        def on_epoch_end(self):
            if self.shuffle:
                random.shuffle(self.flatten_data_list)

        def __getitem__(self, idx):
            batches = self.flatten_data_list[idx * self.batch_size[0]:(idx + 1) * self.batch_size[0]]
            return batches

        def __len__(self):
            return len(self.flatten_data_list) // self.batch_size[0]

if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")
    dataset_train = tf_data_generator(session_train, feature_type=params["feature_type"],
                                      batch_size=params["batch_size"], shuffle=True)
    dataset_test = tf_data_generator(session_test, feature_type=params["feature_type"], batch_size=params["batch_size"],
                                     shuffle=True)

    curr_batch_size = params["batch_size"]
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
    model = AutoEncoder(
        meta_data=ext.meta_data, batch_sz=curr_batch_size, model_save_path=model_save_path, **params
    )
    model.compile(
                  optimizer=tf.keras.optimizers.Adam(1e-4,clipnorm=1.),
                  metrics=['accuracy'])

    eval_results = model.fit(
        train_loader = dataset_train,
        test_loader=dataset_test,
        epoches=params["epoches"],

    )

    dump_final_results(params, eval_results, model)
