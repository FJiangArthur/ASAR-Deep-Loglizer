#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../")
import argparse
import random
import tensorflow as tf
from models.transformer import Transformer
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params

parser = argparse.ArgumentParser()


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


##### Model params
parser.add_argument("--model_name", default="Transformer", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)
parser.add_argument("--nhead", default=2, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str)
parser.add_argument("--use_attention", action="store_true")
parser.add_argument("--label_type", default="next_log", type=str)
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
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)

params = vars(parser.parse_args())

model_save_path = dump_params(params)

if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    dataset_train = tf_data_generator(session_train, feature_type=params["feature_type"],
                                      batch_size=params["batch_size"], shuffle=True)
    dataset_test = tf_data_generator(session_test, feature_type=params["feature_type"], batch_size=params["batch_size"],
                                     shuffle=True)

    curr_batch_size = 1024

    # ext.meta_data = {'num_labels': 14, 'vocab_size': 14}
    model = Transformer(
        meta_data=ext.meta_data, batch_sz=curr_batch_size, model_save_path=model_save_path, **params
    )

    eval_results = model.fit(
        train_loader=dataset_train,
        test_loader=dataset_test,
        epoches=params["epoches"],
    )

    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args_str = "\t".join(
        ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]
    )

    dump_final_results(params, eval_results, model)
