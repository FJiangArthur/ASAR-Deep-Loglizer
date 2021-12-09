#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
from collections import defaultdict
import time

import numpy as np

sys.path.append("../")
import argparse

from models.data_generator import tf_data_generator
from models.new_gan import Discriminator, Generator, sample_noise
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params
import tensorflow as tf

parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="Autoencoder", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--num_layers", default=3, type=int)
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


learning_rate = 1e-3
beta_1 = 0.1
gan_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

def optimize(tape: tf.GradientTape, model: tf.keras.Model, loss: tf.Tensor) -> None:
    gradients = tape.gradient(loss, model.trainable_variables)
    gan_optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def new_GAN_train(discriminator, generator, train_loader,  epoches = 100, callbacks=None, test_loader=None):
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

            item_dict['session_idx'] = np.array(item_dict['session_idx']).reshape((discriminator.batch_sz, -1))
            item_dict['features'] = np.array(item_dict['features']).reshape((discriminator.batch_sz, -1))
            item_dict['window_labels'] = np.array(item_dict['window_labels']).reshape((discriminator.batch_sz, -1))
            item_dict['window_anomalies'] = np.array(item_dict['window_anomalies']).reshape((discriminator.batch_sz, -1))

            y = tf.convert_to_tensor(item_dict['window_labels'], dtype='int32')
            x = tf.convert_to_tensor(item_dict['features'], dtype='int32')


            with tf.GradientTape(persistent=True) as tape:
                # Generator Train
                G_sample_fake = generator.call(x)
                D_sample_fake = tf.reshape(discriminator.call(G_sample_fake), (-1, 1))
                G_sample_real = y
                g_loss = generator.loss_function(tf.cast(D_sample_fake, dtype='float32'), tf.zeros_like(D_sample_fake, dtype='float32'),
                                                 tf.cast(G_sample_fake,dtype='float32'),  tf.cast(x, dtype='float32'))


            optimize(tape, generator, g_loss)

            with tf.GradientTape(persistent=True) as tape:
                # Discriminator Train
                G_sample_fake = generator.call(x)
                D_sample_fake = discriminator.call(G_sample_fake)
                D_sample_real = tf.ones_like(D_sample_fake,dtype=np.int32)
                d_loss = discriminator.loss_function(D_sample_fake, D_sample_real,
                                                 G_sample_fake, tf.zeros_like(G_sample_fake,dtype=np.int32))

            optimize(tape, discriminator, d_loss)

        epoch_loss = epoch_loss / batch_cnt
        print(f"Train Epoch: {epoch} \tLoss: {epoch_loss / len(train_loader):.6f}")
        epoch_time_elapsed = time.time() - epoch_time_start

        logging.info(
            "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
        )
        generator.time_tracker["train"] = epoch_time_elapsed

        if test_loader is not None and (epoch % 1 == 0):
            eval_results = generator.evaluate(test_loader)
            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                best_results = eval_results
                best_results["converge"] = int(epoch)
                # self.save_model()
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= generator.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break

    # tf.keras.models.load_model(self.model_save_file)
    return best_results

if __name__ == "__main__":
    seed_everything(params["random_seed"])
    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    dataset_train = log_dataset(session_train, feature_type=params["feature_type"])

    dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")
    dataset_train = tf_data_generator(session_train, feature_type=params["feature_type"],
                                      batch_size=params["batch_size"], shuffle=True)
    dataset_test = tf_data_generator(session_test, feature_type=params["feature_type"], batch_size=params["batch_size"],
                                     shuffle=True)

    curr_batch_size = 1024

    # ext.meta_data = {'num_labels': 14, 'vocab_size': 14}
    max_input_senquence_len = dataset_train.max_input_size
    feature_len = dataset_train.feature_len
    hidden_size = 256
    num_keys = ext.meta_data['vocab_size']
    emb_dimension = 128

    num_labels = ext.meta_data['num_labels']
    batch_size = 1024

    generator_model = Generator(input_size=feature_len, num_keys=num_keys, emb_dimension=emb_dimension,
                                meta_data=ext.meta_data, batch_sz=curr_batch_size, model_save_path=model_save_path, **params)
    # generator_model.build((None, feature_len))
    # generator_model.build((None, max_input_senquence_len))
    # inputs = tf.keras.Input(shape=(max_input_senquence_len,))
    # abe = generator_model.call(inputs)

    discriminator_model = Discriminator(input_size=feature_len, num_keys=num_keys, emb_dimension=emb_dimension,
                                        meta_data=ext.meta_data, batch_sz=curr_batch_size, model_save_path=model_save_path, **params)
    # discriminator_model.build((None, feature_len))
    # inputs = tf.keras.Input(shape=(max_input_senquence_len,))
    # abe = discriminator_model.call(inputs)


    new_GAN_train(discriminator=discriminator_model, generator=generator_model, epoches=10, train_loader=dataset_train)