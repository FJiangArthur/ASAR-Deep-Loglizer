from random import random

import tensorflow as tf

class tf_data_generator(tf.keras.utils.Sequence):
        def __init__(self, session_dict, batch_size, feature_type="sequentials", shuffle=True):
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
