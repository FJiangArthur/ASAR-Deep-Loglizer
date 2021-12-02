import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square

from models.tf_basemodel import tf_BasedModel


class LSTM(tf_BasedModel):
    def __init__(
        self,
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
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim

        self.linear = Dense(
            self.hidden_size * self.num_directions, activation=None,
        )
        self.lstm = tf.keras.layers.LSTM(self.hidden_size)


    def call(self, input_dict):
        if self.label_type == "anomaly":
            y = tf.convert_to_tensor(input_dict["window_anomalies"])
        elif self.label_type == "next_log":
            y = tf.convert_to_tensor(input_dict["window_labels"])

        x = tf.convert_to_tensor(input_dict["features"])
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        outputs = self.lstm(x)

        logits = self.linear(tf.cast(outputs, dtype=tf.float32))
        y_pred = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, logits)

        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
