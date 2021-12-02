import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square

from models.tf_basemodel import tf_BasedModel


class VariableAutoEncoder(tf_BasedModel):
    def __init__(
        self,
        meta_data,
        batch_sz,
        hidden_size=100,
        num_layers=1,
        num_directions=2,
        embedding_dim=16,
        model_save_path="./vae_models",
        feature_type="sequentials",
        label_type="none",
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
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        self.encoder = Dense(
            self.hidden_size // 2, activation=None,
        )

        self.decoder = Dense(
            self.hidden_size * self.num_directions, activation=None,
        )

        self.layers = tf.keras.Sequential([
                            self.tfEmbedder,
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_size,)),
                            self.encoder,
                            self.decoder,
                        ])




    def call(self, input_dict):
        x = tf.convert_to_tensor(input_dict["features"].tolist())
        x = self.layers(x)

        outputs, _, _ = self.rnn(tf.cast(x, dtype=tf.float32))
        # representation = outputs.mean(dim=1)
        representation = outputs[:, -1, :]

        x_internal = self.encoder(representation)
        x_recst = self.decoder(x_internal)

        pred = tf.reduce_mean(tf.keras.metrics.mean_squared_error(representation, x_recst))

        return_dict = {"loss": loss, "y_pred": pred}
        return return_dict
