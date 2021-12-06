import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square
from models.tf_basemodel import tf_BasedModel

class CNN(tf_BasedModel):
    def __init__(
            self,
            meta_data,
            batch_sz,
            kernel_sizes=[1, 2, 3],
            filters=[2, 3, 4],
            hidden_size=100,
            num_directions=2,
            num_layers=1,
            window_size=None,
            use_attention=False,
            embedding_dim=16,
            model_save_path="./cnn_models",
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
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.num_labels = meta_data["num_labels"]
        self.num_layers = num_layers
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim

        filters1, filters2, filters3 = self.filters
        if isinstance(self.kernel_sizes, str):
            self.kernel_sizes = list(map(int, self.kernel_sizes.split()))
        kernel1, kernel2, kernel3 = self.kernel_sizes

        self.conv2a = tf.keras.layers.Conv2D(filters=filters1,
                                             kernel_size=kernel1,
                                             strides=self.embedding_dim // 2,
                                             input_shape=(self.vocab_size, self.embedding_dim)
                                             )
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters=filters2,
                                             kernel_size=kernel2,
                                             strides=self.embedding_dim // 2,
                                             padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters=filters3,
                                             strides=self.embedding_dim // 2,
                                             kernel_size=kernel3,
                                             padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.linear1 = tf.keras.layers.Dense(self.hidden_size, activation = 'relu')
        self.linear2 = tf.keras.layers.Dense(self.num_labels, activation='softmax')

    def call(self, input_dict):
        if self.label_type == "anomaly":
            y = tf.convert_to_tensor(input_dict["window_anomalies"])
        elif self.label_type == "next_log":
            y = tf.convert_to_tensor(input_dict["window_labels"])

        # y = tf.one_hot(y, self.num_labels)
        # y = tf.squeeze(y)

        x = tf.convert_to_tensor(input_dict["features"])
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        
        x = tf.expand_dims(x, axis=-1)
        conv1 = self.bn2a(self.conv2a(x))
        conv2 = self.bn2b(self.conv2b(conv1))
        conv3 = self.bn2c(self.conv2c(conv2))


        y_pred = self.linear2(self.linear1(conv3))
        y_pred = tf.squeeze(y_pred)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        loss = tf.reduce_sum(loss)

        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
